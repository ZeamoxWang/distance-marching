# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Distance Marching training loop (one-step loss + directional eikonal loss).
import argparse
import gc
import logging
import math
from typing import Iterable

import torch
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training import distributed_mode
from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    """Skewed interpolation-coefficient sampling from EDM (https://arxiv.org/abs/2206.00364)."""
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    # time = 0, pure noise; time = 1, pure data
    return time


def linear_interpolation(t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """x_t = (1 - t) * x_0 + t * x_1 (the CondOT path of the flow_matching library)."""
    t = t.reshape(-1, *([1] * (x_1.dim() - 1)))
    return (1 - t) * x_0 + t * x_1


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)
    batch_os_loss = MeanMetric().to(device, non_blocking=True)
    epoch_os_loss = MeanMetric().to(device, non_blocking=True)
    batch_de_loss = MeanMetric().to(device, non_blocking=True)
    epoch_de_loss = MeanMetric().to(device, non_blocking=True)
    batch_ref_mape = MeanMetric().to(device, non_blocking=True)
    epoch_ref_mape = MeanMetric().to(device, non_blocking=True)
    batch_real_mape = MeanMetric().to(device, non_blocking=True)
    epoch_real_mape = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            batch_os_loss.reset()
            batch_de_loss.reset()
            batch_ref_mape.reset()
            batch_real_mape.reset()
            if data_iter_step > 0 and args.test_run:
                break

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"label": labels}

        # Scaling to [-1, 1] from [0, 1]
        samples = samples * 2.0 - 1.0
        noise = torch.randn_like(samples).to(device)

        if args.skewed_timesteps:
            t = skewed_timestep_sample(samples.shape[0], device=device)
        else:
            t = torch.rand(samples.shape[0]).to(device)

        x_t = linear_interpolation(t=t, x_0=noise, x_1=samples)

        with torch.amp.autocast("cuda", enabled=not args.close_amp):

            grad, dist = model(x=x_t, timesteps=t, extra=conditioning, return_dist=True)
            onestep = -grad * dist.reshape(x_t.shape[0], 1, 1, 1)
            onestep_target = x_t + onestep

            onestep_norm = onestep.reshape(x_t.shape[0], -1).norm(dim=1, keepdim=True)

            # grad is pointing towards the noise
            data_to_x_t = x_t - samples
            norm_data_to_x_t = data_to_x_t.reshape(x_t.shape[0], -1).norm(
                dim=1, keepdim=True
            )

            smooth_norm = torch.sqrt(norm_data_to_x_t**2 + args.c)
            normalized_data_to_x_t = data_to_x_t / smooth_norm.reshape(
                x_t.shape[0], 1, 1, 1
            )

            # Directional eikonal loss
            DE_loss = (
                ((grad - normalized_data_to_x_t) ** 2)
                .reshape(x_t.shape[0], -1)
                .sum(dim=1, keepdim=False)
            )
            DE_loss = DE_loss.mean()

            # One-step loss with weight 1 / (||x - s||^2 + eps)
            if args.OS_loss_ablation:
                OS_loss = (
                    ((onestep_target - samples) ** 2)
                    .reshape(x_t.shape[0], -1)
                    .sum(dim=1, keepdim=False)
                )
            else:
                OS_loss = ((onestep_target - samples) ** 2).reshape(
                    x_t.shape[0], -1
                ).sum(dim=1, keepdim=False) / smooth_norm.squeeze() ** 2

            OS_loss = OS_loss.mean()

            # Only for monitoring: MAPE between the predicted distance and the
            # smoothed ground-truth distance, and between the one-step length
            # and the true distance.
            ref_MAPE = (dist / smooth_norm - 1).abs().mean()
            real_MAPE = (onestep_norm / norm_data_to_x_t - 1).abs().mean()

            loss = args.w_de * DE_loss + args.w_os * OS_loss

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)
        batch_os_loss.update(OS_loss)
        epoch_os_loss.update(OS_loss)
        batch_de_loss.update(DE_loss)
        epoch_de_loss.update(DE_loss)
        batch_ref_mape.update(ref_MAPE)
        epoch_ref_mape.update(ref_MAPE)
        batch_real_mape.update(real_MAPE)
        epoch_real_mape.update(real_MAPE)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            clip_grad=args.clip_grad,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            # Compute metrics on all processes to keep execution paths consistent
            loss_val = batch_loss.compute()
            os_loss_val = batch_os_loss.compute()
            de_loss_val = batch_de_loss.compute()
            ref_mape_val = batch_ref_mape.compute()
            real_mape_val = batch_real_mape.compute()
            # Only log on main process to avoid duplicate output
            if distributed_mode.is_main_process():
                logger.info(
                    f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: "
                    f"loss = {loss_val:.4f}, "
                    f"OS_loss = {os_loss_val:.4f}, "
                    f"DE_loss = {de_loss_val:.4f}, "
                    f"ref_MAPE = {ref_mape_val:.4f}, "
                    f"real_MAPE = {real_mape_val:.4f}, "
                    f"lr = {lr:.6f}"
                )
    lr_schedule.step()
    return {
        "loss": float(epoch_loss.compute().detach().cpu()),
        "OS_loss": float(epoch_os_loss.compute().detach().cpu()),
        "DE_loss": float(epoch_de_loss.compute().detach().cpu()),
        "ref_MAPE": float(epoch_ref_mape.compute().detach().cpu()),
        "real_MAPE": float(epoch_real_mape.compute().detach().cpu()),
    }
