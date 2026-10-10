# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Generation + FID evaluation with the Distance Marching samplers.
import gc
import logging
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from dm.solver import build_sigma_schedule, DistanceMarchingSolver
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from training import distributed_mode
from training.edm_time_discretization import get_time_discretization

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def get_sampling_time_grid(args: Namespace, device: torch.device) -> torch.Tensor:
    """Time grid whose length - 1 is the number of marching steps.

    With ``--edm_schedule`` the EDM discretization with ``args.nfe`` points is used
    (``args.nfe`` steps); otherwise a single step [0, 1].
    """
    if args.edm_schedule:
        return get_time_discretization(nfes=args.nfe)
    return torch.tensor([0.0, 1.0], device=device)


def sample_initial_state(shape, device: torch.device, args: Namespace) -> torch.Tensor:
    """Initial state of the sampler.

    Default: standard Gaussian N(0, I). ``--init_sigma s`` draws N(0, s^2 I) and
    ``--init_uniform r`` draws Uniform(-r, r) per entry (train/test mismatch study).
    """
    init_uniform = getattr(args, "init_uniform", None)
    if init_uniform is not None:
        r = abs(float(init_uniform))
        return (torch.rand(shape, dtype=torch.float32, device=device) * 2.0 - 1.0) * r
    sigma = abs(float(getattr(args, "init_sigma", 1.0)))
    return torch.randn(shape, dtype=torch.float32, device=device) * sigma


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
):
    """Generate ``fid_samples`` images (per process) and compute FID against the
    images streamed from ``data_loader``.

    All processes generate images; the FID statistics are synchronized across
    processes by torchmetrics. Only the main process writes snapshots, the
    ``fid_samples.npy`` memmap and the marching statistics.
    """
    gc.collect()

    model.train(False)

    solver = DistanceMarchingSolver(
        velocity_model=model,
        output_dir=args.output_dir if hasattr(args, "output_dir") else None,
        is_main_process=distributed_mode.is_main_process(),
    )

    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )

    ula_schedule = None
    if args.dm_method in ["gd_ula", "st_ula"] and getattr(args, "dm_ula_schedule", None) is not None:
        raw = {float(k): float(v) for k, v in args.dm_ula_schedule.items()}
        ula_schedule = build_sigma_schedule(raw)

    num_synthetic = 0
    fid_samples_memmap = None
    snapshots_saved = False
    if args.output_dir and distributed_mode.is_main_process():
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        fid_metric.update(samples, real=True)

        if num_synthetic < fid_samples:
            x_0 = sample_initial_state(samples.shape, device=device, args=args)
            time_grid = get_sampling_time_grid(args, device)

            synthetic_samples = solver.sample(
                x_init=x_0,
                step_size=args.dm_step_size,
                method=args.dm_method,
                criteria=args.dm_criteria,
                threshold=args.dm_threshold,
                time_grid=time_grid,
                return_intermediates=False,
                ula_schedule=ula_schedule,
                label=labels,
                cfg_scale=args.cfg_scale,
            )

            # Scaling to [0, 1] from [-1, 1], then quantize to uint8 levels
            synthetic_samples = torch.clamp(
                synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
            )
            synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0

            if distributed_mode.is_main_process():
                logger.info(f"{samples.shape[0]} samples generated.")
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            fid_metric.update(synthetic_samples, real=False)
            num_synthetic += synthetic_samples.shape[0]

            if distributed_mode.is_main_process():
                if not snapshots_saved and args.output_dir:
                    save_image(
                        synthetic_samples,
                        fp=Path(args.output_dir)
                        / "snapshots"
                        / f"{epoch}_{data_iter_step}.png",
                    )
                    snapshots_saved = True

                if args.save_fid_samples and args.output_dir:
                    images_np = (
                        (synthetic_samples * 255.0)
                        .clip(0, 255)
                        .to(torch.uint8)
                        .permute(0, 2, 3, 1)
                        .cpu()
                        .numpy()
                    )
                    if fid_samples_memmap is None:
                        fid_samples_memmap_path = Path(args.output_dir) / "fid_samples.npy"
                        fid_samples_memmap_path.parent.mkdir(parents=True, exist_ok=True)
                        memmap_shape = (fid_samples,) + images_np.shape[1:]
                        fid_samples_memmap = np.lib.format.open_memmap(
                            fid_samples_memmap_path,
                            mode="w+",
                            dtype=np.uint8,
                            shape=memmap_shape,
                        )
                    start_idx = num_synthetic - images_np.shape[0]
                    end_idx = num_synthetic
                    fid_samples_memmap[start_idx:end_idx] = images_np
                    fid_samples_memmap.flush()

        if not args.compute_fid:
            return {}

        if data_iter_step % PRINT_FREQUENCY == 0:
            # Sync fid metric to ensure that the processes dont deviate much.
            gc.collect()
            running_fid = fid_metric.compute()
            if distributed_mode.is_main_process():
                # Cumulative FID over all samples processed so far (not per-batch).
                logger.info(
                    f"Evaluating [{data_iter_step}/{len(data_loader)}] samples generated [{num_synthetic}/{fid_samples}] cumulative fid {running_fid:.4f}"
                )

        if args.test_run:
            break

    if fid_samples_memmap is not None:
        fid_samples_memmap.flush()

    return {"fid": float(fid_metric.compute().detach().cpu())}
