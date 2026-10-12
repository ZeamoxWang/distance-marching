# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Train Distance Marching on CIFAR-10 (optionally with periodic FID evaluation).

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from args import finalize_args, get_train_args_parser
from models.model_configs import instantiate_model

from training import distributed_mode
from training.data_transform import get_train_transform
from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch

logger = logging.getLogger(__name__)


def build_dataloader(args):
    """CIFAR-10 training set (random horizontal flip) with a DistributedSampler."""
    transform_train = get_train_transform()
    # download the dataset on the main process only
    if distributed_mode.is_main_process():
        logger.info("Main process downloading CIFAR-10 dataset...")
        datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
        logger.info("Download complete. Waiting for filesystem sync...")
        time.sleep(2)
    if args.distributed:
        torch.distributed.barrier()
    dataset_train = datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=False,
        transform=transform_train,
    )
    if distributed_mode.is_main_process():
        logger.info(dataset_train)
        logger.info("Intializing DataLoader")

    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if distributed_mode.is_main_process():
        logger.info(str(sampler_train))
    return data_loader_train


def build_model(args, device):
    model = instantiate_model(
        architechture="cifar10",
        use_ema=args.use_ema,
        max_period=args.max_period,
        unified_scale=args.unified_scale,
        ema_decay=args.ema_decay,
    )
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    return model, model_without_ddp


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def fid_samples_for_this_rank(fid_samples: int) -> int:
    """Split the total number of FID samples over the processes."""
    num_tasks = distributed_mode.get_world_size()
    if distributed_mode.is_main_process():
        return fid_samples - (num_tasks - 1) * (fid_samples // num_tasks)
    return fid_samples // num_tasks


def main(args):
    setup_logging()
    distributed_mode.init_distributed_mode(args)

    if distributed_mode.is_main_process():
        logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        logger.info("{}".format(args).replace(", ", ",\n"))
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)

    device = torch.device("cuda")

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if distributed_mode.is_main_process():
        logger.info("Initializing Dataset: cifar10")
    data_loader_train = build_dataloader(args)

    if distributed_mode.is_main_process():
        logger.info("Initializing Model")
    model, model_without_ddp = build_model(args, device)

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )
    if distributed_mode.is_main_process():
        logger.info(f"Learning rate: {args.lr:.2e}")
        logger.info(f"Accumulate grad iterations: {args.accum_iter}")
        logger.info(f"Effective batch size: {eff_batch_size}")

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )
    lr_schedule = torch.optim.lr_scheduler.ConstantLR(
        optimizer, total_iters=args.epochs, factor=1.0
    )

    if distributed_mode.is_main_process():
        logger.info(f"Optimizer: {optimizer}")
        logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    if distributed_mode.is_main_process():
        logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            args=args,
        )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.test_run
        ):
            # Only save checkpoint on main process to avoid duplicate saves
            if distributed_mode.is_main_process():
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
            if args.distributed:
                torch.distributed.barrier()
                data_loader_train.sampler.set_epoch(0)
            eval_stats = eval_model(
                model,
                data_loader_train,
                device,
                epoch=epoch,
                fid_samples=fid_samples_for_this_rank(args.fid_samples),
                args=args,
            )
            log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run:
            break

    if distributed_mode.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = finalize_args(get_train_args_parser().parse_args())
    if args.output_dir and args.timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path(args.output_dir) / timestamp)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
