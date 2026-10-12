# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Generate CIFAR-10 images with a trained Distance Marching checkpoint and
# (optionally) compute FID. Mirrors the evaluation path of train.py.

import datetime
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from args import finalize_args, get_sample_args_parser
from train import build_dataloader, build_model, fid_samples_for_this_rank, setup_logging

from training import distributed_mode
from training.eval_loop import eval_model

logger = logging.getLogger(__name__)


def resolve_checkpoint(args) -> str:
    """Return a local checkpoint path, downloading from Hugging Face if needed."""
    if args.ckpt:
        return args.ckpt
    from huggingface_hub import hf_hub_download

    # Main process downloads first; the other ranks then hit the local cache.
    if distributed_mode.is_main_process():
        logger.info(f"Downloading {args.hf_file} from {args.hf_repo} ...")
        hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file)
    if args.distributed:
        torch.distributed.barrier()
    return hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file)


def main(args):
    setup_logging()
    distributed_mode.init_distributed_mode(args)

    if distributed_mode.is_main_process():
        logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        logger.info("{}".format(args).replace(", ", ",\n"))
        with open(Path(args.output_dir) / "args.json", "w") as f:
            json.dump(vars(args), f)

    device = torch.device("cuda")

    # fix the seed for reproducibility (per rank, as in train.py)
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    data_loader_train = build_dataloader(args)
    model, model_without_ddp = build_model(args, device)

    ckpt_path = resolve_checkpoint(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_without_ddp.load_state_dict(checkpoint["model"])
    if distributed_mode.is_main_process():
        logger.info(f"Loaded checkpoint {ckpt_path} (epoch {checkpoint.get('epoch')})")

    if args.distributed:
        data_loader_train.sampler.set_epoch(0)

    start_time = time.time()
    eval_stats = eval_model(
        model,
        data_loader_train,
        device,
        epoch=checkpoint.get("epoch", 0),
        fid_samples=fid_samples_for_this_rank(args.fid_samples),
        args=args,
    )
    if distributed_mode.is_main_process():
        log_stats = {"epoch": checkpoint.get("epoch"), "ckpt": ckpt_path}
        log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
        total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(f"Result: {log_stats}")
        logger.info(f"Sampling time {total_time}")


if __name__ == "__main__":
    args = finalize_args(get_sample_args_parser().parse_args())
    if args.output_dir and args.timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path(args.output_dir) / timestamp)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
