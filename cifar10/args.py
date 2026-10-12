# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Command-line arguments shared by train.py and sample.py.
import argparse


def add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Keep an EMA copy of the weights and use it for evaluation (always on; the "
        "released checkpoint is EMA-wrapped). Kept for backward compatibility.",
    )
    group.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay.")
    group.add_argument(
        "--unified_scale",
        default=120.0,
        type=float,
        help="The UNet output is divided by this value to obtain the direction v_theta.",
    )
    group.add_argument(
        "--max_period",
        default=1000000.0,
        type=float,
        help="Maximum period of the sinusoidal embedding fed by the distance head.",
    )


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--dm_method",
        default="st",
        type=str,
        choices=["st", "gd"],
        help="'st' sphere tracing, 'gd' gradient descent.",
    )
    group.add_argument(
        "--dm_step_size",
        default=None,
        type=float,
        help="Step size eta. Defaults to the paper values: 0.031 for st, 0.46 for gd.",
    )
    group.add_argument(
        "--nfe",
        default=250,
        type=int,
        help="Number of marching steps.",
    )
    group.add_argument(
        "--init_sigma",
        default=1.0,
        type=float,
        help="Initial state N(0, sigma^2 I). 1.0 is the training noise distribution.",
    )
    group.add_argument(
        "--init_uniform",
        default=None,
        type=float,
        help="If set, initial state Uniform(-r, r) per entry instead of Gaussian "
        "(train/test mismatch experiment).",
    )


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("evaluation")
    group.add_argument(
        "--compute_fid",
        action="store_true",
        help="Compute FID between generated images and the CIFAR-10 training set.",
    )
    group.add_argument(
        "--fid_samples",
        default=50000,
        type=int,
        help="Number of generated images used for FID (total over all processes).",
    )
    group.add_argument(
        "--save_fid_samples",
        action="store_true",
        help="Save the generated uint8 images to <output_dir>/fid_samples.npy.",
    )


def add_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data")
    group.add_argument(
        "--data_path",
        default="./data",
        type=str,
        help="Root folder for torchvision CIFAR-10 (downloaded automatically).",
    )
    group.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * #GPUs).",
    )
    group.add_argument("--num_workers", default=10, type=int)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("run")
    group.add_argument(
        "--output_dir",
        default="./output_dir",
        help="Where to write logs, checkpoints, snapshots and samples.",
    )
    group.add_argument(
        "--timestamp",
        action="store_true",
        help="Append YYYYMMDD_HHMMSS to --output_dir.",
    )
    group.add_argument("--seed", default=0, type=int)
    group.add_argument(
        "--test_run",
        action="store_true",
        help="Run a single batch of training and evaluation (smoke test).",
    )


DEFAULT_STEP_SIZE = {"st": 0.031, "gd": 0.46}


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in values that depend on other arguments."""
    if args.dm_step_size is None:
        args.dm_step_size = DEFAULT_STEP_SIZE[args.dm_method]
    args.start_epoch = 0  # overwritten by load_model() when resuming
    return args


def get_train_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Distance Marching on CIFAR-10: training")
    parser.add_argument("--epochs", default=3000, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Gradient accumulation iterations.",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (constant).")
    parser.add_argument(
        "--optimizer_betas",
        nargs="+",
        type=float,
        default=[0.9, 0.95],
        help="AdamW betas.",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        help="Gradient clipping max norm. If None, no gradient clipping is applied.",
    )
    parser.add_argument(
        "--class_drop_prob",
        type=float,
        default=1.0,
        help="Probability to drop the class label (1.0 = unconditional, the paper setting).",
    )
    parser.add_argument(
        "--skewed_timesteps",
        action="store_true",
        default=True,
        help="Sample the interpolation coefficient from the EDM log-normal distribution "
        "(always on, the paper setting). Kept for backward compatibility.",
    )
    parser.add_argument(
        "--close_amp",
        action="store_true",
        help="Disable mixed precision (AMP) during training.",
    )
    # Distance Marching losses
    parser.add_argument(
        "--c",
        default=4.0,
        type=float,
        help="Smoothing constant: epsilon in the one-step loss and c_0 in the directional "
        "eikonal loss.",
    )
    parser.add_argument("--w_de", default=1.0, type=float, help="Weight of the directional eikonal loss.")
    parser.add_argument("--w_os", default=0.1, type=float, help="Weight of the one-step loss.")
    parser.add_argument(
        "--OS_loss_ablation",
        action="store_true",
        help="Ablation: one-step loss without the inverse-squared-distance reweighting.",
    )
    parser.add_argument(
        "--eval_frequency",
        default=10,
        type=int,
        help="Run FID evaluation (and save a checkpoint) every N epochs. -1 disables it.",
    )
    parser.add_argument("--resume", default="", help="Checkpoint to resume training from.")
    add_model_args(parser)
    add_sampling_args(parser)
    add_eval_args(parser)
    add_data_args(parser)
    add_common_args(parser)
    return parser


def get_sample_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Distance Marching on CIFAR-10: sampling / FID")
    parser.add_argument(
        "--ckpt",
        default="",
        type=str,
        help="Local checkpoint path (checkpoint-<epoch>.pth).",
    )
    parser.add_argument(
        "--hf_repo",
        default="zimowang/Distance-Marching",
        type=str,
        help="Hugging Face repo to download the checkpoint from when --ckpt is not given.",
    )
    parser.add_argument(
        "--hf_file",
        default="cifar10-checkpoint-1809.pth",
        type=str,
        help="File inside --hf_repo.",
    )
    add_model_args(parser)
    add_sampling_args(parser)
    add_eval_args(parser)
    add_data_args(parser)
    add_common_args(parser)
    return parser
