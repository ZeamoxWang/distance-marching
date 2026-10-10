# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Command-line arguments shared by train.py and sample.py.
import argparse
import json


def add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument(
        "--use_ema",
        action="store_true",
        help="Keep an EMA copy of the weights and use it for evaluation.",
    )
    group.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay.")
    group.add_argument(
        "--separate_t",
        action="store_true",
        help="Predict the distance with a separate tiny CNN head (the paper setting) "
        "instead of differentiating a scalar field.",
    )
    group.add_argument(
        "--unified_scale",
        default=1.0,
        type=float,
        help="The UNet output is divided by this value to obtain the direction v_theta "
        "(paper: 120.0).",
    )
    group.add_argument(
        "--max_period",
        default=10000,
        type=float,
        help="Maximum period of the sinusoidal embedding fed by the distance head "
        "(paper: 1000000).",
    )


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--dm_method",
        default="gd",
        type=str,
        choices=["gd", "st", "gd_ula", "st_ula"],
        help="'gd' gradient descent, 'st' sphere tracing; '*_ula' adds annealed Langevin noise.",
    )
    group.add_argument(
        "--dm_step_size",
        default=None,
        type=float,
        required=False,
        help="Step size eta (paper: 0.031 for st, 0.46 for gd).",
    )
    group.add_argument(
        "--dm_criteria",
        default="fixed_steps",
        type=str,
        choices=["fixed_steps", "grad_norm", "dist_value", "step_norm"],
        help="Stopping criterion. 'fixed_steps' always takes --nfe steps.",
    )
    group.add_argument(
        "--dm_threshold",
        default=1e-5,
        type=float,
        help="Threshold for the adaptive stopping criteria.",
    )
    group.add_argument(
        "--dm_ula_schedule",
        default=None,
        type=json.loads,
        help='Sigma schedule for the *_ula methods as a JSON dict {progress: sigma}, '
        'e.g. \'{"0.0": 1.0, "0.3": 0.2, "0.9": 0.01, "1.0": 0.0}\'.',
    )
    group.add_argument(
        "--nfe",
        default=250,
        type=int,
        help="Number of marching steps (with --edm_schedule the EDM time grid has nfe points).",
    )
    group.add_argument(
        "--edm_schedule",
        action="store_true",
        help="Use the EDM time discretization for the sampling grid (only its length matters).",
    )
    group.add_argument(
        "--cfg_scale",
        default=0.0,
        type=float,
        help="Classifier-free guidance scale (CIFAR-10 models are unconditional: keep 0.0).",
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
    group.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in the DataLoader.",
    )
    group.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)


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
    group.add_argument("--device", default="cuda", help="Device to use.")
    group.add_argument("--seed", default=0, type=int)
    group.add_argument(
        "--test_run",
        action="store_true",
        help="Run a single batch of training and evaluation (smoke test).",
    )
    # distributed training parameters (torchrun sets the environment variables)
    group.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    group.add_argument("--local_rank", default=-1, type=int)
    group.add_argument("--dist_on_itp", action="store_true")
    group.add_argument("--dist_url", default="env://", help="url used to set up distributed training")


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
        help="Sample the interpolation coefficient from the EDM log-normal distribution.",
    )
    parser.add_argument(
        "--close_amp",
        action="store_true",
        help="Disable mixed precision (AMP) during training.",
    )
    # Distance Marching losses
    parser.add_argument(
        "--c",
        default=16.0,
        type=float,
        help="Smoothing constant: epsilon in the one-step loss and c_0 in the directional "
        "eikonal loss (paper: 4.0).",
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
        default=50,
        type=int,
        help="Run FID evaluation (and save a checkpoint) every N epochs. -1 disables it.",
    )
    parser.add_argument("--resume", default="", help="Checkpoint to resume training from.")
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="Start epoch (set automatically when resuming).",
    )
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
