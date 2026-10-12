# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Train Distance Marching (SiT backbone + distance head) on ImageNet 256x256 in
the latent space of the frozen SD-VAE, with PyTorch DDP.
"""
import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from dm_losses import training_losses
from models import DM_models
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

# TF32 makes A100 training a lot faster
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    """Logger that writes to a log file and stdout (rank 0 only)."""
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def load_checkpoint(path):
    assert os.path.isfile(path), f"Could not find checkpoint at {path}"
    # checkpoints store the argparse.Namespace of the run
    return torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    n_gpus = torch.cuda.device_count()

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = int(os.environ["LOCAL_RANK"])
    print(f"Found {n_gpus} GPUs, trying to use device index {device}")
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DM-B/2 --> DM-B-2
        experiment_name = f"{experiment_index:03d}-{model_string_name}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Arguments: {vars(args)}")
    else:
        logger = create_logger(None)

    # Load checkpoint (before creating the model) when resuming
    state_dict = None
    resume_step = 0
    resume_epoch = None
    if args.ckpt is not None:
        state_dict = load_checkpoint(args.ckpt)
        if "train_steps" in state_dict:
            resume_step = state_dict["train_steps"]
            logger.info(f"Resuming from step {resume_step} (loaded from checkpoint)")
        if "epoch" in state_dict:
            resume_epoch = state_dict["epoch"]
            logger.info(f"Resuming from epoch {resume_epoch} (loaded from checkpoint)")
        if "args" in state_dict:
            ckpt_args = state_dict["args"]
            for key in ["rescale_energy", "noise_predictor_condition"]:
                if hasattr(ckpt_args, key):
                    setattr(args, key, getattr(ckpt_args, key))
                    logger.info(f"Loaded {key} from checkpoint: {getattr(args, key)}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    logger.info(f"Creating model: {args.model}")
    model = DM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        class_dropout_prob=args.class_dropout_prob,
        rescale_energy=args.rescale_energy,
        noise_predictor_condition=args.noise_predictor_condition,
    ).to(device)

    ema = deepcopy(model).to(device)  # EMA of the model, used for sampling
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    if state_dict is not None:
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"])
            ema.load_state_dict(state_dict["ema"])
            if "opt" in state_dict:
                opt.load_state_dict(state_dict["opt"])
        else:
            model.load_state_dict(state_dict)
            ema.load_state_dict(state_dict)
    requires_grad(ema, False)
    model = DDP(model, device_ids=[device])
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # enables label dropout for classifier-free guidance
    ema.eval()

    steps_per_epoch = len(loader)
    if resume_epoch is not None:
        start_epoch = resume_epoch
    else:
        start_epoch = resume_step // steps_per_epoch if resume_step > 0 else 0
    final_epoch = args.epochs
    start_step_in_epoch = resume_step % steps_per_epoch
    logger.info(
        f"Training from epoch {start_epoch} to {final_epoch} "
        f"({steps_per_epoch} steps per epoch, lambda_OS={args.one_step_loss_weight}, "
        f"lambda_DE={args.eikonal_loss_weight}, c={args.c})"
    )

    train_steps = resume_step
    log_steps = 0
    running_loss = 0
    running_metrics = {}
    start_time = time()

    for epoch in range(start_epoch, final_epoch):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for step_in_epoch, (x, y) in enumerate(loader):
            # Skip already trained steps when resuming
            if epoch == start_epoch and step_in_epoch < start_step_in_epoch:
                continue
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            loss_dict = training_losses(
                model,
                x,
                y,
                one_step_loss_weight=args.one_step_loss_weight,
                eikonal_loss_weight=args.eikonal_loss_weight,
                c=args.c,
            )
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module, decay=args.ema_decay)

            running_loss += loss.item()
            for key in loss_dict:
                if key != "loss":
                    running_metrics[key] = running_metrics.get(key, 0) + loss_dict[key].item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_metrics = {}
                for key, value in running_metrics.items():
                    avg_metric = torch.tensor(value / log_steps, device=device)
                    dist.all_reduce(avg_metric, op=dist.ReduceOp.SUM)
                    avg_metrics[key] = avg_metric.item() / dist.get_world_size()
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if metrics_str:
                    logger.info(f"                         {metrics_str}")
                running_loss = 0
                running_metrics = {}
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path} (epoch={epoch}, train_steps={train_steps})")
                dist.barrier()
            if args.max_steps is not None and train_steps >= args.max_steps:
                break
        if args.max_steps is not None and train_steps >= args.max_steps:
            logger.info(f"Reached --max_steps={args.max_steps}, stopping.")
            break

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="ImageFolder root of the ImageNet train split.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DM_models.keys()), default="DM-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=None, help="Stop after this many optimization steps (for tests).")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1, help="Label dropout for classifier-free guidance.")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--rescale-energy", type=float, default=2.78e-3, help="Rescaling of the predicted direction (paper: 2.78e-3).")
    parser.add_argument("--one-step-loss-weight", type=float, default=1.0, help="lambda_OS")
    parser.add_argument("--eikonal-loss-weight", type=float, default=30.0, help="lambda_DE")
    parser.add_argument("--c", type=float, default=100.0, help="epsilon in the one-step loss and c_0 in the eikonal loss")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to resume from.")
    args = parser.parse_args()
    # The distance head is always conditioned on the class label (paper setting);
    # stored in the checkpoint args so that sample.py rebuilds the same model.
    args.noise_predictor_condition = True
    main(args)
