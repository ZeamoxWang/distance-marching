# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample ImageNet 256x256 images from a Distance Marching checkpoint with
sphere tracing or gradient descent (PyTorch DDP), and write a .npz file
for the ADM evaluation suite (evaluator.py).
"""
import argparse
import math
import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from models import DM_models
from PIL import Image
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def create_npz_from_sample_folder(sample_dir, num=50_000, out_npz_path=None):
    """Builds a single .npz file from a folder of .png samples."""
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = out_npz_path or f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def resolve_checkpoint(args, rank):
    """Local path from --ckpt, otherwise download --hf-file from --hf-repo."""
    if args.ckpt is not None:
        return args.ckpt
    from huggingface_hub import hf_hub_download

    if rank == 0:
        print(f"Downloading {args.hf_file} from {args.hf_repo} ...")
        hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file)
    dist.barrier()
    return hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file)


def main(args):
    assert torch.cuda.is_available(), "Sampling requires at least one GPU."
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, device={device}, world_size={dist.get_world_size()}.")

    timestamp_container = [datetime.now().strftime("%Y%m%d-%H%M%S") if rank == 0 else None]
    dist.broadcast_object_list(timestamp_container, src=0)
    timestamp_suffix = timestamp_container[0]
    args.folder = os.path.join(args.folder, timestamp_suffix)
    # local scratch folder for the intermediate PNGs
    args.tmp_folder = os.path.join(args.tmp_dir, timestamp_suffix)
    if rank == 0:
        os.makedirs(args.tmp_folder, exist_ok=True)
        os.makedirs(args.folder, exist_ok=True)
    dist.barrier()

    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    # Load the checkpoint and its training configuration
    ckpt_path = resolve_checkpoint(args, rank)
    state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
    if "args" in state_dict:
        ckpt_args = state_dict["args"]
        for key in ["rescale_energy", "noise_predictor_condition"]:
            if hasattr(ckpt_args, key):
                setattr(args, key, getattr(ckpt_args, key))
                if rank == 0:
                    print(f"Loaded {key} from checkpoint: {getattr(args, key)}")

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE decoder)."
    latent_size = args.image_size // 8
    print(f"Creating model: {args.model}")
    model = DM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=False,
        ebm="none",
        rescale_energy=args.rescale_energy,
        noise_predictor_condition=args.noise_predictor_condition,
    ).to(device)
    ema = deepcopy(model).to(device)
    if "ema" in state_dict:
        ema.load_state_dict(state_dict["ema"])
    elif "model" in state_dict:
        ema.load_state_dict(state_dict["model"])
    else:
        ema.load_state_dict(state_dict)
    del model
    requires_grad(ema, False)
    ema.eval()
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    print(f"Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    use_cfg = args.cfg_scale >= 0.0
    model_fn = ema.forward_with_cfg if use_cfg else ema.forward

    if rank == 0:
        hyperparams_path = os.path.join(args.folder, "hyperparameters.txt")
        with open(hyperparams_path, "w") as hyper_file:
            for key in sorted(vars(args).keys()):
                hyper_file.write(f"{key}: {vars(args)[key]}\n")
    dist.barrier()

    # Sample a bit more than needed so that every batch is full, then discard the extra samples
    total_samples = int(math.ceil(args.num_fid_samples / args.global_batch_size) * args.global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    n = int(args.global_batch_size // dist.get_world_size())
    iterations = int(total_samples // args.global_batch_size)
    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
    total = 0
    collected_samples = []

    for _ in pbar:
        with torch.no_grad():
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
            y = torch.randint(0, args.num_classes, (n,), device=device)
            t = torch.ones((n,)).to(z).to(device)  # placeholder, ignored by the model
            if use_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([args.num_classes] * n, device=device)
                y = torch.cat([y, y_null], 0)
                t = torch.cat([t, t], 0)
            xt = z
            # The original implementation applies num_sampling_steps - 1 updates.
            for _step in range(args.num_sampling_steps - 1):
                if args.sampler == "gd":
                    if use_cfg:
                        cond_out, cond_energy, uncond_out, uncond_energy = model_fn(
                            xt, t, y, args.cfg_scale, get_energy=True
                        )
                        out = uncond_out + args.cfg_scale * (cond_out - uncond_out)
                        out = torch.cat([out, out], 0)
                    else:
                        out, energy = model_fn(xt, t, y, get_energy=True)
                elif args.sampler == "st":
                    if use_cfg:
                        cond_out, cond_energy, uncond_out, uncond_energy = model_fn(
                            xt, t, y, args.cfg_scale, get_energy=True
                        )
                        uncond_final = uncond_out * uncond_energy.reshape(uncond_out.shape[0], 1, 1, 1)
                        cond_final = cond_out * cond_energy.reshape(cond_out.shape[0], 1, 1, 1)
                        out = uncond_final + args.cfg_scale * (cond_final - uncond_final)
                        out = torch.cat([out, out], 0)
                    else:
                        out, energy = model_fn(xt, t, y, get_energy=True)
                        out = out * energy.reshape(out.shape[0], 1, 1, 1)
                else:
                    raise ValueError(f"Unknown sampler {args.sampler}")
                xt = (xt + out * args.stepsize).detach()
            if use_cfg:
                xt, _ = xt.chunk(2, dim=0)
            samples = vae.decode(xt / 0.18215).sample
            samples = (
                torch.clamp(127.5 * samples + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{args.tmp_folder}/{index:06d}.png")
                if rank == 0 and len(collected_samples) < 16:
                    collected_samples.append(sample)
                    if len(collected_samples) == 16:
                        h, w = sample.shape[:2]
                        grid = np.zeros((h * 4, w * 4, 3), dtype=np.uint8)
                        for idx, img in enumerate(collected_samples):
                            row, col = idx // 4, idx % 4
                            grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = img
                        Image.fromarray(grid).save(f"{args.folder}/grid_4x4.png")
                        print(f"Saved 4x4 grid image to {args.folder}/grid_4x4.png")
        total += args.global_batch_size
        dist.barrier()

    if rank == 0:
        print("Creating .npz file")
        final_npz_path = os.path.join(args.folder, "samples.npz")
        create_npz_from_sample_folder(args.tmp_folder, args.num_fid_samples, out_npz_path=final_npz_path)
        print(f"Saved npz file to {final_npz_path}")
        print("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DM_models.keys()), default="DM-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Classifier-free guidance scale; negative disables CFG.")
    parser.add_argument("--ckpt", type=str, default=None, help="Local checkpoint path. If omitted, download from Hugging Face.")
    parser.add_argument("--hf-repo", type=str, default="zimowang/Distance-Marching")
    parser.add_argument("--hf-file", type=str, default="imagenet-b2-0400000-ema.pt")
    parser.add_argument("--sampler", type=str, default="st", choices=["gd", "st"], help="'st' sphere tracing, 'gd' gradient descent")
    parser.add_argument("--stepsize", type=float, default=0.022, help="Step size eta (paper: 0.022 for st, 0.64 for gd on B/2).")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--folder", type=str, default="samples", help="Output folder (a timestamped subfolder is created).")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/dm_samples", help="Local scratch folder for intermediate PNGs.")
    parser.add_argument("--rescale-energy", type=float, default=2.78e-3, help="Overridden by the checkpoint's training args.")
    parser.add_argument("--noise-predictor-condition", type=lambda x: (str(x).lower() in ["true", "1", "yes"]), default=True)
    args = parser.parse_args()
    main(args)
