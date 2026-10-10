# Distance Marching for Generative Modeling

Official implementation of **Distance Marching**, a generative model that
explicitly learns a distance field of the data manifold and generates images
without any time input. The negative gradient of the field gives the denoising
direction and the predicted distance gives the step size, which yields two
deterministic samplers: **sphere tracing** and **gradient descent**.

The repository contains two self-contained codebases:

| Folder | Data | Backbone | Training | Inference |
| --- | --- | --- | --- | --- |
| [`cifar10/`](cifar10/) | CIFAR-10 32x32, unconditional | UNet (flow_matching CIFAR-10 preset) + distance head | `cifar10/train.py` | `cifar10/sample.py` |
| [`imagenet/`](imagenet/) | ImageNet 256x256, class-conditional, latent space of SD-VAE | SiT-S/B/L/XL-2 + distance head | `imagenet/train.py` | `imagenet/sample.py` |

Each folder has its own README with the exact commands, hyperparameters and
reference numbers. Pretrained checkpoints are hosted at
[`zimowang/Distance-Marching`](https://huggingface.co/zimowang/Distance-Marching)
on Hugging Face and are downloaded automatically by the inference scripts.

| Checkpoint | Sampler | Steps | FID (50k) |
| --- | --- | ---: | ---: |
| `cifar10-checkpoint-1809.pth` | sphere tracing, step 0.031 | 250 | 2.23 |
| `cifar10-checkpoint-1809.pth` | gradient descent, step 0.46 | 250 | 2.54 |
| `imagenet-b2-0400000-ema.pt` (SiT-B/2, CFG 2.0) | sphere tracing, step 0.022 | 250 | 6.59 |
| `imagenet-b2-0400000-ema.pt` (SiT-B/2, CFG 2.0) | gradient descent, step 0.64 | 250 | 6.67 |

## Method in one paragraph

Given a noisy point `x`, the model outputs a direction `v_theta(x)` and a
distance `u_theta(x)`. Training pairs `(x, s)` are built by linear interpolation
between Gaussian noise and a data sample `s`, without any re-matching. Two losses
are used: the **one-step loss** `w * || x - u_theta(x) v_theta(x) - s ||^2` with
the inverse-squared-distance weight `w = 1 / (||x - s||^2 + eps)`, which
emphasizes closer targets, and the **directional eikonal loss**
`|| v_theta(x) - (x - s) / sqrt(||x - s||^2 + c_0) ||^2`, which fixes the offset of
the learned field. At inference, sphere tracing iterates
`x <- x - eta * u_theta(x) * v_theta(x)` and gradient descent iterates
`x <- x - eta * v_theta(x)` for a fixed number of steps.

## Quick start

```bash
# CIFAR-10: generate 50k images with sphere tracing and compute FID
cd cifar10 && pip install -r requirements.txt
python sample.py --use_ema --separate_t --max_period 1000000 --unified_scale 120.0 \
    --edm_schedule --nfe 250 --dm_method st --dm_step_size 0.031 \
    --compute_fid --fid_samples 50000 --batch_size 128 --seed 0 --output_dir ./samples_st

# ImageNet-256 (SiT-B/2): generate 50k images with sphere tracing and compute FID
cd imagenet && pip install -r requirements.txt
torchrun --nproc_per_node=1 sample.py --model DM-B/2 --sampler st --stepsize 0.022 \
    --cfg-scale 2.0 --num-sampling-steps 250 --num-fid-samples 50000 --global-seed 0
```

## Repository layout

```
distance-marching/
├── cifar10/        # pixel-space CIFAR-10 code (training + inference)
├── imagenet/       # latent-space ImageNet code (training + inference)
├── execute/        # launch scripts and Kubernetes (Nautilus) manifests used for the paper runs
└── LICENSE
```

## Citation

```bibtex
@article{wang2026distancemarching,
  title   = {Distance Marching for Generative Modeling},
  author  = {Wang, Zimo and Mehta, Ishit and Lu, Haolin and Huang, Xunpeng and Sun, Chung-En and Yan, Ge and Weng, Tsui-Wei and Li, Tzu-Mao},
  year    = {2026}
}
```

## License

The CIFAR-10 code is derived from the `flow_matching` library and keeps its
CC-BY-NC license (see `LICENSE`); the UNet and the distributed/grad-scaler helpers
are MIT-licensed. The ImageNet code is derived from SiT and Equilibrium Matching and
keeps their licenses. The pretrained VAE `stabilityai/sd-vae-ft-ema` keeps its
upstream license.

## Acknowledgments

We thank Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky T. Q. Chen, David Lopez-Paz, Heli Ben-Hamu and Itai Gat for the [Flow Matching Guide and Code](https://github.com/facebookresearch/flow_matching), Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden and Saining Xie for [SiT](https://github.com/willisma/SiT), and Runqian Wang and Yilun Du for [Equilibrium Matching](https://github.com/raywang4/EqM); this codebase is built on their open-source implementations.
