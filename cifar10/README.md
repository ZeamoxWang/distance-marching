# Distance Marching on CIFAR-10

Unconditional CIFAR-10 generation with Distance Marching: a UNet (the CIFAR-10
preset of the `flow_matching` codebase) predicts the direction `v_theta(x)` and a
tiny CNN head predicts the distance `u_theta(x)`. There is no time input. Sampling
marches Gaussian noise onto the data manifold with sphere tracing
(`x <- x - eta * u_theta(x) * v_theta(x)`) or gradient descent
(`x <- x - eta * v_theta(x)`).

```
cifar10/
├── train.py              # training entry (torchrun-compatible)
├── sample.py             # inference entry: generate 50k images + FID from a checkpoint
├── args.py               # command-line arguments
├── models/               # UNet, distance head (DistanceMarching), EMA
├── training/             # losses (train_loop.py), evaluation/FID (eval_loop.py), helpers
└── dm/solver.py          # sphere tracing / gradient descent samplers
```

## Setup

```bash
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically by torchvision into `--data_path` (default
`./data`).

## Training

The paper configuration (4 x A100 80GB, batch size 128 per GPU, 3000 epochs,
AdamW lr 1e-4, `lambda_OS = 0.1`, `lambda_DE = 1.0`, `epsilon = c_0 = 4.0`,
output rescaling 120, EMA 0.999). A checkpoint and a quick 1024-sample FID are
produced every `--eval_frequency` epochs.

```bash
cd cifar10
torchrun --nproc_per_node=4 train.py \
    --data_path ./data \
    --batch_size 128 --accum_iter 1 --epochs 3000 \
    --lr 1e-4 --optimizer_betas 0.9 0.95 \
    --class_drop_prob 1.0 \
    --use_ema --ema_decay 0.999 \
    --skewed_timesteps \
    --max_period 1000000 --unified_scale 120.0 \
    --c 4.0 --w_os 0.1 --w_de 1.0 \
    --nfe 250 \
    --dm_method st --dm_step_size 0.031 \
    --compute_fid --fid_samples 1024 --eval_frequency 10 \
    --timestamp --output_dir ./output_dir
```

Checkpoints are written to `<output_dir>/<timestamp>/checkpoint-<epoch>.pth`
(plus `checkpoint.pth` for the latest one) and the per-epoch losses/FID to
`log.txt`. Resume with `--resume <checkpoint>.pth`. The paper's checkpoint is
`checkpoint-1809.pth` (epoch index 1809, i.e. 1810 completed epochs).

Smoke test (one training batch + one evaluation batch on a single GPU):

```bash
python train.py --test_run --use_ema --max_period 1000000 \
    --unified_scale 120.0 --c 4.0 --skewed_timesteps --nfe 250 \
    --dm_method st --dm_step_size 0.031 --compute_fid --fid_samples 128 \
    --batch_size 128 --output_dir ./output_smoke
```

## Inference

`sample.py` loads a checkpoint, generates `--fid_samples` images with the chosen
sampler and computes FID against the CIFAR-10 training set. Without `--ckpt`
the checkpoint is downloaded from Hugging Face
(`zimowang/Distance-Marching`, file `cifar10-checkpoint-1809.pth`).

Sphere tracing (paper: FID 2.23 with 50k samples, 250 steps, seed 0):

```bash
cd cifar10
python sample.py \
    --use_ema --max_period 1000000 --unified_scale 120.0 \
    --nfe 250 \
    --dm_method st --dm_step_size 0.031 \
    --compute_fid --fid_samples 50000 --save_fid_samples \
    --batch_size 128 --seed 0 \
    --timestamp --output_dir ./samples_st
```

Gradient descent (paper: FID 2.54):

```bash
python sample.py \
    --use_ema --max_period 1000000 --unified_scale 120.0 \
    --nfe 250 \
    --dm_method gd --dm_step_size 0.46 \
    --compute_fid --fid_samples 50000 --save_fid_samples \
    --batch_size 128 --seed 0 \
    --timestamp --output_dir ./samples_gd
```

Multi-GPU sampling splits `--fid_samples` over the processes:
`torchrun --nproc_per_node=4 sample.py ...`. Use `--ckpt path/to/checkpoint-1809.pth`
for a local checkpoint. The generated images are saved as a uint8 array in
`<output_dir>/fid_samples.npy`, a preview grid in `snapshots/`, and the FID in
`log.txt`.

### Reference numbers

| Checkpoint | Sampler | Steps | Step size | FID (50k, seed 0) |
| --- | --- | ---: | ---: | ---: |
| `cifar10-checkpoint-1809.pth` | sphere tracing (`st`) | 250 | 0.031 | 2.23 |
| `cifar10-checkpoint-1809.pth` | gradient descent (`gd`) | 250 | 0.46 | 2.54 |

Running the two commands above from this repository (seed 0, one A100-80GB,
batch 128, about 2.5 h each) gives FID 2.2439 (`st`) and 2.5511 (`gd`); the
original runs gave 2.2342 and 2.5456 with the same seed. The paper's seed
standard deviations are 0.021 (`st`) and 0.031 (`gd`). The 30-epoch training
command above reproduces the original per-epoch losses to three decimals
(epoch 29: loss 0.1411, 1024-sample FID 70.8 vs 71.8).

FID is computed with `torchmetrics.image.fid.FrechetInceptionDistance(normalize=True)`
(Inception features from `torch-fidelity`) between the generated images and the
50,000 CIFAR-10 training images streamed through the training transform.

### Other options

- `--init_sigma s` / `--init_uniform r`: start sampling from `N(0, s^2 I)` or
  `Uniform(-r, r)` instead of `N(0, I)` (train/test mismatch study).
- `--OS_loss_ablation` (training): one-step loss without the
  inverse-squared-distance weight, the ablation in the paper.
- The defaults of `train.py` / `sample.py` are the paper settings, so the
  model flags above can be omitted; `--dm_step_size` defaults to 0.031 for `st`
  and 0.46 for `gd`.
