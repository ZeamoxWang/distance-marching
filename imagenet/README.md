# Distance Marching on ImageNet 256x256

Class-conditional ImageNet 256x256 generation in the latent space of the frozen
`stabilityai/sd-vae-ft-ema` VAE. The backbone is SiT (S/2, B/2, L/2, XL/2) and
a tiny conditional CNN head predicts the distance; the backbone receives the
predicted distance (divided by 100) in place of the time input. There is no time
input. Sampling marches Gaussian latents onto the data manifold with sphere
tracing or gradient descent and decodes them with the VAE.

```
imagenet/
├── train.py              # training entry (torchrun, DDP)
├── sample.py             # inference entry: generate N images -> samples.npz
├── evaluator.py          # ADM evaluation suite (FID / sFID / IS / precision / recall)
├── models.py             # SiT backbone + distance head (DistanceMarchingSiT, DM_models)
├── dm_losses.py          # one-step loss + directional eikonal loss
└── prepare_imagenet.py   # unpack ILSVRC2012_img_train.tar into an ImageFolder
```

## Setup

```bash
pip install -r requirements.txt
```

`tensorflow` is only needed by `evaluator.py`. The VAE is downloaded from the
Hugging Face hub on first use.

### Data

Download `ILSVRC2012_img_train.tar` from [image-net.org](https://image-net.org/)
and unpack it into an `ImageFolder` layout (one folder per class):

```bash
python prepare_imagenet.py --train-tar /path/to/ILSVRC2012_img_train.tar --output-dir /path/to/imagenet
# -> /path/to/imagenet/train/n01440764/*.JPEG ...
```

For FID, download the ADM reference batch
[`VIRTUAL_imagenet256_labeled.npz`](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz).

## Training

Paper configuration for B/2 (4 x A100 80GB, global batch size 256, 80 epochs,
AdamW lr 1e-4 without weight decay, EMA 0.9999, `lambda_OS = 1.0`,
`lambda_DE = 30.0`, `epsilon = c_0 = 100.0`, direction rescaling 2.78e-3, label
dropout 0.1, interpolation coefficient `t ~ U(0, 0.999)`):

```bash
cd imagenet
torchrun --nnodes=1 --nproc_per_node=4 train.py \
    --model DM-B/2 --data-path /path/to/imagenet/train \
    --global-batch-size 256 --epochs 80 --global-seed 0 \
    --lr 1e-4 --ema-decay 0.9999 --class-dropout-prob 0.1 \
    --rescale-energy 2.78e-3 --one-step-loss-weight 1.0 --eikonal-loss-weight 30.0 --c 100.0 \
    --num-workers 8 --log-every 100 --ckpt-every 50000 \
    --results-dir results
```

Checkpoints (`model`, `ema`, `opt`, `args`, `train_steps`, `epoch`) are written to
`results/<index>-DM-B-2/checkpoints/<step>.pt`; resume with `--ckpt`. The paper's
B/2 checkpoint is step 400,000 (epoch 79). Use `--model DM-S/2`, `DM-L/2` or
`DM-XL/2` for the other sizes. `--max-steps N` stops after `N` optimization
steps (smoke tests).

## Inference

`sample.py` generates `--num-fid-samples` images and writes
`<folder>/<timestamp>/samples.npz` (plus a 4x4 preview grid). Without `--ckpt`
the checkpoint is downloaded from Hugging Face
(`zimowang/Distance-Marching`, file `imagenet-b2-0400000-ema.pt`; it stores the
EMA weights and the training arguments).

Sphere tracing with classifier-free guidance 2.0 (paper: FID 6.59, mean of
seeds 0, 1, 2):

```bash
cd imagenet
torchrun --nnodes=1 --nproc_per_node=1 sample.py \
    --model DM-B/2 --sampler st --stepsize 0.022 --cfg-scale 2.0 \
    --num-sampling-steps 250 --num-fid-samples 50000 \
    --global-batch-size 256 --global-seed 0 \
    --folder samples_st --tmp-dir /tmp/dm_samples
```

Gradient descent (paper: FID 6.67):

```bash
torchrun --nnodes=1 --nproc_per_node=1 sample.py \
    --model DM-B/2 --sampler gd --stepsize 0.64 --cfg-scale 2.0 \
    --num-sampling-steps 250 --num-fid-samples 50000 \
    --global-batch-size 256 --global-seed 0 \
    --folder samples_gd --tmp-dir /tmp/dm_samples
```

Use `--nproc_per_node=N` to split the batch over N GPUs and `--ckpt path.pt`
for a local checkpoint. `--tmp-dir` holds the intermediate PNGs (50k images,
about 10 GB), so point it to a fast local disk. The generated images are also
kept there. Note that `--num-sampling-steps 250` applies 249 marching updates,
matching the original implementation used for the paper.

FID, sFID, Inception Score, precision and recall against the reference batch:

```bash
python evaluator.py VIRTUAL_imagenet256_labeled.npz samples_st/<timestamp>/samples.npz
```

### Reference numbers (B/2, 50k samples, CFG 2.0, 250 steps)

| Sampler | Step size | seed 0 | seed 1 | seed 2 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| sphere tracing (`st`) | 0.022 | 6.562 | 6.597 | 6.611 | 6.59 |
| gradient descent (`gd`) | 0.64 | 6.615 | 6.688 | 6.709 | 6.67 |

Seed 0 was generated with `--global-batch-size 256`, seeds 1 and 2 with 128.
Running the seed-0 commands above from this repository (one A100-80GB, batch
256, about 5.7 h per 50k samples) gives `st` FID 6.5748 / sFID 5.41 / IS 187.8 /
precision 0.848 / recall 0.408 and `gd` FID 6.6276 / sFID 6.39 / IS 188.1 /
precision 0.844 / recall 0.410. The B/2 training command reproduces the original
loss curve over the first 3000 steps within 2% (step 3000: 12.89 vs 12.91) at
2.19 steps/s on 4 A100-80GB.
Paper step sizes for the other sizes: `st` 0.022 for all, `gd` 0.64 (S/2, B/2,
L/2) and 0.68 (XL/2).
