"""Compare outputs of the same run made with two code versions.

Every comparison takes three runs: OLD_A and OLD_B (the reference code run
twice, which measures the GPU run-to-run noise) and NEW (the code under test).
The NEW-vs-OLD_A difference should be no larger than OLD_B-vs-OLD_A.

Usage:
  python3 compare.py images  OLD_A OLD_B NEW   # .npy (CIFAR fid_samples.npy) or .npz (arr_0)
  python3 compare.py ckpt    OLD_A OLD_B NEW   # torch checkpoints: every tensor in model/ema
  python3 compare.py cifarlog    OLD_A OLD_B NEW   # CIFAR log.txt (JSON lines)
  python3 compare.py imagenetlog OLD_A OLD_B NEW   # ImageNet results/*/log.txt
"""
import json
import re
import sys

import numpy as np


def load_images(path):
    if path.endswith(".npz"):
        return np.load(path)["arr_0"]
    return np.load(path)


def diff_arrays(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    d = np.abs(a - b)
    return {
        "identical": bool((d == 0).all()),
        "n_diff": int((d != 0).sum()),
        "n": int(d.size),
        "max_abs": float(d.max()) if d.size else 0.0,
        "mean_abs": float(d.mean()) if d.size else 0.0,
    }


def report(name, ref_noise, new_diff):
    print(f"  {name}")
    print(f"    OLD_B vs OLD_A: {ref_noise}")
    print(f"    NEW   vs OLD_A: {new_diff}")
    if new_diff["identical"]:
        verdict = "IDENTICAL"
    elif new_diff["max_abs"] <= max(ref_noise["max_abs"], 0.0) * 1.5 + 1e-12:
        verdict = "WITHIN RUN-TO-RUN NOISE"
    else:
        verdict = "LARGER THAN RUN-TO-RUN NOISE"
    print(f"    verdict: {verdict}")
    return verdict


def cmd_images(a, b, n):
    A, B, N = load_images(a), load_images(b), load_images(n)
    print(f"images: shape {A.shape}, dtype {A.dtype}")
    return [report("pixels", diff_arrays(B, A), diff_arrays(N, A))]


def flat_tensors(path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for key in ["model", "ema"]:
        if key in ckpt:
            for k, v in ckpt[key].items():
                if hasattr(v, "dtype") and v.dtype.is_floating_point:
                    out[f"{key}.{k}"] = v.double().numpy()
    return out


def cmd_ckpt(a, b, n):
    A, B, N = flat_tensors(a), flat_tensors(b), flat_tensors(n)
    assert A.keys() == B.keys() == N.keys(), "checkpoints have different tensors"
    print(f"checkpoint: {len(A)} floating-point tensors")
    verdicts = []
    for group in sorted({k.split(".")[0] for k in A}):
        keys = [k for k in A if k.split(".")[0] == group]
        cat = lambda D: np.concatenate([D[k].ravel() for k in keys])
        verdicts.append(report(group, diff_arrays(cat(B), cat(A)), diff_arrays(cat(N), cat(A))))
    return verdicts


def read_cifar_log(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    keys = ["train_loss", "train_OS_loss", "train_DE_loss", "train_ref_MAPE", "train_real_MAPE"]
    return [[r[k] for k in keys] for r in rows], keys


def read_imagenet_log(path):
    text = open(path).read()
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    steps = re.findall(r"\(step=(\d+)\) Train Loss: ([0-9.]+)", text)
    metrics = re.findall(
        r"eikonal_loss: ([0-9.]+), one_step_loss: ([0-9.]+), real_distance_MAPE: ([0-9.]+), ref_distance_MAPE: ([0-9.]+)",
        text,
    )
    rows = [[float(l)] + [float(x) for x in m] for (_, l), m in zip(steps, metrics)]
    return rows, ["loss", "eikonal", "one_step", "real_MAPE", "ref_MAPE"], [int(s) for s, _ in steps]


def cmd_log(a, b, n, kind):
    if kind == "cifar":
        (A, keys), (B, _), (N, _) = read_cifar_log(a), read_cifar_log(b), read_cifar_log(n)
        labels = [f"epoch {i}" for i in range(len(A))]
    else:
        A, keys, steps = read_imagenet_log(a)
        B, _, _ = read_imagenet_log(b)
        N, _, _ = read_imagenet_log(n)
        labels = [f"step {s}" for s in steps]
    assert len(A) == len(B) == len(N) > 0, (len(A), len(B), len(N))
    print(f"{'':10s} {'metric':10s} {'OLD_A':>14s} {'OLD_B':>14s} {'NEW':>14s}")
    for lab, ra, rb, rn in zip(labels, A, B, N):
        for k, va, vb, vn in zip(keys, ra, rb, rn):
            print(f"{lab:10s} {k:10s} {va:14.8f} {vb:14.8f} {vn:14.8f}")
    return [report("all logged values", diff_arrays(B, A), diff_arrays(N, A))]


if __name__ == "__main__":
    mode, a, b, n = sys.argv[1:5]
    if mode == "images":
        verdicts = cmd_images(a, b, n)
    elif mode == "ckpt":
        verdicts = cmd_ckpt(a, b, n)
    elif mode == "cifarlog":
        verdicts = cmd_log(a, b, n, "cifar")
    elif mode == "imagenetlog":
        verdicts = cmd_log(a, b, n, "imagenet")
    else:
        raise SystemExit(__doc__)
    print("RESULT", mode, " ".join(verdicts))
