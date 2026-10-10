# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# CIFAR-10 preset of the UNet used by the flow_matching codebase, wrapped by
# the Distance Marching head (models/distance_marching.py).
from models.distance_marching import DistanceMarching
from models.ema import EMA

MODEL_CONFIGS = {
    "cifar10": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2, 2],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}


def instantiate_model(
    architechture: str = "cifar10",
    use_ema: bool = True,
    separate_t: bool = True,
    max_period: float = 10000,
    unified_scale: float = 1.0,
    ema_decay: float = 0.999,
):
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    config = MODEL_CONFIGS[architechture].copy()
    config["separate_t"] = separate_t
    config["max_period"] = max_period
    config["unified_scale"] = unified_scale
    model = DistanceMarching(**config)

    if use_ema:
        return EMA(model=model, decay=ema_decay)
    else:
        return model
