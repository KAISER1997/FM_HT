# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from models.unet import UNetModel
from models.ttfNet import build_ttf_m
import torch
MODEL_CONFIGS = {
    "imagenet": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 3,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 3, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "imagenet_discrete": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.2,
        "channel_mult": [2, 3, 4, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
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
    "cifar10_LT": {
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
    "cifar10LT_tail": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 12,
        "num_res_blocks": 2,  #best at 2
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
    "cifar10_discrete": {
        "in_channels": 3,
        "model_channels": 96,
        "out_channels": 3,
        "num_res_blocks": 5,
        "attention_resolutions": [2],
        "dropout": 0.4,
        "channel_mult": [3, 4, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },



    "pareto": {
        "in_channels": 1,
        "model_channels": 128,
        "out_channels": 1,
        "num_res_blocks": 1,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2],
        "conv_resample": False,
        "dims": 1,
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


    "pareto_tail": {
        "in_channels": 1,
        "model_channels": 128,
        "out_channels": 4,
        "num_res_blocks": 2,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2, 2],
        "conv_resample": False,
        "dims": 1,
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

    "studentT": {
        "in_channels": 1,
        "model_channels": 512,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2],
        "conv_resample": False,
        "dims": 1,
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


    "studentT_tail": {
        "in_channels": 1,
        "model_channels": 512,
        "out_channels": 4,
        "num_res_blocks": 1,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2],
        "conv_resample": False,
        "dims": 1,
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
    architechture: str, is_discrete: bool, use_ema: bool
) -> Union[UNetModel, DiscreteUNetModel]:
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    if is_discrete:
        if architechture + "_discrete" in MODEL_CONFIGS:
            config = MODEL_CONFIGS[architechture + "_discrete"]
        else:
            config = MODEL_CONFIGS[architechture]
        model = DiscreteUNetModel(
            vocab_size=257,
            **config,
        )
    else:
        model = UNetModel(**MODEL_CONFIGS[architechture])

    if use_ema:
        return EMA(model=model)
    else:
        return model

def instantiate_model_TTF(dim,preset_tail=False,dfs=None):
    if preset_tail:
        return build_ttf_m(
        dim,
        model_kwargs=dict(
            fix_tails=False,#ADITYA CHANGED THIS
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in dfs['metadata']['pos_dfs']],  #DIFFFERENT FROM MAIN CODE
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in dfs['metadata']['neg_dfs']],  #DIFFFERENT FROM MAIN CODE
        )
    )
    else:
        return build_ttf_m(
            dim,
            model_kwargs=dict(
                fix_tails=False,
                pos_tail_init=[
                    float(t.cpu()) for t in torch.distributions.Uniform(low=0.05, high=1.0).sample([dim])
                ],
                neg_tail_init=[
                    float(t.cpu()) for t in torch.distributions.Uniform(low=0.05, high=1.0).sample([dim])
                ]
            ))
