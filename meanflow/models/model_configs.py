# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from models.meanflow import MeanFlow

from models.unet import SongUNet

MODEL_ARCHS = {
    "unet": SongUNet,
    "unet_shrimp": SongUNet,
}

MODEL_CONFIGS = {
    "unet": {
        "img_resolution": 32,
        "in_channels": 3,
        "out_channels": 3,
        "channel_mult_noise": 2,
        "resample_filter": [1, 3, 3, 1],
        "channel_mult": [2, 2, 2],
        "encoder_type": "standard",
        "decoder_type": "standard",
    },
    "unet_shrimp": {
        "img_resolution": 128,
        "in_channels": 5,  # radar(1) + sat(4)
        "out_channels": 1,
        "model_channels": 32,
        "channel_mult_noise": 2,
        "resample_filter": [1, 3, 3, 1],
        "channel_mult": [1, 2, 2, 2],
        "encoder_type": "standard",
        "decoder_type": "standard",
    },
}


def instantiate_model(args) -> nn.Module:
    if args.dataset == 'shrimp':
        architechture = args.arch+"_shrimp"
    else:
        architechture = args.arch
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    configs = MODEL_CONFIGS[architechture].copy()  # Create a copy to avoid modifying the original
    configs['dropout'] = args.dropout
    
    # Get base architecture name
    base_arch = args.arch
    arch = MODEL_ARCHS[base_arch]
    
    if args.use_edm_aug:
        configs['augment_dim'] = 6
    model = MeanFlow(arch=arch, net_configs=configs, args=args)

    return model
