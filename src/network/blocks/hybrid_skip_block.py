# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/blocks/hybrid_skip_block.py

Description:
Defines the HybridSkipBlock which replaces simple concatenation with an attention gating mechanism that merges
the upsampled decoder feature and the refined encoder skip feature.
The block includes a reflection-padded refinement convolution for the encoder feature, an attention gating mechanism,
and a Squeeze-and-Excitation block for channel recalibration.
"""

import torch
import torch.nn as nn
from .se_layer import SELayer

class HybridSkipBlock(nn.Module):
    """
    HybridSkipBlock:
      - Applies reflection-padded refinement convolution to the encoder skip feature.
      - Combines the upsampled decoder feature with the refined encoder feature using an attention gating mechanism.
      - Utilizes a Squeeze-and-Excitation block for channel-wise recalibration.
    """
    def __init__(self, dec_ch, enc_ch):
        super(HybridSkipBlock, self).__init__()
        self.border_refine = nn.Sequential(
            nn.Conv2d(enc_ch, enc_ch, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(enc_ch),
            nn.ReLU(inplace=True)
        )

        total_in = dec_ch + enc_ch

        # 1x1 convolution for attention gating
        self.attn_conv = nn.Conv2d(total_in, total_in, kernel_size=1, bias=False)
        self.attn_bn   = nn.BatchNorm2d(total_in)
        self.sigmoid   = nn.Sigmoid()

        # Squeeze-and-Excitation block for channel recalibration
        self.se = SELayer(total_in, reduction=16)

    def forward(self, dec_feat, enc_feat):
        enc_feat_refined = self.border_refine(enc_feat)
        x_cat = torch.cat([dec_feat, enc_feat_refined], dim=1)
        attn = self.attn_conv(x_cat)
        attn = self.attn_bn(attn)
        attn = self.sigmoid(attn)

        x_out = x_cat * attn
        x_out = self.se(x_out)
        return x_out
