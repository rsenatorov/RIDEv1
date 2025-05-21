# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/blocks/upsample_conv.py

Description:
Defines the UpsampleConv module which performs bilinear upsampling by a factor of 2 to match a given target size,
followed by a 3x3 convolution with reflection padding for feature refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleConv(nn.Module):
    """
    UpsampleConv:
      - Performs bilinear upsampling to match a target spatial size.
      - Applies a 3x3 convolution with reflection padding.
      - Activates the output using ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(UpsampleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode='reflect'
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return self.relu(x)
