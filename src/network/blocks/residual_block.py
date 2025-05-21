# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/blocks/residual_block.py

Description:
Defines a simple residual block with two convs and a skip connection.
"""

import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    ResidualBlock:
      - Conv -> LeakyReLU(0.2) -> Conv
      - skip connection from input

    Reflection padding is added to the 3x3 conv layers to help reduce border artifacts.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode='reflect'
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode='reflect'
        )

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out
