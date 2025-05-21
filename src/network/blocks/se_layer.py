# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/blocks/se_layer.py

Description:
Defines the Squeeze-and-Excitation (SE) block used to recalibrate channel-wise feature responses.
The SE block applies global average pooling and learns channel-wise weights through a two-layer fully connected network.
"""

import torch.nn as nn

class SELayer(nn.Module):
    """
    SELayer:
      - Performs global average pooling across spatial dimensions.
      - Uses a two-layer fully connected network with a reduction ratio to compute channel-wise activations.
      - Applies a sigmoid activation to scale the input features.
    """
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
