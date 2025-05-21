# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/blocks/aspp.py

Description:
Defines the Atrous Spatial Pyramid Pooling (ASPP) module for multi-scale context aggregation.
Now enhanced with Strip-Pooling (long 1xK & Kx1 convolutions) for better relative depth ranking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPModule(nn.Module):
    """
    ASPPModule + StripPooling:
    - Parallel branches with different dilation rates
    - Global average pooling branch
    - Long-strip convs: 1xK and Kx1
    - Then 1x1 conv to fuse
    """
    def __init__(self, in_channels, out_channels, dilations=(1,6,12,18), strip_k=21):
        super(ASPPModule, self).__init__()
        # standard ASPP
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.aspp2 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=dilations[0], dilation=dilations[0],
            bias=False, padding_mode='reflect'
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.aspp3 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=dilations[1], dilation=dilations[1],
            bias=False, padding_mode='reflect'
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.aspp4 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=dilations[2], dilation=dilations[2],
            bias=False, padding_mode='reflect'
        )
        self.bn4 = nn.BatchNorm2d(out_channels)

        # global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # strip-pooling branches
        self.strip_h = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, strip_k),
            padding=(0, strip_k//2),
            bias=False,
            padding_mode='reflect'
        )
        self.bn_h = nn.BatchNorm2d(out_channels)

        self.strip_v = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(strip_k, 1),
            padding=(strip_k//2, 0),
            bias=False,
            padding_mode='reflect'
        )
        self.bn_v = nn.BatchNorm2d(out_channels)

        # fuse all branches
        self.conv1 = nn.Conv2d(out_channels * 7, out_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        forward:
           x: [B,in_channels,H,W]
        """
        x1 = self.relu(self.bn1(self.aspp1(x)))
        x2 = self.relu(self.bn2(self.aspp2(x)))
        x3 = self.relu(self.bn3(self.aspp3(x)))
        x4 = self.relu(self.bn4(self.aspp4(x)))

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=False)

        x_h = self.relu(self.bn_h(self.strip_h(x)))
        x_v = self.relu(self.bn_v(self.strip_v(x)))

        x_cat = torch.cat([x1, x2, x3, x4, x5, x_h, x_v], dim=1)
        out = self.relu(self.bn5(self.conv1(x_cat)))
        return out
