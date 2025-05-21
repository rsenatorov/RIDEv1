# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/encoder.py

Description:
Implements a ResNeXt-101 (32x8d)-based encoder for RIDE.
We use torchvision's pretrained ResNeXt101_32x8d with the weights= API,
yielding multi-scale features at 1/2, 1/4, 1/8, 1/16, and 1/32 resolutions.
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights

class ResnetEncoder(nn.Module):
    """
    ResnetEncoder:
      - Uses torchvision ResNeXt-101 (32x8d) pretrained on ImageNet (weights API)
      - Yields multi-scale features at 1/2, 1/4, 1/8, 1/16, 1/32 resolution
    """
    def __init__(self, pretrained=True, num_input_channels=3):
        super().__init__()
        weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnext101_32x8d(weights=weights)

        self.num_ch_enc = np.array([64,256,512,1024,2048])

        if num_input_channels != 3:
            out_ch = backbone.conv1.out_channels
            k,s,p = backbone.conv1.kernel_size, backbone.conv1.stride, backbone.conv1.padding
            bias = backbone.conv1.bias is not None
            new_conv = nn.Conv2d(num_input_channels,out_ch,kernel_size=k,stride=s,padding=p,bias=bias)
            with torch.no_grad():
                new_conv.weight[:,:3] = backbone.conv1.weight
                if num_input_channels>3:
                    new_conv.weight[:,3:].zero_()
            backbone.conv1 = new_conv

        self.layer0 = nn.Sequential(backbone.conv1,backbone.bn1,backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool,backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0,x1,x2,x3,x4]
