# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/blocks/lpg.py

Description:
Defines the Local Planar Guidance (LPG) module for refining depth from plane parameters.
No fixed scaling here; actual fusion weights are learned dynamically in the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalPlanarGuidance(nn.Module):
    """
    LocalPlanarGuidance:
      - Predicts local plane parameters [a,b,c]
      - Evaluates them over a normalized (x,y) grid

    Reflection padding is used in its internal conv layers to reduce border artifacts.
    """
    def __init__(self, in_channels, out_size):
        super(LocalPlanarGuidance, self).__init__()
        self.out_size = out_size

        self.lpg_conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            padding_mode='reflect'
        )
        self.lpg_relu = nn.ReLU(inplace=True)

        self.lpg_conv2 = nn.Conv2d(
            in_channels,
            3,
            kernel_size=5,
            padding=2,
            padding_mode='reflect'
        )

        # create normalized coordinate grids
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0,1,out_size),
            torch.linspace(0,1,out_size),
            indexing='ij'
        )
        self.register_buffer('grid_x', grid_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('grid_y', grid_y.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        """
        x: [B,in_channels,H,W], ideally H=W=out_size
        returns: raw plane contribution [B,1,H,W]
        """
        plane_params = self.lpg_conv2(self.lpg_relu(self.lpg_conv1(x)))
        a = plane_params[:,0:1,:,:]
        b = plane_params[:,1:2,:,:]
        c = plane_params[:,2:3,:,:]

        lpg_depth = a*self.grid_x + b*self.grid_y + c
        return lpg_depth
