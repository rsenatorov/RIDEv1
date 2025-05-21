# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/decoder.py

Description:
Defines the RIDE decoder which merges multi-scale features, uses ASPP+StripPooling for context,
and outputs a single 224x224 relative depth in [0,1], with:
 - ASPP + Strip-Pooling context
 - Residual refinements
 - Scaling head (calib) on coarse output
 - Dynamic LPG Fusion Weights
 - Auxiliary per-pixel refinement
 - Multi-scale side outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.blocks.aspp import ASPPModule
from network.blocks.residual_block import ResidualBlock
from network.blocks.lpg import LocalPlanarGuidance
from network.blocks.upsample_conv import UpsampleConv
from network.blocks.hybrid_skip_block import HybridSkipBlock
from network.blocks.edge_attention_block import EdgeAttentionBlock
from network.blocks.aux_refinement_block import AuxRefinementBlock


class RIDEDecoder(nn.Module):
    """
    RIDEDecoder:
      - Upsample + HybridSkip at 4 scales
      - EdgeAttention on the last upsample
      - ASPP + StripPooling
      - Residual blocks
      - Coarse depth + scaling head
      - LPG fusion + AuxRefinement
      - Multi-scale side outputs
    """
    def __init__(self, num_ch_enc, out_size=224):
        super().__init__()
        self.out_size = out_size

        # upsample + skip connections
        self.up4 = UpsampleConv(num_ch_enc[4], 1024)
        self.skip4 = HybridSkipBlock(1024, num_ch_enc[3])
        self.up3 = UpsampleConv(2048, 512)
        self.skip3 = HybridSkipBlock(512, num_ch_enc[2])
        self.up2 = UpsampleConv(1024, 256)
        self.skip2 = HybridSkipBlock(256, num_ch_enc[1])
        self.up1 = UpsampleConv(512, 128)
        self.skip1 = HybridSkipBlock(128, num_ch_enc[0])
        self.up0 = UpsampleConv(192, 128)

        # edge-focused attention
        self.edge_attn = EdgeAttentionBlock(num_ch_enc[0], 128)

        # ASPP + StripPooling
        self.aspp = ASPPModule(128, 256, dilations=(1,6,12,18), strip_k=21)

        # residual refinements
        self.rb1 = ResidualBlock(256)
        self.rb2 = ResidualBlock(256)

        # coarse depth + scaling head
        self.final_conv = nn.Conv2d(256, 1, kernel_size=3, padding=1, padding_mode='reflect')
        self.calib      = nn.Conv2d(256, 2, kernel_size=1, bias=True)

        # LPG fusion weights & modules
        self.lpg_weight_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.lpg       = LocalPlanarGuidance(256, out_size)
        self.extra_lpg = LocalPlanarGuidance(256, out_size)

        # auxiliary per-pixel refinement
        self.aux_refine = AuxRefinementBlock(256)

        # multi-scale side outputs
        self.side4 = nn.Conv2d(num_ch_enc[4],   1, kernel_size=1)
        self.side3 = nn.Conv2d(num_ch_enc[3],   1, kernel_size=1)
        self.side2 = nn.Conv2d(num_ch_enc[2],   1, kernel_size=1)
        self.side1 = nn.Conv2d(192,             1, kernel_size=1)
        self.side0 = nn.Conv2d(256,             1, kernel_size=1)

    def forward(self, features):
        """
        features: list [x0, x1, x2, x3, x4] from encoder for one image
        returns dict with 'final' and side outputs
        """
        x0, x1, x2, x3, x4 = features

        s4 = self.skip4(self.up4(x4, x3.shape[-2:]), x3)
        s3 = self.skip3(self.up3(s4, x2.shape[-2:]), x2)
        s2 = self.skip2(self.up2(s3, x1.shape[-2:]), x1)
        s1 = self.skip1(self.up1(s2, x0.shape[-2:]), x0)

        up0 = self.up0(s1, (self.out_size, self.out_size))
        up0 = self.edge_attn(up0, x0)

        # ASPP + residuals
        aspp_feat = self.aspp(up0)
        r = self.rb2(self.rb1(aspp_feat))

        # side outputs
        side4 = torch.sigmoid(self.side4(s4))
        side3 = torch.sigmoid(self.side3(s3))
        side2 = torch.sigmoid(self.side2(s2))
        side1 = torch.sigmoid(self.side1(s1))
        side0 = torch.sigmoid(self.side0(r))

        # coarse depth + scaling
        scale, shift = self.calib(r).split(1, dim=1)
        coarse = torch.sigmoid(scale.sigmoid() * self.final_conv(r) + shift.tanh())

        # LPG fusion + refinement
        w1, w2 = torch.chunk(self.lpg_weight_head(r), 2, dim=1)
        l1      = self.lpg(r)
        l2      = self.extra_lpg(r)
        delta   = self.aux_refine(r)

        final = coarse + w1 * l1 + w2 * l2 + delta
        final = torch.clamp(final, 0.0, 1.0)

        return {
            "final": final,
            "side4": side4,
            "side3": side3,
            "side2": side2,
            "side1": side1,
            "side0": side0
        }
