# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/blocks/aux_refinement_block.py

Description:
Defines the AuxRefinementBlock which predicts a per-pixel residual correction
and an attention mask to gate it, boosting local depth accuracy.
"""

import torch
import torch.nn as nn

class AuxRefinementBlock(nn.Module):
    """
    AuxRefinementBlock:
      - Conv -> ReLU -> Conv to predict Δ.
      - 1x1 conv + sigmoid to produce a gating mask.
      - Outputs mask * Δ.
    """
    def __init__(self, in_ch):
        super(AuxRefinementBlock, self).__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feat):
        """
        feat: [B, in_ch, H, W]
        returns: [B,1,H,W] residual to add
        """
        delta = self.refine(feat)
        mask = self.gate(feat)
        return delta * mask
