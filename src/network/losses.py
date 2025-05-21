# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/losses.py

Description:
Defines a simple composite loss using L1 + SSIM, each with a learnable weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim


class CompositeLoss(nn.Module):
    """
    CompositeLoss:
      - w_l1   * L1 loss
      - w_ssim * (1 - SSIM)
    where each w_* = exp(s_*), and s_* are learnable parameters.
    """
    def __init__(self):
        super(CompositeLoss, self).__init__()
        # log-space scalars
        self.s_l1   = nn.Parameter(torch.tensor(0.0))
        self.s_ssim = nn.Parameter(torch.tensor(0.0))

    def forward(self, pred, gt):
        """
        pred: [B,1,H,W], gt: [B,1,H,W]
        """
        # L1 term
        l1_loss = F.l1_loss(pred, gt, reduction="mean")

        # SSIM term
        ssim_val = ssim(pred, gt)
        ssim_loss = 1.0 - ssim_val

        # learnable weights
        w_l1   = torch.exp(self.s_l1)
        w_ssim = torch.exp(self.s_ssim)

        return w_l1 * l1_loss + w_ssim * ssim_loss

