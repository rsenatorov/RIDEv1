# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/blocks/edge_attention_block.py

Description:
Defines the EdgeAttentionBlock which extracts structural cues from encoder features
and uses them to refine decoder features via a learned attention gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAttentionBlock(nn.Module):
    """
    EdgeAttentionBlock:
      - Projects encoder features to decoder channel dims.
      - Projects decoder features to same dims.
      - Computes a sigmoid gate from their sum, and refines decoder features.
    """
    def __init__(self, enc_ch, dec_ch):
        super(EdgeAttentionBlock, self).__init__()
        self.conv_enc = nn.Conv2d(enc_ch, dec_ch, kernel_size=1, bias=False)
        self.conv_dec = nn.Conv2d(dec_ch, dec_ch, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, dec_feat, enc_feat):
        """
        dec_feat: [B, dec_ch, H, W]
        enc_feat: [B, enc_ch, H_enc, W_enc] (e.g. x0 from encoder)
        returns: refined decoder features
        """
        if enc_feat.shape[-2:] != dec_feat.shape[-2:]:
            enc_feat = F.interpolate(
                enc_feat,
                size=dec_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        attn = self.conv_enc(enc_feat) + self.conv_dec(dec_feat)
        attn = self.sigmoid(attn)
        return dec_feat * attn
