# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/network/model.py

Description:
Defines the RIDE model:
 - ResNet34 encoder
 - RIDEDecoder
Outputs multiple depth predictions (side outputs + final) in [0,1],
processing one image at a time.
"""

import torch.nn as nn
from network.encoder import ResnetEncoder
from network.decoder import RIDEDecoder


class RIDE(nn.Module):
    """
    RIDE:
      - ResnetEncoder
      - RIDEDecoder
    """
    def __init__(self, in_channels=3, out_size=224, pretrained=True):
        super().__init__()
        self.encoder = ResnetEncoder(pretrained=pretrained, num_input_channels=in_channels)
        self.decoder = RIDEDecoder(self.encoder.num_ch_enc, out_size=out_size)

    def forward(self, x):
        """
        forward:
         x: [B, in_ch, 224,224]
        returns dict of multi-scale predictions
        """
        feats = self.encoder(x)
        return self.decoder(feats)
