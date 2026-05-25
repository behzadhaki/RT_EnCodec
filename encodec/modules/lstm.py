# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

from torch import nn
import torch

class SLSTM(nn.Module):
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, batch_first=False)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        # Initialize hidden states explicitly for ONNX export
        batch_size = x.size(1)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size,
                         device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size,
                         device=x.device, dtype=x.dtype)

        y, _ = self.lstm(x, (h0, c0))
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y