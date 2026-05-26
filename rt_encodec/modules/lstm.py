# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

import typing as tp

import torch
from torch import nn

LSTMState = tp.Tuple[torch.Tensor, torch.Tensor]


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    Optionally accepts and returns hidden state for streaming use.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, batch_first=False)

    def forward(
        self,
        x: torch.Tensor,
        state: tp.Optional[LSTMState] = None,
    ) -> tp.Tuple[torch.Tensor, LSTMState]:
        x = x.permute(2, 0, 1)
        batch_size = x.size(1)
        if state is None:
            h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size,
                             device=x.device, dtype=x.dtype)
            c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size,
                             device=x.device, dtype=x.dtype)
        else:
            h0, c0 = state
        y, (h_n, c_n) = self.lstm(x, (h0, c0))
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y, (h_n, c_n)
