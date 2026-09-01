"""Causal dilated TCN over the time axis of node embeddings."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp: int) -> None:
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : x.size(-1) - self.chomp] if self.chomp else x


class TemporalBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, kernel, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(F.relu(self.chomp1(self.conv1(x))))
        y = self.dropout(F.relu(self.chomp2(self.conv2(y))))
        return F.relu(y + self.down(x))


class TemporalConvNet(nn.Module):
    def __init__(self, in_dim: int, channels: list[int], kernel: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [in_dim, *channels]
        for i in range(len(channels)):
            layers.append(TemporalBlock(dims[i], dims[i + 1], kernel, dilation=2**i, dropout=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*N, T, F] -> [B*N, T, C]."""
        z = x.transpose(1, 2)
        z = self.net(z)
        return z.transpose(1, 2)
