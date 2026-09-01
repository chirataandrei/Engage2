from __future__ import annotations

import torch
from torch import nn

from kastgat.data.schema import INTENT_CLASSES


class TrajectoryHead(nn.Module):
    def __init__(self, in_dim: int, horizon: int, hidden: int = 64) -> None:
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * 3),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).view(*h.shape[:-1], self.horizon, 3)


class ConflictHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class IntentHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int | None = None, hidden: int = 64) -> None:
        super().__init__()
        n_classes = n_classes or len(INTENT_CLASSES)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
