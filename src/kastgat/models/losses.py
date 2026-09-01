from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        target = target.long()
        nll = F.nll_loss(logp.reshape(-1, logp.size(-1)), target.reshape(-1), reduction="none")
        p = torch.exp(-nll)
        loss = ((1 - p) ** self.gamma) * nll
        w = mask.reshape(-1)
        denom = w.sum().clamp_min(1.0)
        return (loss * w).sum() / denom


def trajectory_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred/target [B, N, H, 3], mask [B, N]
    err = (pred - target) ** 2
    w = mask.unsqueeze(-1).unsqueeze(-1)
    return (err * w).sum() / w.sum().clamp_min(1.0) / 3.0


def physics_penalty(pred: torch.Tensor, mask: torch.Tensor, dt_s: float = 10.0) -> torch.Tensor:
    """Penalise implied ground speeds / climb rates that a jet cannot fly."""
    # pred: [B, N, H, 3] lat, lon, alt_ft
    if pred.size(2) < 2:
        return pred.new_zeros(())
    dlat = pred[:, :, 1:, 0] - pred[:, :, :-1, 0]
    dlon = pred[:, :, 1:, 1] - pred[:, :, :-1, 1]
    dalt = pred[:, :, 1:, 2] - pred[:, :, :-1, 2]
    dist_nm = torch.sqrt((dlat * 60.0) ** 2 + (dlon * 60.0) ** 2)
    gs_kt = dist_nm * (3600.0 / dt_s)
    vs_fpm = dalt * (60.0 / dt_s)
    gs_excess = F.relu(gs_kt - 600.0)
    vs_excess = F.relu(vs_fpm.abs() - 4000.0)
    w = mask.unsqueeze(-1)
    pen = (gs_excess + vs_excess * 0.01) * w
    return pen.sum() / w.sum().clamp_min(1.0)
