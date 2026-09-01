from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from kastgat.models.heads import ConflictHead, IntentHead, TrajectoryHead
from kastgat.models.spatial import SpatialStack, edges_to_adj
from kastgat.models.temporal import TemporalConvNet


def decode_absolute_traj(delta: torch.Tensor, last_pos: torch.Tensor) -> torch.Tensor:
    """delta: [B,N,H,3] in (NM north, NM east, FL). last_pos: [B,N,3] lat,lon,alt_ft."""
    lat0 = last_pos[..., 0].unsqueeze(-1)
    lon0 = last_pos[..., 1].unsqueeze(-1)
    alt0 = last_pos[..., 2].unsqueeze(-1)
    coslat = torch.cos(torch.deg2rad(lat0)).clamp(min=0.2)
    lat = lat0 + delta[..., 0] / 60.0
    lon = lon0 + delta[..., 1] / (60.0 * coslat)
    alt = alt0 + delta[..., 2] * 100.0
    return torch.stack([lat, lon, alt], dim=-1)


def encode_delta_traj(future: torch.Tensor, last_pos: torch.Tensor) -> torch.Tensor:
    lat0 = last_pos[..., 0].unsqueeze(-1)
    lon0 = last_pos[..., 1].unsqueeze(-1)
    alt0 = last_pos[..., 2].unsqueeze(-1)
    coslat = torch.cos(torch.deg2rad(lat0)).clamp(min=0.2)
    d_north = (future[..., 0] - lat0) * 60.0
    d_east = (future[..., 1] - lon0) * 60.0 * coslat
    d_fl = (future[..., 2] - alt0) / 100.0
    return torch.stack([d_north, d_east, d_fl], dim=-1)


class KASTGAT(nn.Module):
    """Knowledge-aware spatial-temporal graph attention network.

    Input x: [B, T, N, F], node_mask: [B, T, N]
    edge_index: list over batch, each a list over T of [2, E] tensors.
    """

    def __init__(
        self,
        in_dim: int = 7,
        hidden_dim: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        tcn_channels: list[int] | None = None,
        tcn_kernel: int = 3,
        dropout: float = 0.1,
        horizon: int = 6,
        n_intent: int = 8,
    ) -> None:
        super().__init__()
        tcn_channels = tcn_channels or [64, 64]
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.spatial = SpatialStack(hidden_dim, hidden_dim, gat_heads, gat_layers, dropout)
        self.temporal = TemporalConvNet(hidden_dim, tcn_channels, kernel=tcn_kernel, dropout=dropout)
        tcn_out = tcn_channels[-1]
        self.traj_head = TrajectoryHead(tcn_out, horizon)
        self.conflict_head = ConflictHead(tcn_out)
        self.intent_head = IntentHead(tcn_out, n_classes=n_intent)

    def _adj_from_batch(self, edge_index, b: int, n: int, t: int, device, dtype) -> torch.Tensor:
        # returns [B, N, N] for a single timestep t
        adjs = []
        for bi in range(b):
            ei = edge_index[bi][t].to(device)
            adjs.append(edges_to_adj(ei, n, device, dtype))
        return torch.stack(adjs, dim=0)

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor, edge_index) -> dict[str, torch.Tensor]:
        b, t, n, f = x.shape
        h = self.input_proj(x)
        spatial_seq = []
        last_alpha = None
        for ti in range(t):
            adj = self._adj_from_batch(edge_index, b, n, ti, x.device, h.dtype)
            ht, alpha = self.spatial(h[:, ti], adj, node_mask[:, ti])
            spatial_seq.append(ht)
            last_alpha = alpha
        spatial = torch.stack(spatial_seq, dim=1)  # [B, T, N, H]
        bn = b * n
        tmp = spatial.permute(0, 2, 1, 3).reshape(bn, t, -1)
        tmp = self.temporal(tmp)
        tmp = tmp.reshape(b, n, t, -1)
        last = tmp[:, :, -1, :]  # [B, N, C]
        delta = self.traj_head(last)
        last_pos = x[:, -1, :, :3]
        traj = decode_absolute_traj(delta, last_pos)
        return {
            "traj": traj,
            "traj_delta": delta,
            "conflict_logits": self.conflict_head(last),
            "intent_logits": self.intent_head(last),
            "attention": last_alpha,
            "embeddings": last,
        }
