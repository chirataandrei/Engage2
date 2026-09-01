"""Spatial encoder: dense GATv2 (default) and optional PyG sparse GATv2."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DenseGATv2(nn.Module):
    """GATv2 over a dense adjacency derived from proximity edges.

    Returns node embeddings and attention weights [B, H, N, N] for XAI.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        assert out_dim % heads == 0
        self.head_dim = out_dim // heads
        self.lin = nn.Linear(in_dim, heads * self.head_dim, bias=False)
        self.att = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, F], adj: [B, N, N] {0,1}, mask: [B, N]
        b, n, _ = x.shape
        h = self.lin(x).view(b, n, self.heads, self.head_dim)
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1, -1)
        cat = torch.cat([h_i, h_j], dim=-1)
        e = F.leaky_relu(self.att(cat).squeeze(-1), negative_slope=0.2)  # [B, N, N, H]
        e = e.permute(0, 3, 1, 2)  # [B, H, N, N]
        valid = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)  # [B,1,N,N]
        adj_h = adj.unsqueeze(1) * valid
        e = e.masked_fill(adj_h <= 0, torch.finfo(e.dtype).min)
        alpha = torch.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        h_t = h.permute(0, 2, 1, 3)  # [B, H, N, D]
        out = torch.matmul(alpha, h_t)  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        out = out + self.bias
        return F.elu(out), alpha


def edges_to_adj(edge_index: torch.Tensor, n: int, device, dtype) -> torch.Tensor:
    adj = torch.eye(n, device=device, dtype=dtype)
    if edge_index.numel() == 0:
        return adj
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


class SpatialStack(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        mods: list[DenseGATv2] = []
        dim = in_dim
        for _ in range(layers):
            mods.append(DenseGATv2(dim, hidden_dim, heads=heads, dropout=dropout))
            dim = hidden_dim
        self.layers = nn.ModuleList(mods)

    def forward(self, x, adj, mask) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = None
        h = x
        for layer in self.layers:
            h, alpha = layer(h, adj, mask)
        assert alpha is not None
        return h, alpha


class PyGGATv2Stack(nn.Module):
    """Optional sparse path used when encoder: pyg is set."""

    def __init__(self, in_dim: int, hidden_dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        self.convs = nn.ModuleList()
        dim = in_dim
        for i in range(layers):
            concat = i < layers - 1
            out = hidden_dim // heads if concat else hidden_dim
            self.convs.append(
                GATv2Conv(dim, out, heads=heads, dropout=dropout, concat=concat, add_self_loops=True)
            )
            dim = hidden_dim if concat else hidden_dim

    def forward(self, x, edge_index):
        alpha = None
        h = x
        for conv in self.convs:
            h, (ei, alpha) = conv(h, edge_index, return_attention_weights=True)
            h = F.elu(h)
        return h, (ei, alpha)
