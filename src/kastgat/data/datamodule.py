"""Lightning-friendly dataset wrapping temporal graphs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from kastgat.data.graphs import TemporalGraph, build_windows


class GraphWindowDataset(Dataset):
    def __init__(self, samples: list[TemporalGraph]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "x": s.x,
            "node_mask": s.node_mask,
            "y_traj": s.y_traj,
            "y_conflict": s.y_conflict,
            "y_intent": s.y_intent,
            "edge_index": s.edge_index,
            "edge_attr": s.edge_attr,
        }


def collate_windows(batch: list[dict]) -> dict:
    """Pad N is already fixed per sample; stack the tensor fields.

    edge_index stays as a list-of-lists because E varies.
    """
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "node_mask": torch.stack([b["node_mask"] for b in batch], dim=0),
        "y_traj": torch.stack([b["y_traj"] for b in batch], dim=0),
        "y_conflict": torch.stack([b["y_conflict"] for b in batch], dim=0),
        "y_intent": torch.stack([b["y_intent"] for b in batch], dim=0),
        "edge_index": [b["edge_index"] for b in batch],
        "edge_attr": [b["edge_attr"] for b in batch],
    }


def make_loaders(
    samples: list[TemporalGraph],
    batch_size: int = 4,
    val_frac: float = 0.2,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    n = len(samples)
    n_val = max(1, int(n * val_frac)) if n > 1 else 0
    train = samples[: max(1, n - n_val)]
    val = samples[n - n_val :] if n_val else samples[:1]
    train_ds = GraphWindowDataset(train)
    val_ds = GraphWindowDataset(val)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_windows
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_windows
    )
    return train_loader, val_loader


def save_samples(samples: list[TemporalGraph], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, path)


def load_samples(path: str | Path) -> list[TemporalGraph]:
    return torch.load(path, weights_only=False)
