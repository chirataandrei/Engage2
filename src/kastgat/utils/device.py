from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Never assume CUDA. auto prefers cuda, then mps, then cpu."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def amp_enabled(device: torch.device, requested: bool) -> bool:
    return bool(requested and device.type == "cuda")
