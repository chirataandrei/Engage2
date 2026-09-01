"""State-based conflict detection (the BlueSky / TCAS-style geometric baseline)."""

from __future__ import annotations

import pandas as pd

from kastgat.data.labels import pairwise_conflict_mask
from kastgat.data.schema import ensure_canonical
from kastgat.reasoning.rules import evaluate_snapshot


def detect(snap: pd.DataFrame, separation_nm: float = 5.0, separation_ft: float = 1000.0, horizon_s: float = 60.0):
    snap = ensure_canonical(snap)
    flags = pairwise_conflict_mask(
        snap, separation_nm=separation_nm, separation_ft=separation_ft, horizon_s=horizon_s
    )
    out = snap[["flight_id", "lat", "lon", "alt_ft"]].copy()
    out["conflict"] = flags
    return out


def explain(snap: pd.DataFrame, **kwargs):
    return evaluate_snapshot(snap, **kwargs)
