"""Conflict and ATCO-intent labels from geometry and optional event logs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kastgat.data.features import cpa_tcpa, haversine_nm
from kastgat.data.schema import INTENT_TO_ID, ensure_canonical


def _intent_from_future(now: pd.Series, future: pd.Series) -> str:
    dalt = float(future["alt_ft"] - now["alt_ft"])
    dgs = float(future["gs_kt"] - now["gs_kt"])
    dhdg = (float(future["track_deg"] - now["track_deg"]) + 180.0) % 360.0 - 180.0
    if dalt > 400:
        return "CLIMB"
    if dalt < -400:
        return "DESCEND"
    if dhdg > 15:
        return "HEADING_RIGHT"
    if dhdg < -15:
        return "HEADING_LEFT"
    if dgs < -20:
        return "SPEED_REDUCE"
    if dgs > 20:
        return "SPEED_INCREASE"
    return "NONE"


def attach_intent_ids(df: pd.DataFrame, horizon_steps: int = 6) -> pd.DataFrame:
    df = ensure_canonical(df).sort_values(["flight_id", "ts"]).reset_index(drop=True)
    intents: list[str] = []
    for _, g in df.groupby("flight_id", sort=False):
        events = g["atco_event"].astype(str).str.upper().to_numpy()
        for i in range(len(g)):
            raw = events[i]
            if raw in INTENT_TO_ID and raw != "NONE":
                intents.append(raw)
                continue
            j = min(i + horizon_steps, len(g) - 1)
            intents.append(_intent_from_future(g.iloc[i], g.iloc[j]))
    out = df.copy()
    out["intent"] = intents
    out["intent_id"] = out["intent"].map(INTENT_TO_ID).fillna(0).astype(int)
    return out


def pairwise_conflict_mask(
    snap: pd.DataFrame,
    *,
    separation_nm: float = 5.0,
    separation_ft: float = 1000.0,
    horizon_s: float = 60.0,
) -> np.ndarray:
    """Return [N] bool: aircraft will lose separation within horizon under linear motion."""
    n = len(snap)
    flags = np.zeros(n, dtype=bool)
    if n < 2:
        return flags
    lat0 = float(snap["lat"].mean())
    lon0 = float(snap["lon"].mean())
    lat = snap["lat"].to_numpy()
    lon = snap["lon"].to_numpy()
    gs = snap["gs_kt"].to_numpy()
    tr = snap["track_deg"].to_numpy()
    alt = snap["alt_ft"].to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            if abs(alt[i] - alt[j]) >= separation_ft:
                continue
            cpa, tcpa = cpa_tcpa(lat[i], lon[i], gs[i], tr[i], lat[j], lon[j], gs[j], tr[j], lat0, lon0)
            dist = haversine_nm(lat[i], lon[i], lat[j], lon[j])
            imminent = (0.0 <= float(tcpa) <= horizon_s and float(cpa) < separation_nm) or dist < separation_nm
            if imminent:
                flags[i] = True
                flags[j] = True
    return flags
