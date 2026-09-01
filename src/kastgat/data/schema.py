"""Canonical traffic schema — the internal contract for the whole kit."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CANONICAL_REQUIRED = (
    "flight_id",
    "ts",
    "lat",
    "lon",
    "alt_ft",
    "gs_kt",
    "track_deg",
    "vert_rate_fpm",
)

CANONICAL_OPTIONAL = (
    "aircraft_type",
    "atco_event",
    "callsign",
    "onground",
)

INTENT_CLASSES = (
    "NONE",
    "CLIMB",
    "DESCEND",
    "HEADING_LEFT",
    "HEADING_RIGHT",
    "SPEED_REDUCE",
    "SPEED_INCREASE",
    "DIRECT_TO",
)

INTENT_TO_ID = {name: i for i, name in enumerate(INTENT_CLASSES)}
ID_TO_INTENT = {i: name for name, i in INTENT_TO_ID.items()}

FT_PER_M = 3.280839895
KT_PER_MPS = 1.943844492
FPM_PER_MPS = 196.8503937
NM_PER_DEG_LAT = 60.0
M_PER_NM = 1852.0


@dataclass(frozen=True)
class CanonicalFrame:
    """Thin wrapper so pipeline steps can assert they received canonical data."""

    df: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in CANONICAL_REQUIRED if c not in self.df.columns]
        if missing:
            raise ValueError(f"Canonical frame missing columns: {missing}")


def ensure_canonical(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CANONICAL_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")
    out = df.copy()
    out["flight_id"] = out["flight_id"].astype(str)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    for col in ("lat", "lon", "alt_ft", "gs_kt", "track_deg", "vert_rate_fpm"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "aircraft_type" not in out.columns:
        out["aircraft_type"] = "B738"
    if "atco_event" not in out.columns:
        out["atco_event"] = "NONE"
    out["atco_event"] = out["atco_event"].fillna("NONE").astype(str).str.upper()
    out["onground"] = out["onground"].fillna(False) if "onground" in out.columns else False
    return out.dropna(subset=["flight_id", "ts", "lat", "lon"])
