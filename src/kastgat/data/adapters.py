"""Dataset-agnostic column mapping onto the canonical schema.

At the hackathon this is the only file you should have to touch besides
`config/hackathon.yaml`. Everything downstream consumes canonical columns.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from kastgat.data.schema import (
    FPM_PER_MPS,
    FT_PER_M,
    KT_PER_MPS,
    ensure_canonical,
)
from kastgat.utils.config import load_yaml

SYNONYMS: dict[str, tuple[str, ...]] = {
    "flight_id": ("flight_id", "icao24", "acid", "callsign", "id", "hexident", "track_id"),
    "ts": ("ts", "time", "timestamp", "datetime", "simt", "t", "time_epoch"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "lng", "long", "x"),
    "alt_ft": ("alt_ft", "altitude_ft", "alt", "baroaltitude_ft", "flight_level"),
    "alt_m": ("alt_m", "baroaltitude", "geoaltitude", "altitude_m", "altitude"),
    "gs_kt": ("gs_kt", "groundspeed", "gs", "tas", "speed_kt"),
    "gs_mps": ("gs_mps", "velocity", "speed", "spd"),
    "track_deg": ("track_deg", "heading", "track", "hdg", "cog"),
    "vert_rate_fpm": ("vert_rate_fpm", "roc", "vs", "vertical_rate_fpm"),
    "vert_rate_mps": ("vert_rate_mps", "vertrate", "vertical_rate", "vz"),
    "aircraft_type": ("aircraft_type", "type", "typecode", "actype"),
    "atco_event": ("atco_event", "intent", "command", "clearance", "reso_cmd"),
    "callsign": ("callsign", "cs"),
    "onground": ("onground", "on_ground", "ground"),
}


def guess_mapping(columns: list[str]) -> dict[str, str]:
    lower = {c.lower().strip(): c for c in columns}
    mapping: dict[str, str] = {}
    for canonical, names in SYNONYMS.items():
        for name in names:
            if name.lower() in lower:
                mapping[canonical] = lower[name.lower()]
                break
    return mapping


def mapping_from_config(path_or_dict: str | dict[str, Any]) -> dict[str, str]:
    data = path_or_dict if isinstance(path_or_dict, dict) else load_yaml(path_or_dict)
    raw = data.get("mapping", data)
    return {k: v for k, v in raw.items() if v not in (None, "TODO", "")}


def adapt_frame(
    df: pd.DataFrame,
    mapping: dict[str, str] | None = None,
    *,
    auto: bool = True,
) -> pd.DataFrame:
    """Rename/convert a raw table into the canonical schema."""
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    resolved = dict(mapping or {})
    if auto:
        guessed = guess_mapping(list(work.columns))
        for key, value in guessed.items():
            resolved.setdefault(key, value)

    renamed: dict[str, str] = {}
    extras: dict[str, str] = {}
    for canonical, source in resolved.items():
        if source not in work.columns:
            continue
        if canonical in {
            "alt_m",
            "gs_mps",
            "vert_rate_mps",
        }:
            extras[canonical] = source
        else:
            renamed[source] = canonical
    out = work.rename(columns=renamed)

    if "alt_ft" not in out.columns and "alt_m" in extras:
        out["alt_ft"] = pd.to_numeric(work[extras["alt_m"]], errors="coerce") * FT_PER_M
    if "gs_kt" not in out.columns and "gs_mps" in extras:
        out["gs_kt"] = pd.to_numeric(work[extras["gs_mps"]], errors="coerce") * KT_PER_MPS
    if "vert_rate_fpm" not in out.columns and "vert_rate_mps" in extras:
        out["vert_rate_fpm"] = pd.to_numeric(work[extras["vert_rate_mps"]], errors="coerce") * FPM_PER_MPS

    if "ts" in out.columns:
        ts = out["ts"]
        if pd.api.types.is_numeric_dtype(ts):
            sample = float(pd.to_numeric(ts, errors="coerce").dropna().iloc[0]) if len(ts) else 0.0
            if sample > 1e17:
                unit = "ns"
            elif sample > 1e14:
                unit = "us"
            elif sample > 1e11:
                unit = "ms"
            else:
                unit = "s"
            out["ts"] = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")

    return ensure_canonical(out)
