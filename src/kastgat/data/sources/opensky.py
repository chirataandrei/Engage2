"""OpenSky live REST + weekly-snapshot ingest."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from kastgat.data.adapters import adapt_frame
from kastgat.data.schema import ensure_canonical

# OpenSky live state-vector column order (REST /states/all).
LIVE_COLUMNS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "lon",
    "lat",
    "baroaltitude",
    "onground",
    "velocity",
    "heading",
    "vertrate",
    "sensors",
    "geoaltitude",
    "squawk",
    "spi",
    "position_source",
]


def live_states_url(bbox: dict[str, float] | None = None) -> str:
    base = "https://opensky-network.org/api/states/all"
    if not bbox:
        return base
    return f"{base}?{urlencode(bbox)}"


def fetch_live_states(bbox: dict[str, float] | None = None, timeout: int = 30) -> dict:
    with urlopen(live_states_url(bbox), timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def snapshot_to_frame(payload: dict) -> pd.DataFrame:
    rows = payload.get("states") or []
    n_cols = len(LIVE_COLUMNS)
    trimmed = [r[:n_cols] + [None] * max(0, n_cols - len(r)) for r in rows]
    df = pd.DataFrame(trimmed, columns=LIVE_COLUMNS)
    df["time"] = payload.get("time")
    return df


def load_opensky_json(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text())
    raw = snapshot_to_frame(payload)
    return ensure_canonical(
        adapt_frame(
            raw,
            {
                "flight_id": "icao24",
                "ts": "time",
                "lat": "lat",
                "lon": "lon",
                "alt_m": "baroaltitude",
                "gs_mps": "velocity",
                "track_deg": "heading",
                "vert_rate_mps": "vertrate",
                "callsign": "callsign",
                "onground": "onground",
            },
        )
    )


def load_opensky_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return adapt_frame(df, auto=True)


def try_trino_history(start: str, stop: str, bbox: tuple[float, float, float, float]):
    """Optional path — requires approved Trino credentials and pyopensky."""
    try:
        from pyopensky.trino import Trino
    except ImportError as exc:
        raise RuntimeError("Install extras: uv sync --extra opensky") from exc
    west, south, east, north = bbox
    return Trino().history(start, stop, bounds=(west, south, east, north))
