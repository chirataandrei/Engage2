"""Physics-ish sector generator with injected conflicts and ATCO events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kastgat.data.schema import INTENT_CLASSES, NM_PER_DEG_LAT

AIRCRAFT_TYPES = ("B738", "A320", "A20N", "B77W", "E190", "A359")


@dataclass
class SectorSpec:
    lat0: float = 45.81  # Zagreb
    lon0: float = 16.03
    n_aircraft: int = 10
    duration_s: int = 600
    dt_s: int = 10
    seed: int = 42


def _offset(lat0: float, lon0: float, north_nm: float, east_nm: float) -> tuple[float, float]:
    lat = lat0 + north_nm / NM_PER_DEG_LAT
    lon = lon0 + east_nm / (NM_PER_DEG_LAT * np.cos(np.deg2rad(lat0)))
    return lat, lon


def _step(lat, lon, gs_kt, track_deg, alt_ft, vs_fpm, dt_s: float) -> tuple[float, float, float]:
    dist_nm = gs_kt * (dt_s / 3600.0)
    rad = np.deg2rad(track_deg)
    dlat = (dist_nm * np.cos(rad)) / NM_PER_DEG_LAT
    dlon = (dist_nm * np.sin(rad)) / (NM_PER_DEG_LAT * np.cos(np.deg2rad(lat)))
    return lat + dlat, lon + dlon, alt_ft + vs_fpm * (dt_s / 60.0)


def generate_synthetic_traffic(spec: SectorSpec | None = None) -> pd.DataFrame:
    spec = spec or SectorSpec()
    rng = np.random.default_rng(spec.seed)
    t0 = pd.Timestamp("2026-10-20T12:00:00Z")
    n_steps = spec.duration_s // spec.dt_s + 1
    rows: list[dict] = []

    # Two head-on pairs at the same FL to guarantee labeled conflicts.
    forced = [
        dict(north=-40, east=-8, track=90.0, alt=35000.0, gs=450.0, vs=0.0, event_at=18, event="DESCEND"),
        dict(north=-40, east=40, track=270.0, alt=35000.0, gs=460.0, vs=0.0, event_at=18, event="CLIMB"),
        dict(north=25, east=-30, track=135.0, alt=37000.0, gs=430.0, vs=0.0, event_at=22, event="HEADING_RIGHT"),
        dict(north=-15, east=35, track=315.0, alt=37000.0, gs=440.0, vs=0.0, event_at=22, event="HEADING_LEFT"),
    ]
    extras = spec.n_aircraft - len(forced)
    for _ in range(max(0, extras)):
        forced.append(
            dict(
                north=float(rng.uniform(-50, 50)),
                east=float(rng.uniform(-50, 50)),
                track=float(rng.uniform(0, 360)),
                alt=float(rng.choice([31000, 33000, 35000, 37000, 39000])),
                gs=float(rng.uniform(400, 490)),
                vs=0.0,
                event_at=None,
                event="NONE",
            )
        )

    for i, ac in enumerate(forced):
        lat, lon = _offset(spec.lat0, spec.lon0, ac["north"], ac["east"])
        alt, gs, track, vs = ac["alt"], ac["gs"], ac["track"], ac["vs"]
        fid = f"SYN{i:03d}"
        actype = AIRCRAFT_TYPES[i % len(AIRCRAFT_TYPES)]
        event_step = ac["event_at"]
        pending = ac["event"]
        for k in range(n_steps):
            event = "NONE"
            if event_step is not None and k == event_step:
                event = pending
                if pending == "CLIMB":
                    vs = 1500.0
                elif pending == "DESCEND":
                    vs = -1500.0
                elif pending == "HEADING_RIGHT":
                    track = (track + 25.0) % 360.0
                elif pending == "HEADING_LEFT":
                    track = (track - 25.0) % 360.0
                elif pending == "SPEED_REDUCE":
                    gs *= 0.92
            lat, lon, alt = _step(lat, lon, gs, track, alt, vs, spec.dt_s)
            # small process noise
            lat += float(rng.normal(0, 0.0003))
            lon += float(rng.normal(0, 0.0003))
            rows.append(
                {
                    "flight_id": fid,
                    "ts": t0 + pd.Timedelta(seconds=k * spec.dt_s),
                    "lat": lat,
                    "lon": lon,
                    "alt_ft": alt,
                    "gs_kt": gs + float(rng.normal(0, 1.5)),
                    "track_deg": track,
                    "vert_rate_fpm": vs,
                    "aircraft_type": actype,
                    "atco_event": event,
                    "callsign": fid,
                    "onground": False,
                }
            )
    return pd.DataFrame(rows)


def generate_unseen_raw_csv(path, spec: SectorSpec | None = None) -> None:
    """Rehearsal dataset with *different* column names and mixed units."""
    spec = spec or SectorSpec(seed=99, n_aircraft=8, duration_s=420)
    df = generate_synthetic_traffic(spec)
    raw = pd.DataFrame(
        {
            "icao": df["flight_id"],
            "time_epoch": (df["ts"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds(),
            "latitude": df["lat"],
            "longitude": df["lon"],
            "altitude_m": df["alt_ft"] / 3.280839895,
            "speed_mps": df["gs_kt"] / 1.943844492,
            "heading": df["track_deg"],
            "vz_mps": df["vert_rate_fpm"] / 196.8503937,
            "typecode": df["aircraft_type"],
            "clearance": df["atco_event"],
        }
    )
    path = str(path)
    raw.to_csv(path, index=False)


# referenced so ruff does not flag unused
_ = INTENT_CLASSES
