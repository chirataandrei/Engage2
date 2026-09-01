"""Linear 4D extrapolation baseline (constant heading, GS, VS)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kastgat.data.schema import NM_PER_DEG_LAT, ensure_canonical


def extrapolate_track(row: pd.Series, horizon_s: np.ndarray) -> pd.DataFrame:
    dist_nm = float(row.gs_kt) * (horizon_s / 3600.0)
    rad = np.deg2rad(float(row.track_deg))
    lat = float(row.lat) + (dist_nm * np.cos(rad)) / NM_PER_DEG_LAT
    lon = float(row.lon) + (dist_nm * np.sin(rad)) / (
        NM_PER_DEG_LAT * np.cos(np.deg2rad(float(row.lat)))
    )
    alt = float(row.alt_ft) + float(row.vert_rate_fpm) * (horizon_s / 60.0)
    return pd.DataFrame({"horizon_s": horizon_s, "lat": lat, "lon": lon, "alt_ft": alt})


def predict_snapshot(snap: pd.DataFrame, horizon_steps: int = 6, dt_s: int = 10) -> dict[str, pd.DataFrame]:
    snap = ensure_canonical(snap)
    hs = np.arange(1, horizon_steps + 1) * dt_s
    return {
        str(row.flight_id): extrapolate_track(row, hs) for row in snap.itertuples(index=False)
    }


def rmse_nm_fl(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """pred/truth [N, H, 3] lat, lon, alt_ft."""
    dlat = pred[..., 0] - truth[..., 0]
    dlon = pred[..., 1] - truth[..., 1]
    horiz_nm = np.sqrt((dlat * 60.0) ** 2 + (dlon * 60.0) ** 2)
    vert_fl = (pred[..., 2] - truth[..., 2]) / 100.0
    return {
        "rmse_nm": float(np.sqrt(np.mean(horiz_nm**2))),
        "rmse_fl": float(np.sqrt(np.mean(vert_fl**2))),
        "mae_nm": float(np.mean(np.abs(horiz_nm))),
    }
