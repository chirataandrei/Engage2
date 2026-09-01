"""Resample, interpolate, unit-homogenise, and Z-score canonical tracks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from kastgat.data.schema import ensure_canonical


def _spline_resample(g: pd.DataFrame, freq: str, k: int) -> pd.DataFrame:
    g = g.sort_values("ts").drop_duplicates("ts")
    numeric = ["lat", "lon", "alt_ft", "gs_kt", "vert_rate_fpm"]
    if len(g) < 4:
        num = g.set_index("ts")[numeric].resample(freq).interpolate(method="linear")
        out = num.reset_index()
        out["flight_id"] = g["flight_id"].iloc[0]
        out["track_deg"] = g["track_deg"].iloc[-1]
        out["aircraft_type"] = g["aircraft_type"].iloc[0] if "aircraft_type" in g.columns else "B738"
        out["atco_event"] = "NONE"
        return out

    t0 = g["ts"].iloc[0]
    x = (g["ts"] - t0).dt.total_seconds().to_numpy()
    grid = pd.date_range(g["ts"].iloc[0], g["ts"].iloc[-1], freq=freq, tz="UTC")
    if len(grid) == 0:
        return g.reset_index(drop=True)
    xg = (grid - t0).total_seconds().to_numpy()

    out = pd.DataFrame({"ts": grid, "flight_id": g["flight_id"].iloc[0]})
    numeric = ["lat", "lon", "alt_ft", "gs_kt", "vert_rate_fpm"]
    for col in numeric:
        y = g[col].to_numpy(dtype=float)
        if np.all(~np.isfinite(y)):
            out[col] = np.nan
            continue
        mask = np.isfinite(y)
        k_use = min(k, int(mask.sum()) - 1)
        if k_use < 1:
            out[col] = np.interp(xg, x[mask], y[mask])
        else:
            spl = CubicSpline(x[mask], y[mask], bc_type="natural")
            out[col] = spl(xg)

    # unwrap heading then interpolate, wrap back to [0, 360)
    heading = np.unwrap(np.deg2rad(g["track_deg"].to_numpy(dtype=float)))
    heading_g = np.interp(xg, x, heading)
    out["track_deg"] = np.rad2deg(heading_g) % 360.0
    if "aircraft_type" in g.columns:
        out["aircraft_type"] = g["aircraft_type"].iloc[0]
    if "atco_event" in g.columns:
        events = g.set_index("ts")["atco_event"].reindex(grid, method="ffill")
        out["atco_event"] = events.to_numpy()
    if "callsign" in g.columns:
        out["callsign"] = g["callsign"].iloc[0]
    if "onground" in g.columns:
        out["onground"] = False
    return out


def resample_tracks(df: pd.DataFrame, resample_s: int = 10, spline_k: int = 3) -> pd.DataFrame:
    df = ensure_canonical(df)
    freq = f"{int(resample_s)}s"
    parts = [_spline_resample(g, freq, spline_k) for _, g in df.groupby("flight_id", sort=False)]
    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
    return ensure_canonical(out)


def drop_ground_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_canonical(df)
    if "onground" in out.columns:
        out = out.loc[~out["onground"].astype(bool)]
    out = out.loc[out["alt_ft"].between(500, 60_000)]
    out = out.loc[out["gs_kt"].between(50, 700)]
    out = out.loc[out["lat"].between(-90, 90) & out["lon"].between(-180, 180)]
    return out.reset_index(drop=True)


def attach_trig_heading(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rad = np.deg2rad(out["track_deg"].to_numpy(dtype=float))
    out["track_sin"] = np.sin(rad)
    out["track_cos"] = np.cos(rad)
    return out


class ZScoreScaler:
    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns or ["lat", "lon", "alt_ft", "gs_kt", "vert_rate_fpm"]
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> ZScoreScaler:
        for col in self.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            self.mean_[col] = float(series.mean())
            std = float(series.std(ddof=0))
            self.std_[col] = std if std > 1e-6 else 1.0
        return self

    def transform(self, df: pd.DataFrame, clip: float = 6.0) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            z = (pd.to_numeric(out[col], errors="coerce") - self.mean_[col]) / self.std_[col]
            out[f"{col}_z"] = z.clip(-clip, clip)
        return out

    def fit_transform(self, df: pd.DataFrame, clip: float = 6.0) -> pd.DataFrame:
        return self.fit(df).transform(df, clip=clip)


def clean_tracks(df: pd.DataFrame, resample_s: int = 10, spline_k: int = 3) -> pd.DataFrame:
    out = drop_ground_and_outliers(df)
    out = resample_tracks(out, resample_s=resample_s, spline_k=spline_k)
    out = attach_trig_heading(out)
    scaler = ZScoreScaler(["gs_kt", "vert_rate_fpm"]).fit(out)
    return scaler.transform(out)
