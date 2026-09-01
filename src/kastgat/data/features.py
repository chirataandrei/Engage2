"""Pairwise geometric features: range, CPA, TCPA, relative kinematics."""

from __future__ import annotations

import numpy as np

from kastgat.data.schema import M_PER_NM, NM_PER_DEG_LAT

KT_TO_NM_S = 1.0 / 3600.0


def haversine_nm(lat1, lon1, lat2, lon2) -> np.ndarray:
    r_nm = 3440.065  # mean Earth radius in nautical miles
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2 * r_nm * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def local_xy_nm(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    x = (lon - lon0) * NM_PER_DEG_LAT * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * NM_PER_DEG_LAT
    return x, y


def cpa_tcpa(
    lat_i, lon_i, gs_i, track_i,
    lat_j, lon_j, gs_j, track_j,
    lat0: float,
    lon0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear CPA/TCPA in a local tangent plane. TCPA in seconds, CPA in NM."""
    xi, yi = local_xy_nm(lat_i, lon_i, lat0, lon0)
    xj, yj = local_xy_nm(lat_j, lon_j, lat0, lon0)
    tr_i = np.deg2rad(track_i)
    tr_j = np.deg2rad(track_j)
    vxi = gs_i * np.sin(tr_i) * KT_TO_NM_S
    vyi = gs_i * np.cos(tr_i) * KT_TO_NM_S
    vxj = gs_j * np.sin(tr_j) * KT_TO_NM_S
    vyj = gs_j * np.cos(tr_j) * KT_TO_NM_S
    rx = xj - xi
    ry = yj - yi
    vx = vxj - vxi
    vy = vyj - vyi
    vv = vx * vx + vy * vy
    tcpa = np.where(vv > 1e-12, -(rx * vx + ry * vy) / vv, 0.0)
    tcpa = np.clip(tcpa, -3600.0, 3600.0)
    cpa = np.sqrt((rx + vx * tcpa) ** 2 + (ry + vy * tcpa) ** 2)
    return cpa, tcpa


def pair_features(nodes: dict[str, np.ndarray], i: int, j: int, lat0: float, lon0: float) -> np.ndarray:
    dist = haversine_nm(nodes["lat"][i], nodes["lon"][i], nodes["lat"][j], nodes["lon"][j])
    rel_gs = nodes["gs_kt"][j] - nodes["gs_kt"][i]
    cpa, tcpa = cpa_tcpa(
        nodes["lat"][i], nodes["lon"][i], nodes["gs_kt"][i], nodes["track_deg"][i],
        nodes["lat"][j], nodes["lon"][j], nodes["gs_kt"][j], nodes["track_deg"][j],
        lat0, lon0,
    )
    dalt = nodes["alt_ft"][j] - nodes["alt_ft"][i]
    return np.array([dist, rel_gs, float(cpa), float(tcpa), dalt], dtype=np.float32)


# keep M_PER_NM imported for callers that convert
_ = M_PER_NM
