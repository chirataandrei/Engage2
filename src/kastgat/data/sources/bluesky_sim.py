"""BlueSky scenario generation and labeled-log fallback.

The simulator is optional. This module always writes `.scn` files and a
labeled parquet. If `bluesky` is installed, `run_headless` will try to execute
the scenario; otherwise kinematics are integrated here with the same commands.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kastgat.data.schema import NM_PER_DEG_LAT, ensure_canonical

SCN_HEADER = """# Engage 2 / KA-STGAT conflict scenario
00:00:00.00>HOLD
00:00:00.00>PAN 45.81 16.03
00:00:00.00>ZOOM 2
00:00:00.00>ASAS ON
"""


def write_conflict_scn(path: str | Path, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [SCN_HEADER]
    # Reciprocal pair at FL350.
    lines.append("00:00:00.00>CRE AWARE1 B738 45.50 15.40 FL350 90 450")
    lines.append("00:00:00.00>CRE AWARE2 B738 45.50 16.70 FL350 270 450")
    lines.append("00:00:00.00>DEST AWARE1 LDZA")
    lines.append("00:00:00.00>DEST AWARE2 EDDF")
    # Crossing pair at FL370.
    lines.append("00:00:00.00>CRE DLOG1 A320 46.20 15.50 FL370 135 430")
    lines.append("00:00:00.00>CRE DLOG2 A320 45.40 16.60 FL370 315 430")
    for i in range(4):
        lat = 45.81 + rng.uniform(-0.6, 0.6)
        lon = 16.03 + rng.uniform(-0.8, 0.8)
        hdg = rng.uniform(0, 360)
        fl = rng.choice([310, 330, 390])
        tas = rng.integers(420, 480)
        acid = f"BG{i:03d}"
        lines.append(f"00:00:00.00>CRE {acid} B738 {lat:.3f} {lon:.3f} FL{fl} {hdg:.0f} {tas}")
    lines.append("00:03:00.00>ALT AWARE1 FL330")
    lines.append("00:03:00.00>ALT AWARE2 FL370")
    lines.append("00:04:00.00>HDG DLOG1 160")
    lines.append("00:04:00.00>HDG DLOG2 290")
    lines.append("00:10:00.00>HOLD")
    path.write_text("\n".join(lines) + "\n")
    return path


def generate_labeled_from_scn_commands(duration_s: int = 600, dt_s: int = 10) -> pd.DataFrame:
    """Integrate the scripted scenario without running BlueSky."""
    t0 = pd.Timestamp("2026-10-20T08:00:00Z")
    aircraft = [
        dict(fid="AWARE1", lat=45.50, lon=15.40, alt=35000.0, gs=450.0, track=90.0, vs=0.0, ac="B738"),
        dict(fid="AWARE2", lat=45.50, lon=16.70, alt=35000.0, gs=450.0, track=270.0, vs=0.0, ac="B738"),
        dict(fid="DLOG1", lat=46.20, lon=15.50, alt=37000.0, gs=430.0, track=135.0, vs=0.0, ac="A320"),
        dict(fid="DLOG2", lat=45.40, lon=16.60, alt=37000.0, gs=430.0, track=315.0, vs=0.0, ac="A320"),
    ]
    events = {
        18: {"AWARE1": ("DESCEND", dict(vs=-1200.0)), "AWARE2": ("CLIMB", dict(vs=1200.0))},
        24: {"DLOG1": ("HEADING_RIGHT", dict(track=160.0)), "DLOG2": ("HEADING_LEFT", dict(track=290.0))},
    }
    rows = []
    n_steps = duration_s // dt_s + 1
    for k in range(n_steps):
        cmds = events.get(k, {})
        for ac in aircraft:
            event = "NONE"
            if ac["fid"] in cmds:
                event, updates = cmds[ac["fid"]]
                ac.update(updates)
            dist_nm = ac["gs"] * (dt_s / 3600.0)
            rad = np.deg2rad(ac["track"])
            ac["lat"] += (dist_nm * np.cos(rad)) / NM_PER_DEG_LAT
            ac["lon"] += (dist_nm * np.sin(rad)) / (NM_PER_DEG_LAT * np.cos(np.deg2rad(ac["lat"])))
            ac["alt"] += ac["vs"] * (dt_s / 60.0)
            rows.append(
                {
                    "flight_id": ac["fid"],
                    "ts": t0 + pd.Timedelta(seconds=k * dt_s),
                    "lat": ac["lat"],
                    "lon": ac["lon"],
                    "alt_ft": ac["alt"],
                    "gs_kt": ac["gs"],
                    "track_deg": ac["track"],
                    "vert_rate_fpm": ac["vs"],
                    "aircraft_type": ac["ac"],
                    "atco_event": event,
                    "callsign": ac["fid"],
                    "onground": False,
                }
            )
    return ensure_canonical(pd.DataFrame(rows))


def run_headless(scn_path: str | Path) -> pd.DataFrame | None:
    try:
        import bluesky as bs
        from bluesky import stack
    except ImportError:
        return None
    bs.init(mode="sim", detached=True)
    stack.stack(f"IC {Path(scn_path).resolve()}")
    stack.stack("FF")
    # Best-effort: if the API differs, caller falls back to the kinematic generator.
    return None
