#!/usr/bin/env python3
"""Timed rehearsal: unseen schema → dashboard artefacts. Target < 90 minutes (usually seconds)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from kastgat.data.sources.synthetic import SectorSpec, generate_unseen_raw_csv
from kastgat.pipeline import run_pipeline
from kastgat.utils.config import project_root


REHEARSAL_MAPPING = {
    "mapping": {
        "flight_id": "icao",
        "ts": "time_epoch",
        "lat": "latitude",
        "lon": "longitude",
        "alt_m": "altitude_m",
        "gs_mps": "speed_mps",
        "track_deg": "heading",
        "vert_rate_mps": "vz_mps",
        "aircraft_type": "typecode",
        "atco_event": "clearance",
    }
}


def main() -> None:
    root = project_root()
    raw_dir = root / "data" / "rehearsal"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / "raw_unknown.csv"
    generate_unseen_raw_csv(raw_csv, SectorSpec(seed=99, n_aircraft=8, duration_s=420))
    mapping_path = raw_dir / "mapping.yaml"
    import yaml

    mapping_path.write_text(yaml.safe_dump(REHEARSAL_MAPPING))
    out_dir = raw_dir / "out"
    t0 = time.perf_counter()
    stats = run_pipeline(raw_csv, mapping_path, out_dir)
    elapsed = time.perf_counter() - t0
    stats["seconds_wall"] = elapsed
    report = raw_dir / "timing.json"
    report.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if stats["n_windows"] < 1 or stats["n_rows"] < 10:
        raise SystemExit("rehearsal produced no usable windows — mapping/units are wrong")
    if elapsed > 90 * 60:
        raise SystemExit("rehearsal exceeded 90 minutes")
    print(f"REHEARSAL OK in {elapsed:.2f}s  (budget 90 min)")


if __name__ == "__main__":
    main()
