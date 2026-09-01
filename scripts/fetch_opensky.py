#!/usr/bin/env python3
"""Fetch OpenSky data: live snapshot (no auth) or optional weekly/Trino paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kastgat.data.sources.opensky import fetch_live_states, load_opensky_json, snapshot_to_frame, try_trino_history
from kastgat.utils.config import load_yaml, project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", default=True)
    parser.add_argument("--weekly", default=None, help="YYYY-MM-DD (Monday snapshot if the public bucket has it)")
    parser.add_argument("--trino-start", default=None)
    parser.add_argument("--trino-stop", default=None)
    args = parser.parse_args()
    root = project_root()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(root / "config" / "opensky.yaml")
    bbox = cfg.get("bbox", {})

    if args.trino_start and args.trino_stop:
        bounds = (bbox["lomin"], bbox["lamin"], bbox["lomax"], bbox["lamax"])
        hist = try_trino_history(args.trino_start, args.trino_stop, bounds)
        out = raw_dir / "opensky_trino.parquet"
        hist.to_parquet(out)
        print(f"Wrote {out}")
        return

    if args.weekly:
        print(
            "Weekly Monday files are listed at https://opensky-network.org/datasets/states/ "
            f"— look for {args.weekly}. After download, run:\n"
            "  uv run kastgat-pipeline data/raw/STATES.csv --mapping config/opensky.yaml"
        )
        return

    payload = fetch_live_states(bbox)
    snap_path = raw_dir / "opensky_live_snapshot.json"
    snap_path.write_text(json.dumps(payload))
    frame = snapshot_to_frame(payload)
    csv_path = raw_dir / "opensky_live_snapshot.csv"
    frame.to_csv(csv_path, index=False)
    canonical = load_opensky_json(snap_path)
    out = root / "data" / "processed" / "opensky_live_canonical.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(out, index=False)
    print(f"Live aircraft: {len(canonical)}  time={payload.get('time')}  -> {out}")


if __name__ == "__main__":
    main()
