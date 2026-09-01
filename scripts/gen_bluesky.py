#!/usr/bin/env python3
"""Write a BlueSky .scn plus a labeled parquet (works without BlueSky installed)."""

from __future__ import annotations

from pathlib import Path

from kastgat.data.sources.bluesky_sim import (
    generate_labeled_from_scn_commands,
    run_headless,
    write_conflict_scn,
)
from kastgat.utils.config import project_root


def main() -> None:
    root = project_root()
    scn = write_conflict_scn(root / "data" / "raw" / "engage2_conflicts.scn")
    print(f"Wrote scenario {scn}")
    ran = run_headless(scn)
    if ran is not None:
        df = ran
        print("Used BlueSky headless run")
    else:
        df = generate_labeled_from_scn_commands()
        print("BlueSky not installed — used kinematic fallback with the same commands")
    out = root / "data" / "processed" / "bluesky_labeled.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Labeled rows={len(df)} flights={df.flight_id.nunique()} -> {out}")


if __name__ == "__main__":
    main()
