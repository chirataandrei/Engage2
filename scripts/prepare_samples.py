#!/usr/bin/env python3
"""Materialise synthetic + BlueSky labeled parquets used by training and the dashboard."""

from kastgat.data.clean import clean_tracks
from kastgat.data.sources.bluesky_sim import generate_labeled_from_scn_commands, write_conflict_scn
from kastgat.data.sources.synthetic import SectorSpec, generate_synthetic_traffic, generate_unseen_raw_csv
from kastgat.utils.config import project_root


def main() -> None:
    root = project_root()
    syn = clean_tracks(generate_synthetic_traffic(SectorSpec(n_aircraft=10, duration_s=600, seed=42)))
    blu = clean_tracks(generate_labeled_from_scn_commands())
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    syn.to_parquet(root / "data" / "processed" / "synthetic.parquet", index=False)
    blu.to_parquet(root / "data" / "processed" / "bluesky_labeled.parquet", index=False)
    import pandas as pd

    combined = pd.concat([syn, blu], ignore_index=True)
    combined.to_parquet(root / "data" / "processed" / "canonical.parquet", index=False)
    write_conflict_scn(root / "data" / "raw" / "engage2_conflicts.scn")
    (root / "data" / "rehearsal").mkdir(parents=True, exist_ok=True)
    generate_unseen_raw_csv(root / "data" / "rehearsal" / "raw_unknown.csv")
    print(f"synthetic={len(syn)} bluesky={len(blu)} combined={len(combined)}")


if __name__ == "__main__":
    main()
