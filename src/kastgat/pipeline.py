"""One-command raw CSV → canonical parquet → graph windows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from kastgat.data.adapters import adapt_frame, mapping_from_config
from kastgat.data.clean import clean_tracks
from kastgat.data.datamodule import save_samples
from kastgat.data.graphs import build_windows
from kastgat.data.schema import CANONICAL_REQUIRED
from kastgat.utils.config import load_config, project_root


def _load_raw(raw: Path, mapping_yaml: str | Path | None) -> pd.DataFrame:
    suffix = raw.suffix.lower()
    mapping = mapping_from_config(mapping_yaml) if mapping_yaml else {}
    if suffix == ".json":
        from kastgat.data.sources.opensky import load_opensky_json

        return load_opensky_json(raw)
    if suffix == ".parquet":
        df = pd.read_parquet(raw)
    else:
        df = pd.read_csv(raw)
    if all(col in df.columns for col in CANONICAL_REQUIRED) and not mapping:
        return df
    return adapt_frame(df, mapping, auto=True)


def run_pipeline(
    raw_path: str | Path,
    mapping_yaml: str | Path | None,
    out_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, float]:
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    df = _load_raw(Path(raw_path), mapping_yaml)
    t_adapt = time.perf_counter()
    df = clean_tracks(df, resample_s=int(cfg.get("resample_s", 10)))
    t_clean = time.perf_counter()
    samples = build_windows(
        df,
        history_steps=int(cfg["history_steps"]),
        horizon_steps=int(cfg["horizon_steps"]),
        max_nodes=int(cfg["max_nodes"]),
        proximity_nm=float(cfg["proximity_nm"]),
        separation_nm=float(cfg["separation_nm"]),
        separation_ft=float(cfg["separation_ft"]),
    )
    t_graphs = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "canonical.parquet", index=False)
    save_samples(samples, out / "windows.pt")
    t_end = time.perf_counter()
    return {
        "n_rows": float(len(df)),
        "n_flights": float(df["flight_id"].nunique()),
        "n_windows": float(len(samples)),
        "seconds_total": t_end - t0,
        "seconds_read_adapt": t_adapt - t0,
        "seconds_clean": t_clean - t_adapt,
        "seconds_graphs": t_graphs - t_clean,
        "seconds_write": t_end - t_graphs,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    root = project_root()
    out = args.out or (root / "data" / "processed")
    stats = run_pipeline(args.raw, args.mapping, out, args.config)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
