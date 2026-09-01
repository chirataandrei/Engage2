#!/usr/bin/env python3
"""CPU smoke test: ingest → graphs → forward pass → rules → baselines. Target < 60 s."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from kastgat.baselines.linear import predict_snapshot, rmse_nm_fl
from kastgat.baselines.state_based import detect
from kastgat.data.clean import clean_tracks
from kastgat.data.datamodule import collate_windows
from kastgat.data.graphs import build_windows
from kastgat.data.sources.synthetic import SectorSpec, generate_synthetic_traffic
from kastgat.models.kastgat import KASTGAT
from kastgat.reasoning.rules import evaluate_snapshot
from kastgat.train.lit_module import KASTGATModule
from kastgat.utils.config import load_config, project_root


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    spec = SectorSpec(n_aircraft=6, duration_s=240, dt_s=10, seed=1)
    df = clean_tracks(generate_synthetic_traffic(spec), resample_s=10)
    samples = build_windows(
        df,
        history_steps=8,
        horizon_steps=4,
        max_nodes=8,
        proximity_nm=30.0,
        stride=4,
    )
    assert samples, "expected at least one window"
    batch = collate_windows(
        [
            {
                "x": s.x,
                "node_mask": s.node_mask,
                "y_traj": s.y_traj,
                "y_conflict": s.y_conflict,
                "y_intent": s.y_intent,
                "edge_index": s.edge_index,
                "edge_attr": s.edge_attr,
            }
            for s in samples[:2]
        ]
    )
    model = KASTGAT(in_dim=7, hidden_dim=32, gat_heads=4, gat_layers=1, tcn_channels=[32], horizon=4)
    model.eval()
    with torch.no_grad():
        out = model(batch["x"], batch["node_mask"], batch["edge_index"])
    assert out["traj"].shape[-1] == 3
    assert out["attention"] is not None

    snap = df[df["ts"] == df["ts"].max()].drop_duplicates("flight_id")
    _ = detect(snap)
    _ = evaluate_snapshot(snap)
    preds = predict_snapshot(snap, horizon_steps=4)
    assert preds

    # Lightning module constructs
    tiny_cfg = dict(cfg)
    tiny_cfg["horizon_steps"] = 4
    tiny_cfg["features"] = {"node": ["a"] * 7}
    tiny_cfg["model"] = {"hidden_dim": 32, "gat_heads": 4, "gat_layers": 1, "tcn_channels": [32], "dropout": 0.0}
    _ = KASTGATModule(tiny_cfg)

    elapsed = time.perf_counter() - t0
    print(f"SMOKE OK  windows={len(samples)}  traj={tuple(out['traj'].shape)}  {elapsed:.2f}s")
    if elapsed > 60:
        raise SystemExit(f"smoke test too slow: {elapsed:.1f}s")
    root = project_root()
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
