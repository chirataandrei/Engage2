from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from kastgat.data.clean import clean_tracks
from kastgat.data.datamodule import make_loaders, save_samples
from kastgat.data.graphs import build_windows
from kastgat.data.sources.synthetic import generate_synthetic_traffic
from kastgat.train.lit_module import KASTGATModule, precision_from_cfg
from kastgat.utils.config import load_config, project_root


def build_samples(cfg: dict, parquet_path: Path | None = None):
    if parquet_path and parquet_path.exists():
        import pandas as pd

        df = pd.read_parquet(parquet_path)
    else:
        df = generate_synthetic_traffic()
    df = clean_tracks(df, resample_s=int(cfg.get("resample_s", 10)))
    samples = build_windows(
        df,
        history_steps=int(cfg["history_steps"]),
        horizon_steps=int(cfg["horizon_steps"]),
        max_nodes=int(cfg["max_nodes"]),
        proximity_nm=float(cfg["proximity_nm"]),
        separation_nm=float(cfg["separation_nm"]),
        separation_ft=float(cfg["separation_ft"]),
    )
    return df, samples


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train KA-STGAT")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None, help="Optional canonical parquet")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.cpu:
        cfg["device"] = "cpu"
        cfg["train"]["amp"] = False
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs

    L.seed_everything(int(cfg.get("seed", 42)))
    data_path = Path(args.data) if args.data else None
    df, samples = build_samples(cfg, data_path)
    if not samples:
        raise SystemExit("No training windows — generate longer tracks or reduce history_steps.")
    root = project_root()
    save_samples(samples, root / "data" / "processed" / "windows.pt")
    df.to_parquet(root / "data" / "processed" / "canonical.parquet", index=False)

    train_loader, val_loader = make_loaders(
        samples,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )
    module = KASTGATModule(cfg)
    ckpt_dir = root / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="kastgat-{epoch:02d}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=int(cfg["train"]["early_stopping_patience"]),
            mode="min",
        ),
    ]
    trainer = L.Trainer(
        max_epochs=int(cfg["train"]["max_epochs"]),
        accelerator="cpu" if args.cpu else "auto",
        devices=1,
        precision=precision_from_cfg(cfg),
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=1,
        default_root_dir=str(root / cfg["paths"]["logs"]),
        limit_train_batches=cfg["train"].get("limit_train_batches"),
    )
    trainer.fit(module, train_loader, val_loader)
    last = ckpt_dir / "last.ckpt"
    trainer.save_checkpoint(last)
    print(f"Saved checkpoint to {last}")


if __name__ == "__main__":
    main()
