# KA-STGAT — Engage 2 Hackathon Starter Kit

Starter kit for the SESAR **Engage 2** hackathon (Zagreb, 20–21 October 2026). The
kit is built so that the 24-hour clock is spent on **schema mapping, fine-tuning,
and the pitch** — not on scaffolding.

The model is a **Knowledge-Aware Spatial-Temporal Graph Attention Network**:

1. **Canonical adapter** — organiser CSV → internal schema (`flight_id, ts, lat, lon, …`).
2. **Knowledge / rules layer** — ICAO 5 NM / 1000 ft safety net over neural outputs.
3. **GATv2 + TCN** — spatial attention (explainable) and fast temporal encoding.
4. **Multi-task heads** — 4D trajectory regression + conflict + ATCO intent.
5. **Streamlit + PyDeck dashboard** — traffic, predicted trails, attention arcs, EASA-style explanations.

## Quick start

Python **3.12** (the default Homebrew 3.14 cannot install this stack). `uv` is required.

```bash
cd Engage2
uv sync
uv run python scripts/smoke_test.py
uv run python scripts/gen_bluesky.py
uv run python scripts/fetch_opensky.py --live
uv run python -m kastgat.train.cli --cpu --max-epochs 3
uv run streamlit run app/dashboard.py
```

## What to do on day 1

Trino historical access is gated. Follow [docs/opensky_access.md](docs/opensky_access.md)
**today**. Until it lands, the kit trains on:

- a **synthetic Zagreb-FIR sector** with injected conflicts and ATCO events
- **BlueSky-style labeled scenarios** (`scripts/gen_bluesky.py`)
- a **live OpenSky REST snapshot** (real columns, not trajectories)

## Hackathon 24h

Fill in [config/hackathon.yaml](config/hackathon.yaml) and run:

```bash
uv run kastgat-pipeline data/raw/hackathon.csv --mapping config/hackathon.yaml
uv run python -m kastgat.train.cli --data data/processed/canonical.parquet
uv run streamlit run app/dashboard.py
```

Full hour-by-hour playbook: [RUNBOOK_24H.md](RUNBOOK_24H.md).

## Layout

```
config/            base + source-specific column maps
src/kastgat/       adapters, graphs, model, rules, training, XAI
app/dashboard.py   Streamlit ATCO view
scripts/           OpenSky fetch, BlueSky gen, smoke, rehearsal
notebooks/colab_train.ipynb
```

## Design constraints

- Device is `auto` (`cuda` → `mps` → `cpu`). Nothing assumes CUDA.
- `scripts/smoke_test.py` must pass on CPU in under 60 seconds.
- The dashboard works **without** a trained checkpoint (geometric baseline + inverse-distance attention).
