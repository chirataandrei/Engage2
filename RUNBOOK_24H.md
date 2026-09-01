# 24-hour runbook — Engage 2 Zagreb

The kit is already wired. Do not rewrite the model. Touch **one YAML** and fine-tune.

## Golden rule

Every hour there is a demoable artefact. If training is still running, the
**linear baseline + dashboard** is the demo.

## H+0 – H+2  Ingest

1. Copy the organiser files into `data/raw/`.
2. `uv run python -c "import pandas as pd; df=pd.read_csv('data/raw/hackathon.csv'); print(df.dtypes); print(df.head())"`
3. Fill `config/hackathon.yaml` (`flight_id`, `ts`, `lat`, `lon`, altitude, speed, heading, vertical rate).
4. Units: if altitude is metres use `alt_m`; if speed is m/s use `gs_mps`. The adapter converts.
5. `uv run kastgat-pipeline data/raw/hackathon.csv --mapping config/hackathon.yaml`
6. Confirm `data/processed/canonical.parquet` has more than one flight and 10 s sampling.

## H+2 – H+4  Baseline (mandatory checkpoint)

```bash
uv run python - <<'PY'
import pandas as pd
from kastgat.baselines import detect, predict_snapshot
df = pd.read_parquet("data/processed/canonical.parquet")
snap = df[df.ts==df.ts.max()].drop_duplicates("flight_id")
print(detect(snap).conflict.value_counts())
print(list(predict_snapshot(snap))[:5])
PY
uv run streamlit run app/dashboard.py
```

From this moment you can present *something* to the jury.

## H+4 – H+10  Fine-tune KA-STGAT

On Colab GPU (see `notebooks/colab_train.ipynb`) or locally:

```bash
uv run python -m kastgat.train.cli --data data/processed/canonical.parquet --max-epochs 15
```

Starts from whatever is in `checkpoints/last.ckpt` if you copy it next to the new run
(Lightning `save_last=True`). Watch `val/mse` and `val/conflict`.

If loss explodes: drop `train.lr` to `3e-4` in `config/base.yaml` and cut `max_nodes`.

## H+10 – H+16  Metrics vs baseline

Compare horizontal RMSE (NM) and conflict precision against the linear / state-based
baselines. Put a three-row table in the pitch:

| Method | RMSE NM | Conflict P / R | Explainable |
| --- | --- | --- | --- |
| Linear extrapolation | | n/a | no |
| State-based CD | n/a | | rules only |
| KA-STGAT + ICAO net | | | GAT attention |

## H+16 – H+20  Dashboard on real data

`uv run streamlit run app/dashboard.py`

Check:

- predicted trails sit on the map
- red aircraft = conflict flag
- cyan→red arcs = attention
- right panel sentence: *"DESCEND for AWARE1 because the model allocated 85% of its attention to AWARE2"*
- safety net blocks illegal clearances

## H+20 – H+23  Pitch

90 seconds:

1. Problem — ATCO out-of-the-loop + nuisance alerts (AWARE / DIALOG).
2. Demo — one conflict, one explanation, one blocked illegal climb.
3. Compliance — EASA AI Roadmap Level 2, human stays in the loop.
4. Numbers — the three-row table.

## H+23 – H+24  Freeze

No more training. Zip `checkpoints/last.ckpt`, `data/processed/canonical.parquet`,
and this repo. Rehearse the demo once more.

## If the dataset is hostile

| Symptom | Move |
| --- | --- |
| Unknown column names | `guess_mapping(df.columns)` then write YAML |
| One snapshot, no trajectories | dashboard + state-based CD only; skip TCN fine-tune |
| No ATCO labels | conflict head still trains from geometry; intent becomes heuristic |
| GPU missing | `--cpu --max-epochs 5`; demo the baseline |
| Package install blocked | the dashboard imports only pandas/pydeck/streamlit + this repo |
