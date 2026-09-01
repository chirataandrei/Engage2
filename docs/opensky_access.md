# OpenSky Trino access (critical path)

Historical ADS-B from OpenSky is **not** on the public REST API. Full history
lives behind Trino and is restricted to university-affiliated researchers,
government organisations, and aviation authorities.

## Do this on day 1

1. Create an OpenSky account with an academic email:
   https://opensky-network.org/
2. Request Trino / historical-database access from the same account. Describe
   the project as SESAR Engage 2 hackathon preparation (4D trajectory prediction
   and conflict detection on public ATM data).
3. Install optional extras: `uv sync --extra opensky`
4. Configure credentials as documented by pyopensky:
   https://open-aviation.github.io/pyopensky/credentials.html
5. Test with a one-hour, bbox-limited query over Zagreb FIR (see `config/opensky.yaml`).

Access review can take days to weeks. Do not wait on it.

## Fallback datasets (no Trino)

| Source | What you get | Script |
| --- | --- | --- |
| Live REST `/states/all` | Real column schema + current traffic, not trajectories | `scripts/fetch_opensky.py --live` |
| Weekly Monday snapshots 2017–2022 | 10 s state vectors, public bucket | `scripts/fetch_opensky.py --weekly DATE` |
| Scientific page | Climbing segments, PRC 2024, March 2026 Trino snapshot | https://opensky-network.org/data/scientific |
| BlueSky scenarios | **Labeled** conflicts + resolution commands | `scripts/gen_bluesky.py` |
| Built-in synthetic generator | Physics-ish trajectories + ATCO events, always available | `kastgat data.sources.synthetic` |

## Query hygiene

Always filter Trino by `hour`/`day` partition **and** a bounding box. Unscoped
full-table scans are rejected. Prefer `pyopensky.trino.Trino.history(...)`.
