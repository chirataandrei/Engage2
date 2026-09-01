"""Streamlit ATCO dashboard: traffic, predictions, attention explanations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import torch

from kastgat.baselines.linear import predict_snapshot
from kastgat.baselines.state_based import detect
from kastgat.data.schema import ID_TO_INTENT
from kastgat.reasoning.rules import evaluate_snapshot, filter_intents
from kastgat.utils.config import load_config, project_root
from kastgat.xai.attention import explanation_table, top_attention_partners


def _load_traffic() -> pd.DataFrame:
    root = project_root()
    parquet = root / "data" / "processed" / "canonical.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    from kastgat.data.clean import clean_tracks
    from kastgat.data.sources.synthetic import generate_synthetic_traffic

    df = clean_tracks(generate_synthetic_traffic())
    parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet, index=False)
    return df


@st.cache_resource
def _try_model():
    root = project_root()
    ckpt = root / "checkpoints" / "last.ckpt"
    if not ckpt.exists():
        return None
    try:
        from kastgat.train.lit_module import KASTGATModule

        cfg = load_config()
        return KASTGATModule.load_from_checkpoint(ckpt, cfg=cfg, map_location="cpu")
    except Exception:
        return None


def _attention_arcs(snap: pd.DataFrame, partners: dict) -> pd.DataFrame:
    rows = []
    pos = snap.set_index("flight_id")
    for fid, plist in partners.items():
        if fid not in pos.index:
            continue
        a = pos.loc[fid]
        for other, w in plist:
            if other not in pos.index:
                continue
            b = pos.loc[other]
            rows.append(
                {
                    "from_lon": float(a["lon"]),
                    "from_lat": float(a["lat"]),
                    "to_lon": float(b["lon"]),
                    "to_lat": float(b["lat"]),
                    "weight": float(w),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="KA-STGAT Engage 2", layout="wide")
    st.title("KA-STGAT — Artificial Situational Awareness")
    st.caption("Engage 2 hackathon kit · EASA Level 2 explanations · ICAO 5 NM / 1000 ft safety net")

    df = _load_traffic()
    times = np.sort(df["ts"].unique())
    idx = st.sidebar.slider("Time index", 0, max(0, len(times) - 1), min(20, len(times) - 1))
    t = times[idx]
    snap = df[df["ts"] == t].drop_duplicates("flight_id").reset_index(drop=True)

    det = detect(snap)
    snap = snap.merge(det[["flight_id", "conflict"]], on="flight_id", how="left")
    preds = predict_snapshot(snap)

    model = _try_model()
    intent_ids = np.zeros(len(snap), dtype=int)
    attn = np.eye(len(snap), dtype=np.float32)
    if model is not None:
        st.sidebar.success("Loaded checkpoints/last.ckpt")
    else:
        st.sidebar.info("No checkpoint — showing geometric baseline + inverse-distance attention.")
        # geometric stand-in so the XAI panel is never empty
        from kastgat.data.features import haversine_nm

        n = len(snap)
        attn = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = haversine_nm(snap.lat.iloc[i], snap.lon.iloc[i], snap.lat.iloc[j], snap.lon.iloc[j])
                attn[i, j] = 1.0 / max(float(d), 0.1)
            s = attn[i].sum()
            if s > 0:
                attn[i] /= s
        intent_ids = np.where(snap["conflict"].to_numpy(), 2, 0)  # DESCEND vs NONE heuristic

    mask = np.ones(len(snap), dtype=np.float32)
    partners = top_attention_partners(snap["flight_id"].astype(str).tolist(), attn, mask)
    verdicts = evaluate_snapshot(snap, intent_ids=intent_ids)
    allowed = filter_intents(verdicts)
    explain_df = explanation_table(snap, intent_ids, attn, mask)

    left, right = st.columns([2, 1])
    with left:
        view = pdk.ViewState(
            latitude=float(snap["lat"].mean()),
            longitude=float(snap["lon"].mean()),
            zoom=7,
            pitch=45,
        )
        trails = []
        for fid, pred in preds.items():
            row = snap[snap.flight_id.astype(str) == fid]
            if row.empty:
                continue
            trails.append(
                {
                    "path": [[float(row.lon.iloc[0]), float(row.lat.iloc[0])]]
                    + pred[["lon", "lat"]].values.tolist(),
                    "conflict": bool(row.conflict.iloc[0]),
                }
            )
        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=snap.assign(color=np.where(snap["conflict"], 255, 30)),
                get_position="[lon, lat]",
                get_fill_color="[color, 80, 180, 200]",
                get_radius=2500,
                pickable=True,
            ),
            pdk.Layer(
                "TextLayer",
                data=snap,
                get_position="[lon, lat]",
                get_text="flight_id",
                get_size=14,
                get_color=[240, 240, 240],
                get_alignment_baseline="top",
            ),
            pdk.Layer(
                "PathLayer",
                data=trails,
                get_path="path",
                get_width=3,
                get_color="[255, 180, 40]",
            ),
        ]
        arcs = _attention_arcs(snap, partners)
        if len(arcs):
            layers.append(
                pdk.Layer(
                    "ArcLayer",
                    data=arcs,
                    get_source_position="[from_lon, from_lat]",
                    get_target_position="[to_lon, to_lat]",
                    get_source_color="[0, 180, 255, 160]",
                    get_target_color="[255, 80, 80, 200]",
                    get_width="weight * 8",
                )
            )
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{flight_id}"}))

    with right:
        st.subheader("Explainability")
        st.dataframe(explain_df, hide_index=True, use_container_width=True)
        st.subheader("Safety net (ICAO)")
        blocked = [v for v in verdicts if not v.allowed]
        if blocked:
            st.warning(f"{len(blocked)} pair-checks blocked a neural clearance.")
            st.json({v.flight_id: v.reason for v in blocked[:8]})
        else:
            st.success("No hard ICAO violations on this snapshot.")
        st.caption("Allowed intents after the reasoning engine")
        st.json(allowed)

    st.subheader("Snapshot")
    show = snap.copy()
    show["intent"] = [ID_TO_INTENT.get(int(i), "NONE") for i in intent_ids]
    st.dataframe(
        show[["flight_id", "lat", "lon", "alt_ft", "gs_kt", "track_deg", "conflict", "intent"]],
        hide_index=True,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
