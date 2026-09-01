"""Turn GAT attention into human-readable ATCO-style explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kastgat.data.schema import ID_TO_INTENT


def top_attention_partners(
    flight_ids: list[str],
    attention: np.ndarray,
    node_mask: np.ndarray,
    top_k: int = 3,
) -> dict[str, list[tuple[str, float]]]:
    """attention: [H, N, N] or [N, N]."""
    if attention.ndim == 3:
        attn = attention.mean(axis=0)
    else:
        attn = attention
    n = min(len(flight_ids), attn.shape[0])
    out: dict[str, list[tuple[str, float]]] = {}
    for i in range(n):
        if node_mask[i] < 0.5 or not flight_ids[i]:
            continue
        scores = []
        for j in range(n):
            if i == j or node_mask[j] < 0.5 or not flight_ids[j]:
                continue
            scores.append((flight_ids[j], float(attn[i, j])))
        scores.sort(key=lambda x: x[1], reverse=True)
        total = sum(s for _, s in scores) or 1.0
        out[flight_ids[i]] = [(fid, s / total) for fid, s in scores[:top_k]]
    return out


def explain_clearance(
    flight_id: str,
    intent_id: int,
    partners: list[tuple[str, float]],
    tcpa_s: float | None = None,
) -> str:
    intent = ID_TO_INTENT.get(int(intent_id), "NONE")
    if not partners:
        return f"{flight_id}: {intent} (no nearby traffic in the attention field)."
    top_id, top_w = partners[0]
    pct = 100.0 * top_w
    time_bit = f", conflict predicted in {tcpa_s:.0f}s" if tcpa_s is not None and tcpa_s > 0 else ""
    return (
        f"{intent} for {flight_id} because the model allocated {pct:.0f}% of its attention "
        f"to {top_id}{time_bit}."
    )


def explanation_table(
    snap: pd.DataFrame,
    intent_ids: np.ndarray,
    attention: np.ndarray,
    node_mask: np.ndarray,
    tcpa: dict[str, float] | None = None,
) -> pd.DataFrame:
    ids = snap["flight_id"].astype(str).tolist()
    partners = top_attention_partners(ids, attention, node_mask)
    rows = []
    for i, fid in enumerate(ids):
        text = explain_clearance(fid, int(intent_ids[i]), partners.get(fid, []), (tcpa or {}).get(fid))
        rows.append({"flight_id": fid, "intent": ID_TO_INTENT.get(int(intent_ids[i]), "NONE"), "explanation": text})
    return pd.DataFrame(rows)
