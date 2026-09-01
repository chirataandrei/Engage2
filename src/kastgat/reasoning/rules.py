"""Deterministic ICAO separation safety net over neural outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kastgat.data.features import cpa_tcpa, haversine_nm
from kastgat.data.schema import ID_TO_INTENT


@dataclass
class RuleVerdict:
    flight_id: str
    partner_id: str
    dist_nm: float
    dalt_ft: float
    cpa_nm: float
    tcpa_s: float
    loss_of_separation: bool
    neural_intent: str
    allowed: bool
    reason: str


def evaluate_snapshot(
    snap: pd.DataFrame,
    intent_ids: np.ndarray | None = None,
    *,
    separation_nm: float = 5.0,
    separation_ft: float = 1000.0,
) -> list[RuleVerdict]:
    """Audit every pair. Block a neural clearance if it would *create* a LoS."""
    n = len(snap)
    lat0 = float(snap["lat"].mean()) if n else 0.0
    lon0 = float(snap["lon"].mean()) if n else 0.0
    intents = ["NONE"] * n
    if intent_ids is not None:
        intents = [ID_TO_INTENT.get(int(i), "NONE") for i in intent_ids]
    verdicts: list[RuleVerdict] = []
    ids = snap["flight_id"].astype(str).tolist()
    lat = snap["lat"].to_numpy()
    lon = snap["lon"].to_numpy()
    gs = snap["gs_kt"].to_numpy()
    tr = snap["track_deg"].to_numpy()
    alt = snap["alt_ft"].to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(haversine_nm(lat[i], lon[i], lat[j], lon[j]))
            dalt = abs(float(alt[i] - alt[j]))
            cpa, tcpa = cpa_tcpa(lat[i], lon[i], gs[i], tr[i], lat[j], lon[j], gs[j], tr[j], lat0, lon0)
            los = dist < separation_nm and dalt < separation_ft
            for idx in (i, j):
                intent = intents[idx]
                allowed, reason = _allow(intent, los, float(cpa), float(tcpa), dalt, separation_nm, separation_ft)
                verdicts.append(
                    RuleVerdict(
                        flight_id=ids[idx],
                        partner_id=ids[j if idx == i else i],
                        dist_nm=dist,
                        dalt_ft=dalt,
                        cpa_nm=float(cpa),
                        tcpa_s=float(tcpa),
                        loss_of_separation=los,
                        neural_intent=intent,
                        allowed=allowed,
                        reason=reason,
                    )
                )
    return verdicts


def _allow(intent, los, cpa, tcpa, dalt, sep_nm, sep_ft) -> tuple[bool, str]:
    if los:
        return False, "HARD_LOS: current geometry already violates ICAO 5 NM / 1000 ft"
    if intent in {"CLIMB", "DESCEND"} and cpa < sep_nm and 0 <= tcpa <= 120 and dalt < sep_ft * 1.5:
        # vertical instruction toward a pair that is already horizontally close
        return False, "BLOCKED: vertical clearance into an imminent horizontal conflict"
    return True, "OK"


def filter_intents(verdicts: list[RuleVerdict]) -> dict[str, str]:
    """Per-flight: if any pair blocks the intent, emit NONE (human stays in the loop)."""
    out: dict[str, str] = {}
    blocked: set[str] = set()
    for v in verdicts:
        out.setdefault(v.flight_id, v.neural_intent)
        if not v.allowed:
            blocked.add(v.flight_id)
    for fid in blocked:
        out[fid] = "NONE"
    return out
