"""Build padded temporal graphs (nodes = aircraft, edges = proximity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from kastgat.data.features import haversine_nm, pair_features
from kastgat.data.labels import attach_intent_ids, pairwise_conflict_mask
from kastgat.data.schema import ensure_canonical

NODE_FEATURES = ["lat", "lon", "alt_ft", "gs_kt_z", "track_sin", "track_cos", "vert_rate_fpm_z"]


@dataclass
class TemporalGraph:
    """One training window over a sector.

    x: [T, N, F], node_mask: [T, N], y_traj: [N, H, 3], y_conflict: [N], y_intent: [N]
    edge_index_t / edge_attr_t: length-T lists.
    """

    x: torch.Tensor
    node_mask: torch.Tensor
    edge_index: list[torch.Tensor]
    edge_attr: list[torch.Tensor]
    y_traj: torch.Tensor
    y_conflict: torch.Tensor
    y_intent: torch.Tensor
    flight_ids: list[str]
    t0: pd.Timestamp
    attention_index: torch.Tensor  # last-step edges, useful for XAI even before training


def _snapshot_graph(snap: pd.DataFrame, proximity_nm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(snap)
    x = snap.loc[:, list(NODE_FEATURES)].to_numpy(dtype=np.float32)
    if n == 0:
        return x, np.zeros((2, 0), dtype=np.int64), np.zeros((0, 5), dtype=np.float32)
    lat0 = float(snap["lat"].mean())
    lon0 = float(snap["lon"].mean())
    src: list[int] = []
    dst: list[int] = []
    attrs: list[np.ndarray] = []
    nodes = {col: snap[col].to_numpy(dtype=float) for col in ("lat", "lon", "alt_ft", "gs_kt", "track_deg")}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = haversine_nm(nodes["lat"][i], nodes["lon"][i], nodes["lat"][j], nodes["lon"][j])
            if dist <= proximity_nm:
                src.append(i)
                dst.append(j)
                attrs.append(pair_features(nodes, i, j, lat0, lon0))
    edge_index = np.vstack([src, dst]).astype(np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.stack(attrs, axis=0) if attrs else np.zeros((0, 5), dtype=np.float32)
    return x, edge_index, edge_attr


def _pad_nodes(x: np.ndarray, max_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    n, f = x.shape
    out = np.zeros((max_nodes, f), dtype=np.float32)
    mask = np.zeros((max_nodes,), dtype=np.float32)
    n_keep = min(n, max_nodes)
    out[:n_keep] = x[:n_keep]
    mask[:n_keep] = 1.0
    return out, mask


def build_windows(
    df: pd.DataFrame,
    *,
    history_steps: int = 12,
    horizon_steps: int = 6,
    max_nodes: int = 24,
    proximity_nm: float = 20.0,
    separation_nm: float = 5.0,
    separation_ft: float = 1000.0,
    stride: int = 3,
) -> list[TemporalGraph]:
    df = attach_intent_ids(ensure_canonical(df), horizon_steps=horizon_steps)
    if "track_sin" not in df.columns:
        from kastgat.data.clean import attach_trig_heading

        df = attach_trig_heading(df)
    if "gs_kt_z" not in df.columns:
        from kastgat.data.clean import ZScoreScaler

        df = ZScoreScaler(["gs_kt", "vert_rate_fpm"]).fit_transform(df)

    times = np.sort(df["ts"].unique())
    samples: list[TemporalGraph] = []
    last_start = len(times) - history_steps - horizon_steps
    if last_start < 0:
        return samples

    for start in range(0, last_start + 1, stride):
        hist_times = times[start : start + history_steps]
        fut_times = times[start + history_steps : start + history_steps + horizon_steps]
        hist = df[df["ts"].isin(hist_times)]
        t_last = hist_times[-1]
        snap = hist[hist["ts"] == t_last].drop_duplicates("flight_id")
        if len(snap) < 2:
            continue
        snap = snap.sort_values("flight_id").head(max_nodes).reset_index(drop=True)
        ids = snap["flight_id"].tolist()
        id_set = set(ids)

        xs: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        edges: list[torch.Tensor] = []
        eattrs: list[torch.Tensor] = []
        for t in hist_times:
            sl = hist[(hist["ts"] == t) & (hist["flight_id"].isin(id_set))].drop_duplicates("flight_id")
            sl = sl.set_index("flight_id").reindex(ids).reset_index()
            present = sl[NODE_FEATURES].notna().all(axis=1).to_numpy()
            sl_valid = sl.loc[present].copy()
            # reindex features in id order with zeros for missing
            feat = np.zeros((len(ids), len(NODE_FEATURES)), dtype=np.float32)
            if len(sl_valid):
                pos = {fid: i for i, fid in enumerate(ids)}
                raw_x, e_idx, e_attr = _snapshot_graph(sl_valid.reset_index(drop=True), proximity_nm)
                local_ids = sl_valid["flight_id"].tolist()
                for li, fid in enumerate(local_ids):
                    feat[pos[fid]] = raw_x[li]
                remap = np.array([pos[fid] for fid in local_ids], dtype=np.int64)
                if e_idx.shape[1]:
                    e_idx = remap[e_idx]
            else:
                e_idx = np.zeros((2, 0), dtype=np.int64)
                e_attr = np.zeros((0, 5), dtype=np.float32)
            padded, mask = _pad_nodes(feat, max_nodes)
            mask[: len(ids)] = sl[NODE_FEATURES].notna().all(axis=1).to_numpy()[:max_nodes]
            xs.append(padded)
            masks.append(mask)
            edges.append(torch.from_numpy(e_idx))
            eattrs.append(torch.from_numpy(e_attr.astype(np.float32)))

        fut = df[(df["ts"].isin(fut_times)) & (df["flight_id"].isin(id_set))]
        y_traj = np.zeros((max_nodes, horizon_steps, 3), dtype=np.float32)
        for ni, fid in enumerate(ids):
            track = fut[fut["flight_id"] == fid].sort_values("ts")
            coords = track[["lat", "lon", "alt_ft"]].to_numpy(dtype=np.float32)
            h = min(len(coords), horizon_steps)
            if h:
                y_traj[ni, :h] = coords[:h]
                if h < horizon_steps:
                    y_traj[ni, h:] = coords[h - 1]

        y_conflict = np.zeros((max_nodes,), dtype=np.float32)
        y_conflict[: len(ids)] = pairwise_conflict_mask(
            snap,
            separation_nm=separation_nm,
            separation_ft=separation_ft,
            horizon_s=horizon_steps * 10.0,
        ).astype(np.float32)
        y_intent = np.zeros((max_nodes,), dtype=np.int64)
        y_intent[: len(ids)] = snap["intent_id"].to_numpy(dtype=np.int64)

        samples.append(
            TemporalGraph(
                x=torch.from_numpy(np.stack(xs, axis=0)),
                node_mask=torch.from_numpy(np.stack(masks, axis=0)),
                edge_index=edges,
                edge_attr=eattrs,
                y_traj=torch.from_numpy(y_traj),
                y_conflict=torch.from_numpy(y_conflict),
                y_intent=torch.from_numpy(y_intent),
                flight_ids=ids + [""] * (max_nodes - len(ids)),
                t0=pd.Timestamp(hist_times[0]),
                attention_index=edges[-1],
            )
        )
    return samples


def to_pyg_data(sample: TemporalGraph, t: int = -1) -> Data:
    """Sparse snapshot at one timestep, for the optional PyG encoder path."""
    mask = sample.node_mask[t] > 0.5
    idx = mask.nonzero(as_tuple=False).view(-1)
    x = sample.x[t][idx]
    mapping = {int(old): new for new, old in enumerate(idx.tolist())}
    ei = sample.edge_index[t]
    keep = []
    for e in range(ei.shape[1]):
        s, d = int(ei[0, e]), int(ei[1, e])
        if s in mapping and d in mapping:
            keep.append([mapping[s], mapping[d]])
    edge_index = torch.tensor(keep, dtype=torch.long).t().contiguous() if keep else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=sample.y_conflict[idx])
