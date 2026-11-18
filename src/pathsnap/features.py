"""Candidate generation and feature engineering (team 111).

For every GPS observation we generate a small candidate set of road arcs and
describe each (observation, candidate) pair with geometric + kinematic +
context features. A binary classifier then scores P(candidate is the
reference arc).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Trip, angle_diff, gps_heading_speed

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands",
    "head_diff", "speed", "move_dist", "arc_len", "t_proj", "end_dist",
    "has_reverse", "ctx_dist_mean", "ctx_dist_max", "ctx_head_diff",
    "dt_prev", "out_deg_to", "in_deg_from",
    # team-111 self-calibrating noise features
    "trip_sigma", "dist_z", "ctx_dist_z", "head_consistency",
]

DEFAULT_CAND = dict(radius=50.0, k_max=8, k_fallback=3, ctx=3)


def generate_candidates(trip: Trip, radius=50.0, k_max=8, k_fallback=3):
    """Return arrays (point_idx, arc_id, dist, t_proj, fallback_flag)."""
    near = trip.arcs_near(trip.xy, radius)
    P, A, D, T, F = [], [], [], [], []
    for i, arcs in enumerate(near):
        p = trip.xy[i]
        if len(arcs):
            d, tt = trip.point_arc_distance(np.repeat(p[None], len(arcs), 0), arcs)
            m = d <= radius
            arcs, d, tt = arcs[m], d[m], tt[m]
        fb = 0
        if len(arcs) == 0:  # missing candidates -> fall back to the k nearest arcs
            arcs = trip.nearest_arcs(p, k_fallback)
            d, tt = trip.point_arc_distance(np.repeat(p[None], len(arcs), 0), arcs)
            fb = 1
        o = np.argsort(d, kind="stable")[:k_max]
        P.append(np.full(len(o), i)); A.append(arcs[o]); D.append(d[o]); T.append(tt[o])
        F.append(np.full(len(o), fb))
    return (np.concatenate(P), np.concatenate(A).astype(int), np.concatenate(D),
            np.concatenate(T), np.concatenate(F))


def build_features(trip: Trip, radius=50.0, k_max=8, k_fallback=3, ctx=3) -> pd.DataFrame:
    pi, arc, dist, tproj, fb = generate_candidates(trip, radius, k_max, k_fallback)
    N = len(trip.xy)
    heading, speed, move = gps_heading_speed(trip.xy, trip.t)
    df = pd.DataFrame({"pt": pi, "arc": arc, "dist": dist, "t_proj": tproj, "fallback": fb})
    g = df.groupby("pt")["dist"]
    df["dist_rank"] = g.rank(method="first").astype(int) - 1
    mn = g.transform("min")
    df["dist_minus_min"] = df["dist"] - mn
    df["dist_ratio"] = df["dist"] / (mn + 1.0)
    df["n_cands"] = g.transform("size")
    df["head_diff"] = angle_diff(heading[pi], trip.arc_bearing[arc])
    df["speed"] = speed[pi]
    df["move_dist"] = move[pi]
    df["arc_len"] = trip.arc_len[arc]
    df["end_dist"] = np.minimum(tproj, 1 - tproj) * trip.arc_len[arc]
    df["has_reverse"] = trip.has_reverse[arc]
    # temporal context: how well the same arc explains neighbouring observations
    cd, ch = [], []
    for o in range(-ctx, ctx + 1):
        j = np.clip(pi + o, 0, N - 1)
        d, _ = trip.point_arc_distance(trip.xy[j], arc)
        cd.append(d)
        ch.append(angle_diff(heading[j], trip.arc_bearing[arc]))
    cd = np.vstack(cd); ch = np.vstack(ch)
    df["ctx_dist_mean"] = cd.mean(0)
    df["ctx_dist_max"] = cd.max(0)
    df["ctx_head_diff"] = ch.mean(0)
    dtp = np.concatenate([[1.0], np.diff(trip.t)])
    df["dt_prev"] = dtp[pi]
    df["out_deg_to"] = trip.out_deg[trip.arcs[arc, 1]]
    df["in_deg_from"] = trip.in_deg[trip.arcs[arc, 0]]
    # ---- self-calibrating GPS noise model (team 111) -------------------------
    # Robust per-trip noise scale: 1.4826 * median of each point's nearest-arc
    # distance (MAD-style estimator). Uses only the GPS + map, never the route,
    # so it is computed identically at inference time.
    sigma = float(np.clip(1.4826 * np.median(mn.groupby(df["pt"]).first()), 2.0, 80.0))
    df["trip_sigma"] = sigma
    df["dist_z"] = df["dist"] / sigma
    df["ctx_dist_z"] = df["ctx_dist_mean"] / sigma
    # heading consistency: share of window points whose heading agrees (<45 deg) with the arc
    df["head_consistency"] = (ch < 45).mean(0)
    df["trip"] = trip.trip_id
    return df


def attach_labels(df: pd.DataFrame, label_arc: np.ndarray, label_dist: np.ndarray, max_label_dist=100.0):
    """y=1 if candidate arc equals the aligned reference arc of its point."""
    df["label_arc"] = label_arc[df["pt"].values]
    df["y"] = (df["arc"].values == df["label_arc"].values).astype(np.int8)
    df["label_dist"] = label_dist[df["pt"].values]
    df["label_ok"] = (df["label_dist"] <= max_label_dist).astype(np.int8)
    return df
