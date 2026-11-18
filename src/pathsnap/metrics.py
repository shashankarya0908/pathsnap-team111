"""Evaluation metrics shared by experiments and the API."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(
        macro_f1=f1_score(y, yhat, average="macro"),
        macro_precision=precision_score(y, yhat, average="macro", zero_division=0),
        macro_recall=recall_score(y, yhat, average="macro", zero_division=0),
        log_loss=log_loss(y, pc, labels=[0, 1]),
    )


def top1_arcs(df: pd.DataFrame, score_col="p") -> pd.Series:
    """Highest-scoring candidate arc per observation (index = pt)."""
    idx = df.groupby("pt")[score_col].idxmax()
    return pd.Series(df.loc[idx, "arc"].values, index=df.loc[idx, "pt"].values)


def route_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Share of consecutive observation pairs whose predicted arcs are equal or
    directly connected (head of first == tail of second)."""
    a, b = pred_arcs[:-1], pred_arcs[1:]
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(ok.mean()) if len(ok) else 1.0


def collapse(seq):
    out = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(int(s))
    return out


def route_set_metrics(pred_route, true_route, arc_len):
    """Arc-set precision / recall / F1 (length weighted) between predicted and reference routes."""
    P, T = set(map(int, pred_route)), set(map(int, true_route))
    inter = P & T
    lp = arc_len[list(P)].sum() if P else 0.0
    lt = arc_len[list(T)].sum() if T else 0.0
    li = arc_len[list(inter)].sum() if inter else 0.0
    prec = li / lp if lp else 0.0
    rec = li / lt if lt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(route_precision=float(prec), route_recall=float(rec), route_f1=float(f1))
