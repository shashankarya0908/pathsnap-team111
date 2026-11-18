"""Shared helpers for experiment scripts."""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.features import FEATURES  # noqa: E402

SEED = 111
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
MOD = os.path.join(ROOT, "models")
for d in (RES, FIG, MOD):
    os.makedirs(d, exist_ok=True)

# Trips whose GPS is systematically far (>100 m) from the reference route for most
# points (found in EDA / label_quality.csv). Their labels are unreliable, so they are
# excluded from training (they are still reported in EDA).
BAD_LABEL_TRIP_FRAC = 0.30


def load_train(step=3):
    df = pd.read_pickle(os.path.join(PROC, "train.pkl"))
    q = pd.read_csv(os.path.join(RES, "label_quality.csv"))
    bad = set(q[(q.partition == "train") & (q.frac_label_gt100m > BAD_LABEL_TRIP_FRAC)].trip.astype(str).str.zfill(8))
    df = df[~df.trip.isin(bad)]
    df = df[df.label_ok == 1]
    has_pos = df.groupby(["trip", "pt"])["y"].transform("max") == 1
    df = df[has_pos]
    if step > 1:
        df = df[df.pt % step == 0]
    return df.reset_index(drop=True), sorted(bad)


def load_student_test():
    return pd.read_pickle(os.path.join(PROC, "student_test.pkl")).reset_index(drop=True)


def top1_accuracy(df, p):
    d = df[["trip", "pt", "arc", "label_arc"]].copy(); d["p"] = p
    idx = d.groupby(["trip", "pt"])["p"].idxmax()
    return float((d.loc[idx, "arc"].values == d.loc[idx, "label_arc"].values).mean())
