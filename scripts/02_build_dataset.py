"""Build candidate-level feature tables for the train and student_test partitions.

Usage: python scripts/02_build_dataset.py
Output: data/processed/{train,student_test}.pkl  and results/label_quality.csv
The evaluator_test partition is never touched here.
"""
import os, sys, time, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.data import load_trip, load_route, list_trips
from pathsnap.labels import align_route
from pathsnap.features import build_features, attach_labels, DEFAULT_CAND

RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

rows = []
for part in ["train", "student_test"]:
    frames = []
    for tid in list_trips(os.path.join(RAW, part)):
        t0 = time.time()
        tdir = os.path.join(RAW, part, tid)
        trip = load_trip(tdir)
        route = load_route(tdir)
        lab, pos, ldist = align_route(trip, route)
        df = build_features(trip, **DEFAULT_CAND)
        df = attach_labels(df, lab, ldist)
        df["partition"] = part
        hit = df.groupby("pt")["y"].max()
        rows.append(dict(partition=part, trip=tid, n_points=len(trip.xy), n_raw=trip.n_raw_points,
                         route_len=len(route), n_rows=len(df), mean_cands=len(df) / len(trip.xy),
                         label_dist_median=float(np.median(ldist)), label_dist_p95=float(np.percentile(ldist, 95)),
                         frac_label_gt100m=float((ldist > 100).mean()),
                         candidate_miss_rate=float(1 - hit.mean()),
                         fallback_rate=float(df.groupby("pt")["fallback"].max().mean()),
                         route_coverage=float(len(np.unique(pos)) / len(route)),
                         secs=time.time() - t0))
        frames.append(df)
        print(part, tid, f"pts={len(trip.xy)} rows={len(df)} miss={rows[-1]['candidate_miss_rate']:.3f} {rows[-1]['secs']:.1f}s", flush=True)
    pd.concat(frames, ignore_index=True).to_pickle(os.path.join(OUT, f"{part}.pkl"))

q = pd.DataFrame(rows)
q.to_csv(os.path.join(ROOT, "results", "label_quality.csv"), index=False)
print(q.groupby("partition")[["n_points", "mean_cands", "candidate_miss_rate", "label_dist_median", "route_coverage"]].mean())
with open(os.path.join(ROOT, "results", "candidate_config.json"), "w") as f:
    json.dump(DEFAULT_CAND, f, indent=2)
