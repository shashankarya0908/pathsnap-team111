"""Live demo: test all 8 classifiers on student_test trips, alone and with the HMM.

Usage (from the project folder, venv active):
    python scripts\\demo_models.py                      # trips 00000077 00000007 00000065
    python scripts\\demo_models.py 00000076 00000009    # any student_test trips
    python scripts\\demo_models.py all                  # all 10 student_test trips
    python scripts\\demo_models.py --retrain 00000077   # force retraining the models

First run: builds the feature tables if missing (~1 min) and trains the 8 models
(~2 min, saved to models/compare/). Later runs only load and predict (seconds).
Configurations are the same as scripts/03_train_compare.py; training here uses every
6th training observation so the demo stays fast (numbers differ slightly from the report).
"""
import os, sys, time, json, subprocess
import numpy as np
import pandas as pd
import joblib

from common import ROOT, SEED, PROC, MOD, FEATURES, load_train, load_student_test
from pathsnap.data import load_trip
from pathsnap.hmm import RoadGraph, Decoder, candidates_from_frame
from pathsnap.metrics import candidate_metrics, route_continuity

args = [a for a in sys.argv[1:] if a != "--retrain"]
RETRAIN = "--retrain" in sys.argv
CMP = os.path.join(MOD, "compare"); os.makedirs(CMP, exist_ok=True)

if not os.path.exists(os.path.join(PROC, "train.pkl")):
    print("Feature tables not found -> running 02_build_dataset.py (about 1 minute)...", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "02_build_dataset.py")], check=True)

from model_zoo import MODELS  # exactly the same definitions as the comparison experiment

# ---------------------------------------------------------------- train or load
models, train_s = {}, {}
need = [m for m in MODELS if RETRAIN or not os.path.exists(os.path.join(CMP, f"{m}.joblib"))]
if need:
    tr, _ = load_train(step=6)
    X, y = tr[FEATURES].values, tr.y.values
    rng = np.random.RandomState(SEED)
    print(f"Training {len(need)} models on {len(tr):,} candidate rows from the training trips...", flush=True)
    for name in need:
        _, _, make, sub = MODELS[name]
        Xs, ys = X, y
        if sub and len(X) > sub:
            i = rng.choice(len(X), sub, replace=False); Xs, ys = X[i], y[i]
        t0 = time.time(); m = make().fit(Xs, ys); train_s[name] = time.time() - t0
        joblib.dump((m, train_s[name]), os.path.join(CMP, f"{name}.joblib"))
        print(f"  trained {name:13s} in {train_s[name]:6.1f}s", flush=True)
for name in MODELS:
    models[name], train_s[name] = joblib.load(os.path.join(CMP, f"{name}.joblib"))
tuned = joblib.load(os.path.join(MOD, "final_model.joblib"))
models["HistGB_tuned (final)"] = tuned; train_s["HistGB_tuned (final)"] = float("nan")

# ---------------------------------------------------------------- decoder (deployed config)
hcfg = json.load(open(os.path.join(MOD, "hmm_config.json")))
tm = joblib.load(os.path.join(MOD, "transition_model.joblib"))
mode = hcfg["selected_mode"]
dec = Decoder(mode=mode, trans_model=tm, unreach_penalty=hcfg["unreach_penalty"], min_p=hcfg["min_p"], **hcfg["modes"][mode])

te = load_student_test()
all_trips = sorted(te.trip.unique())
trips = all_trips if args == ["all"] else (args or ["00000077", "00000007", "00000065"])
bad = [t for t in trips if t not in all_trips]
if bad:
    sys.exit(f"Unknown student_test trip(s): {bad}. Choose from {all_trips}")

pd.set_option("display.width", 200)
summary = []
for tid in trips:
    fr = te[te.trip == tid].sort_values(["pt", "dist_rank"]).reset_index(drop=True)
    trip = load_trip(os.path.join(ROOT, "data", "raw", "student_test", tid)); G = RoadGraph(trip)
    lab = fr.groupby("pt").label_arc.first().values
    base = candidates_from_frame(fr.assign(p=1.0))
    pre = Decoder.precompute(trip, base, G)          # transitions are model-independent: compute once
    rows = []
    for name, m in models.items():
        t0 = time.time(); p = m.predict_proba(fr[FEATURES].values)[:, 1]; inf = time.time() - t0
        cm = candidate_metrics(fr.y.values, p)
        fr["p"] = p
        cands = candidates_from_frame(fr)
        clf = np.array([c[0][int(np.argmax(c[2]))] for c in cands])
        t0 = time.time(); _, arcs, conf = dec.decode(trip, cands, G, pre); hs = time.time() - t0
        rows.append({"model": name, "top1_acc": (clf == lab).mean(), "macro_f1": cm["macro_f1"],
                     "log_loss": cm["log_loss"], "continuity": route_continuity(clf, trip.arcs),
                     "HMM_acc": (arcs == lab).mean(), "HMM_continuity": route_continuity(arcs, trip.arcs),
                     "predict_ms": 1000 * inf, "HMM_s": hs, "train_s": train_s[name]})
    df = pd.DataFrame(rows)
    print(f"\n=== student_test trip {tid}: {len(lab)} GPS points, {len(fr)} candidate rows, decoder={mode} ===")
    show = df.copy()
    for c in ["top1_acc", "continuity", "HMM_acc", "HMM_continuity"]:
        show[c] = (100 * show[c]).map("{:.2f}%".format)
    for c in ["macro_f1", "log_loss"]:
        show[c] = show[c].map("{:.4f}".format)
    show["predict_ms"] = show.predict_ms.map("{:.0f}".format); show["HMM_s"] = show.HMM_s.map("{:.2f}".format)
    show["train_s"] = show.train_s.map(lambda v: "-" if np.isnan(v) else f"{v:.1f}")
    print(show.to_string(index=False))
    best = df.sort_values("top1_acc").iloc[-1]
    print(f"Best classifier alone: {best.model} ({100*best.top1_acc:.2f}%). "
          f"HMM gain for the final model: {100*(df.iloc[-1].HMM_acc - df.iloc[-1].top1_acc):+.2f} points.")
    summary.append(df.assign(trip=tid, n=len(lab)))

if len(summary) > 1:
    s = pd.concat(summary)
    agg = s.groupby("model", sort=False).apply(
        lambda g: pd.Series({"top1_acc": np.average(g.top1_acc, weights=g.n), "HMM_acc": np.average(g.HMM_acc, weights=g.n),
                             "macro_f1": g.macro_f1.mean(), "log_loss": g.log_loss.mean()}), include_groups=False)
    agg["HMM_gain"] = agg.HMM_acc - agg.top1_acc
    print(f"\n=== Overall on {len(trips)} trips (accuracy weighted by points) ===")
    print((agg * [100, 100, 1, 1, 100]).round(3).rename(columns={"top1_acc": "top1_%", "HMM_acc": "HMM_%", "HMM_gain": "gain_pts"}).to_string())
