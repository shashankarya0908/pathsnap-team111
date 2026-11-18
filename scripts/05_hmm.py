"""HMM / Viterbi experiments.

1. Out-of-fold classifier probabilities for EVERY training observation
   (5-fold GroupKFold, tuned HistGB) so decoder tuning never sees in-sample scores.
2. Train the learned transition model on training trips of half "A".
3. Grid-search decoder weights on training trips of half "B" (classic + learned).
4. Evaluate on the 10 student_test trips: classifier-only vs classic HMM vs learned HMM.
Outputs: models/transition_model.joblib, models/hmm_config.json,
         results/hmm_tuning.csv, results/hmm_comparison.csv, results/hmm_per_trip.csv
"""
import os, json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ROOT, SEED, PROC, RES, MOD, FEATURES, BAD_LABEL_TRIP_FRAC
from pathsnap.data import load_trip, load_route
from pathsnap.hmm import RoadGraph, Decoder, transition_features, stitch_route, candidates_from_frame, TRANS_FEATURES
from pathsnap.metrics import route_continuity, route_set_metrics

RAW = os.path.join(ROOT, "data", "raw")
best = json.load(open(os.path.join(RES, "bayes_opt_best.json")))["best_params"]
q = pd.read_csv(os.path.join(RES, "label_quality.csv"))
q["trip"] = q.trip.astype(str).str.zfill(8)
good = set(q[(q.partition == "train") & (q.frac_label_gt100m <= BAD_LABEL_TRIP_FRAC)].trip)

# ---------------- 1. OOF probabilities for all train observations
full = pd.read_pickle(os.path.join(PROC, "train.pkl"))
full = full[full.trip.isin(good)].reset_index(drop=True)
oof = np.zeros(len(full))
for a, b in GroupKFold(n_splits=5).split(full, groups=full.trip):
    fa = full.iloc[a]
    fa = fa[(fa.label_ok == 1) & (fa.pt % 3 == 0) & (fa.groupby(["trip", "pt"]).y.transform("max") == 1)]
    m = HistGradientBoostingClassifier(random_state=SEED, **best).fit(fa[FEATURES].values, fa.y.values)
    oof[b] = m.predict_proba(full.iloc[b][FEATURES].values)[:, 1]
full["p"] = oof
print("OOF done", flush=True)

trips = sorted(good)
rng = np.random.RandomState(SEED); rng.shuffle(trips)
half_A, half_B = trips[: len(trips) // 2], trips[len(trips) // 2:]
cache = {}


def trip_bundle(part, tid, frame):
    key = (part, tid)
    if key not in cache:
        trip = load_trip(os.path.join(RAW, part, tid)); G = RoadGraph(trip)
        fr = frame[frame.trip == tid].sort_values(["pt", "dist_rank"])
        cache[key] = dict(trip=trip, G=G, fr=fr, route=load_route(os.path.join(RAW, part, tid)),
                          lab=fr.groupby("pt").label_arc.first().values)
    return cache[key]


# ---------------- 2. learned transition model (half A, every 2nd step, trips <= 6000 pts)
XT, YT = [], []
for tid in half_A:
    fr0 = full[full.trip == tid]
    if fr0.pt.max() > 6000:
        continue
    B = trip_bundle("train", tid, full); trip, G, fr = B["trip"], B["G"], B["fr"]
    groups = [g for _, g in fr.groupby("pt")]; lab = B["lab"]
    for t in range(1, len(groups), 2):
        g0, g1 = groups[t - 1], groups[t]
        if lab[t - 1] not in g0.arc.values or lab[t] not in g1.arc.values:
            continue
        d_gc = float(np.hypot(*(trip.xy[t] - trip.xy[t - 1]))); dt = float(trip.t[t] - trip.t[t - 1])
        F = transition_features(G, g0.arc.values, g0.t_proj.values, g1.arc.values, g1.t_proj.values, d_gc, dt)
        i = list(g0.arc.values).index(lab[t - 1])  # condition on the true previous state
        XT.append(F[i]); YT.append((g1.arc.values == lab[t]).astype(int))
XT, YT = np.vstack(XT), np.concatenate(YT)
tm = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, class_weight=None)).fit(XT, YT)
joblib.dump(tm, os.path.join(MOD, "transition_model.joblib"))
coef = dict(zip(TRANS_FEATURES, tm[-1].coef_[0].round(4).tolist()))
print("transition model rows", len(YT), "pos", YT.mean(), coef, flush=True)

# ---------------- 3. decoder tuning on half B (OOF probabilities)
tune_trips = [t for t in half_B if full[full.trip == t].pt.max() <= 6000][:16]
grid = [("classic", dict(beta=b, w_emit=w)) for b in [1, 3, 10, 30] for w in [0.5, 1, 2]] + \
       [("learned", dict(w_trans=1.0, w_emit=w)) for w in [0.25, 0.5, 1, 2, 4]] + \
       [("hybrid", dict(beta=b, w_learn=wl, w_emit=w)) for b in [3, 10] for wl in [0.25, 0.5, 1.0] for w in [0.5, 1]]
# note: Viterbi is invariant to scaling all log-scores, so for "learned" only the
# ratio w_emit / w_trans matters -> we fix w_trans = 1 and vary w_emit.
pres = {}
for tid in tune_trips:
    B = trip_bundle("train", tid, full); B["cands"] = candidates_from_frame(B["fr"])
    pres[tid] = Decoder.precompute(B["trip"], B["cands"], B["G"])
rows = []
for mode, kw in grid:
    dec = Decoder(mode=mode, trans_model=tm, **kw)
    acc, n = 0, 0
    for tid in tune_trips:
        B = cache[("train", tid)]
        _, arcs, _ = dec.decode(B["trip"], B["cands"], B["G"], pres[tid])
        acc += (arcs == B["lab"]).sum(); n += len(arcs)
    rows.append(dict(mode=mode, **kw, point_acc=acc / n)); print(rows[-1], flush=True)
tun = pd.DataFrame(rows); tun.to_csv(os.path.join(RES, "hmm_tuning.csv"), index=False)
best_cfg = {}
KEYS = {"classic": ["beta", "w_emit"], "learned": ["w_trans", "w_emit"], "hybrid": ["beta", "w_learn", "w_emit"]}
for mode in KEYS:
    r = tun[tun["mode"] == mode].sort_values("point_acc").iloc[-1]
    best_cfg[mode] = {k: float(r[k]) for k in KEYS[mode]}
# deployed decoder = best mode on the TRAINING tuning trips (student_test not used for this choice)
selected = tun.sort_values("point_acc").iloc[-1]["mode"]
# classifier-only accuracy on the same tuning trips for reference
co = np.mean(np.concatenate([np.array([c[0][np.argmax(c[2])] for c in cache[("train", t)]["cands"]]) == cache[("train", t)]["lab"] for t in tune_trips]))
json.dump(dict(selected_mode=selected, modes=best_cfg, unreach_penalty=-15.0, min_p=1e-4,
               tuning_trips=tune_trips, tuning_classifier_only_acc=float(co),
               transition_features=TRANS_FEATURES, transition_coef=coef),
          open(os.path.join(MOD, "hmm_config.json"), "w"), indent=2)
print("best", best_cfg, "selected", selected, "clf-only on tuning trips", co, flush=True)

# ---------------- 4. student_test comparison
te = pd.read_pickle(os.path.join(PROC, "student_test.pkl"))
te["p"] = np.load(os.path.join(PROC, "test_HistGB_tuned.npy"))
per = []
for tid in sorted(te.trip.unique()):
    B = trip_bundle("student_test", tid, te); trip, G = B["trip"], B["G"]
    cands = candidates_from_frame(B["fr"]); lab = B["lab"]
    t0 = time.time(); pre = Decoder.precompute(trip, cands, G); tpre = time.time() - t0
    variants = {"classifier_only": (np.array([c[0][np.argmax(c[2])] for c in cands]), 0.0)}
    for mode in ["classic", "learned", "hybrid"]:
        t0 = time.time(); _, arcs, conf = Decoder(mode=mode, trans_model=tm, **best_cfg[mode]).decode(trip, cands, G, pre)
        variants[f"hmm_{mode}"] = (arcs, time.time() - t0 + tpre)
    for v, (arcs, secs) in variants.items():
        route, filled, breaks = stitch_route(G, arcs)
        r = dict(trip=tid, variant=v, n_points=len(arcs), point_acc=float((arcs == lab).mean()),
                 continuity=route_continuity(arcs, trip.arcs), gaps_filled=filled, breaks=breaks,
                 n_transitions=int((arcs[1:] != arcs[:-1]).sum()), decode_s=secs)
        r.update(route_set_metrics(route, B["route"], trip.arc_len)); per.append(r)
    print(tid, {r["variant"]: round(r["point_acc"], 4) for r in per[-4:]}, flush=True)
per = pd.DataFrame(per); per.to_csv(os.path.join(RES, "hmm_per_trip.csv"), index=False)
w = per.assign(correct=per.point_acc * per.n_points)
comp = w.groupby("variant").agg(point_acc_micro=("correct", "sum"), n=("n_points", "sum"),
                                point_acc_trip_mean=("point_acc", "mean"), continuity=("continuity", "mean"),
                                route_precision=("route_precision", "mean"), route_recall=("route_recall", "mean"),
                                route_f1=("route_f1", "mean"), breaks=("breaks", "sum"),
                                transitions=("n_transitions", "sum"), decode_s=("decode_s", "sum"))
comp["point_acc_micro"] /= comp.pop("n")
comp.to_csv(os.path.join(RES, "hmm_comparison.csv")); print(comp.to_string())
