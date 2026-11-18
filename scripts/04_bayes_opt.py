"""Bayesian hyperparameter optimisation of the selected classifier (HistGradientBoosting).

Our own Gaussian-process Bayesian optimiser (Matern-5/2 GP surrogate +
Expected Improvement acquisition), built on scikit-learn's
GaussianProcessRegressor, so every step can be explained in the viva.

Objective : mean macro-F1 over 3-fold GroupKFold (groups = trip) on the 80
            training trips (subsampled every 9th observation for speed).
Trials    : 6 random initial points + 14 GP-EI iterations = 20.
Outputs   : results/bayes_opt_trials.csv, results/bayes_opt_best.json,
            results/tuned_vs_untuned.csv, models/final_model.joblib,
            models/model_config.json, data/processed/{oof,test}_HistGB_tuned.npy
"""
import os, time, json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from common import ROOT, SEED, PROC, RES, MOD, FEATURES, load_train, load_student_test, top1_accuracy
from pathsnap.metrics import candidate_metrics
from pathsnap.features import DEFAULT_CAND

SPACE = [  # name, low, high, log-scale, integer
    ("learning_rate", 0.01, 0.3, True, False),
    ("max_leaf_nodes", 8, 128, True, True),
    ("min_samples_leaf", 10, 400, True, True),
    ("l2_regularization", 1e-4, 10.0, True, False),
    ("max_iter", 100, 600, False, True),
    ("max_features", 0.4, 1.0, False, False),
]
UNTUNED = dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31)


def decode(u):
    out = {}
    for x, (n, lo, hi, lg, it) in zip(u, SPACE):
        v = np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))) if lg else lo + x * (hi - lo)
        out[n] = int(round(v)) if it else float(v)
    return out


def make(params):
    return HistGradientBoostingClassifier(random_state=SEED, **params)


bo_df, _ = load_train(step=9)
X, y, g = bo_df[FEATURES].values, bo_df.y.values, bo_df.trip.values
folds = list(GroupKFold(n_splits=3).split(X, y, g))


def objective(params):
    s = []
    for a, b in folds:
        m = make(params).fit(X[a], y[a])
        s.append(f1_score(y[b], (m.predict_proba(X[b])[:, 1] >= .5).astype(int), average="macro"))
    return float(np.mean(s)), float(np.std(s))


rng = np.random.RandomState(SEED)
U, Y, trials = [], [], []
N_INIT, N_ITER = 6, 14
t_all = time.time()
for it in range(N_INIT + N_ITER):
    if it < N_INIT:
        u = rng.rand(len(SPACE)); how = "random-init"
    else:
        gp = GaussianProcessRegressor(ConstantKernel(1.0) * Matern(length_scale=np.ones(len(SPACE)), nu=2.5)
                                      + WhiteKernel(1e-4), normalize_y=True, random_state=SEED, n_restarts_optimizer=3)
        gp.fit(np.array(U), np.array(Y))
        cand = rng.rand(4000, len(SPACE))
        mu, sd = gp.predict(cand, return_std=True)
        best = max(Y); z = (mu - best - 1e-4) / np.maximum(sd, 1e-9)
        ei = (mu - best - 1e-4) * norm.cdf(z) + sd * norm.pdf(z)
        u = cand[int(np.argmax(ei))]; how = "gp-ei"
    p = decode(u); t0 = time.time(); f, fs = objective(p)
    U.append(u); Y.append(f)
    trials.append(dict(trial=it, strategy=how, cv3_macro_f1=f, cv3_std=fs, best_so_far=max(Y), secs=time.time() - t0, **p))
    print(f"trial {it:2d} {how:11s} f1={f:.4f} best={max(Y):.4f} {p}", flush=True)
    pd.DataFrame(trials).to_csv(os.path.join(RES, "bayes_opt_trials.csv"), index=False)
bo_time = time.time() - t_all
best = trials[int(np.argmax(Y))]
best_params = {n: best[n] for n, *_ in SPACE}
json.dump(dict(best_params=best_params, best_cv3_macro_f1=best["cv3_macro_f1"], n_trials=len(trials),
               total_time_s=bo_time, space=[dict(name=n, low=lo, high=hi, log=lg, integer=it) for n, lo, hi, lg, it in SPACE]),
          open(os.path.join(RES, "bayes_opt_best.json"), "w"), indent=2)

# ---------- untuned vs tuned with identical protocol (5-fold GroupKFold, step=3, student_test)
tr, _ = load_train(step=3); te = load_student_test()
Xtr, ytr, gtr, Xte, yte = tr[FEATURES].values, tr.y.values, tr.trip.values, te[FEATURES].values, te.y.values
rows = []
for name, params in [("HistGB_untuned", UNTUNED), ("HistGB_tuned", best_params)]:
    oof = np.zeros(len(tr)); f1s = []
    for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, gtr):
        m = make(params).fit(Xtr[a], ytr[a]); oof[b] = m.predict_proba(Xtr[b])[:, 1]
        f1s.append(f1_score(ytr[b], (oof[b] >= .5).astype(int), average="macro"))
    t0 = time.time(); m = make(params).fit(Xtr, ytr); ft = time.time() - t0
    t0 = time.time(); p = m.predict_proba(Xte)[:, 1]; inf = time.time() - t0
    cm = candidate_metrics(yte, p)
    rows.append(dict(model=name, cv_macro_f1=np.mean(f1s), cv_macro_f1_std=np.std(f1s), cv_top1=top1_accuracy(tr, oof),
                     test_top1=top1_accuracy(te, p), test_macro_f1=cm["macro_f1"], test_log_loss=cm["log_loss"],
                     train_time_s=ft, inference_time_s=inf, **{f"param_{k}": v for k, v in params.items()}))
    print(rows[-1], flush=True)
    if name == "HistGB_tuned":
        np.save(os.path.join(PROC, "oof_HistGB_tuned.npy"), oof); np.save(os.path.join(PROC, "test_HistGB_tuned.npy"), p)
        joblib.dump(m, os.path.join(MOD, "final_model.joblib"))
        json.dump(dict(model="HistGradientBoostingClassifier", params=best_params, features=FEATURES,
                       candidates=DEFAULT_CAND, seed=SEED, trained_on="80 train trips (excluding bad-label trips), every 3rd observation"),
                  open(os.path.join(MOD, "model_config.json"), "w"), indent=2)
pd.DataFrame(rows).to_csv(os.path.join(RES, "tuned_vs_untuned.csv"), index=False)
