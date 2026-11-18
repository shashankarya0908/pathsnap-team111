"""Train and compare candidate classifiers (>= 5 model families).

* 5-fold GroupKFold over the 80 training trips (groups = trip) -> CV macro F1,
  CV Top-1 accuracy, plus out-of-fold probabilities (used later to tune the HMM).
* Refit on all training trips -> evaluate on the 10 student_test trips.
Outputs: results/model_comparison.csv, results/cv_folds.csv,
         data/processed/oof_<model>.npy, data/processed/test_<model>.npy,
         results/mlp_history.csv, results/hgb_history.csv
"""
import os, time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

from common import ROOT, SEED, PROC, RES, FEATURES, load_train, load_student_test, top1_accuracy
from pathsnap.data import load_trip
from pathsnap.metrics import candidate_metrics, route_continuity

from model_zoo import MODELS

tr, bad = load_train(step=3)
te = load_student_test()
print("train rows", len(tr), "excluded trips", bad, "test rows", len(te), flush=True)
Xtr, ytr, gtr = tr[FEATURES].values, tr.y.values, tr.trip.values
Xte, yte = te[FEATURES].values, te.y.values
rng = np.random.RandomState(SEED)

test_arcs = {t: load_trip(os.path.join(ROOT, "data", "raw", "student_test", t)).arcs for t in te.trip.unique()}


def fit(name, X, y):
    _, _, mk, sub = MODELS[name]
    m = mk()
    if sub and len(X) > sub:
        i = rng.choice(len(X), sub, replace=False); X, y = X[i], y[i]
    t0 = time.time(); m.fit(X, y)
    return m, time.time() - t0


def continuity(df, p):
    d = df[["trip", "pt", "arc"]].copy(); d["p"] = p
    idx = d.groupby(["trip", "pt"])["p"].idxmax()
    best = d.loc[idx].sort_values(["trip", "pt"])
    return float(np.mean([route_continuity(g.arc.values, test_arcs[t]) for t, g in best.groupby("trip")]))


gkf = GroupKFold(n_splits=5)
fold_rows, summary = [], []
for name, (family, config, _, _) in MODELS.items():
    oof = np.zeros(len(tr))
    for k, (a, b) in enumerate(gkf.split(Xtr, ytr, gtr)):
        m, ft = fit(name, Xtr[a], ytr[a])
        oof[b] = m.predict_proba(Xtr[b])[:, 1]
        cm = candidate_metrics(ytr[b], oof[b])
        fold_rows.append(dict(model=name, fold=k, cv_macro_f1=cm["macro_f1"], cv_log_loss=cm["log_loss"],
                              cv_top1=top1_accuracy(tr.iloc[b], oof[b]), fit_s=ft))
        print(f"  {name} fold{k} f1={cm['macro_f1']:.4f} top1={fold_rows[-1]['cv_top1']:.4f} {ft:.1f}s", flush=True)
    np.save(os.path.join(PROC, f"oof_{name}.npy"), oof)
    folds = pd.DataFrame([r for r in fold_rows if r["model"] == name])
    m, ft = fit(name, Xtr, ytr)
    t0 = time.time(); p = m.predict_proba(Xte)[:, 1]; inf = time.time() - t0
    np.save(os.path.join(PROC, f"test_{name}.npy"), p)
    cm = candidate_metrics(yte, p)
    row = dict(model=name, family=family, config=config,
               cv_macro_f1_mean=folds.cv_macro_f1.mean(), cv_macro_f1_std=folds.cv_macro_f1.std(),
               cv_top1_mean=folds.cv_top1.mean(), cv_log_loss_mean=folds.cv_log_loss.mean(),
               test_top1=top1_accuracy(te, p), test_macro_f1=cm["macro_f1"],
               test_macro_precision=cm["macro_precision"], test_macro_recall=cm["macro_recall"],
               test_log_loss=cm["log_loss"], test_route_continuity=continuity(te, p),
               train_top1_full_fit=top1_accuracy(tr, m.predict_proba(Xtr)[:, 1]),
               train_time_s=ft, inference_time_s=inf, inference_ms_per_1k_obs=1000 * inf / te.pt.groupby(te.trip).nunique().sum() * 1)
    summary.append(row)
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "config"}), flush=True)
    pd.DataFrame(summary).to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(RES, "cv_folds.csv"), index=False)

# --------- epoch-wise histories (only for models that genuinely have them)
# MLP: manual epochs with partial_fit, train vs validation log loss on held-out trips
trips = np.unique(gtr); rng2 = np.random.RandomState(SEED); val_trips = rng2.choice(trips, 12, replace=False)
va = np.isin(gtr, val_trips)
sc = StandardScaler().fit(Xtr[~va])
Xa, ya, Xv, yv = sc.transform(Xtr[~va]), ytr[~va], sc.transform(Xtr[va]), ytr[va]
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, random_state=SEED, batch_size=512)
hist = []
for ep in range(1, 31):
    perm = rng2.permutation(len(Xa))
    for s in range(0, len(Xa), 20000):
        j = perm[s:s + 20000]; mlp.partial_fit(Xa[j], ya[j], classes=[0, 1])
    hist.append(dict(epoch=ep, train_log_loss=log_loss(ya, mlp.predict_proba(Xa)[:, 1]),
                     val_log_loss=log_loss(yv, mlp.predict_proba(Xv)[:, 1])))
pd.DataFrame(hist).to_csv(os.path.join(RES, "mlp_history.csv"), index=False)
# HistGB: per-iteration loss with group-held-out validation trips
hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, random_state=SEED)
hgb.fit(Xtr[~va], ytr[~va])
h = [dict(iteration=i + 1, train_log_loss=log_loss(ytr[~va], pa[:, 1]), val_log_loss=log_loss(yv, pv[:, 1]))
     for i, (pa, pv) in enumerate(zip(hgb.staged_predict_proba(Xtr[~va]), hgb.staged_predict_proba(Xtr[va]))) if i % 5 == 4]
pd.DataFrame(h).to_csv(os.path.join(RES, "hgb_history.csv"), index=False)
print("done")
