"""All result figures for the report, generated only from saved experiment outputs."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from common import PROC, RES, FIG, load_student_test

C1, C2, C3 = "#3b6ea5", "#d1495b", "#edae49"
cmp_ = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
n_test_obs = load_student_test().groupby("trip").pt.nunique().sum()
cmp_["inference_ms_per_1k_obs"] = 1e6 * cmp_.inference_time_s / n_test_obs  # corrected unit
cmp_.to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
folds = pd.read_csv(os.path.join(RES, "cv_folds.csv"))
x = np.arange(len(cmp_))

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, cmp_.cv_macro_f1_mean, yerr=cmp_.cv_macro_f1_std, color=C1, capsize=4)
for i, m in enumerate(cmp_.model):
    f = folds[folds.model == m].cv_macro_f1.values; ax.scatter(np.full(len(f), i), f, color="k", s=8, zorder=3)
ax.set(xticks=x, xticklabels=cmp_.model, ylim=(0.8, 1.0), ylabel="macro F1", title="5-fold GroupKFold CV macro F1 (dots = folds)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "cmp_cv_macro_f1.png"), dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(10, 4)); w = .38
ax.bar(x - w / 2, cmp_.test_top1, w, label="Top-1 accuracy", color=C1)
ax.bar(x + w / 2, cmp_.test_macro_f1, w, label="Macro F1", color=C2)
ax.set(xticks=x, xticklabels=cmp_.model, ylim=(0.8, 0.96), title="Student-test: observation Top-1 accuracy and candidate macro F1"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "cmp_test_top1_f1.png"), dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(x, cmp_.test_log_loss, color=C3); ax[0].set(xticks=x, title="Student-test log loss (lower is better)")
ax[0].set_xticklabels(cmp_.model, rotation=30)
ax[1].bar(x, cmp_.inference_ms_per_1k_obs, color=C1); ax[1].set(xticks=x, yscale="log", title="Inference time, ms per 1k observations (log)")
ax[1].set_xticklabels(cmp_.model, rotation=30)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "cmp_logloss_time.png"), dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - w / 2, cmp_.train_top1_full_fit, w, label="train (in-sample) Top-1", color="#999999")
ax.bar(x + w / 2, cmp_.cv_top1_mean, w, label="CV Top-1", color=C1)
ax.set(xticks=x, xticklabels=cmp_.model, ylim=(0.8, 1.0), title="Over-fitting check: in-sample vs cross-validated Top-1"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "cmp_overfit.png"), dpi=130); plt.close()

# confusion matrices (candidate level, threshold 0.5) on student_test
te = load_student_test()
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for axx, m in zip(axs.ravel(), cmp_.model):
    p = np.load(os.path.join(PROC, f"test_{m}.npy"))
    ConfusionMatrixDisplay(confusion_matrix(te.y, p >= .5), display_labels=["not ref", "ref arc"]).plot(ax=axx, colorbar=False, values_format="d")
    axx.set_title(m)
plt.suptitle("Student-test confusion matrices (candidate level)"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "cmp_confusion.png"), dpi=110); plt.close()

# training histories (only models with genuine epoch/iteration histories)
for f, xcol, name in [("mlp_history.csv", "epoch", "MLP"), ("hgb_history.csv", "iteration", "HistGB")]:
    p = os.path.join(RES, f)
    if os.path.exists(p):
        h = pd.read_csv(p); fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h[xcol], h.train_log_loss, label="train"); ax.plot(h[xcol], h.val_log_loss, label="validation (12 held-out trips)")
        ax.set(xlabel=xcol, ylabel="log loss", title=f"{name} training vs validation loss"); ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(FIG, f"history_{name}.png"), dpi=130); plt.close()

# Bayesian optimisation
p = os.path.join(RES, "bayes_opt_trials.csv")
if os.path.exists(p):
    t = pd.read_csv(p); fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(t.trial, t.cv3_macro_f1, c=np.where(t.strategy == "gp-ei", C2, "#999999"), label="trial")
    ax.plot(t.trial, t.best_so_far, color="k", label="best so far")
    ax.set(xlabel="trial", ylabel="3-fold group CV macro F1", title="Bayesian optimisation (grey = random init, red = GP-EI)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "bayes_opt_convergence.png"), dpi=130); plt.close()
p = os.path.join(RES, "tuned_vs_untuned.csv")
if os.path.exists(p):
    t = pd.read_csv(p); cols = ["cv_macro_f1", "test_top1", "test_macro_f1"]
    fig, ax = plt.subplots(figsize=(7, 4)); xx = np.arange(len(cols))
    for k, (_, r) in enumerate(t.iterrows()):
        ax.bar(xx + (k - .5) * .38, r[cols].values.astype(float), .38, label=r.model, color=[C1, C2][k])
    ax.set(xticks=xx, xticklabels=cols, ylim=(0.85, 0.97), title="HistGB untuned vs Bayesian-tuned"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "tuned_vs_untuned.png"), dpi=130); plt.close()

# HMM comparison
p = os.path.join(RES, "hmm_per_trip.csv")
if os.path.exists(p):
    h = pd.read_csv(p); piv = h.pivot(index="trip", columns="variant", values="point_acc")
    fig, ax = plt.subplots(figsize=(10, 4)); xx = np.arange(len(piv))
    for k, (v, c) in enumerate([("classifier_only", "#999999"), ("hmm_classic", C3), ("hmm_learned", C2), ("hmm_hybrid", C1)]):
        ax.bar(xx + (k - 1.5) * .2, piv[v], .2, label=v, color=c)
    ax.set(xticks=xx, xticklabels=piv.index, ylim=(0.5, 1.0), ylabel="point accuracy", title="Student-test point accuracy per trip")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "hmm_per_trip.png"), dpi=130); plt.close()
    comp = pd.read_csv(os.path.join(RES, "hmm_comparison.csv"), index_col=0)
    cols = ["point_acc_micro", "continuity", "route_precision", "route_recall", "route_f1"]
    fig, ax = plt.subplots(figsize=(9, 4)); xx = np.arange(len(cols))
    for k, (v, c) in enumerate([("classifier_only", "#999999"), ("hmm_classic", C3), ("hmm_learned", C2), ("hmm_hybrid", C1)]):
        ax.bar(xx + (k - 1.5) * .2, comp.loc[v, cols].values, .2, label=v, color=c)
    ax.set(xticks=xx, xticklabels=cols, ylim=(0.5, 1.0), title="Classifier-only vs HMM decoding (student_test)"); ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "hmm_comparison.png"), dpi=130); plt.close()
print("figures written to", FIG)
