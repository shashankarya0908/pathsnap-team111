#!/usr/bin/env bash
# Reproduce every experiment and artifact from the raw data (approx. 30-40 min on a laptop).
set -e
cd "$(dirname "$0")/scripts"
python 02_build_dataset.py     # candidates + features + reference labels
python 01_eda.py               # EDA tables + figures
python 03_train_compare.py     # 8 models, GroupKFold CV, student-test metrics
python 04_bayes_opt.py         # GP-EI Bayesian optimisation, final model export
python 05_hmm.py               # transition model, decoder tuning, HMM comparison
python 07_plots.py             # all report figures
python 06_maps.py              # Folium + static maps
