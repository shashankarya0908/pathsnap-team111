# PathSnap — Team 111
Map-matching and route reconstruction from noisy GPS: candidate classifier + **learned-transition HMM**.

## Final result (evaluator test, 10 unseen trips, 19,674 GPS points)
| Metric | Classifier only | **Classifier + hybrid HMM** |
|---|---|---|
| Point accuracy | 90.5% | **96.7%** |
| Route continuity | 92.8% | **96.3%** |
| Route precision / recall / F1 | — | **97.0% / 99.5% / 0.982** |
| Unbridgeable route breaks | — | 0 |
| Inference time per trip | — | 1.4 s |

Frozen models (checksums verified before and after), reference routes blocked during prediction:
`notebooks/06_evaluator_test.ipynb`, `results/evaluator_test_results.csv`, `results/evaluator_test_overall.csv`.

## Deliverables → where to find them (executed notebooks = proof)
| Deliverable | Notebook (with outputs) | Production code |
|---|---|---|
| Data processing (read, inspect, clean, preprocess) | `notebooks/01_data_processing.ipynb` (+ `02b_stage2-4_full_analysis.ipynb`) | `src/pathsnap/data.py`, `labels.py`, `scripts/01_eda.py`, `scripts/02_build_dataset.py` |
| Feature engineering | `notebooks/02_feature_engineering.ipynb` | `src/pathsnap/features.py` |
| Model training (8 models) + evaluation | `notebooks/03_model_training_evaluation.ipynb` | `scripts/model_zoo.py`, `scripts/03_train_compare.py`, `scripts/07_plots.py` |
| Bayesian hyperparameter optimisation | `notebooks/04_bayesian_optimisation.ipynb` | `scripts/04_bayes_opt.py` |
| HMM + Viterbi route reconstruction | `notebooks/05_hmm_viterbi.ipynb` | `src/pathsnap/hmm.py`, `scripts/05_hmm.py` |
| Final evaluation on evaluator test | `notebooks/06_evaluator_test.ipynb` | `src/pathsnap/pipeline.py` |
| Geographic visualisation | `maps/*.html`, `figures/map_*.png` | `scripts/06_maps.py` |
| API + UI | — | `api/main.py`, `ui/index.html` |

Notebook run order: 01 → 02 → 03 → 04 → 05 (each **Run all**; also works in Google Colab, which asks for `Student_folder.zip`).
06 needs the repository (`src/`, `models/`) and the evaluator trips in `data/raw/evaluator_test/`.
Demo helpers: `python scripts/demo_models.py` (all 8 models live, alone and with the HMM), `python scripts/preprocessing_walkthrough.py` (preprocessing step by step).

Note on numbers: the deployed system (`models/`, used by the API and notebook 06) was produced by `scripts/`; the notebooks reproduce every experiment step by step (`models/notebook/`). Values agree to within about 0.01; the notebooks are the reference for the report tables.

## What is different about our approach
Most map-matchers use "nearest road + Newson–Krumm HMM". Ours has three extra pieces:

1. **Self-calibrating noise model.** Each trip estimates its own GPS noise scale σ from the GPS and the map only (σ = 1.4826 × median nearest-arc distance, a MAD-style estimator). Distance features are also given in units of σ (`dist_z`, `ctx_dist_z`), so one classifier works for both clean and badly offset trips. A **temporal context window** (±3 observations: mean/max distance to the same arc, heading agreement) lets the classifier see how a whole stretch of track fits each arc, not just one point.
2. **Hybrid learned-transition HMM.** A logistic transition model is trained on the reference transitions of half the training trips. Its features are |route distance − GPS distance|, their ratio, same-arc, U-turn, moving backwards on an arc, turn angle, time gap, and unreachable. We compared three decoders: *classic* (Newson–Krumm exponential), *learned* (logistic model only) and *hybrid* (classic distance prior + learned correction). Decoder weights were tuned on held-out training trips using out-of-fold classifier scores. **Hybrid won the tuning and is deployed.** On student-test, point accuracy was classifier-only 89.2% → classic 94.7% → learned 94.2% → hybrid 95.5% (`results/hmm_comparison.csv`).
3. **Confidence + stitched route.** Forward–backward gives a posterior confidence for each observation (the UI can colour points by it). Gaps between non-adjacent decoded arcs are filled with shortest paths, so the output is one continuous route you could actually drive.

## Repository layout
```
src/pathsnap/      data.py (loading, projection, spatial index) · labels.py (route→per-point labels)
                   features.py (candidates + features) · hmm.py (Viterbi, forward-backward, stitching)
                   metrics.py · pipeline.py (inference used by the API)
scripts/           01_eda.py 02_build_dataset.py 03_train_compare.py 04_bayes_opt.py
                   05_hmm.py 06_maps.py 07_plots.py api_client.py common.py model_zoo.py
                   demo_models.py preprocessing_walkthrough.py
notebooks/         01–06 executed notebooks (proof), 02b full stage 2–4 analysis
api/main.py        FastAPI backend (Laptop 1)
ui/index.html      evaluator UI (Laptop 2), Leaflet
models/            final_model.joblib, transition_model.joblib, model_config.json, hmm_config.json (deployed)
models/notebook/   models produced by the notebooks
results/           every CSV / JSON / log produced by the scripts
figures/           every plot used in the report     maps/  Folium HTML maps
data/raw/          train/ (80) · student_test/ (10) · evaluator_test/ (10, received at evaluation)
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
The saved `.joblib` models were built with Python 3.13 and scikit-learn 1.9.1. If your scikit-learn version differs, re-create the artifacts with `./run_all.sh`, or run scripts 02 → 04 → 05 (about 30 min).

## Reproduce the experiments
```bash
./run_all.sh          # or run scripts/0*.py one by one, in the order listed in run_all.sh
```
Seeds are fixed (`SEED = 111`). The evaluator partition is never read by any training or tuning script.

## Run the two-laptop demo
**Laptop 1 (API + model)**
```bash
# evaluator trips are in data/raw/evaluator_test/ (the API serves them by default)
uvicorn api.main:app --host 0.0.0.0 --port 8000
# optional: PATHSNAP_EVAL_DIR=/path/to/eval uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Find Laptop 1's IP (`ipconfig` / `ip a`) and allow port 8000 through its firewall. Each request is logged in this terminal (`REQUEST …` and `PREDICT trip=… acc=…`). Those log lines are the "API logs on the receiving laptop".

**Laptop 2 (UI)**
```bash
cd ui && python -m http.server 8080
# open http://localhost:8080, enter http://<laptop1-ip>:8000, click "Connect & load trips", pick a trip
```
Command-line check from Laptop 2: `python scripts/api_client.py http://<laptop1-ip>:8000`

### API
| Method | Path | Description |
|---|---|---|
| GET | `/health` | status, model, decoder |
| GET | `/trips` | evaluation trip ids found in `PATHSNAP_EVAL_DIR` |
| POST | `/predict` `{"trip_id": "..."}` | full pipeline; returns GPS, per-point arc + confidence, stitched route polyline, metrics |
| GET | `/predict/{trip_id}` | same as POST |

Reference routes are read only in `PathSnapPredictor.evaluate()`, on the server, after prediction. Only aggregate metrics are returned. No reference geometry or labels are sent to the UI.

## Pipeline summary
1. **Clean**: drop non-finite / out-of-range coordinates and non-increasing timestamps (none found in train / student_test, but handled).
2. **Project**: per-trip local equirectangular projection (metres), centred on the trip mean.
3. **Reference labels**: monotonic DP alignment of GPS points to the ordered `.route` arcs (`labels.py`). Trips 33 and 54 have most points more than 100 m from their route, so they are excluded from training as unreliable labels.
4. **Candidates**: arcs within 50 m (KD-tree over points sampled along arcs), at most 8 per point. If none are within 50 m, fall back to the 3 nearest arcs.
5. **Features** (22): distance, rank, gap to nearest, heading difference, speed, arc length, projection position, two-way flag, context-window distances and headings, time gap, node degrees, noise-normalised distances.
6. **Classifier**: 8 model families compared with 5-fold GroupKFold (groups = trips). HistGradientBoosting is selected and tuned with our own GP + Expected-Improvement Bayesian optimiser.
7. **Decoder**: hybrid learned-transition HMM (Viterbi), forward–backward confidence, shortest-path stitching.
