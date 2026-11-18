# PathSnap — Technical Report (DRAFT) — Team 111

> **Draft built from the experiment outputs in this repository. Every number comes from a file in `results/`.**
> Before you submit: rewrite the explanations in your own words, add the API/UI screenshots from your laptops, add a screenshot of a Folium map (`maps/*.html`), and disclose AI assistance as the event policy requires. Every team member must be able to explain each section.

## 2.2 Dataset understanding and EDA (partitions: train + student_test; evaluator_test never opened)
**Inventory** (`results/eda_inventory.csv`, `results/eda_summary.json`): 80 train trips and 10 student-test trips, 208,417 and 19,160 GPS observations. All trips have the four files. The trips come from many countries (figures/eda_trip_centres.png), so a global planar projection would distort distances.

**Missing / invalid values**: 0 missing values, 0 out-of-range coordinates, 0 non-increasing timestamps. Cleaning code is still in `data.load_trip` for the evaluator data.

**Sampling**: 96.7% of steps are exactly 1 s apart. 42 trips have gaps over 10 s, and the largest gap is 153064 s (figures/eda_distributions.png).

**Road network**: median 9305 nodes / 18762 arcs per trip, no isolated nodes, a median of 19 weakly connected components, and about 92% of arcs have a reverse twin (two-way roads). In 99.94% of cases, consecutive reference arcs are directly connected (figures/eda_network.png).

**Key observations that shaped the design**
- Two-way roads mean the nearest arc is right only about 50% of the time (two arcs at the same distance). Heading is therefore essential.
- The `.route` file is an ordered arc list, not one label per GPS row. We built per-observation labels with a monotonic DP alignment (`src/pathsnap/labels.py`).
- Reference limitations: trips 33 and 54 have most points more than 100 m from their reference route (`results/label_quality.csv`). They are excluded from training and points with a label distance over 100 m are dropped.

**Preprocessing**: cleaning → per-trip local equirectangular projection in metres (origin = trip mean; error far below GPS noise at trip scale) → candidate arcs within 50 m, at most 8 (KD-tree over points sampled every 10 m along arcs), falling back to the 3 nearest arcs if none are within 50 m. Mean candidates per point: 6.2. Candidate miss rate: train 3.1%, student-test 0.4%.

**Features (22)** — `src/pathsnap/features.py`: point-to-arc distance, rank, gap to nearest, ratio; heading difference; speed and moved distance; arc length, projection position, distance to arc end; two-way flag; ±3-observation context (mean/max distance to the same arc, mean heading difference, heading-consistency share); time gap; node degrees; **self-calibrating noise features** (per-trip robust σ, distance / σ, context distance / σ).

Figures: eda_distributions, eda_trip_centres, eda_network, eda_trip_00000048 / 00000076 (GPS + network + reference), eda_candidate_miss, plus the Folium maps in maps/.

## 3. Model experimentation and comparison
Task: binary classification of (observation, candidate arc) pairs → P(reference arc). Protocol: 5-fold GroupKFold over training trips (groups = trip), every 3rd observation used for fitting, refit on all training trips, evaluate on student-test. Seed 111.
Metrics: candidate-level macro F1 / precision / recall / log loss at threshold 0.5; observation-level Top-1 accuracy (argmax per observation); route continuity of the argmax sequence. **Primary selection metric: CV macro F1**, with Top-1 accuracy and log loss as tie-breakers because the HMM uses the probabilities.

### 3.1 Overall comparison (`results/model_comparison.csv`)
| Model        | Family              | Key configuration                                                                              | CV macro F1     |   CV Top-1 |   Test Top-1 |   Test macro F1 |   Test macro P |   Test macro R |   Test log loss |   Route continuity (argmax) |   Train top-1 (in-sample) |   Train s |   Inference ms/1k obs |
|:-------------|:--------------------|:-----------------------------------------------------------------------------------------------|:----------------|-----------:|-------------:|----------------:|---------------:|---------------:|----------------:|----------------------------:|--------------------------:|----------:|----------------------:|
| LogReg       | Linear              | StandardScaler + L2 logistic regression, C=1.0                                                 | 0.9098 ± 0.0226 |     0.8919 |       0.8666 |          0.904  |         0.9045 |         0.9034 |          0.1347 |                      0.9208 |                    0.8914 |       1.1 |                   0.6 |
| GaussianNB   | Probabilistic       | Gaussian naive Bayes, var_smoothing=1e-9                                                       | 0.8587 ± 0.0173 |     0.8792 |       0.8548 |          0.8432 |         0.8098 |         0.9006 |          0.6293 |                      0.9205 |                    0.879  |       0.1 |                   1.1 |
| KNN          | Instance-based      | StandardScaler + distance-weighted KNN, k=25 (fit on 60k-row subsample)                        | 0.9189 ± 0.0185 |     0.8888 |       0.8633 |          0.9136 |         0.921  |         0.9067 |          0.1557 |                      0.9269 |                    0.9338 |       0   |                 583.9 |
| DecisionTree | Tree                | CART, max_depth=12, min_samples_leaf=20                                                        | 0.9318 ± 0.0182 |     0.8992 |       0.8787 |          0.9218 |         0.9197 |         0.9238 |          0.1383 |                      0.9292 |                    0.935  |       6   |                   0.5 |
| RandomForest | Bagging ensemble    | 100 trees, max_depth=20, min_samples_leaf=5, max_features=sqrt                                 | 0.9400 ± 0.0179 |     0.9084 |       0.8929 |          0.9351 |         0.9402 |         0.9301 |          0.0906 |                      0.9363 |                    0.9679 |      43.7 |                  20.1 |
| ExtraTrees   | Randomised ensemble | 150 trees, max_depth=20, min_samples_leaf=5                                                    | 0.9331 ± 0.0175 |     0.9025 |       0.8849 |          0.9265 |         0.9406 |         0.9137 |          0.0994 |                      0.9292 |                    0.9261 |      11.8 |                  36.5 |
| HistGB       | Boosting            | HistGradientBoosting lr=0.1, max_iter=200, max_leaf_nodes=31 (untuned defaults)                | 0.9400 ± 0.0175 |     0.9082 |       0.8912 |          0.9342 |         0.9337 |         0.9347 |          0.0925 |                      0.9363 |                    0.943  |       4.1 |                  16.4 |
| MLP          | Neural network      | StandardScaler + MLP(64,32) ReLU, Adam, alpha=1e-4, early stopping (fit on 200k-row subsample) | 0.9338 ± 0.0170 |     0.9035 |       0.8793 |          0.9243 |         0.9321 |         0.917  |          0.1108 |                      0.9392 |                    0.9253 |      15.5 |                   2.7 |

**Selection**: RandomForest and HistGB tie on CV macro F1 (0.9400). We chose **HistGradientBoosting**: about 10× faster to train, lower variance across folds, a smaller train-vs-CV gap (0.943 vs 0.908 compared with RF's 0.968 vs 0.908, so less overfitting) and comparable log loss, which matters for the HMM emissions.

### 3.2 Per-model notes (expand each into your own subsection: suitability, strengths/limitations, hyperparameters, loss, student-test results, over/underfitting evidence)
- LogReg — linear boundary on the features; log-loss objective. Underfits slightly (train ≈ CV ≈ 0.89).
- GaussianNB — independence assumption broken by correlated distance features; poorly calibrated (log loss 0.63).
- KNN — no training loss; slowest inference (584 ms/1k observations); overfits (train 0.934 vs CV 0.889).
- DecisionTree — Gini impurity; depth limit 12 controls overfitting.
- RandomForest — strongest but overfits most (train 0.968); slowest to train.
- ExtraTrees — more randomness, so lower variance, but slightly lower accuracy.
- HistGB — log-loss boosting; see figures/history_HistGB.png for the train/validation loss.
- MLP — binary cross-entropy with Adam; see figures/history_MLP.png (train/validation loss by epoch).

### 3.3 Visualisations
figures/cmp_cv_macro_f1.png, cmp_test_top1_f1.png, cmp_logloss_time.png, cmp_overfit.png, cmp_confusion.png, history_MLP.png, history_HistGB.png (training curves only for the two models that produce them).

## 4. Bayesian hyperparameter optimisation (HistGB)
Our own GP-based optimiser (`scripts/04_bayes_opt.py`): Matern-5/2 Gaussian-process surrogate + Expected Improvement, 6 random initial points + 14 GP-EI iterations = **20 trials** (203 s). Objective: mean macro F1 over 3-fold GroupKFold on training trips (every 9th observation).
Search space: learning_rate 0.01–0.3 (log), max_leaf_nodes 8–128 (log, int), min_samples_leaf 10–400 (log, int), l2_regularization 1e-4–10 (log), max_iter 100–600, max_features 0.4–1.0.
Best: learning_rate=0.0129, max_leaf_nodes=97, min_samples_leaf=187, l2=1.595, max_iter=508, max_features=0.995. Trials: `results/bayes_opt_trials.csv`; figures/bayes_opt_convergence.png.

| model          |   cv_macro_f1 |   cv_top1 |   test_top1 |   test_macro_f1 |   test_log_loss |   train_time_s |   inference_time_s |
|:---------------|--------------:|----------:|------------:|----------------:|----------------:|---------------:|-------------------:|
| HistGB_untuned |        0.94   |    0.907  |      0.8912 |          0.9342 |          0.0925 |         4.7791 |             0.3183 |
| HistGB_tuned   |        0.9404 |    0.9072 |      0.8921 |          0.9353 |          0.0895 |        19.2697 |             1.1714 |

Tuning gave a small but consistent improvement (CV F1 +0.0004, test log loss 0.0925 → 0.0895). The classifier was already close to a plateau; the decoder gave the large gain (section 5). The cost is about 4× slower inference (more, smaller trees).

## 5. HMM and Viterbi route reconstruction (`src/pathsnap/hmm.py`, `scripts/05_hmm.py`)
1. **Hidden states**: the candidate arcs of each observation (arc id + projection fraction), so the candidate ↔ arc mapping is kept.
2. **Emission**: w_e · log(p_clf / Σ p_clf over the observation's candidates), from the tuned HistGB.
3. **Transition**: route distance d_r (along the arc / shortest path, Dijkstra with cutoff) vs GPS distance d_g.
   - classic: −|d_r − d_g| / (β · max(1, Δt))
   - learned: log-sigmoid of a logistic model trained on reference transitions (half A of the training trips). Coefficients: {'log_diff': -5.0081, 'ratio': 0.0431, 'same': 0.987, 'uturn': -0.8099, 'backward': 0.611, 'turn': -2.0218, 'log_dt': 0.0236, 'unreach': 0.8599}
   - **hybrid (deployed)**: classic + w_l · learned. Selected parameters: {'classic': {'beta': 3.0, 'w_emit': 0.5}, 'learned': {'w_trans': 1.0, 'w_emit': 1.0}, 'hybrid': {'beta': 3.0, 'w_learn': 1.0, 'w_emit': 1.0}}
4. **Missing candidates / gaps**: fallback to the 3 nearest arcs, so every observation has a state. β scales with Δt, and the Dijkstra cutoff grows with d_g after long gaps.
5. **Disconnected segments**: unreachable pairs get a fixed penalty (-15.0) instead of −∞, so the chain acts like a soft restart and never fails. After decoding, gaps are stitched with shortest paths; any break that cannot be bridged is counted.
6. **Tuning**: grid search on 16 held-out training trips (half B) using **out-of-fold** classifier probabilities (`results/hmm_tuning.csv`). Mode and weights were chosen there, not on student-test.

**Student-test results** (`results/hmm_comparison.csv`):
| variant         |   point_acc_micro |   point_acc_trip_mean |   continuity |   route_precision |   route_recall |   route_f1 |   breaks |   transitions |   decode_s |
|:----------------|------------------:|----------------------:|-------------:|------------------:|---------------:|-----------:|---------:|--------------:|-----------:|
| classifier_only |            0.8921 |                0.9066 |       0.9385 |            0.6786 |         0.9979 |     0.7972 |      131 |          4391 |     0      |
| hmm_classic     |            0.9468 |                0.9509 |       0.9771 |            0.9326 |         0.9966 |     0.9598 |        3 |          3291 |     4.0048 |
| hmm_hybrid      |            0.9553 |                0.9555 |       0.9771 |            0.9079 |         0.9972 |     0.9454 |        3 |          3158 |     8.4638 |
| hmm_learned     |            0.9416 |                0.9473 |       0.9767 |            0.9188 |         0.9965 |     0.9536 |        3 |          3160 |     8.0979 |

Per trip (`results/hmm_per_trip.csv`, figures/hmm_per_trip.png):
|   trip |   classifier_only |   hmm_classic |   hmm_hybrid |   hmm_learned |
|-------:|------------------:|--------------:|-------------:|--------------:|
|      0 |            0.9521 |        0.9792 |       0.9816 |        0.9696 |
|      7 |            0.9291 |        1      |       1      |        0.9948 |
|      9 |            0.878  |        0.9047 |       0.9066 |        0.9094 |
|     40 |            0.9472 |        0.9856 |       0.9803 |        0.981  |
|     52 |            0.9733 |        1      |       0.9809 |        0.9809 |
|     56 |            0.9121 |        0.9679 |       0.9697 |        0.9603 |
|     65 |            0.8815 |        0.8815 |       0.8997 |        0.9036 |
|     76 |            0.9314 |        0.965  |       0.9724 |        0.961  |
|     77 |            0.7622 |        0.8681 |       0.9039 |        0.8601 |
|     91 |            0.8993 |        0.9565 |       0.96   |        0.9519 |

Discussion points: the HMM raises point accuracy by 6.3 points and route F1 from 0.80 to about 0.95. Unbridgeable breaks fall from 131 to 3. The pure learned transition was *worse* than classic: the logistic model is trained on independent pairs rather than as a conditional distribution. The hybrid keeps the physical prior and uses the learned terms for turns and U-turns, and wins on point accuracy (biggest gain on trip 77: 0.762 → 0.904). Where the HMM did not help: trip 65 classic gives no gain, and trip 52 hybrid is below classic. Classic has the highest route precision (0.933). The hybrid costs about 2× the decode time.

## 6. Folium maps
`scripts/06_maps.py` → maps/trip_<id>.html (layers: road network, GPS, reference, classifier-only, HMM route, low-confidence points; layer control and legend) and figures/map_<id>.png. **Insert a screenshot of one HTML map here.**

## 7. API, UI and deployment
FastAPI (`api/main.py`): GET /health, GET /trips, POST /predict. The UI (`ui/index.html`, Leaflet) lists trips, sends only the trip ID and shows the route, GPS points (optionally coloured by HMM confidence) and metrics. Reference routes are used only server-side for scoring. **Insert UI screenshots and the Laptop-1 terminal log lines (`PREDICT trip=…`).**

## 8. AI-use disclosure
(Fill in as required by the event policy.)

## 9. Final evaluation on the evaluator test set (10 trips, 19,674 GPS points)
Run with the frozen deployed pipeline in `notebooks/06_evaluator_test.ipynb`: model checksums recorded before and after (unchanged), model files last modified before the evaluator data arrived, and `.route` files blocked during prediction and read only for scoring.

| Metric | Classifier only | Classifier + hybrid HMM |
|---|---|---|
| Point accuracy (all points) | 90.49% | 96.74% |
| Route continuity (trip mean) | 92.79% | 96.33% |
| Route precision / recall / F1 (trip mean) | — | 0.970 / 0.995 / 0.982 |
| Unbridgeable breaks | — | 0 |

The HMM improved all 10 trips (largest gain: trip 15, 82.1% → 95.7%; hardest trip: 30, 89.3%). Evaluator accuracy is close to the student-test result (95.5%), so the development estimate generalised. Per-trip results: `results/evaluator_test_results.csv`; maps: `figures/evaluator_test_maps.png`.
