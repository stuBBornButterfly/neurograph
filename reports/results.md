# Results (Phases 1–5)

Every number below comes from a committed Kaggle notebook run; the commit that produced it is listed per step.
All model metrics are **person-level ROC AUC, mean ± SD over the same 50 folds** (StratifiedGroupKFold by person, 5 folds × 10 repeats, seed 97) unless stated otherwise.

## 1. Data (ds003478 v1.1.0)

| | n | women | BDI-II mean (range) | STAI mean (range) |
|---|---|---|---|---|
| CTL (stable BDI < 7) | 75 | 40 (53%) | 1.7 (0–7) | 31.1 (22–47) |
| DEP (BDI ≥ 13) | 45 | 33 (73%) | 22.1 (13–30) | 55.7 (38–69) |

- 122 subjects in the release → 120 analysed: sub-038 excluded by the authors (unstable BDI), sub-107 has < 2 eyes-closed or eyes-open blocks.
- Groups follow Cavanagh et al. 2019 (PMID 31149639). BDI and STAI correlate strongly (ρ = 0.83), so anxiety cannot be separated from depression in this sample.
- **Sex is a confound:** DEP is 73% female vs 53% in CTL.

## 2. Preprocessing QC (commit `4531cf3`, 120/120 subjects, 0 failures)

| metric | median | range |
|---|---|---|
| epochs rejected | 1.6% | 0–44.9% |
| eyes-closed / eyes-open epochs kept | — | ≥ 56 / ≥ 46 per person |
| ICA components removed (ICLabel p ≥ 0.7) | 8.5 | — |
| brain components kept | 13 | — |
| alpha O1 eyes-closed / eyes-open | 7.4× | — |

- CTL vs DEP do not differ on any QC metric (p = 0.29–0.90): cleaning did not favour one group.
- 7 subjects flagged and kept (5 with high rejection, 2 with weak eyes-closed alpha); a sensitivity run without them is in step 4.1.
- Data-quality fixes found on the way: EKG channel in 65 recordings, HEOG/VEOG labels swapped in all recordings, junk at the end of run-01.

## 3. Features (commit `17b8949`)

- Welch PSD (0.5 Hz bins), 5 bands (delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz), absolute (log10 µV²) and relative power, 5 regions, 3 FAA pairs, individual alpha frequency (IAF): **157 features per person**.
- Sanity: IAF median 10.0 Hz (98% within 8–12 Hz); alpha back > front in 100%, eyes-closed > eyes-open (Berger effect) in 99% of people.
- **FAA not replicated:** F4–F3 DEP +0.04 vs CTL (p = 0.035 uncorrected, 3 pairs tested, opposite to the classic direction).
- Gamma excess in DEP is frontal (+0.12 log10, p = 0.009) and absent at temporal sites: not jaw or temple muscle; frontal muscle vs brain unresolved.
- **Sex effect dominates:** female − male central beta +0.21 / +0.27 log10 vs DEP − CTL within sex +0.07 (F) / +0.01 (M), n.s.
- Differential entropy is identical to log band power (r = 1.000), so it is kept only as a named biomarker in the graph.
- Descriptive (graph query 6): eyes-closed relative alpha is lower in DEP in every region (≈ −0.05).

## 4. Models

### 4.1 Baselines and the headline model (commit `d855274`)

| model | features | AUC |
|---|---|---|
| majority class | — | 0.500 |
| sex only | 1 | 0.602 ± 0.104 |
| EEG, eyes closed | 54 | 0.667 ± 0.102 |
| EEG, both conditions (exploratory) | 107 | 0.712 ± 0.095 |
| **sex + EEG eyes closed (pre-declared headline)** | 55 | **0.671 ± 0.105** |
| headline without the 7 QC-flagged people | 55 | 0.657 ± 0.108 |
| women only (n = 73) | 54 | 0.648 ± 0.114 |

Paired over the same folds: headline − sex only ΔAUC +0.069 ± 0.154 (EEG better in 70% of folds).

### 4.2 Permutation tests, XGBoost, feature importance (commits `bfcd77b`, `f4bd9fb`)

| model | real AUC | shuffled-label 95th pct | p (200 shuffles) |
|---|---|---|---|
| EEG eyes closed | 0.667 | 0.618 | 0.010 |
| EEG both conditions | 0.712 | 0.617 | 0.005 |
| women only | 0.648 | 0.659 | 0.075 (n.s.) |
| **EEG eyes closed, sex removed** | 0.633 | 0.605 | **0.030** |
| EEG both conditions, sex removed | 0.670 | 0.628 | 0.010 |

- "Sex removed": sex's linear effect is regressed out of every feature inside each training fold and sex is not an input. About 20% of EEG's above-chance signal overlaps with sex; the rest survives.
- XGBoost does not beat logistic regression (eyes closed 0.664, both 0.682).
- Drivers (LR weights and tree SHAP agree, all weights sign-stable across 50 folds but small): more eyes-closed frontal delta, less eyes-open temporal/parietal theta, more frontal/central beta (+ DEP). Alpha and FAA contribute little.
- Confound candidates: frontal delta = drowsy eye rolling; beta = forehead muscle or sex.

### 4.3 Calibration and risk tiers (commits `2a3b53e`, `f4bd9fb`)

| model | Brier | mean P (prevalence 0.375) | calibration slope | intercept |
|---|---|---|---|---|
| headline, uncalibrated | 0.232 | 0.472 | 0.66 | −0.45 |
| **headline, Platt-calibrated inside folds** | **0.217** | **0.377** | **1.04** | **0.01** |
| EEG both, calibrated | 0.211 | 0.382 | 1.28 | 0.09 |
| EEG eyes closed, sex removed, calibrated | 0.224 | 0.376 | 0.89 | −0.06 |

Risk tiers from calibrated P(DEP) with pre-declared cut-offs 0.33 / 0.66:

| tier | n | CTL | DEP | share DEP | women |
|---|---|---|---|---|---|
| low | 46 | 33 | 13 | 28% | 35% |
| moderate | 72 | 42 | 30 | 42% | 78% |
| high | 2 | 0 | 2 | 100% | 50% |

- Without sex (exploratory): low 45 (27% DEP), moderate 75 (44% DEP), high 0.
- Severity is not tracked: within DEP, Spearman ρ(P(DEP), BDI) = −0.20 (p = 0.20), ρ(P(DEP), STAI) = −0.14 (p = 0.37).
- **Conclusion:** the signal is real but too weak for individual risk tiers; tiers are shown as a pipeline demonstration, not a usable screen.

### 4.4 EEGNet on raw EEG (commit `8fda339`, GPU T4)

| model (same 15 folds: first 3 repeats) | AUC |
|---|---|
| EEGNet, raw eyes-closed epochs (11,357 × 59 ch × 500 samples) | 0.653 ± 0.111 (epoch-level 0.618) |
| logistic regression on band power | 0.665 ± 0.090 |

Paired ΔAUC −0.013 ± 0.143 (EEGNet better in 40% of folds). EEGNet overfits within ~2 epochs (best epoch median 2): with 81 training people per fold, deep learning adds nothing over band power.

### 4.5 Smaller caps (commit `e0b9da4`)

Features recomputed from the cleaned epochs with only the layout's channels, re-referenced to the layout's own average.

| layout | channels | EEG eyes closed | sex + EEG | sex removed |
|---|---|---|---|---|
| full cap | 62 | 0.667 ± 0.102 | 0.671 ± 0.105 | 0.633 ± 0.100 |
| BioSemi 32 | 32 | 0.667 ± 0.099 | 0.669 ± 0.100 | 0.645 ± 0.095 |
| 10-20 (Mumtaz layout) | 19 | 0.658 ± 0.105 | 0.658 ± 0.104 | 0.635 ± 0.107 |

Paired ΔAUC vs the full cap within ±0.013. Sanity: the full layout reproduces the step 3 features to 4.8e-8.

### 4.6 Experiment tracking

All 36 Phase 4 model runs are logged in MLflow (`derivatives/mlflow.db` in the `neurograph-42-channels` output: one parent run per step, one child run per model with per-fold AUC, figures attached) and summarised in `derivatives/models/all_results.csv`.

## 5. Study graph (commit `9782432`, Neo4j AuraDB 5.27)

19,952 nodes / 38,963 relationships: 120 participants → sessions → recordings → pipeline runs (git hash, ICs removed, % rejected); 18,838 feature values linked to 79 biomarkers (measure × band × region, 3 FAA, IAF) → 5 frequency bands / 5 brain regions; 31 models with AUC; logistic-regression weights; 240 risk estimates. All 10 hand-written queries reproduce the numbers above (70–170 ms each); see `src/neurograph/graph/queries.py`.
