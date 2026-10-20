# Results

Phases 1–5: every number comes from a committed Kaggle notebook run; the commit that produced it is listed per step, and section 4.7 checks them with an independent local rerun. Phases 6–9: notebook 60 on Kaggle and a local run of the same code (2026-10-21, one RTX 5060 Ti 16 GB) for the QLoRA fine-tune, the agent, the demo and the evaluation.
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

### 4.7 Independent local rerun of Phases 1–5 (2026-10-21)

The whole EEG pipeline was re-run from the raw OpenNeuro download on a different machine (RTX 5060 Ti 16 GB, 12 CPU cores; Python 3.12, MNE 1.13.2, scikit-learn 1.7.2) to rebuild the study graph in a local Neo4j 5.26.

| check | Kaggle (above) | local rerun |
|---|---|---|
| labels | 120 (75 CTL / 45 DEP) | identical |
| preprocessing QC | rejected median 1.6% (max 44.9%), 8.5 ICs, alpha ×7.4, same 7 flagged | identical |
| features | 120 × 1186, IAF median 10.0 Hz, OZ closed rel. alpha 0.63 | identical (channel-subset sanity 4.9e-8) |
| sex only | 0.602 ± 0.104 | 0.602 ± 0.104 (identical folds) |
| EEG eyes closed / both | 0.667 ± 0.102 / 0.712 ± 0.095 | 0.664 ± 0.098 / 0.710 ± 0.095 |
| headline sex + EEG closed | 0.671 ± 0.105 | 0.667 ± 0.105 |
| sex removed (permutation p) | 0.633 (p 0.030) | 0.631 (p 0.035) |
| women only (permutation p) | 0.648 (p 0.075) | 0.640 (p 0.085) |
| calibrated headline: Brier / slope | 0.217 / 1.04 | 0.216 / 1.05 |
| risk tiers low / moderate / high | 46 / 72 / 2 | identical (sex-free 45 / 75 / 0 identical) |
| 32 / 19-channel caps (headline) | 0.669 / 0.658 | 0.670 / 0.660 |
| EEGNet vs band-power LR (15 folds) | 0.653 vs 0.665 (Δ −0.013 ± 0.143) | 0.686 vs 0.665 (Δ +0.021 ± 0.125) |

Every conclusion holds. The logistic-regression AUCs differ by ≤ 0.008 (a tenth of the fold-to-fold SD; library versions). EEGNet moved by +0.033 on different GPU hardware with the same seed and folds, and its paired difference to LR changed sign: both runs are inside the noise, which is itself the finding: at n = 120, EEGNet is not stable enough to beat band power. The rebuilt study graph has the same 20,092 nodes as the AuraDB graph, and all 10 Cypher queries return the expected answers (the best model is eeg_all_cal at 0.728 vs 0.727). **scikit-learn ≥ 1.8 changes `StratifiedGroupKFold`'s shuffled splits** (sex-only becomes 0.600 ± 0.073 with the same seed), so the repo pins scikit-learn < 1.8 for local runs.

## 5. Study graph (commit `9782432`, Neo4j AuraDB 5.27)

19,952 nodes / 38,963 relationships: 120 participants → sessions → recordings → pipeline runs (git hash, ICs removed, % rejected); 18,838 feature values linked to 79 biomarkers (measure × band × region, 3 FAA, IAF) → 5 frequency bands / 5 brain regions; 31 models with AUC; logistic-regression weights; 240 risk estimates. All 10 hand-written queries reproduce the numbers above (70–170 ms each); see `src/neurograph/graph/queries.py`.

## 6. The university intake, search index and graphs

### 6.1 Public university pages → PII redaction → chunks (`scripts/ingest_pages.py`)

| | |
|---|---|
| approved public URLs | 23 (4 skipped: 3 JavaScript-only pages, 1 page with links only) |
| ingested | 19 documents (14 HTML, 5 policy PDFs), 19,181 words → 100 chunks (≈ 240 words, 40-word overlap) |
| redacted at intake (Presidio + rules) | 80 person names, 51 personal emails; role-based office contacts kept on purpose |
| leak check after redaction | 0 suspicious strings |
| crawl etiquette | listed pages only, one request every 2 s, identifying user agent; raw pages (with names) never indexed or saved |

Identical on Kaggle (notebook 60) and in the local rerun of 2026-10-21.

### 6.2 Search index (`scripts/build_index.py`)

18 hand-written student questions, hit = a chunk from the right page:

| method | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| **dense (BGE-small, Qdrant)** | **0.89** | **0.94** | **0.94** | **0.907** |
| keyword (BM25) | 0.50 | 0.67 | 0.78 | 0.599 |
| hybrid (reciprocal rank fusion) | 0.67 | 0.83 | 0.89 | 0.755 |

On 100 chunks, BM25 adds noise rather than recall, so the agent uses dense retrieval only (plus three query expansions from student words to university words, e.g. "ChatGPT" → "artificial intelligence (AI) tools policy").

### 6.3 Policy graph (`scripts/build_policy_graph.py`)

19 `Document` → 100 `Policy` sections; 6 `Service` nodes (psychosocial counselling, Room 301 → depression, anxiety, stress); 57 `REFERS_TO`; 9 curated `Rule` nodes (probation at CGPA < 2.25, dismissal < 1.0, retake once, leave ≤ 3 semesters, D lowest pass, W, no re-evaluation). Every rule carries a quote that the loader refuses unless it appears verbatim in the redacted page text.

### 6.4 Literature graph (`scripts/build_literature_graph.py`)

| | |
|---|---|
| PubMed query | resting EEG × depression × band or asymmetry, English abstracts, no reviews: 200 abstracts (1983–2026) |
| candidate sentences | 73 from 53 papers (band word + depressed group + comparison word; connectivity etc. excluded) |
| extracted into the fixed schema (Qwen2.5-3B, self-hosted) | 23 claims {band, measure, region, direction} |
| verified by NLI (DeBERTa-v3-base: the claim restated as a sentence must be entailed by its source sentence) | **16 (70%) from 14 papers** |
| graph | 18 `Paper`, 23 `Claim` (16 verified), 95 `ABOUT` links to the shared `Biomarker` nodes of the study graph |

Replication of each published direction in our eyes-closed data (Welch t-test and Cohen's d computed in Python, verdict rules in `agent/replication.py`):

| biomarker | papers (higher / lower) | ours: DEP − CTL | d | p | verdict |
|---|---|---|---|---|---|
| frontal alpha asymmetry | 2 / 1 | +0.073 | 0.45 | 0.016 | not replicated: significant, opposite direction |
| frontal alpha power | 3 / 1 | −0.070 | −0.16 | 0.39 | not replicated: opposite, n.s. |
| global alpha power | 1 / 0 | −0.068 | −0.16 | 0.38 | not replicated: opposite, n.s. |
| posterior alpha power | 0 / 1 | −0.059 | −0.13 | 0.47 | same direction, n.s. |
| frontal beta power | 0 / 1 | +0.116 | 0.47 | 0.013 | not replicated: significant, opposite direction |
| frontal delta power | 1 / 1 | +0.050 | 0.25 | 0.18 | literature mixed |
| temporal gamma power | 0 / 1 | +0.051 | 0.20 | 0.28 | opposite, n.s. |
| frontal / global theta power | 1 / 0, 2 / 0 | −0.023 / −0.052 | −0.08 / −0.17 | 0.66 / 0.37 | opposite, n.s. |

None of the published directions replicates in this sample. Extraction caveat: one verified claim (PMID 36792002) compares *remitted* depression with controls; the prompt excludes remitted groups but the 3B model kept it and NLI accepted it, so a few claims need a human check.

## 7. QLoRA fine-tune (local run, RTX 5060 Ti 16 GB)

| step | result |
|---|---|
| 7.1 Q&A pairs | teacher Qwen2.5-7B-Instruct (self-hosted, 4-bit on a 16 GB card): 548 pairs from 99 chunks → **427 kept** (answer entailed by its chunk, DeBERTa-v3-base NLI ≥ 0.5) |
| held-out split | by chunk: 20 of 99 chunks; held-out chunks never appear in training prompts (not even as distractors); 25 test pairs dropped because a neighbouring training chunk (40-word overlap) states the same answer → **train 347 (54 with the answer removed) / test 55 (11)** |
| 7.2 training | Qwen2.5-3B-Instruct in 4-bit NF4 + LoRA r 16, α 32 on all 7 projections (29.9 M trainable parameters, 1.7%); 2 epochs, lr 2e-4, effective batch 16, 338 examples (9 longer than 2,048 tokens dropped, never cut); 17 min, 12.1 GB peak |
| loss | train 0.75 → 0.09; held-out 0.197 (epoch 1) → 0.176 (epoch 2) |
| 7.4 export | adapter merged into fp16, llama.cpp `Q4_K_M` GGUF **1.93 GB**; CPU smoke test 29 tokens/s on 4 threads, answers the one-passage test with "The lowest passing grade is a grade of D. [1]" |

Base vs fine-tuned on the 55 held-out examples (same prompt layout as the agent: the right passage among 3 retrieved distractors from other pages, or removed; greedy decoding):

| metric | base | QLoRA |
|---|---|---|
| token F1 vs reference answer | 0.596 | **0.664** |
| faithful (answer entailed by the right chunk, NLI) | 0.614 | **0.841** |
| cites the passage that holds the answer | 0.182 | **0.886** |
| wrongly says "not in the passages" | **0.000** | 0.045 |
| correctly says "not in the passages" (answer removed) | 0.455 | **0.818** |
| median words per answer | 20 | 19 |

The adapter learned the task's format (cite, refuse when the answer is missing) more than new facts. Side effect: 25% of its answers are fragments of ≤ 5 words ("80% and above [1]"), against none for the base model.

## 8. Agent and demo app

### 8.1 Agent on the 24 gold questions (`scripts/run_agent_eval.py`; Neo4j with the study, policy and literature graphs)

Safety → router → dense retrieval + graph rules / study query / verified claims → LLM → verifier (numbers must appear in the evidence; each sentence checked by DeBERTa-v3-small NLI) → sources.

| LLM | route | safety → referral | policy facts | study facts | literature facts | all non-safety | verified (policy) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B base | 24/24 | 6/6 | 10/11 | 4/4 | 2/3 | 16/18 | 0.55 |
| + QLoRA on every route | 24/24 | 6/6 | **11/11** | 3/4 | 1/3 | 15/18 | **0.73** |
| **+ QLoRA on the policy route only** | 24/24 | 6/6 | **11/11** | **4/4** | 2/3 | **17/18** | **0.73** |

- No normal question went to the safety referral; every distress or crisis message did (fixed text, contacts from the graph, 999).
- The adapter was trained on university policy prompts only. On the study and literature routes it answers in fragments and drops PMID citations, so the agent and the demo keep it to the policy route (the base model answers research questions). This setting was chosen after seeing these 24 answers, so it is a design decision, not an independent test.
- Remaining misses: a23 (literature: no PMID cited, all three set-ups); base a08 ("GenAI" instead of the expected "AI"; its content is reasonable).
- Gold set fix: a20 expected Kaggle's 0.727 and the keyword "eeg_all", which cannot match the model name "eeg_all_cal"; the expectation now also accepts "eeg_all_cal" and 0.728 (the local graph's value), for every model alike.
- **Verifier false positive found in the demo test:** the CPU GGUF answered "using ChatGPT for assignments is not permitted [3]", but the AI policy permits limited, cited use; the NLI check still marked it supported. Sentence-level NLI catches invented numbers and statements far from the passages, not wrong generalisations of a nuanced rule.

### 8.2 Demo app (`src/neurograph/app/`, `space/`)

FastAPI (`/ask`, `/feedback`, `/stats`, `/health`) behind a Streamlit chat page, in one Docker image for a free Hugging Face Space (CPU): the fine-tuned Q4_K_M GGUF for university questions and Qwen's own base Q4_K_M GGUF for research questions (llama.cpp), BGE-small + Qdrant, the DeBERTa NLI verifier, and the knowledge graph either live (Neo4j secrets) or from a 19 kB snapshot of aggregates (`scripts/export_graph_snapshot.py`: counts, means and verdicts, never per-participant values). Monitoring: every answer and 👍/👎 vote goes to a JSON-lines log with PII redacted by the same Presidio rules as the intake (distress and crisis messages are not stored); the sidebar shows answers per route, verification rate, latency and feedback, and the log can be pushed to a private HF dataset every 10 minutes.

## 9. Evaluation (local run, 2026-10-21; `scripts/run_benchmark.py`, `run_ragas.py`, `make_audit_sheet.py`)

Same prompts, same retrieved passages and graph facts for every model; greedy decoding; one request at a time (the latency a user sees). The **reference** is Qwen3.5-9B served by Ollama on the same GPU: no commercial API key was available, so it stands in for the commercial-API comparison. The harness talks to any OpenAI-compatible API, and an `external: true` reference only receives public university questions. "Deployed" = the routed setup as Q4_K_M GGUFs in llama.cpp on 8 CPU threads.

### 9.1 Held-out university questions (55: 44 answerable, 11 with the right passage removed; chunks never seen in training)

| model | token F1 | faithful (NLI) | cites right passage | wrong refusals | correct refusals | words | s / answer |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B base | 0.60 | 0.61 | 0.18 | 0.00 | 0.46 | 23 | 0.8 |
| Qwen2.5-3B + QLoRA | 0.66 | 0.84 | 0.89 | 0.04 | 0.82 | 18 | 1.0 |
| routed, Q4_K_M GGUFs on CPU (as deployed) | 0.72 | 0.84 | 0.91 | 0.00 | 0.73 | 20 | 9.2 |
| Qwen3.5-9B via Ollama (self-hosted API) | 0.47 | 0.41 | 0.98 | 0.09 | 0.91 | 45 | 1.7 |

`faithful`: the whole answer is entailed by the right chunk (NLI), which penalises long answers such as the reference's. `cites right passage`: the answer cites the passage number that holds the answer. Refusals: says "the passages don't answer this" (wrong when the answer is there, correct when it was removed). The policy-only row is identical to the QLoRA row here (all held-out prompts are policy prompts).

### 9.2 Agent end to end on the 24 gold questions (18 facts + 6 distress/crisis messages)

| model | policy | study | literature | facts (18) | verified | routing + safety | s / question |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B base | 10/11 | 4/4 | 2/3 | 16/18 | 0.61 | 24/24 | 1.1 |
| Qwen2.5-3B + QLoRA | 11/11 | 3/4 | 1/3 | 15/18 | 0.61 | 24/24 | 1.1 |
| QLoRA on policy, base on study/literature | 11/11 | 4/4 | 2/3 | 17/18 | 0.72 | 24/24 | 1.1 |
| routed, Q4_K_M GGUFs on CPU (as deployed) | 10/11 | 3/4 | 1/3 | 14/18 | 0.78 | 24/24 | 9.8 |
| Qwen3.5-9B via Ollama (self-hosted API) | 10/11 | 4/4 | 3/3 | 17/18 | 0.50 | 24/24 | 2.5 |

- Every set-up routed all 24 questions correctly, sent all 6 distress/crisis messages to the fixed referral, and sent none of the 18 normal questions there.
- Misses: base a08, a23; QLoRA everywhere a19, a23, a24 (terse study/literature answers without PMIDs); **policy-only routing a23 only**; deployed GGUFs a08, a19, a23, a24; reference a08 ("GenAI" instead of the expected "AI": its answer is the most complete on what the AI policy allows).
- `verified` = share of the 18 answers that passed the verifier. It is a pass rate, not accuracy: the reference writes longer answers that fail more NLI checks but are often better (a08), and the deployed model passes 0.78 of its answers while getting the fewest facts right (14/18). The direction check flags its swapped FAA answer (a19); its wrong a08 answer still passes.
- The 4-bit CPU deployment loses 3 gold facts against the fp16 routed setup (14 vs 17 of 18), two of them on the research routes (base GGUF), while doing as well on the held-out policy questions (token F1 0.72 vs 0.66, faithful 0.84 for both).

### 9.3 Cost and compute per 1,000 questions

| model | tokens in / out per question | API $ per 1,000 | compute h per 1,000 | GPU kWh per 1,000 |
|---|---|---|---|---|
| Qwen2.5-3B base | 1164 / 35 | 0.00 | 0.24 | 0.04 |
| Qwen2.5-3B + QLoRA | 1164 / 25 | 0.00 | 0.26 | 0.05 |
| QLoRA on policy, base on study/literature | 1164 / 28 | 0.00 | 0.27 | 0.05 |
| routed, Q4_K_M GGUFs on CPU (as deployed) | 1164 / 28 | 0.00 | 2.52 | CPU |
| Qwen3.5-9B via Ollama (self-hosted API) | 1196 / 70 | 0.00 | 0.51 | 0.09 |

All rows are self-hosted, so the API bill is $0. For a commercial API with list prices in $ per million tokens, 1,000 questions cost (input tokens × input price + output tokens × output price) / 1,000 dollars, with the per-question token counts above (about 1,200 in and 25–70 out per question; prices change, so none are hard-coded). Compute hours are wall-clock model time on one RTX 5060 Ti (GPU rows, 180 W board power for the energy estimate) or 8 CPU threads (deployed row). On the free 2-vCPU Space the deployed setup takes ~100 s per university answer and 40–50 s per research answer.

### 9.4 RAGAS (judge Gemma 2 9B through Ollama, a different model family from every contestant)

| set | model | faithfulness | answer relevancy | context precision |
|---|---|---|---|---|
| gold (18) | Qwen2.5-3B base | 0.97 ± 0.12 (n 18) | 0.80 ± 0.30 (n 18) | - |
| gold (18) | Qwen2.5-3B + QLoRA | 0.88 ± 0.27 (n 18) | 0.71 ± 0.30 (n 18) | - |
| gold (18) | QLoRA on policy only | 0.89 ± 0.26 (n 18) | 0.71 ± 0.31 (n 18) | - |
| gold (18) | deployed Q4_K_M GGUFs | 0.80 ± 0.38 (n 18) | 0.70 ± 0.36 (n 18) | - |
| gold (18) | Qwen3.5-9B (reference) | 0.91 ± 0.24 (n 18) | 0.62 ± 0.40 (n 18) | - |
| held-out (30) | Qwen2.5-3B base | 0.90 ± 0.30 (n 30) | 0.89 ± 0.18 (n 30) | 0.63 ± 0.30 (n 30) |
| held-out (30) | Qwen2.5-3B + QLoRA | 0.97 ± 0.19 (n 29) | 0.69 ± 0.27 (n 30) | 0.63 ± 0.30 (n 30) |
| held-out (30) | deployed Q4_K_M GGUFs | 0.98 ± 0.09 (n 29) | 0.74 ± 0.24 (n 30) | 0.63 ± 0.30 (n 30) |
| held-out (30) | Qwen3.5-9B (reference) | 0.96 ± 0.15 (n 30) | 0.78 ± 0.28 (n 30) | 0.63 ± 0.30 (n 30) |

Context precision is identical for every model by construction: it scores the passages in the prompt, which are the same for all models, with the right passage placed at a random position among four. It says nothing about the models. The two automatic faithfulness measures disagree on the gold policy answers (RAGAS: base 0.95, QLoRA 0.80; NLI verifier pass rate: base 0.55, QLoRA 0.73). The judge also makes visible mistakes: it scored a verbatim quote from the page (a07, "Re-evaluation of Scripts of Final Examinations is not allowed.") as 0.0 faithful, and identical base and routed literature answers got 1.00 vs 0.94. Differences under about 0.1 are therefore not meaningful at these sample sizes. Clearer signals: QLoRA's terse answers lower answer relevancy on held-out questions (0.69 vs 0.89), and the 4-bit deployment is the least faithful on the gold set (0.80). n 29 = a fragment answer with no checkable statement.

### 9.5 Human audit

`eval/human_audit.csv` holds 50 answers in random order, with the model hidden (key in `eval/human_audit_key.csv`): the 6 distress/crisis referrals, plus per model 6 gold answers (3 policy, 2 study, 1 literature) and 5 held-out answers (1 with the passage removed). The rubric is in `eval/human_audit_guide.md`: correct 0–2, grounded 0–2, safe 0/1, helpful 1–5. **Ratings pending** (by the project owner); `python -m scripts.summarise_audit` prints the table for this section.
