# DSGym Trace Analysis V4 — gpt-5.1 / dags & nodags

> Analysed from `evaluation_results/` conversation logs, Jun 12 2026.  
> Benchmark: **DSPredict-Hard** (full benchmark — 53 dags runs, 54 nodags runs).  
> Model: **gpt-5.1 · both modes**

---

## Quick Summary

| Run | n | Complete | Error | No-Submit | Submit Rate |
|-----|---|----------|-------|-----------|-------------|
| gpt-5.1 / dags   | 53 | 28 (53%) | 12 (23%) | 13 (25%) | **53%** |
| gpt-5.1 / nodags | 54 | 42 (78%) | 10 (19%) |  2  (4%) | **78%** |

**Nodags dramatically outperforms dags (78% vs 53%)** — the reverse of the pattern seen for Claude Sonnet 4.5 (where dags boosted submit rate from 26% to 40%). For GPT 5.1, the DAG protocol overhead is causing additional failure modes that do not appear in the no-DAG mode. This is the central finding of this run.

Compared to V2 baseline (35-competition sample):

| Run | V2 | V4 |
|-----|:---:|:---:|
| gpt-5.1 / dags | 71% | 53% |
| gpt-5.1 / nodags | 83% | 78% |

Both modes have regressed from V2, but dags has regressed more severely. The full benchmark (53–54 tasks vs 35 sampled) includes harder competitions, but the regression is also driven by new failure modes (see §1).

---

## 1 · No-Submit Failures (13 dags, 2 nodags)

### DAGS — no-submit breakdown

| Competition | Turns | Exec time | Primary cause |
|-------------|:-----:|:---------:|---------------|
| march-machine-learning-mania-2023 | 2 | 122 min | **500 alloc error** at turn 1 |
| march-machine-learning-mania-2025 | 2 | 122 min | **500 alloc error** at turn 1 |
| mens-march-mania-2022 | 2 | 122 min | **500 alloc error** at turn 1 |
| microsoft-malware-prediction | 2 | 121 min | **500 alloc error** at turn 1 |
| sp-society-camera-model-identification | 2 | 121 min | **500 alloc error** at turn 1 |
| humpback-whale-identification | 2 | 75 min | **ContextWindowExceededError** (26 occurrences, 10 python blocks) |
| understanding_cloud_organization | 7 | 75 min | **ContextWindowExceededError** (16 occurrences) |
| ieee-fraud-detection | 12 | 51 min | LGBM API incompatibility + schema AssertionError |
| m5-forecasting-uncertainty | 11 | 188 min | LGBM API TypeError, NameError (state loss after error) |
| statoil-iceberg-classifier | 6 | 56 min | Read-only filesystem (`.7z` extraction to `/data/`) |
| tensorflow-speech-recognition | 6 | 45 min | Read-only filesystem + no 7z extractor |
| google-smartphone-decimeter | 11 | 46 min | ValueError (feature mismatch), FileNotFoundError |
| inclusive-images-challenge | 1 | 61 min | Run stopped after DAG emission only (1 traj entry) |

**Sub-categories:**

| Cause | Count |
|-------|:-----:|
| Container 500 allocation error | **5** |
| ContextWindowExceededError | **2** |
| LGBM API incompatibility | **2** |
| Read-only filesystem / 7z | **2** |
| Data/code error (no timeout) | **2** |
| Unknown (1-turn stop) | **1** |

### Container 500 Allocation Errors (5 dags cases, 0 nodags)

All 5 runs hit `Server error '500 Internal Server Error' for url 'http://localhost:5000/allocate'` at turn 1 (the first code execution). The DAG was emitted (turn 0), then the first `<python>` block failed to allocate a container, and the run returned `total_turns=2` with no submission. Exec times of ~2 hours suggest the run then waited in a timeout loop rather than immediately stopping.

These 5 failures appear in dags mode only — the same 5 competitions (march-mania-2023, march-mania-2025, mens-march-mania-2022, microsoft-malware, sp-society-camera) all completed successfully in nodags mode. This suggests the dags batch was run under different infrastructure conditions (pool exhausted) from the nodags batch.

**Key difference from V3 (Claude):** In V3, 500 errors produced `zero-turn` outcomes (turns=0) with real work lost. Here GPT correctly returns turns=2 (the DAG emission turn is counted), but no code was executed and no submission was made. The infrastructure bug is the same; the accounting differs.

### ContextWindowExceededError (2 dags cases, 0 nodags)

`humpback-whale-identification` and `understanding_cloud_organization` both hit repeated `litellm.ContextWindowExceededError` mid-run (26 and 16 occurrences respectively). These are image-heavy competitions where the agent loaded large amounts of file listing output, model output, and intermediate data into context. In both cases the run reported `total_turns=2` (only the initial DAG turn was completed before context overflow began filling the queue).

**This is a dags-specific failure mode**: the DAG JSON emitted at turn 0 is ~300–600 tokens of additional context per run. Over 15 turns this accumulates ~4,500–9,000 extra tokens. On already large-context tasks (image datasets, long file listings), this overhead pushes the model over the context limit earlier. The same competitions succeeded in nodags mode.

### NODAGS — no-submit breakdown (only 2)

| Competition | Turns | Exec time | Cause |
|-------------|:-----:|:---------:|-------|
| elo-merchant-category-recommendation | 15 | 103 min | LGBM API TypeError + `cards` NameError; ran all 15 turns without submitting |
| youtube8m-2019 | 2 | 58 min | TFRecord format; agent assessed task as infeasible given environment constraints, wrote a summary but no submission |

---

## 2 · Submission Errors (12 dags, 10 nodags)

Errors are submissions that passed local format validation but were rejected by the Kaggle evaluator.

### Recurring errors (both modes)

| Competition | Error pattern | Both modes? |
|-------------|---------------|:-----------:|
| `champs-scalar-coupling` | Wrong row count (Kaggle expects 2,505,542) | ✓ |
| `data-science-bowl-2018` | Submitted IDs don't match solution IDs | ✓ |
| `liverpool-ion-switching` | Wrong `time` key values (200,000 expected) | ✓ |
| `web-traffic-time-series-forecasting` | Wrong row count (expects 8,993,906) | ✓ |
| `planttraits2024` | 5 rows submitted vs 6,545 expected | ✓ |
| `mens-machine-learning-competition-2018` | Wrong row count (expects 2,278) | ✓ |

### DAGS-only errors

| Competition | Error pattern |
|-------------|---------------|
| `gendered-pronoun-resolution` | Stage 1 submission (wrong rows); expects 12,359 rows — same V2 bug |
| `imaterialist-challenge-fashion-2018` | Wrong column names (`id`, `predicted` instead of `image_id`, `label_id`) |
| `m5-forecasting-accuracy` | CSV submitted with `scalar_coupling_constant` column — likely stale file from a previous competition |
| `recruit-restaurant-visitor-forecasting` | CSV submitted with `rle_mask` column — wrong competition's format |
| `ventilator-pressure-prediction` | CSV submitted with `Class`, `Segments` columns instead of `id`, `pressure` |
| `womens-machine-learning-competition-2019` | Wrong row count (expects 2,016 rows) |

### NODAGS-only errors

| Competition | Error pattern |
|-------------|---------------|
| `march-machine-learning-mania-2025` | Wrong row count (expects 131,407) |
| `mens-machine-learning-competition-2019` | CSV has `scalar_coupling_constant` column — cross-contamination |
| `santander-value-prediction-challenge` | CSV contains `inf` values — numeric overflow not caught before submission |
| `tensorflow-speech-recognition-challenge` | Wrong columns (`ID`, `target` instead of `fname`, `label`) |

### Cross-competition contamination pattern (NEW)

Three error cases show columns from entirely different competitions in the submitted CSV:
- `m5-forecasting-accuracy` (dags): submission had `scalar_coupling_constant` — from molecular chemistry competition
- `mens-machine-learning-competition-2019` (nodags): same, `scalar_coupling_constant`  
- `recruit-restaurant-visitor-forecasting` (dags): submission had `rle_mask` — from image segmentation competition

These are not in-run logic errors (the trajectory shows valid m5/restaurant code being written). They indicate **stale submission files** from parallel container runs — a file from a previously-run competition was present in the container's `/submission/` directory and was picked up as the current submission. The stale-file fix from V2 has not fully eliminated cross-container contamination.

---

## 3 · Per-Cell Timeouts

| Metric | dags | nodags |
|--------|:----:|:------:|
| Runs with ≥1 timeout | 15/53 (28%) | 14/54 (26%) |
| Total timeout events | 30 | 30 |
| Avg per run (all) | 0.6 | 0.6 |
| Avg per complete run | 0.6 | — |
| Avg per no-submit run | 0.2 | — |
| Avg per error run | 1.0 | — |

**GPT handles timeouts much better than Claude.** Only 1 no-submit (m5-uncertainty, dags) is primarily caused by timeout — and even there the agent revised its plan twice before the timeout killed the LightGBM run. Most runs with timeouts still submitted: timeouts are a recoverable obstacle, not a fatal one. GPT immediately pivots to simpler approaches (recent-mean baselines, logistic regression, rule-based methods) after the first timeout.

The 30 timeout events in dags runs are spread across 15 competitions. In contrast, V3 (Claude/dags, n=35) had 51 timeout events in 20/35 runs. GPT uses smaller, more incremental code cells (avg 20.5 python blocks/run vs Claude's 13.7 larger blocks) that are less likely to trigger the 300-second per-cell limit.

---

## 4 · DAG Format & Instruction Following

| Pattern | gpt-5.1 / dags | gpt-5.1 / nodags |
|---------|:---:|:---:|
| `<reasoning>` tag | 53/53 | 54/54 |
| `<python>` tag | 53/53 | 54/54 |
| `<answer>` block | **53/53** | **54/54** |
| DAG (raw JSON, reasoning) | **53/53** | 0/54 |
| DAG (raw JSON, implementation) | **53/53** | 0/54 |
| `PLAN REVISION` text | **53/53** | 0/54 |
| `Step N/M` format | **53/53** | 0/54 |
| `Execution summary` | **53/53** | 0/54 |
| Avg plan revisions (dags) | **4.5** | — |

GPT follows every format instruction at 100% compliance in both modes. All DAG-mode markers (`Step N/M`, `PLAN REVISION`, `Execution summary`) are consistently present in all 53 dags runs and absent in all 54 nodags runs — clean mode separation.

The `<answer>` block is written in every single run regardless of mode, confirming GPT's strong habit of structured self-reporting (consistent with V2 findings).

**DAG format**: GPT continues to emit raw JSON objects rather than ` ```json ``` ` fenced blocks, as noted in V2. Both DAG objects are concatenated in the first assistant turn. Functionally equivalent for analysis, but diverges from the spec.

### Plan revision behaviour

| Metric | dags / complete | dags / no-submit | dags / error |
|--------|:---:|:---:|:---:|
| Avg plan revisions | 5.0 | 3.5 | 4.6 |
| Max revisions | 12 | 8 | 7 |

Completing runs have more revisions (5.0 vs 3.5 for no-submit), the inverse of Claude's pattern in V3 (where failing runs had more revisions). For GPT, plan revision is a sign of proactive adaptation rather than distress — runs that successfully submitted revised their plans more often because they had more turns in which to do so.

---

## 5 · DAG Consistency: Reasoning → Implementation → Code

### Implementation DAG → code keyword coverage

| Outcome | Avg coverage | n |
|---------|:-----------:|:-:|
| complete | **79.1%** | 28 |
| no_submit | 64.6% | 13 |
| error | 78.7% | 12 |
| **overall** | **75.5%** | 53 |

Coverage buckets: >80%: 20/53, 50–80%: 32/53, <50%: 1/53.

Coverage predicts outcome more strongly here than in V2 (where GPT showed near-zero correlation): complete runs average 79.1% vs no-submit 64.6% (14.5 pp gap). No-submit runs with low coverage are primarily the 500-alloc and context-window cases where execution never began — they have high plan coverage by construction (implementation DAG was written but no code ran), which artificially depresses the no-submit coverage figure. Removing these 7 infrastructure failures, remaining no-submit runs average ~74% coverage, narrowing the gap to ~5 pp.

### Reasoning → Implementation DAG semantic overlap

| Outcome | Keyword overlap | Jaccard | n |
|---------|:---:|:---:|:-:|
| complete | 45.7% | 25.1% | 28 |
| no_submit | 48.5% | 25.7% | 13 |
| **overall** | **46.6%** | **25.3%** | 41+ |

R→I overlap is virtually identical to V2 GPT findings (V2: 45.9% keyword, 26.0% Jaccard). Both models consistently rewrites ~54% of node labels between reasoning and implementation, replacing abstract planning vocabulary with concrete technical steps. This stability across runs and versions suggests it is a fundamental property of GPT's planning style rather than run-specific variation.

---

## 6 · Successful Runs — what worked

### DAGS (28/53 complete)

The 28 complete dags runs span a wide range of competition types. Notable highlights:

| Competition | Score | Turns | Pub percentile |
|-------------|:-----:|:-----:|:--------------:|
| home-data-for-ml-course | 14,253.68 | 11 | 90.4% ✓ |
| mens-machine-learning-competition-2019 | 0.499 | 12 | 52.3% ✓ |
| nlp-getting-started | 0.810 | 7 | 51.2% ✓ |
| spaceship-titanic | 0.798 | 15 | 33.8% |
| digit-recognizer | 0.975 | 7 | 30.1% |
| youtube8m-2018 | 0.052 | 7 | 40.1% |

Only 3/28 complete dags runs beat the public leaderboard median (`above_median=True`): `home-data-for-ml-course`, `mens-ml-2019`, `nlp-getting-started`. The median percentile across all complete dags runs is ~16%.

### NODAGS (42/54 complete)

| Competition | Score | Turns | Pub percentile |
|-------------|:-----:|:-----:|:--------------:|
| m5-forecasting-accuracy | 0.0 | 4 | 93.6% ✓✓ |
| trec-covid-information-retrieval | 0.189 | 11 | 85.0% ✓✓ |
| spaceship-titanic | 0.801 | 8 | 59.9% ✓ |
| home-data-for-ml-course | 16,602.04 | 10 | 68.5% ✓ |
| nlp-getting-started | 0.798 | 7 | 43.1% |
| imaterialist-fashion-2018 | 0.346 | 8 | 45.5% |

5/42 nodags complete runs beat median (vs 3/28 for dags). Nodags runs are faster on average: 8.0 avg turns vs 10.8 for dags complete runs, 78.6 min vs 87.8 min. GPT completes the same work in fewer turns without the DAG overhead.

### Competitions where mode matters

| Competition | dags result | nodags result |
|-------------|:-----------:|:-------------:|
| ieee-fraud-detection | no_submit | **complete** (0.927, 28%ile) |
| humpback-whale-identification | no_submit | **complete** (0.239) |
| understanding_cloud_org | no_submit | **complete** (0.478) |
| statoil-iceberg-classifier | no_submit | **complete** (0.318) |
| microsoft-malware-prediction | no_submit | **complete** (0.500) |
| imaterialist-fashion-2018 | error | **complete** (0.346) |
| m5-forecasting-accuracy | error | **complete** (0.0, 93.6%ile) |
| elo-merchant-category | **complete** (3.923) | no_submit |
| santander-value-prediction | **complete** (1.563) | error |
| youtube8m-2019 | **complete** (0.0001) | no_submit |
| mens-ml-2019 | **complete** (0.499, 52%ile) | error |

**17 competitions where nodags wins** (dags failed, nodags completed) vs **4 where dags wins**. This 4:1 ratio is the clearest evidence that the DAG protocol is net-negative for GPT 5.1 at this benchmark scale.

---

## 7 · Key Failure Mode Summary

### DAGS

| Failure mode | Count | % of 53 |
|-------------|:-----:|:--------:|
| Container 500 alloc error | 5 | 9% |
| Cross-contamination (wrong submission) | 3 | 6% |
| Wrong row count / submission format | 9 | 17% |
| ContextWindowExceededError | 2 | 4% |
| LGBM API incompatibility | 2 | 4% |
| Read-only filesystem (7z) | 2 | 4% |
| Data/code error | 2 | 4% |
| Complete | 28 | 53% |

### NODAGS

| Failure mode | Count | % of 54 |
|-------------|:-----:|:--------:|
| Wrong row count / submission format | 8 | 15% |
| Cross-contamination (wrong submission) | 2 | 4% |
| Numeric overflow (inf values) | 1 | 2% |
| Turn limit (LGBM loop) | 1 | 2% |
| TFRecord / infeasible format | 1 | 2% |
| Complete | 42 | 78% |

---

## 8 · Behavioural Differences: DAGS vs NODAGS (GPT 5.1)

| Metric | dags | nodags |
|--------|:----:|:------:|
| n | 53 | 54 |
| Submit rate | 53% | **78%** |
| Avg turns (complete) | 10.8 | **8.0** |
| Avg exec time (complete) | 87.8 min | **78.6 min** |
| Avg python blocks | 20.5 | **14.4** |
| Avg plan revisions | 4.5 | 0 |
| Context window errors | 3 | **0** |
| 500 alloc errors | 5 | **0** |
| Above-median complete | 3/28 (11%) | 5/42 (12%) |

The DAG mode does not improve GPT's quality (above-median rate is identical at ~11–12%) and significantly reduces its completion rate through infrastructure and context overhead. GPT nodags produces:

- **Fewer code blocks** (14.4 vs 20.5): more focused, monolithic cell-by-cell progression
- **Fewer turns on successful runs** (8.0 vs 10.8): faster to a submission
- **Zero context window overflows**: eliminating the DAG JSON from context keeps the conversation compact enough for image-heavy and large-dataset tasks

The most likely explanation: GPT 5.1's base instruction-following and planning is already strong enough that explicit DAG scaffolding adds burden without adding reasoning quality. The time and context spent generating, revising, and narrating the plan could be spent executing code.

---

## 9 · Recurring Environment Blockers

These are unchanged from V2/V3 — same root causes, same competitions:

| Blocker | Affected competitions | Status |
|---------|-----------------------|--------|
| `.7z` extraction (no binary or read-only write target) | `statoil-iceberg-classifier`, `tensorflow-speech-recognition` | Blocks dags, resolved in nodags (`statoil` completed via fallback) |
| Read-only `/data/` filesystem | `data-science-bowl-2018`, `statoil`, `tensorflow-speech`, `imaterialist-furniture` | Agent tries to write intermediates to `/data/` |
| LGBM API incompatibility (`verbose`, `early_stopping_rounds` kwargs) | `ieee-fraud-detection`, `m5-uncertainty`, `elo-merchant`, `santander-value` | Affects ~15 runs; most recover by switching to sklearn wrapper |
| Container pool exhaustion (`500 /allocate`) | 5 dags competitions | Infrastructure; appears isolated to the dags batch |

---

## 10 · Recommendations

### Immediate (infra)

1. **Stale submission file isolation**: Three error cases show columns from unrelated competitions in submitted CSVs (scalar_coupling_constant in m5, rle_mask in recruit-restaurant). The container's `/submission/` directory must be wiped between tasks. File-level isolation (rename output files to task-specific names before submission) does not prevent an old file from being present alongside the new one if the evaluator picks by path rather than name.

2. **Container pool sizing for dags batch**: The 5 container 500 errors in dags mode were absent in nodags. If both modes were run in parallel, the dags runs likely arrived at the allocator when all containers were busy. Log allocator queue depth at job start and fail fast with a re-queue rather than waiting 2 hours.

3. **Context budget awareness**: The DAG JSON adds ~500+ tokens per run. For image, video, or genomics competitions (where turn content is large), this overhead causes context window errors. Consider compressing or summarising the DAG after initial emission, or omitting it for tasks with large-content environments.

### Prompt-level

4. **LGBM API guard**: Add explicit fallback in the prompt: "If `LGBMClassifier.fit()` raises a `TypeError` about `verbose` or `early_stopping_rounds`, retry using `lgb.train()` without those arguments or switch to the sklearn wrapper." This affects ~15 runs per batch; most recover but waste 1–2 turns.

5. **Write-path discipline**: Reinstate prompt instruction: "Never write to `/data/`. Use `/tmp/` for intermediate work and `/submission/submission.csv` for final output." Five runs per mode hit read-only filesystem errors writing to `/data/`.

6. **Row count validation**: Inject expected submission row count and column schema into the system prompt. Six submissions were rejected for wrong row count (a consistent V2/V3/V4 failure). `gendered-pronoun-resolution` (12,359 rows expected) continues to hit Stage 1 vs Stage 2 confusion — the correct row count would resolve this immediately.

7. **inf/NaN submission guard**: `santander-value-prediction` (nodags) was rejected for `inf` values. Add pre-submission check: `assert not df.isin([np.inf, -np.inf]).any().any() and not df.isnull().any().any()`.

### For DAG protocol

8. **Reconsider DAG mode for GPT 5.1**: The data strongly suggests that DAG scaffolding is net-negative for this model (53% vs 78% completion, no quality uplift). GPT 5.1 already self-structures its work with `Step N/M` headers, `PLAN REVISION` markers, and `Execution summary` blocks even in nodags mode (these appeared 0% in nodags by design — they are dags-only — but the model's reasoning coherence is clearly maintained without them). A lighter-weight alternative — requiring only a single bullet-point plan at turn 0 rather than a full reasoning+implementation DAG — may preserve planning benefits without the context and infrastructure overhead.

9. **Bind DAG node IDs to execution**: Coverage analysis shows 64.6% implementation DAG coverage in no-submit runs vs 79.1% in complete runs, but the gap is partially inflated by infrastructure failures. To make the DAG more actionable, require agents to quote the current node ID in each `<reasoning>` block (e.g., `executing node i3`). This would make plan-execution coupling explicit and allow per-node status tracking.

---

## 11 · Nodags Deep Dive: Library Usage, Behaviour & Hypotheses on DAG Uplift

### Library Usage

GPT 5.1 nodags defaults heavily to a pragmatic tabular stack. Almost no deep learning was attempted:

| Library | Used in | % of 54 | Complete | Error | No-Submit |
|---------|:-------:|:-------:|:--------:|:-----:|:---------:|
| sklearn | 41 | 75% | 32 | 8 | 1 |
| lightgbm | 30 | 55% | 22 | 7 | 1 |
| xgboost | 27 | 50% | 19 | 7 | 1 |
| tensorflow/keras | 6 | 11% | 5 | 1 | 0 |
| scipy | 6 | 11% | 5 | 1 | 0 |
| PIL/skimage | 5 | 9% | 4 | 1 | 0 |
| transformers | 4 | 7% | 4 | 0 | 0 |
| catboost | 2 | 3% | 2 | 0 | 0 |
| **pytorch** | **1** | **2%** | 1 | 0 | 0 |
| **cv2** | **0** | **0%** | — | — | — |

**PyTorch**: used in exactly **1/54** nodags runs — `tgs-salt-identification-challenge`, where a lightweight U-Net was appropriate for image segmentation. It completed successfully. GPT treats neural nets as a genuine last resort, not a first instinct.

**cv2 / OpenCV**: **zero actual usages** in nodags. The single regex hit (`open-problems-single-cell-perturbations`) is a false positive — the string `CV2` appears as an abbreviation for "cross-validation fold 2", not an OpenCV import. When tasks involve image data, GPT uses PIL, skimage, or rule-based methods instead. It never even attempts to import cv2.

**Primary strategy in complete nodags runs (42):**
- sklearn only (RandomForest, GradientBoosting, LinearRegression): **14 runs**
- LightGBM-led: **12 runs**
- XGBoost/other: **10 runs**
- TF/Keras: **2 runs**
- PyTorch: **1 run**

The model systematically prefers fast, stable tabular methods. This is not prompt-induced — nodags has no explicit library preference instruction. It appears to be a native GPT 5.1 disposition: start with something that will finish within the turn budget and improve from there.

**Comparison to dags mode:**

| Library | dags % | nodags % | Δ |
|---------|:------:|:--------:|:-:|
| sklearn | 86% | 75% | −11 |
| lightgbm | 64% | 55% | −9 |
| pytorch | 11% | 2% | −9 |
| PIL/skimage | 15% | 9% | −6 |

Dags mode shows higher library breadth — especially more PyTorch and PIL/skimage attempts. But those extra attempts mostly correspond to no-submit or error outcomes (3/6 pytorch dags runs fail). The nodags agent is more conservative and that conservatism pays off in completion rate.

---

### Why There Is No Quality Uplift from DAGs

The head-to-head score comparison on 24 competitions completed by both modes tells the clearest story:

| Outcome | Count |
|---------|:-----:|
| Nodags better (>2 pp higher percentile) | 10 |
| Dags better (>2 pp higher percentile) | 9 |
| Tie (within 2 pp) | 5 |
| Avg percentile diff (dags − nodags) | **−3.6 pp** |

This is a statistical coin flip with a slight lean toward nodags. Three complementary hypotheses explain why the DAG adds no quality signal:

#### Hypothesis 1: The DAG consumes 41% of the turn budget

In complete dags runs (avg 10.8 turns): **1 turn is the DAG emission** + **3.4 turns are plan revisions** = **4.4 turns spent on planning**. That leaves only **6.4 productive code turns** per run. Nodags complete runs average **8.0 code turns** — 25% more execution time within the same turn limit. Fewer iterations means less model tuning, fewer CV evaluations, and less time to recover from errors.

#### Hypothesis 2: The DAG locks the agent into a turn-0 strategy

The most striking example is `trec-covid-information-retrieval`:

| Mode | Score | Percentile | Turns | Approach |
|------|:-----:|:----------:|:-----:|----------|
| dags | 0.0 | 27.5% | 14 | TF-IDF only |
| nodags | 0.189 | **85.0%** | 11 | **BM25** + TF-IDF |

The dags agent committed to TF-IDF in its implementation DAG at turn 0 and followed it for 14 turns. Every plan revision built on TF-IDF. The nodags agent explored freely, identified BM25 as appropriate for information retrieval, and reached the 85th percentile in 3 fewer turns.

`store-sales-time-series-forecasting` shows the reverse overhead cost: dags used **35 code cells** over 15 turns exploring LightGBM + XGBoost combinations (10.8th percentile). Nodags used **13 cells** over 8 turns with a focused LightGBM baseline (35.7th percentile). The dags agent over-engineered in response to plan revisions; nodags shipped a simpler solution faster.

The DAG functions as a commitment device. When it locks the agent onto the right approach, there is no harm. But when the right approach only becomes apparent mid-run (after seeing data statistics, CV scores, or environment constraints), the plan revision overhead costs turns and the initial commitment biases subsequent decisions.

#### Hypothesis 3: GPT 5.1 already self-plans natively — the DAG is redundant

**47/54 nodags runs** contain numbered steps, structured phases, or explicit section headers in free-form reasoning — without being prompted to do so. First-turn reasoning in complete nodags runs averages **4,431 characters** of structured planning, immediately followed by **7,086 characters of code** in the same turn. The agents already produce a coherent multi-step plan; they simply express it in prose rather than JSON.

Reasoning block depth differs between modes:

| Mode | Total reasoning blocks | Avg chars / block |
|------|:----------------------:|:-----------------:|
| dags | 1,381 | 619 |
| nodags | 980 | **843** |

Nodags agents write fewer but deeper reasoning blocks. The planning capacity is identical; it is just distributed differently. The DAG doesn't create additional planning — it fragments and formalises planning that was already happening, at the cost of context budget and turn overhead.

Nodags also adapts without formal revision markers: complete nodags runs average **2.0 pivot/fallback events** per run (language like "fallback", "switch to", "alternative approach" in reasoning), demonstrating that mid-run adaptation doesn't require the explicit `PLAN REVISION` tag.

---

### The Score Floor: Why Both Modes Land in the Bottom Third

Even the better nodags mode achieves only **median 18.8th percentile** across completed runs, with **5/42 above the leaderboard median**. This is not a planning problem — it is a compute constraint problem:

- The 300-second per-cell timeout prevents training large gradient boosting models or running more than a few epochs of neural network training on competition-scale datasets
- The agent compensates by using smaller data samples, simpler models, or naive baselines (recent-mean forecasts, rule-based predictions) that finish within the time limit but sacrifice accuracy
- Competitors on the real leaderboard had days or weeks of GPU time; the agent gets ~90 minutes per task with a hard per-cell ceiling

This ceiling is environment-level, not agent-level. Better prompting or planning protocols will not move the percentile floor until the execution environment allows longer-running cells.
