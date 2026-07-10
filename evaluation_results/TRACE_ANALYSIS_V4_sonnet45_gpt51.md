# DSGym Trace Analysis V4 — Claude Sonnet 4.5 & GPT-5.1 / DSPredict-Sampled

> Analysed from `evaluation_results/dspredict-sampled_litellm_*_results.json` conversation logs.  
> Benchmark: **DSPredict-Sampled** (35 Kaggle competitions).  
> Models: **claude-sonnet-4-5** · **gpt-5.1** · Modes: **dags** · **nodags**  
> Note: claude-sonnet-4-5 dags run uses 900-second per-cell timeout; all other runs use 900s as well based on run times observed.

---

## Quick Summary

| Run | n | Complete | Error | No-Submit | Submit Rate |
|-----|---|----------|-------|-----------|-------------|
| claude-sonnet-4-5 / dags   | 35 | 17 | 4  | 14 | **49%** |
| claude-sonnet-4-5 / nodags | 35 | 11 | 4  | 20 | **31%** |
| gpt-5.1 / dags             | 35 | 25 | 3  |  7 | **71%** |
| gpt-5.1 / nodags           | 35 | 29 | 3  |  3 | **83%** |

GPT-5.1 maintains a **1.7–2.7× higher submission rate** than Claude Sonnet 4.5. The DAG protocol helps Claude (+18pp) more than GPT (+12pp, whose nodags baseline is already very high). Claude's main failure mode is no-submit from turn-limit exhaustion; GPT's is almost exclusively Kaggle-side format rejection.

---

## Per-Competition Status

| Competition | c45/dags | c45/nodags | gpt/dags | gpt/nodags |
|-------------|:---:|:---:|:---:|:---:|
| LANL-Earthquake-Prediction | ✓ 1.782 | — | ✓ 1.651 | ✓ 1.668 |
| ashrae-energy-prediction | ✓ 1.407 | — | — | — |
| career-con-2019 | ✓ 0.128 | ✓ 0.128 | ✓ 0.128 | ✓ 0.131 |
| digit-recognizer | ✓ 0.969 | ✓ 0.967 | ✓ 0.973 | ✓ 0.973 |
| elo-merchant-category | — | — | ✓ 3.884 | ✓ 3.926 |
| **gendered-pronoun-resolution** | **ERR** | **ERR** | **ERR** | **ERR** |
| geolifeclef-2024 | — | — | ✓ 0.243 | ✓ 0.118 |
| home-credit-default-risk | ✓ 0.778 | — | ✓ 0.749 | ✓ 0.748 |
| humpback-whale-identification | ✓ 0.275 | — | ✓ 0.156 | ✓ 0.285 |
| ieee-fraud-detection | — | — | — | ✓ 0.922 |
| imaterialist-fashion-2018 | ✓ 0.351 | ✓ 0.375 | ✓ 0.346 | ✓ 0.351 |
| imaterialist-furniture-2018 | ✓ 0.955 | — | ✓ 0.985 | ✓ 0.993 |
| liverpool-ion-switching | — | **ERR** | — | **ERR** |
| m5-forecasting-accuracy | — | — | ✓ 5.446 | ✓ 0.838 |
| mens-machine-learning-2018 | **ERR** | **ERR** | — | **ERR** |
| mens-march-mania-2022 | **ERR** | **ERR** | — | ✓ 0.660 |
| microsoft-malware-prediction | — | — | — | ✓ 0.620 |
| nlp-getting-started | ✓ 0.795 | ✓ 0.794 | ✓ 0.803 | ✓ 0.792 |
| novozymes-enzyme-stability | ✓ 0.046 | — | ✓ 0.029 | ✓ 0.026 |
| predict-ai-model-runtime | — | ✓ 0.278 | ✓ 0.241 | ✓ −2.894 |
| rsna-pneumonia-detection | ✓ 0.000 | ✓ 0.000 | ✓ 0.000 | ✓ 0.008 |
| santander-customer-transaction | ✓ 0.895 | ✓ 0.898 | ✓ 0.895 | ✓ 0.891 |
| santander-value-prediction | ✓ 1.507 | — | ✓ 1.511 | ✓ 1.514 |
| siim-acr-pneumothorax | ✓ 0.484 | — | ✓ 0.600 | ✓ 0.600 |
| sp-society-camera-id | — | — | ✓ 0.146 | — |
| stanford-covid-vaccine | ✓ 0.363 | — | ✓ 1.111 | ✓ 0.401 |
| store-sales-time-series | — | — | ✓ 1.078 | ✓ 2.248 |
| talkingdata-adtracking | — | ✓ 0.607 | ✓ 0.905 | ✓ 0.834 |
| trec-covid-information | ✓ 0.188 | — | ✓ 0.217 | ✓ 0.000 |
| understanding-cloud-org | **ERR** | ✓ 0.478 | ✓ 0.433 | ✓ 0.478 |
| web-traffic-time-series | — | ✓ 46.568 | **ERR** | ✓ 41.580 |
| womens-ml-competition-2019 | — | — | **ERR** | ✓ 0.430 |
| youtube8m-2018 | — | ✓ 0.052 | ✓ 0.031 | ✓ 0.020 |
| youtube8m-2019 | ✓ 0.000 | — | ✓ 0.000 | ✓ 0.000 |

---

## 1 · Submission Errors (SUBMISSIONSTATUS.ERROR) — 14 total

All errors involve submissions that **pass local format validation** but are rejected by Kaggle's server-side evaluator. Every single error case contains evidence of multi-stage confusion or structural format mismatch.

### Error map

| Competition | c45/dags | c45/nodags | gpt/dags | gpt/nodags | Root cause |
|-------------|:---:|:---:|:---:|:---:|-----------|
| `gendered-pronoun-resolution` | ERR | ERR | ERR | ERR | Stage-1 sample used (2,000 rows) instead of Stage-2 (12,359 rows) |
| `mens-machine-learning-competition-2018` | ERR | ERR | — | ERR | Stage-1 matchup scope (2014–2017 only) vs full required seasons |
| `mens-march-mania-2022` | ERR | ERR | — | ✓ | Stage-1 scope (11,390 rows) — GPT nodags correctly uses Stage-2 |
| `liverpool-ion-switching` | — | ERR | — | ERR | Wrong column format or out-of-range `open_channels` predictions |
| `womens-machine-learning-competition-2019` | — | — | ERR | — | Stage-1 submission (wrong scope) |
| `web-traffic-time-series-forecasting` | — | — | ERR | — | Wrong page-key IDs or row scope |
| `understanding_cloud_organization` | ERR | — | — | — | Empty-mask baseline: wrong `Image_Label` column format |

### Root cause analysis

**Stage-1 vs Stage-2 confusion** drives 11 of 14 errors (79%). In every affected competition, both `sample_submission_stage_1.*` and `sample_submission_stage_2.*` files exist in the data directory. Agents load whichever they find first — typically Stage-1, which is alphabetically earlier and substantially smaller. The local validation passes because the model builds against the Stage-1 template. Kaggle rejects because the Stage-2 submission is expected at evaluation time.

Crucially, `gendered-pronoun-resolution` hits **all four runs** with this bug. Even GPT dags, which finishes in only 5 turns, submits Stage-1. The error is deterministic and model-agnostic: when two sample files exist, agents pick the wrong one.

**`liverpool-ion-switching`** is a distinct error. Both Claude nodags and GPT nodags produce format-validated files, but Kaggle rejects them. The agent reasoning suggests this is a value-range issue — `open_channels` is an integer 0–10, but training a continuous regressor and rounding can produce values outside this range. GPT nodags explicitly acknowledges: *"all attempts to train LightGBM on even modestly sized samples hit the 900s timeout — created a baseline submission."* The placeholder zero-prediction passes local validation but may have disallowed values.

**`understanding_cloud_organization`** (Claude dags only): The competition requires run-length encoded segmentation masks in an `EncodedPixels` column. Claude submits an empty-mask baseline with `Image_Label` and `Image` columns (7,960 rows) instead of the required `Image_Label` / `EncodedPixels` format. The model validates its own column schema against its own template and misses the mismatch.

**Hypothesis**: Agents treat local format validation (column names present, row count matches sample) as sufficient signal that the submission is correct. They do not independently verify which stage the task expects, nor check value distributions against domain constraints.

---

## 2 · No-Submit Failure Modes

### Error breakdown

| Error type | c45/dags | c45/nodags | gpt/dags | gpt/nodags |
|-----------|:---:|:---:|:---:|:---:|
| **TimeoutError** (turn-limit exhaustion) | 8 | 7 | 4 | 2 |
| **ImportError** (missing package) | 2 | 5 | 0 | 1 |
| **ContextWindowExceeded** | 2 | 1† | 0 | 0 |
| **OtherTraceback** (non-timeout) | 2 | 4 | 0 | 0 |
| **NoTraceback** (ran out of turns, no crash) | 0 | 3 | 0 | 0 |
| **MemoryError** | 0 | 0 | 1 | 0 |
| **FileNotFoundError** | 0 | 0 | 1 | 0 |
| **ValueError** | 0 | 0 | 1 | 0 |

† Classified as "NoTraceback" in raw output but the error message is `request_too_large` / `AnthropicException`.

---

### 2a · TimeoutError

**Timeouts are the dominant failure mode for both models but manifest differently.**

**Claude (both modes)**: All 15 timeout cases hit the **turn limit (t=15)**, not the per-cell execution timeout. Average run time at turn limit: 2,654s (dags), 2,121s (nodags). The model is still making forward progress at turn 15 — training is ongoing, feature engineering is progressing — but there are no remaining turns to save a submission. Claude treats every task as requiring thorough feature engineering before saving, even when the deadline is visible.

Example competitions always hitting turn limit for Claude:
- `ashrae-energy-prediction`: large dataset, model training in progress at turn 15
- `talkingdata-adtracking-fraud-detection` (dags): 184M-row dataset, re-loading data from scratch each turn
- `elo-merchant-category-recommendation`: large transaction dataset, still encoding features at turn 15
- `m5-forecasting-accuracy`: hierarchical time-series, feature engineering phase incomplete
- `store-sales-time-series-forecasting`: oil/holiday merges complete, LightGBM training in progress

**GPT (both modes)**: Only 6 timeout cases across both modes. GPT dags averages 10.5 turns when timing out — it hits per-cell timeout within 10–13 turns and recognises it cannot complete the task, whereas Claude keeps trying different approaches until turns run dry. GPT nodags averages 9 turns on its 2 timeouts.

**Hypothesis — why Claude timeouts > GPT timeouts**: GPT writes significantly more incremental code blocks per run (20.3 avg in dags vs Claude's 13.3) despite completing in fewer turns. This means GPT chunks work into smaller cells, each of which is less likely to exhaust the per-cell limit. When GPT does hit a timeout, it immediately revises the plan and falls back to a simpler model. Claude writes larger monolithic cells (complex training pipelines in a single block) and persists with the same approach rather than cutting scope.

---

### 2b · ImportError

All ImportErrors involve packages that are genuinely absent from the execution environment:

| Package | Affected runs | Competition | Notes |
|---------|:---:|-----------|-------|
| `py7zr` / `7z` | c45/dags, c45/nodags, gpt/nodags | `statoil-iceberg-classifier` | `.7z` archive needs decompressor — environment blocker |
| `tensorflow` | c45/dags, c45/nodags | CV/video tasks | claude attempts DL first, GPT falls back to sklearn |
| `torch` | c45/nodags | Image tasks | same pattern |
| `pydicom` | c45/nodags | `rsna-pneumonia-detection` | DICOM file reader — not installed |

**Pattern**: `py7zr` / `statoil` is an environment-level blocker that no agent can solve — it affects all models. All other ImportErrors are Claude-only (nodags) because Claude aggressively attempts deep-learning libraries before falling back. GPT's library preference order is effectively `sklearn → lightgbm → xgboost`, making it less susceptible to missing DL packages.

**Hypothesis**: Claude in nodags mode (no structured plan) is more likely to attempt a "best" solution immediately (transformer/CNN) rather than a conservative baseline. Without the DAG forcing a pragmatic implementation plan, Claude's ambition first hits the missing-library wall.

---

### 2c · ContextWindowExceeded

Unique to Claude (Anthropic 200k token limit). GPT has no such failures.

| Competition | Mode | Turn | Tokens | What happened |
|------------|------|:---:|:---:|---------------|
| `geolifeclef-2024` | dags | 1 | 209,844 | Geospatial dataset exploration (55k-row PA data) produced massive stdout at turn 1, filling context before any training |
| `trec-covid-information-retrieval` | dags | 10 | 212,584 | 40-topic TF-IDF ranking output accumulated across 10 turns |
| `geolifeclef-2024` | nodags | 12–15 | >200k | Multiple turns of large-output cells accumulated until request too large |

Both competitions produce large outputs per cell: geospatial data with tens of thousands of survey rows, and IR tasks with large similarity matrices across 128k documents. With 900s per cell, agents can now execute cells that would previously time out — but this creates far more stdout that accumulates in context.

**Hypothesis**: The longer timeout directly introduced this failure mode. With 300s cells, geospatial exploration was cut before it could fill 200k tokens; with 900s, the full exploration completes and floods the context. A stdout cap (e.g., 50k characters per cell) would eliminate these cases without limiting useful computation.

---

### 2d · OtherTraceback (non-timeout, non-import)

Exclusively Claude nodags (4 cases):

| Competition | Error | What happened |
|------------|-------|---------------|
| `stanford-covid-vaccine` | `JSONDecodeError: Extra data` | JSON data is newline-delimited (JSONL format); agent tried `json.load()` on the full file instead of line-by-line parsing; never recovered |
| `ashrae-energy-prediction` | `ChainedAssignmentError` + `TypeError: 'Dataset' object is not iterable` | LightGBM Dataset object passed to pandas operation; persistent state confusion between turns |
| `store-sales-time-series-forecasting` | Merge shape mismatch | Duplicate key rows in training data caused feature explosion; model never stabilised |
| `trec-covid-information-retrieval` | Submission format mismatch (column/shape) | TF-IDF ranking produced 400 rows but sample format had different structure; agent ran out of turns fixing it |

**Hypothesis**: These are all **stateful confusion errors** — the agent loses track of what variables exist across turns, or makes incorrect assumptions about data format. Claude nodags shows 2× more of these than Claude dags, likely because the DAG plan provides a reference for what state should exist at each step. Without the plan, Claude re-derives data structures from scratch each turn, introducing inconsistency.

---

### 2e · NoTraceback (ran out of turns, no crash)

Three cases in Claude nodags — agent had correct predictions in memory but ran out of turns before writing the submission file:

| Competition | Time | What happened |
|------------|:---:|---------------|
| `santander-value-prediction-challenge` | 2326s | Ensemble trained (LGBM ×2, RMSLE ~0.94), predictions generated, but at turn 15 still running format-check loops instead of saving |
| `home-credit-default-risk` | 2041s | Feature matrix ready (307k×271 rows), still processing missing values at turn 15 |
| `mens-march-mania-2022` | 415s | Model trained (5-fold CV log loss 0.43), but turn 15 spent parsing submission ID format rather than writing the file |

All three had **working predictions but no `to_csv` call before the turn limit**. The agents prioritise correctness checks over actually saving. In the santander case, the agent was still running `print(f"Any NaN values: {sub.isnull().any().any()}")` at turn 15 instead of executing the already-formed submission save.

---

### 2f · GPT singleton errors

| Competition | Mode | Error | Root cause |
|------------|------|-------|------------|
| `ieee-fraud-detection` | dags | MemoryError | Turn 1: attempted to load the full 590k-row transaction dataset into memory with all features; OOM immediately |
| `sp-society-camera-model-identification` | dags | ValueError: "Unexpected band size" | Turn 1: tif image files have non-standard band count; `PIL.Image` raises ValueError before any useful code ran |
| `mens-march-mania-2022` | dags | FileNotFoundError | Looked for `sample_submission.csv` (generic name); actual file is `MSampleSubmissionStage1.csv` — agent never found it and ran 15 turns without a submission template |

All three are **turn-1 or very-early failures** that GPT does not recover from. The MemoryError and ValueError are environment-level blockers. The FileNotFoundError is an agent error — GPT should have listed directory contents before assuming filenames.

---

## 3 · Behavioural Metrics

| Metric | c45/dags | c45/nodags | gpt/dags | gpt/nodags |
|--------|:---:|:---:|:---:|:---:|
| Avg turns (all) | 12.5 | 14.5 | 9.3 | 8.2 |
| Avg turns (complete) | 11.2 | 14.5 | 9.6 | 8.1 |
| Avg turns (no-submit) | 13.6 | 14.3 | 8.4 | 7.7 |
| Avg time (all, s) | 1,712 | 1,376 | 1,201 | 867 |
| Avg time (no-submit, s) | 2,034 | 1,447 | 1,280 | 1,109 |
| Avg code blocks (all) | 13.3 | 17.1 | 20.3 | 15.3 |
| Avg code blocks (complete) | 11.1 | 16.4 | 20.3 | 15.0 |
| Avg code blocks (no-submit) | 15.3 | 17.5 | 14.3 | 15.7 |
| Top libraries | sklearn, lgbm, scipy, tf, keras | sklearn, lgbm, xgb, scipy, torch | sklearn, lgbm, xgb, scipy, tf | sklearn, lgbm, xgb, tf, scipy |

### Key observations

**Turns vs code blocks divergence**: GPT completes in ~9 turns but writes ~20 code blocks per run — roughly 2.2 code blocks per turn. Claude dags uses ~12 turns with 13 code blocks — approximately 1 block per turn. This confirms GPT writes many small, incremental cells while Claude writes fewer but larger cells. For GPT, no-submit runs have *fewer* code blocks (14.3 vs 20.3 for complete), suggesting failed runs collapse early. For Claude, no-submit runs have *more* code blocks (15.3 vs 11.1 for complete), confirming Claude keeps retrying failed approaches rather than cutting scope.

**Claude nodags uses more turns than Claude dags** despite submitting less (14.5 avg vs 12.5). Without a DAG plan, Claude exhausts all available turns whether it succeeds or fails — there is no plan signal to tell it when it's done.

**GPT is faster even when failing**: GPT no-submit runs average 8.4 turns and 1,280s (dags). Claude no-submit runs average 13.6 turns and 2,034s. GPT gives up or hits hard limits sooner; Claude persists much longer without result.

---

## 4 · Model-Specific Error Hypotheses

### Claude Sonnet 4.5

**Hypothesis 1 — Completeness bias causes timeout saturation**  
Claude treats each task as requiring exhaustive feature engineering before saving a submission. Even when a deadline (turn limit) is visible in context, it continues building features rather than writing a conservative placeholder. The `santander-value` and `home-credit` NoTraceback cases are the clearest evidence: predictions existed in memory at turn 14 but were never saved. Root cause: Claude optimises for solution quality over solution delivery.

**Hypothesis 2 — Stateful confusion under nodags**  
Without a DAG plan, Claude loses track of what variables exist across turns. This manifests as `TypeError: 'Dataset' object is not iterable` (passing a LightGBM dataset to a pandas operation), `JSONDecodeError` (misidentifying data format), and repeated re-reading of the sample submission file in the last turns. The DAG protocol partially mitigates this — Claude dags has 2 OtherTraceback cases vs 4 for nodags.

**Hypothesis 3 — Context accumulation from verbose cells**  
With 900s per cell, Claude's exploration cells (listing all columns, printing full DataFrames, verbose training logs) can produce 20–50k tokens of stdout in a single turn. After 10 turns of such cells, the context exceeds 200k tokens. GPT avoids this either because it writes more compact print statements or because its context limit is higher.

**Hypothesis 4 — Ambition-first library selection in nodags**  
Claude nodags attempts DL solutions (tensorflow, torch, keras) before sklearn. This is reasonable ambition but burns turns on ImportErrors when packages are unavailable. Claude dags avoids this more often — the DAG plan forces it to reason about which library to use before writing code, and it chooses sklearn/lightgbm for the initial implementation node.

### GPT-5.1

**Hypothesis 5 — Stage-1/Stage-2 disambiguation failure on all models**  
`gendered-pronoun-resolution` produces errors across all 4 runs including both GPT modes. GPT dags gets there in only 5 turns — the fastest of any run — and immediately uses `sample_submission_stage_1.csv` without checking whether stage-2 is the required format. This is not a GPT-specific failure but it is notable that even GPT's DAG planning doesn't prevent it. Both models see two sample submission files and anchor on the first/smaller one.

**Hypothesis 6 — Early hard failures on exotic data types**  
GPT's 3 singleton no-submit cases (MemoryError, ValueError, FileNotFoundError) all occur at turn 1 or within 2 turns. GPT does not explore the data directory before assuming file names (`sample_submission.csv` vs `MSampleSubmissionStage1.csv`), and it does not check image band counts before loading. These are turn-0 planning failures. GPT's DAG encourages a "load data → process → train" linear sequence, but doesn't include "validate environment" as a first node.

**Hypothesis 7 — Incremental cell pattern prevents timeout saturation**  
GPT writes 20+ code blocks per successful run, averaging ~2 code blocks per turn. These are small: a single merge, a single model fit, a single evaluation. Each cell fits within the 900s timeout easily. Claude's larger cells (complex training pipelines) occasionally hit cell-level timeouts as well as turn-level limits. GPT's habit of chunking into small steps also means it accumulates fewer tokens per turn, avoiding context window issues.

---

## 5 · Structural Pattern: Who Fails Where

### Competitions where both models fail (all 4 runs)
Only `gendered-pronoun-resolution` achieves 0/4 submissions, and all 4 are Kaggle errors (not crashes). This is a pure stage-confusion problem — both models always produce a file, but it's always Stage-1.

### Competitions where Claude fails but GPT succeeds (3+ runs)
- `elo-merchant-category-recommendation`: Claude dags/nodags — turn timeout on large transaction data; GPT completes both
- `geolifeclef-2024`: Claude — ContextWindowExceeded; GPT completes both
- `m5-forecasting-accuracy`: Claude — timeout; GPT completes both  
- `microsoft-malware-prediction`: Claude — timeout; GPT nodags completes
- `store-sales-time-series-forecasting`: Claude — OtherTraceback (store-sales nodags); GPT completes both
- `ieee-fraud-detection`: Claude — timeout; GPT nodags completes (GPT dags MemoryError)

All Claude-only failures are **dataset-scale problems**: large datasets, complex time-series structures, or memory-intensive operations. GPT's more incremental approach handles these better.

### Competitions where GPT fails but Claude succeeds
- `understanding_cloud_organization`: Claude nodags ✓ (0.478), but Claude dags ERR; GPT dags ✓, GPT nodags ✓ — mixed
- `web-traffic-time-series-forecasting`: GPT dags ERR, all others ✓ or Claude —  
- `stanford-covid-vaccine`: Claude dags ✓ (0.363), GPT dags ✓ but worse (1.111 vs 0.363 — lower is better for MAE)

GPT-specific failures are rare. The `web-traffic` GPT dags error is a stage/key-ID confusion similar to gendered-pronoun. The `sp-society` ValueError is environment-specific.

---

## 6 · Public vs Private Percentile Gap — Claude Sonnet 4.5 / DAGs

### Summary numbers

| Run | Avg pub pct | Avg priv pct | Δ (priv − pub) | Median pub | Median priv | Median Δ |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| claude-4-5 / dags | 23.8 | 28.6 | **+4.74** | 20.4 | 29.7 | **+9.34** |
| claude-4-5 / nodags | 26.4 | 27.5 | +1.11 | 21.0 | 24.6 | +3.62 |
| gpt-5.1 / dags | 24.2 | 24.7 | +0.51 | 23.9 | 21.1 | −2.76 |
| gpt-5.1 / nodags | 20.6 | 19.7 | −0.96 | 21.1 | 14.5 | −6.61 |

Claude Sonnet 4.5 with DAGs shows the largest public-to-private improvement (+4.74pp avg, +9.34pp median). The other three runs are flat or slightly negative. This section explains why.

### Three competitions drive the entire gap

Excluding `youtube8m-2019`, `LANL-Earthquake-Prediction`, and `novozymes-enzyme-stability-prediction`, the pub/priv delta for claude-4-5/dags collapses to **+0.39pp** — statistically indistinguishable from the other three runs. All three are concentrated in claude-dags specifically because claude-dags completed all three, claude-nodags completed none of them (both LANL and novozymes timed out at turn 15 without submitting), and GPT's larger submission pool dilutes their weight.

| Competition | c45/dags Δ | gpt/dags Δ | gpt/nodags Δ | Mechanism |
|-------------|:---:|:---:|:---:|-----------|
| youtube8m-2019 | **+24.6pp** | −1.6pp | −1.6pp | Scoring artifact — small competition, degenerate metric |
| LANL-Earthquake-Prediction | **+23.3pp** | +11.2pp | +25.5pp | Genuine private LB shake-up |
| novozymes-enzyme-stability-prediction | **+23.1pp** | +6.6pp | −13.9pp | Genuine private LB shake-up (GPT-nodags reverses) |

### Competition-by-competition diagnosis

**`youtube8m-2019` — scoring artifact, not real performance (+24.6pp)**

The agent couldn't parse TFRecord video files (no `tensorflow` in the environment), so it submitted a purely random baseline that scored **exactly 0.0** on the public leaderboard. This is a 284-team competition with a degenerate score distribution (public median ≈ 0.0001), where the difference between a score of 0.0 and 0.0001 spans roughly 175 positions in the public ranking:

- Claude-dags: score = 0.0000, pub_rank = 276/284 → **pub_pct = 1.6%** (beats only 8 teams)
- GPT-dags/nodags: score = 0.0001, pub_rank ≈ 75/284 → **pub_pct = 27.8%** (beats ~75 teams)

Privately both GPT runs land at priv_pct = 26.2% and claude-dags also lands at 26.2% — **identical private performance**. The 24.6pp "improvement" is entirely a public-leaderboard artefact: claude's exactly-zero score sits at the absolute bottom of a competition where nearly every team scores near zero, while GPT's marginal non-zero score places it in the 27th percentile publicly. There is no real quality difference.

The root cause: without `tensorflow`, claude cannot parse TFRecord format. GPT generated a marginally non-zero random prediction (0.0001). In a 284-team competition with extreme score compression near zero, this 0.0001 gap produced a 26pp public percentile swing while the private outcome was identical.

**`LANL-Earthquake-Prediction` — genuine private shake-up (+23.3pp)**

The LANL competition is well-documented for a massive private leaderboard shake-up: the public test set was a small subsample that teams exploited to produce inflated public scores that didn't generalise. Evidence from the leaderboard statistics:

- Public mean = **81.96** MAE, public median = **1.51** MAE → extreme right skew, many teams with garbage/overfit predictions pulling the mean to 82
- Private mean = **82.49**, private median = **2.62** MAE → the private median degraded from 1.51 to 2.62, meaning the typical team got **73% worse** privately

Claude-dags submitted a vanilla LightGBM model with 33 statistical features (mean, std, quantiles, skewness) on the acoustic signal — a legitimate approach that trained once and never saw public scores. Its MAE of 1.78 was better than the private median (2.62), placing it at the 41.5th percentile privately vs 18.1th publicly. The improvement reflects competitors' public-overfit solutions degrading on private, not any improvement in claude's approach.

This effect is real and appears in all runs that completed LANL: GPT-dags also improves (+11.2pp) and GPT-nodags improves even more (+25.5pp). The difference between runs is how much each team's approach overfit, not anything specific to claude.

**`novozymes-enzyme-stability-prediction` — genuine private shake-up, model-dependent (+23.1pp)**

Novozymes also had documented public LB manipulation: participants could submit many times and tune to public test signal. Evidence:

- Public median Spearman = 0.401, private median = 0.373 → field degraded slightly
- Claude-dags used XGBoost with 20 amino acid composition features + mutation-specific features, validated with proper train/val split → generalises to private
- Claude-dags: pub=19.4 → priv=42.5 (+23.1pp)
- GPT-dags: pub=15.7 → priv=22.4 (+6.6pp) — smaller shake-up benefit
- GPT-nodags: pub=15.0 → priv=**1.2** (−13.9pp) — **reversed**: GPT-nodags's approach apparently overfit to the public test in some way and collapsed privately

The fact that GPT-nodags degrades badly (−13.9pp) while claude-dags improves strongly (+23.1pp) on the same competition suggests claude-dags's feature set was more robustly generalising. Both LANL and novozymes represent legitimate ML outcomes where sound methodology is rewarded privately.

### Why the gap is larger for claude-dags than the other runs

Three compounding factors:

1. **Smaller denominator amplifies signal**: Claude-dags completed 17 tasks; GPT completed 25–29. The three big-gap competitions are 17.6% of claude-dags's pool but only 12% (gpt/dags) and 10.3% (gpt/nodags) of GPT's pool. The same absolute gap produces a larger average shift when divided by 17 instead of 25.

2. **Claude-nodags misses both shake-up competitions**: LANL and novozymes both timed out for claude-nodags before a submission was saved (both exhausted 15 turns on large-scale training). It captures neither the shake-up benefit nor the youtube8m artifact, explaining why its Δ is a smaller +1.11pp.

3. **GPT-nodags novozymes reversal partially cancels**: GPT-nodags's −13.9pp on novozymes offsets the +25.5pp from LANL, leaving its overall Δ near zero. Claude-dags has no such cancelling negative outlier.

### What this means for interpretation

The summary-level pub/priv delta is **not a reliable signal of generalisation quality** for claude-dags:
- The single biggest contributor (+24.6pp youtube8m) is pure noise from a 284-team competition with a degenerate score distribution — one where submitting 0.0001 instead of 0.0 changes the public rank by 200 positions
- The other two competitions (+23pp each) do reflect genuine LB shake-up dynamics, but these affect all runs that complete those tasks — claude-dags just happens to weight them more heavily due to its smaller completion pool

After removing the three outlier competitions, all four runs sit within ±1.7pp of parity between public and private — consistent with the expectation that none of the agents can see public leaderboard scores during the run and therefore cannot preferentially overfit to the public test set.

---

## 7 · Recommendations

### High priority

1. **Stage disambiguation guard**: Inject into the task prompt: *"If multiple sample submission files exist (stage_1, stage_2), always use the highest-numbered stage file."* Also add a pre-submission check: verify row count against the largest available sample file. This alone would fix 11/14 Kaggle errors.

2. **Mandatory early-exit submission**: Add a rule: *"By turn 10, save whatever predictions you have to `/submission/submission.csv`. You may overwrite it in later turns if you improve."* The three NoTraceback cases (Claude nodags) had valid predictions at turn 14 but no save call. An early checkpoint would guarantee a fallback submission.

3. **Stdout truncation**: Cap each cell's stdout at 50k characters. This would prevent the ContextWindowExceeded failures in geolifeclef and trec-covid without affecting computation. Claude generates verbose output that fills 200k tokens when cells run for 900s.

### Library and environment

4. **Library fallback ladder in prompt**: Specify `sklearn → lightgbm → xgboost → (skip torch/tensorflow if unavailable)`. Claude nodags loses 5 tasks to ImportErrors that GPT avoids by defaulting to the lighter stack.

5. **Environment introspection first node**: For GPT, the DAG plan should always include an `i0: List data directory and identify all available files` node before assuming filenames. The `mens-march-mania` FileNotFoundError would not have occurred if GPT had listed `ls` before assuming `sample_submission.csv`.

### For Claude specifically

6. **Turn-budget awareness**: Add to the system prompt: *"You have N turns total. At turn N−3, if no submission file exists, immediately write a placeholder submission using the simplest available prediction (e.g., mean target, zero). Refine it in the remaining turns."* Claude consistently exhausts turns building towards a perfect solution that never arrives.

7. **Cell size discipline**: Prompt Claude to write one operation per cell (load, then process, then train, then save) rather than composing complex pipelines in single blocks. This reduces per-cell runtime, reduces context accumulation, and makes turn-by-turn state management tractable.

