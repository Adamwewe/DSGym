# DSGym Trace Analysis v3 — Claude Sonnet 4.5 / DAGs / Timeout ×3

> Analysed from `dspredict-sampled_litellm_claude-sonnet-4-5_dags_results.json` (updated 2026-05-31 00:44)  
> Change from previous run: **EXECUTION_TIMEOUT 300 → 900 seconds**  
> Benchmark: **DSPredict-Sampled** · 35 Kaggle competitions

---

## Quick Summary vs Previous Run

| Metric | Before (300s) | After (900s) | Δ |
|--------|:---:|:---:|:---:|
| Complete | 14/35 (40%) | **17/35 (49%)** | +3 |
| Error | 2/35 (6%) | 3/35 (9%) | +1 |
| No-submit | 19/35 (54%) | **15/35 (43%)** | -4 |
| **Submit rate** | **40%** | **49%** | **+9pp** |
| Avg time (complete) | 20.8 min | **36.4 min** | +15.6 min |
| Avg time (no-submit) | 17.2 min | 31.4 min | +14.2 min |
| Avg pub percentile | — | 20.6 | — |
| Above median | — | 0/17 | — |
| Gold medal | 0 | **1** (gendered-pronoun-resolution) | +1 |

The tripled timeout converted **4 no-submits into completions**. The agent is clearly using the extra time productively — avg completion time jumped from 20.8→36.4 minutes, showing tasks that previously timed out mid-training are now finishing. However, 5 tasks still hit the timeout at 15 turns despite having 900s per cell.

---

## 1 · Submission Errors (SUBMISSIONSTATUS.ERROR)

Three errors, all passing local format validation but rejected by Kaggle:

| Competition | Turns | Time | What happened |
|-------------|:---:|:---:|---------------|
| `liverpool-ion-switching` | 14 | 34.5min | Submitted valid format; Kaggle rejected — likely wrong row count or out-of-range `open_channels` values |
| `womens-machine-learning-competition-2019` | 10 | 23.5min | Stage 1 submission (10,080 rows, format validated) — same multi-stage confusion as before |
| `mens-machine-learning-competition-2018` | 14 | 40.7min | Format fully validated (`✓ Submission format validated successfully`), saved to disk — Kaggle rejects, likely wrong season/row count |

**Pattern unchanged**: All 3 errors are competitions with multiple submission stages or non-obvious row-count requirements. The agent's local validation passes but Kaggle's server-side check rejects. The `womens-machine-learning-competition-2019` error is identical to the previous run — this competition needs explicit stage disambiguation in the prompt.

---

## 2 · No-Submit Failure Modes

| Error type | Count | Notes |
|-----------|:---:|-------|
| **TimeoutError** | 5 | Down from 10 — tripled timeout directly fixed half the timeout cases |
| **ImportError** | 5 | Same as before; now exclusively deep-learning libraries |
| **FileNotFoundError** | 3 | Data files or tools not present in environment |
| **ContextWindowExceeded** | 2 | **New failure mode** — not seen in previous run |

### Timeout (5 cases, all at 15 turns)
All 5 timeout cases hit the **turn limit** (15), not the per-cell limit — they ran until the agent ran out of moves. These are structurally hard tasks where even 900s per cell isn't enough to converge:

| Competition | Time | Pattern |
|------------|:---:|---------|
| `ashrae-energy-prediction` | 34.8min | Large dataset, keeps losing variables between cells due to kernel resets |
| `imaterialist-challenge-furniture-2018` | 20.9min | Image classification, iterating through slow Random Forest baseline |
| `talkingdata-adtracking-fraud-detection` | 37.4min | Huge dataset (184M rows), model reloading on every turn |
| `novozymes-enzyme-stability-prediction` | 41.5min | Training full LightGBM on protein embeddings, conservative params still slow |
| `predict-ai-model-runtime` | **72.1min** | Longest run; processes XLA tile configs iteratively, never generates a complete submission |

### ImportError (5 cases — deep learning libraries only)
Every ImportError is now a **computer vision or video understanding** task needing `torch`, `tensorflow`, or `cv2`:

| Competition | Missing modules |
|------------|----------------|
| `understanding-cloud-organization` | `torch`, `tensorflow`, `cv2` |
| `sp-society-camera-model-identification` | `cv2`, `torch`, `tensorflow`, `skimage` |
| `youtube8m-2018` | `tensorflow` |
| `youtube8m-2019` | `tensorflow` |
| `statoil-iceberg-classifier` | (also `7z` for decompression — environment blocker) |

These are **environment-level blockers**, not agent failures. No amount of prompt improvement resolves them — the image requires these packages or `statoil` needs a decompressor.

### FileNotFoundError (3 cases)

| Competition | Missing path | Root cause |
|------------|-------------|------------|
| `imaterialist-challenge-fashion-2018` | `train.json` | Data file not in sampled dataset |
| `web-traffic-time-series-forecasting` | `train_1.csv` | Data file not provided (compressed format) |
| `mens-march-mania-2022` | `MSampleSubmissionStage1.csv` | Agent expected specific filename, actual name differs |

### ContextWindowExceeded (2 cases — new in this run)

| Competition | Turns | What happened |
|------------|:---:|---------------|
| `geolifeclef-2024` | 1 | Hit 200k token limit immediately at turn 1. The trajectory output from geospatial data exploration produced >200k tokens in a single cell output, filling the context before any model training |
| `trec-covid-information-retrieval` | 15 | Accumulated context (40 topics × multi-step ranking output) eventually exceeded 200k tokens |

**This is a new failure mode introduced by the longer timeout.** With 900s per cell, agents can now produce vastly more stdout output (large dataframe prints, verbose training logs, iterative search results) that fills the context window. The `geolifeclef` case is particularly striking — it failed at turn 1 because a single exploration cell produced too much output.

**Recommendation**: Add stdout truncation in the execution environment (e.g., cap cell output at 50k characters), or inject a reminder in the system prompt: "Always limit print output — use `.head(5)`, `[:10]` slices, and `print(f'shape: {df.shape}')` rather than printing full dataframes."

---

## 3 · Instruction Following

| Pattern | Before (300s) | After (900s) | Δ |
|---------|:---:|:---:|:---:|
| `<reasoning>` tag | 35/35 | 35/35 | — |
| `<python>` tag | 35/35 | 35/35 | — |
| `<answer>` block | 15/35 (43%) | **18/35 (51%)** | +3 |
| `PLAN REVISION` | 25/35 (71%) | 18/35 (51%) | -7 |
| Execution summary | 14/35 (40%) | 17/35 (49%) | +3 |
| `Step N/M` format | 35/35 | 35/35 | — |

**`<answer>` block usage** improved slightly (43→51%). Still not consistent — this should be 35/35 to match GPT.

**Plan revisions dropped** (71→51%, avg 1.1/run vs 2.2 before). With more execution time, agents are less likely to hit dead ends that force replanning. More importantly, the revision rate is now **equal across complete and no-submit runs** (1.0 vs 1.2 respectively), unlike before where no-submits had nearly 2× more revisions — the "revisions = distress signal" correlation has weakened.

---

## 4 · DAG Consistency

| Metric | Before (300s) | After (900s) |
|--------|:---:|:---:|
| Reasoning DAG present | 35/35 | 35/35 |
| Impl DAG present | 35/35 | 35/35 |
| Avg nodes (reasoning) | 5.9 | 6.0 |
| Avg nodes (impl) | 6.0 | 6.0 |

### Implementation DAG → code keyword coverage

| Metric | Before | After |
|--------|:---:|:---:|
| Avg coverage | 77.6% | **77.5%** |
| [complete] avg | 81.4% | **79.9%** |
| [no_submit] avg | 74.8% | 73.3% |
| [error] avg | 77.6% | **85.1%** |

Coverage is essentially unchanged — the DAG accurately represents the plan regardless of timeout. The coverage→outcome correlation persists: complete runs cover more of the plan (79.9% vs 73.3% for no-submits).

### Reasoning → implementation semantic overlap

| Metric | Before | After |
|--------|:---:|:---:|
| Avg R→I overlap | 46.4% | **47.9%** |
| [complete] avg | 42.7% | **47.9%** |
| [no_submit] avg | 49.4% | 47.1% |

The previously notable pattern — where failing runs had *higher* R→I overlap (suggesting they over-planned and under-concretised) — has **disappeared**. Complete and no-submit runs now show identical R→I overlap (47.9% vs 47.1%). The extra time removes the scenario where an agent runs out of turns mid-translation and the implementation DAG stays close to the abstract reasoning DAG.

---

## 5 · Notable Individual Results

**Gold medal — `gendered-pronoun-resolution`** (score=0.0, 50th percentile): Score of 0.0 log-loss would be perfect; 50th percentile is suspicious — possibly this competition evaluates on a small leaderboard where 0.0 is tied with many entrants. Only took 4.7 minutes (7 turns) — the agent used a simple heuristic approach that happened to be optimal for the available data.

**`predict-ai-model-runtime`** (72.1 min, 15 turns, no submit): The longest run by far. The agent processed XLA tile configurations iteratively across 15 turns, accumulating context, but never managed to generate a complete submission file. This competition likely needs a fundamentally different approach (batch processing all configs in one cell rather than iterating).

**`microsoft-malware-prediction`** (76.2 min, 14 turns, complete): The longest successful run. Agent persisted through multiple LightGBM training iterations with large data, eventually submitting at 18.6th percentile.

**`m5-forecasting-accuracy`** (56.7 min, 15 turns, complete, 1.7th percentile): Submitted despite using all 15 turns — the extended timeout allowed completing a forecasting pipeline that previously would not have finished.

---

## 6 · What Still Needs Fixing

### Unchanged problems
1. **Multi-stage competition confusion** (`womens-machine-learning`, `mens-2018`) — same 3 competitions as before still erroring on server-side validation. Needs expected row count in prompt.
2. **ImportError (deep learning)** — 4–5 competitions are pure environment blockers (`torch`, `tensorflow`, `cv2`, `7z`). Nothing prompt-level will fix these.
3. **`<answer>` block** inconsistency — still only ~51% of runs, should be 100%.

### New problems introduced by longer timeout
4. **ContextWindowExceeded** — 2 cases appeared that didn't exist with 300s timeout. Agents now produce more verbose output per cell, filling the 200k context window. The `geolifeclef` case failed at turn 1.

### Diminishing-returns timeout cases
5. Five competitions still time out even with 900s. These are not 300s→900s problems; they are **architecture problems**: the agent pattern of reloading data and models from scratch each turn means large-dataset tasks are fundamentally inefficient. A stateful execution environment (persistent kernel) would help more than another timeout increase.
