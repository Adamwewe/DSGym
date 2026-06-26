# DSGym Trace Analysis V3 — claude-sonnet-4-5 / dags

> Analysed from `evaluation_results/` conversation logs, Jun 10 2026.  
> Benchmark: **DSPredict-Sampled** (35 Kaggle competitions).  
> Model: **claude-sonnet-4-5 · dags mode only**

---

## Quick Summary

| Metric | V2 (claude-sonnet-4-5 / dags) | V3 (this run) |
|--------|:---:|:---:|
| n | 35 | 35 |
| Complete (scored) | 14 (40%) | **8 (23%)** |
| Zero-turn failures | ~0 | **15 (43%)** |
| Turn-limit no-submit | ~19 | 9 (26%) |
| Other no-submit | — | 3 (9%) |

**Submit rate has dropped from 40% to 23%.** The regression is driven almost entirely by a new category of failure — **zero-turn outcomes** — which did not appear in V2 and now accounts for 15/35 runs.

---

## 1 · Zero-Turn Failures (NEW — dominant issue)

**15/35 runs** returned `total_turns=0` and `prediction=""` despite the agent completing significant work (execution times of 2.5–4 h). This is the dominant new failure pattern and was absent in V2.

### What actually happened

These runs are NOT silent crashes. The trajectory shows the agent ran up to 29 steps, often executing large training jobs. The failure is a **result-capture bug** — the agent code path exited via the outer exception handler before setting `actual_turns`, so the evaluator received `solution=""` even though the container had done real work.

**Two sub-types:**

| Sub-type | Count | Cause |
|----------|:-----:|-------|
| **Container allocation 500 error** | 11 | Manager returned `500 Internal Server Error` for `http://localhost:5000/allocate` — container pool was exhausted or crashed at the start of the run. The env eventually continued (or retried) so code ran, but an exception leaked past the for-loop, zeroing `actual_turns`. |
| **Overall execution timeout / system kill** | 4 | No 500 error; task ran normally but hit a system-level timeout (exec times: LANL 2.87h, ieee-fraud 3.24h, web-traffic 3.07h, predict-ai-runtime 3.93h). Agent was killed mid-run, exception caught by outer handler. |

### Submissions lost in zero-turn failures

Of the 15 zero-turn cases, **10 had a submission.csv saved inside the container** and **6 had fully validated submissions** (format, row count, column names all confirmed inside the agent):

| Competition | Sub saved | Sub validated | Cause |
|-------------|:---------:|:-------------:|-------|
| LANL-Earthquake-Prediction | ✓ | ✓ | System timeout |
| humpback-whale-identification | ✓ | ✓ | 500 alloc error |
| nlp-getting-started | ✓ | ✓ | 500 alloc error |
| store-sales-time-series-forecasting | ✓ | ✓ | 500 alloc error |
| womens-machine-learning-competition-2019 | ✓ | ✓ | 500 alloc error |
| youtube8m-2018 | ✓ | ✓ | 500 alloc error |
| ieee-fraud-detection | ✓ | — | System timeout |
| mens-march-mania-2022 | ✓ | — | 500 alloc error |
| novozymes-enzyme-stability | ✓ | — | 500 alloc error |
| web-traffic-time-series | ✓ | — | System timeout |

These 6 fully-validated cases are **scored as failures with `prediction=""`** even though the agent completed its task. If captured, they would push the true submission rate to ~40% — matching V2.

### Root causes

1. **Container pool exhaustion**: 11/15 zero-turn runs had a `500 Internal Server Error` at the `/allocate` endpoint at turn 0 or 1. This happens when all containers are busy. Running 8 containers in parallel with tasks averaging 2–4 h means any task starting in the second half of the queue waits >1 h for allocation.

2. **`actual_turns` not saved on exception path**: The outer `except Exception` handler returns `"turns": actual_turns` which is 0 because the exception fires before or between loop iterations. Even though the agent loop DID run (trajectory has 23–29 steps), the return dict loses this information. The submission file exists on disk but the evaluator gets `solution=""`.

3. **System-level kill / EXECUTION_TIMEOUT**: 4 runs with exec times >2.5 h and no 500 errors appear to have been killed by an OS or orchestration timeout, with the exception caught by the outer handler.

---

## 2 · Per-Cell Timeout (unchanged — #1 in-run failure)

**20/35 files** had at least one `Code execution error: Failed to execute code: timed out` in the trajectory. There were **51 total timeout events** across the run.

| Competition | Timeouts | Outcome |
|-------------|:--------:|---------|
| microsoft-malware-prediction | 5 | no-submit (turn limit) |
| digit-recognizer | 5 | **complete** (0.9608) |
| ashrae-energy-prediction | 3 | no-submit (turn limit) |
| home-credit-default-risk | 3 | no-submit (turn limit) |
| stanford-covid-vaccine | 3 | no-submit (turn limit) |
| liverpool-ion-switching | 4 | no-submit |
| (11 others) | 1–2 each | mixed |

Notable: `digit-recognizer` hit 5 timeouts and still submitted — the agent recovered well by switching to a simpler approach. `microsoft-malware-prediction` hit 5 timeouts and exhausted all 15 turns without submitting.

Timeout is still caused by the same pattern as V2: agents write large monolithic training cells (LightGBM/neural nets on full datasets) that exceed the 300s per-cell limit. Claude rarely reduces data size or switches to a lighter model on the first timeout — it typically retries with the same code, consuming multiple turns before pivoting.

---

## 3 · Turn-Limit No-Submits (9 cases)

These ran all 15 turns and still didn't submit. Error breakdown:

| Competition | Timeouts | Primary error | Notes |
|-------------|:--------:|---------------|-------|
| microsoft-malware-prediction | 5 | — | Timeout loop, never pivoted |
| ashrae-energy-prediction | 3 | `KeyError: 'building_id'` | Data merge issue, then timeout |
| home-credit-default-risk | 3 | — | Large dataset, timeout |
| stanford-covid-vaccine | 3 | `JSONDecodeError`, `NameError` | Malformed target parse, state lost |
| elo-merchant-category-recommendation | 2 | `ValueError: NaN to int` | NaN handling bug in feature engineering |
| imaterialist-challenge-fashion-2018 | 2 | `FileNotFoundError`, `KeyError` | Data path wrong then schema mismatch |
| imaterialist-challenge-furniture-2018 | 2 | — | Timeout on image URL dataset |
| understanding-cloud-organization | 1 | `ValueError` (Keras shape) | CNN dimension mismatch |
| youtube8m-2019 | 0 | `OSError: Read-only filesystem` | Agent tried to write to `/data/` |

**`stanford-covid-vaccine`** stands out: the agent parses the multi-target JSON incorrectly (`JSONDecodeError: Extra data`), then loses variable state (`NameError: X_train not defined`) across code cells. Classic context-fragmentation bug — variables defined in one cell are not available in a later cell after an error caused a kernel restart.

**`youtube8m-2019`**: Agent attempted `pip install` to `/home/...` (read-only in the container), then tried writing output to `/data/` (also read-only). Never pivoted to using `/submission/` or `/tmp/`.

---

## 4 · Other No-Submits (3 cases)

| Competition | Turns | Issue |
|-------------|:-----:|-------|
| gendered-pronoun-resolution | 13 | No submission path extracted — same root cause as submission_dir bug (see TRACE_ANALYSIS_V2 gotcha) |
| liverpool-ion-switching | 13 | 4 timeouts, ran out of turns before submitting |
| mens-machine-learning-competition-2018 | 13 | No error — agent wrote a summary/answer but never saved submission.csv |

---

## 5 · Missing Packages (environment blockers)

| Package | Missing in | Workaround used |
|---------|-----------|----------------|
| `cv2` (OpenCV) | `understanding-cloud-organization`, `humpback-whale` | No — both blocked |
| `pydicom` | `rsna-pneumonia-detection-challenge`, `siim-acr-pneumothorax` | rsna still submitted via fallback; siim-acr partially worked |
| `py7zr` | `statoil-iceberg-classifier` | Blocked — `.7z` archive not extractable |

These are unchanged from V2. The same three packages cause failures in the same categories of tasks.

---

## 6 · Read-Only Filesystem Errors (NEW)

**4 runs** hit `OSError: [Errno 30] Read-only file system` when trying to write to `/data/`:

- `imaterialist-challenge-fashion-2018`: tried writing a cache file to `/data/dspredict-sampled/...`
- `statoil-iceberg-classifier`: tried extracting `.7z` to `/data/`
- `youtube8m-2019`: tried writing model outputs to `/data/`
- `web-traffic-time-series-forecasting`: tried writing processed data to `/data/`

Agents should be writing all outputs to `/tmp/` or `/submission/`, never to `/data/`. This suggests the prompt doesn't clearly enough prohibit writes to the data directory.

---

## 7 · Successful Runs — what worked

The 8 complete runs share common traits:

| Competition | Score | Turns | Strategy |
|-------------|:-----:|:-----:|---------|
| digit-recognizer | 0.9608 | 9 | CNN → fallback to sklearn when timeout hit |
| santander-customer-tx | 0.8772 | 13 | LightGBM, feature engineering |
| siim-acr-pneumothorax-seg | 0.6000 | 15 | DICOM fallback (no pydicom) |
| santander-value-prediction | 1.4781 | 11 | Log-transform + LightGBM |
| m5-forecasting-accuracy | 1.0814 | 12 | LightGBM with lag features |
| trec-covid-info-retrieval | 0.1949 | 15 | BM25 + fallback to TF-IDF |
| career-con-2019 | 0.1331 | 11 | Random Forest, small dataset |
| rsna-pneumonia-detection | 0.0353 | 12 | Rule-based fallback (no pydicom) |

Successful runs on this model tend to be: **small/medium tabular datasets** (santander ×2, career-con, m5), or tasks where **library absence forced a simpler fallback** that happened to work (rsna, siim-acr using rule-based), or **NLP/retrieval** tasks with small memory footprint (trec-covid, nlp would have been here too if the submission had been captured).

---

## Summary of Failure Modes

| Failure mode | Count | % of 35 |
|-------------|:-----:|:--------:|
| Zero-turn (allocation 500 error) | 11 | 31% |
| Zero-turn (system timeout / kill) | 4 | 11% |
| Per-cell timeout → turn limit hit | 9 | 26% |
| Per-cell timeout → gave up early | 3 | 9% |
| Missing package (hard block) | 2 | 6% |
| Data/schema error | 2 | 6% |
| Read-only filesystem | 1 | 3% |
| Complete | 8 | 23% |

*(Some runs have multiple failure modes; above assigns primary cause.)*

---

## Recommendations

### Immediate (infra bugs causing silent losses)

1. **Fix zero-turn result capture**: The outer `except` handler should check whether `submission_file` exists on disk before returning `solution=""`. If the file is present and non-empty, return it as a partial success regardless of `actual_turns`. This alone would recover ~6 confirmed submissions lost this run.

2. **Container pool sizing**: 11/35 tasks hit `500 Internal Server Error` at allocation, meaning all 8 containers were occupied. With average task duration of 2–4 h and 35 tasks running in sequence, the pool is fine — but if tasks are run in parallel batches, the pool size needs to match the batch size. Log queue depth at start of each task.

3. **System-level timeout vs agent timeout**: 4 tasks ran 2.8–4 h and were killed by the orchestrator. Align the orchestrator timeout with the agent's `max_turns × avg_turn_time` ceiling, or add a heartbeat so the orchestrator can detect genuine hangs vs slow legitimate work.

### Prompt-level fixes

4. **Write-path discipline**: Add explicit instruction: "Never write files to `/data/`. Use `/tmp/` for intermediate work and `/submission/submission.csv` for your final output." This would resolve 4 OSError read-only failures.

5. **Timeout recovery heuristic**: Same as V2 rec #2 — "If a cell runs >250 s, immediately simplify: cut dataset to 20%, switch to `RandomForestClassifier(n_estimators=50)` or `LinearRegression`." The 51 timeout events (20/35 runs) show this hasn't been addressed.

6. **Package fallback ladder**: State explicitly that `cv2`, `py7zr`, and `pydicom` are not available. Provide alternatives: use `PIL`/`skimage` instead of `cv2`, `zipfile`/`tarfile` instead of `py7zr`, and metadata-only approaches for DICOM challenges.

7. **State persistence awareness**: The `stanford-covid-vaccine` failure shows classic kernel-restart state loss (`NameError: X_train not defined`). Prompt agents to re-run prerequisite cells after any error rather than assuming prior state is available.
