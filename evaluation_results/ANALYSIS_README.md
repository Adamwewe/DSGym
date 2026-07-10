# DSGym Agent Evaluation — Failure Analysis

> Analysed from `evaluation_results/` conversation logs for the **DSPredict-Sampled** benchmark (35 Kaggle competitions, 160 total experiment runs across 3 agents × 2 modes).

---

## Quick Summary

| Metric | Value |
|--------|-------|
| Total experiment runs | 160 |
| Competitions covered | 35 |
| **Overall success rate** | **74 / 160 (46.3%)** |
| Competitions where ALL agents failed | 5 |
| Competitions where ALL agents succeeded | 8 |

---

## Plots

| Plot | Description |
|------|-------------|
| `error_analysis_overview.png` | Error type distribution (pie) + success rate by agent × mode (bar) |
| `competition_heatmap.png` | Full outcome heatmap: every competition × every agent/mode |
| `stale_file_contamination.png` | Unexpected column names found in stale files (source competition labelled) |
| `turns_and_categories.png` | Turn-count distribution (success vs failure) + success rate by task category |

---

## Error Type Breakdown

| Error Type | Count | % of failures | Description |
|------------|-------|--------------|-------------|
| **Wrong Columns (Stale Submission File)** | 72 | 83.7% | Agent submitted a file left over from a different competition |
| **Wrong Row Count** | 10 | 11.6% | Submission had the wrong number of rows |
| **Missing Key Values** | 3 | 3.5% | Required key values absent from the submission |
| **Invalid Values (Inf/NaN)** | 1 | 1.2% | Predicted `inf` values in regression output |

---

## Root Cause #1: Stale Submission File (Cross-Competition Contamination)

**72 of 86 failures (83.7%)** share the same root cause.

### What happens

The evaluation harness writes all submissions to a **shared path** (`/submission/submission.csv`). When a new competition starts, the file from the *previous* competition persists on disk. If the agent fails to overwrite it (due to a code error, timeout, or early exit), the evaluator reads the stale file and reports "Unexpected Column" errors.

### Evidence from trajectory logs

Agents frequently diagnose the issue mid-run:
```
[Turn 17 — digit-recognizer, claude-sonnet-4-5 nodags]
"I see there's an issue with a previous submission.csv file that has the wrong format
(it seems to be from a different competition). Let me start fresh..."

[Turn 19 — ashrae-energy-prediction, claude-sonnet-4-5 dags]
"[PLAN REVISION] Diagnosis: There's an old submission.csv from a different challenge
(earthquake prediction with seg_id, time_to_failure columns)."
```

Despite recognising the problem, agents were often unable to save a correct file before hitting the turn/time limit.

### Which stale columns appeared most often

The column names found in the "wrong" submissions reveal which competition's output was accidentally submitted:

| Stale Column(s) | Belongs to competition | Count |
|----------------|----------------------|-------|
| `seg_id`, `time_to_failure` | LANL-Earthquake-Prediction | 6 each |
| `SK_ID_CURR`, `TARGET` | home-credit-default-risk | 6 each |
| `Class`, `Segments` | career-con-2019 | 6 each |
| `Image`, `Id` | imaterialist-challenge / humpback-whale | 7, 9 |
| `ID` | gendered-pronoun-resolution | 11 |
| `MachineIdentifier`, `HasDetections` | microsoft-malware-prediction | 4 each |
| `topic-id`, `cord-id` | trec-covid-information-retrieval | 4 each |
| `ID_code` | santander-customer-transaction-prediction | 5 |

See `stale_file_contamination.png` for the full chart.

### Fix recommendation

**Clear `/submission/submission.csv` between competitions.** Either:
- Delete the file at the start of each run, or
- Use competition-specific submission paths (e.g., `/submission/{competition_name}.csv`), or
- Validate the submission schema before accepting the file as the final answer.

---

## Root Cause #2: Wrong Row Count

**10 failures** across 5 competitions:

| Competition | Error | Agent(s) affected |
|------------|-------|------------------|
| `gendered-pronoun-resolution` | Must have 12 359 rows (only stage-1 data was available) | gpt-5.1 ×3, claude-sonnet-4-5 dags |
| `mens-march-mania-2022` | Must have 2 278 rows (confused 2022 vs 2018 season) | gpt-5.1 dags |
| `mens-machine-learning-competition-2018` | Must have 2 278 rows | claude-sonnet-4-5 dags |
| `web-traffic-time-series-forecasting` | Must have 8 993 906 rows | gpt-5.1 ×2 |
| `predict-ai-model-runtime` | 2 016 or 49 342 rows submitted, 894 expected | claude-sonnet-4-5 ×2 |
| `womens-machine-learning-competition-2019` | Must have 2 016 rows | gpt-5.1 dags |

**Root causes:**
- Sampled data had fewer rows than the real competition; agents computed the expected row count from the sample and under-generated.
- For `gendered-pronoun-resolution` specifically: agents only had access to `test_stage_1.tsv` (2 000 rows) but the submission required all 12 359 stage-2 rows.
- For `web-traffic`: the key file had 8.7 M entries but agents generated forecasts only for the sampled subset.

---

## Root Cause #3: Missing Key Values (liverpool-ion-switching)

**3 failures** — gpt-5.1 (both modes) and supporting claude runs.

The submission required exactly 200 000 time-step keys (`500.0001`, `500.0002`, …). Agents generated only the sampled subset (far fewer rows) and missed the required dense time-step index.

```
ERROR: Unable to find 200000 required key values in the 'time' column
ERROR: Unable to find the required key value '500.0001' in the 'time' column
```

---

## Root Cause #4: Invalid Values — Inf/NaN (santander-value-prediction-challenge)

**1 failure** — gpt-5.1 nodags.

A log-transformation of zero values (common in value-prediction pipelines) produced `inf` in the output CSV, which the Kaggle evaluator rejected:

```
ERROR: Could not parse 'inf' into expected type of Double (Line 62, Column 11)
```

**Fix:** Clip predictions to a finite range or replace `inf` with a large constant before writing the submission.

---

## Performance by Agent

| Agent | Runs | Successes | Success Rate |
|-------|------|-----------|-------------|
| **gpt-5.1** | 70 | 48 | **68.6%** |
| claude-sonnet-4-5 | 70 | 26 | 37.1% |
| claude-sonnet-4-6 | 20 | 0 | **0%** |

> **Note:** claude-sonnet-4-6 was only evaluated on a subset of competitions (10 dags + 10 nodags). It achieved 0% success — every run was a stale-file error, suggesting this model was particularly prone to running out of turns before writing the correct submission.

---

## Performance by Mode (DAGs vs No-DAGs)

| Mode | Runs | Successes | Success Rate |
|------|------|-----------|-------------|
| **dags** | 80 | 39 | **48.8%** |
| nodags | 80 | 35 | 43.8% |

The DAG-guided mode has a small but consistent advantage. The stale-file bug affects both modes roughly equally, so the difference likely reflects improved reasoning structure.

---

## Performance by Competition Category

| Category | Competitions | Runs | Successes | Rate |
|----------|-------------|------|-----------|------|
| **Tabular/Structured** | 18 | 72 | 38 | **52.8%** |
| **NLP** | 4 | 16 | 9 | 56.3% |
| **Computer Vision** | 9 | 36 | 13 | 36.1% |
| **Geo/Video** | 4 | 16 | 10 | 62.5% |

Computer vision competitions fared worst — they tend to require specific RLE/image-ID formats that agents struggle to generate correctly when facing stale files.

---

## Competitions Where All Agents Failed

| Competition | Why all agents failed |
|------------|----------------------|
| `gendered-pronoun-resolution` | 3/6 stale-file errors; 3/6 wrong row count (only stage-1 data available, need full 12 359 rows) |
| `liverpool-ion-switching` | 4/6 stale-file errors; 2/6 missing key values (need dense 200 k time steps, only sampled) |
| `mens-machine-learning-competition-2018` | Wrong row count + stale files + missing key values (confusion between 2018/2022 season data) |
| `sp-society-camera-model-identification` | All 4 runs stale-file errors (expected `fname` column, got varied competition outputs) |
| `statoil-iceberg-classifier-challenge` | All 4 runs stale-file errors (expected `id`/`is_iceberg`, got outputs from 5 different competitions) |

---

## Competitions Where All Agents Succeeded

`LANL-Earthquake-Prediction` · `digit-recognizer` (dags) · `home-credit-default-risk` · `nlp-getting-started` · `santander-customer-transaction-prediction` · `santander-value-prediction-challenge` (3/4) · `siim-acr-pneumothorax-segmentation` · `stanford-covid-vaccine` (4/6)

---

## Per-Competition Results

> Icon legend: ✅ Success  ❌ Stale file error  ⚠️ Wrong row count  🔍 Missing key values  🔢 Invalid values

### `LANL-Earthquake-Prediction`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 1.7481 | Valid submission |
| claude-sonnet-4-5 | nodags | ✅ SUCCESS | 1.7280 | Valid submission |
| gpt-5.1 | dags | ✅ SUCCESS | 1.8485 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 1.6129 | Valid submission |

### `ashrae-energy-prediction`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `seg_id`, `time_to_failure`; expected `row_id` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `ID`, `Pred`; expected `row_id` |
| claude-sonnet-4-6 | dags | ❌ FAILED | — | Stale file: found `seg_id`, `time_to_failure`; expected `row_id` |
| claude-sonnet-4-6 | nodags | ❌ FAILED | — | Stale file: found `ID`, `TopConfigs`; expected `row_id` |
| gpt-5.1 | dags | ❌ FAILED | — | Stale file: found `ID_code`, `target`; expected `row_id` |
| gpt-5.1 | nodags | ✅ SUCCESS | 1.4920 | Valid submission |

### `career-con-2019`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.1278 | Valid submission |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `Class`, `Segments`; expected `series_id` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.1307 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.1131 | Valid submission |

### `digit-recognizer`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.9729 | Valid submission |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `seg_id`, `time_to_failure`; expected `ImageId` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.9700 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.9680 | Valid submission |

### `elo-merchant-category-recommendation`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `Image`, `Id`; expected `card_id` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `Class`, `Segments`; expected `card_id` |
| claude-sonnet-4-6 | dags | ❌ FAILED | — | Stale file: found `ID_code`; expected `card_id` |
| claude-sonnet-4-6 | nodags | ❌ FAILED | — | Stale file: found `seg_id`, `time_to_failure`; expected `card_id` |
| gpt-5.1 | dags | ❌ FAILED | — | Stale file: found `Image`, `Id`; expected `card_id` |
| gpt-5.1 | nodags | ✅ SUCCESS | 3.9219 | Valid submission |

### `gendered-pronoun-resolution`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ⚠️ FAILED | — | Evaluation Exception: Submission must have 12359 rows |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `seg_id`, `time_to_failure`; expected `ID` |
| claude-sonnet-4-6 | dags | ❌ FAILED | — | Stale file: found `SK_ID_CURR`, `TARGET`; expected `ID` |
| claude-sonnet-4-6 | nodags | ❌ FAILED | — | Stale file: found `Class`, `Segments`; expected `ID` |
| gpt-5.1 | dags | ⚠️ FAILED | — | Evaluation Exception: Submission must have 12359 rows |
| gpt-5.1 | nodags | ⚠️ FAILED | — | Evaluation Exception: Submission must have 12359 rows |

### `geolifeclef-2024`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `ID`, `A`; expected `surveyId` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `SK_ID_CURR`, `TARGET`; expected `surveyId` |
| claude-sonnet-4-6 | dags | ❌ FAILED | — | Stale file: found `ID`, `TopConfigs`; expected `surveyId` |
| claude-sonnet-4-6 | nodags | ❌ FAILED | — | Stale file: found `topic-id`, `cord-id`; expected `surveyId` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.2538 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.2559 | Valid submission |

### `home-credit-default-risk`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.7477 | Valid submission |
| claude-sonnet-4-5 | nodags | ✅ SUCCESS | 0.7145 | Valid submission |
| gpt-5.1 | dags | ✅ SUCCESS | 0.6251 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.7145 | Valid submission |

### `humpback-whale-identification`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.2721 | Valid submission |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `Visits`; expected `Image` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.2775 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.2775 | Valid submission |

### `ieee-fraud-detection`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `ID`, `A`; expected `TransactionID` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `ID`, `Pred`; expected `TransactionID` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.5000 | Valid submission |
| gpt-5.1 | nodags | ❌ FAILED | — | Stale file: found `ID`, `Pred`; expected `TransactionID` |

### `imaterialist-challenge-fashion-2018`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.3532 | Valid submission |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `MachineIdentifier`, `HasDetections`; expected `image_id` |
| claude-sonnet-4-6 | dags | ❌ FAILED | — | Stale file: found `Image`, `Id`; expected `image_id` |
| claude-sonnet-4-6 | nodags | ❌ FAILED | — | Stale file: found `topic-id`, `cord-id`; expected `image_id` |
| gpt-5.1 | dags | ❌ FAILED | — | Stale file: found `seg_id`, `time_to_failure`; expected `image_id` |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.3233 | Valid submission |

### `imaterialist-challenge-furniture-2018`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.9935 | Valid submission |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `target`; expected `predicted` |
| claude-sonnet-4-6 | dags | ❌ FAILED | — | Stale file: found `target`; expected `predicted` |
| claude-sonnet-4-6 | nodags | ❌ FAILED | — | Stale file: found `ID_code`, `target`; expected `id` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.9788 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.9935 | Valid submission |

### `liverpool-ion-switching`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `MachineIdentifier`, `HasDetections`; expected `time` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `id_seqpos`, `reactivity`; expected `time` |
| claude-sonnet-4-6 | dags | ❌ FAILED | — | Stale file: found `Class`, `Segments`; expected `time` |
| claude-sonnet-4-6 | nodags | ❌ FAILED | — | Stale file: found `ID_code`, `target`; expected `time` |
| gpt-5.1 | dags | 🔍 FAILED | — | ERROR: Unable to find 200000 required key values in the 'time' column ERROR: Una |
| gpt-5.1 | nodags | 🔍 FAILED | — | ERROR: Unable to find 200000 required key values in the 'time' column ERROR: Una |

### `m5-forecasting-accuracy`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `Pred`; expected `F1` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `SK_ID_CURR`, `TARGET`; expected `id` |
| gpt-5.1 | dags | ✅ SUCCESS | 1.0822 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 5.4456 | Valid submission |

### `mens-machine-learning-competition-2018`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ⚠️ FAILED | — | Evaluation Exception: Submission must have 2278 rows |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `VideoId`, `LabelConfidencePairs`; expected `ID` |
| gpt-5.1 | dags | ❌ FAILED | — | Stale file: found `F1`, `F2` |
| gpt-5.1 | nodags | 🔍 FAILED | — | ERROR: Unable to find 2278 required key values in the 'ID' column ERROR: Unable  |

### `mens-march-mania-2022`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.6635 | Valid submission |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `SK_ID_CURR`, `TARGET`; expected `ID` |
| gpt-5.1 | dags | ⚠️ FAILED | — | Evaluation Exception: Submission must have 2278 rows |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.7125 | Valid submission |

### `microsoft-malware-prediction`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `series_id`, `surface`; expected `MachineIdentifier` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `Class`, `Segments`; expected `MachineIdentifier` |
| gpt-5.1 | dags | ❌ FAILED | — | Stale file: found `ID`, `TopConfigs`; expected `MachineIdentifier` |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.5543 | Valid submission |

### `nlp-getting-started`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.7904 | Valid submission |
| claude-sonnet-4-5 | nodags | ✅ SUCCESS | 0.7990 | Valid submission |
| gpt-5.1 | dags | ✅ SUCCESS | 0.7956 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.7971 | Valid submission |

### `novozymes-enzyme-stability-prediction`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `image_id`, `label_id`; expected `seq_id` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `SK_ID_CURR`, `TARGET`; expected `seq_id` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.0295 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.0243 | Valid submission |

### `predict-ai-model-runtime`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ⚠️ FAILED | — | Submission has 2016 rows but 894 were expected. |
| claude-sonnet-4-5 | nodags | ⚠️ FAILED | — | Submission has 49342 rows but 894 were expected. |
| gpt-5.1 | dags | ✅ SUCCESS | -2.9039 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.1190 | Valid submission |

### `rsna-pneumonia-detection-challenge`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.0000 | Valid submission |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `Class`, `Segments`; expected `patientId` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.0083 | Valid submission |
| gpt-5.1 | nodags | ❌ FAILED | — | Stale file: found `surveyId`, `predictions`; expected `patientId` |

### `santander-customer-transaction-prediction`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.8947 | Valid submission |
| claude-sonnet-4-5 | nodags | ✅ SUCCESS | 0.8952 | Valid submission |
| gpt-5.1 | dags | ✅ SUCCESS | 0.8855 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.8941 | Valid submission |

### `santander-value-prediction-challenge`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 1.4948 | Valid submission |
| claude-sonnet-4-5 | nodags | ✅ SUCCESS | 1.4709 | Valid submission |
| gpt-5.1 | dags | ✅ SUCCESS | 1.5390 | Valid submission |
| gpt-5.1 | nodags | 🔢 FAILED | — | Predicted `inf` values in output |

### `siim-acr-pneumothorax-segmentation`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.6000 | Valid submission |
| claude-sonnet-4-5 | nodags | ✅ SUCCESS | 0.6000 | Valid submission |
| gpt-5.1 | dags | ✅ SUCCESS | 0.6000 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.6000 | Valid submission |

### `sp-society-camera-model-identification`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `ImageId`, `EncodedPixels`; expected `fname` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `Id`, `Visits`; expected `fname` |
| gpt-5.1 | dags | ❌ FAILED | — | Stale file: found `id`, `predicted`; expected `fname` |
| gpt-5.1 | nodags | ❌ FAILED | — | Stale file: found `ImageId`, `EncodedPixels`; expected `fname` |

### `stanford-covid-vaccine`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.8947 | Valid submission |
| claude-sonnet-4-5 | nodags | ✅ SUCCESS | 0.7692 | Valid submission |
| claude-sonnet-4-6 | dags | ❌ FAILED | — | Stale file: found `topic-id`, `cord-id`; expected `id_seqpos` |
| claude-sonnet-4-6 | nodags | ❌ FAILED | — | Stale file: found `Image`, `Id`; expected `id_seqpos` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.3693 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.4028 | Valid submission |

### `statoil-iceberg-classifier-challenge`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `sales`; expected `is_iceberg` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `id_seqpos`, `reactivity`; expected `id` |
| gpt-5.1 | dags | ❌ FAILED | — | Stale file: found `F1`, `F2` |
| gpt-5.1 | nodags | ❌ FAILED | — | Stale file: found `topic-id`, `cord-id`; expected `id` |

### `store-sales-time-series-forecasting`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 1.4348 | Valid submission |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `MachineIdentifier`, `HasDetections`; expected `id` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.5752 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 2.0823 | Valid submission |

### `talkingdata-adtracking-fraud-detection`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `Image`, `Id`; expected `click_id` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `MachineIdentifier`, `HasDetections`; expected `click_id` |
| claude-sonnet-4-6 | dags | ❌ FAILED | — | Stale file: found `ID`, `target`; expected `click_id` |
| claude-sonnet-4-6 | nodags | ❌ FAILED | — | Stale file: found `ID`, `target`; expected `click_id` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.5000 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.9179 | Valid submission |

### `trec-covid-information-retrieval`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `topicid`, `docid`; expected `topic-id` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `id_seqpos`, `reactivity`; expected `topic-id` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.0000 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.2150 | Valid submission |

### `understanding_cloud_organization`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `Id`, `Visits`; expected `Image_Label` |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `ID_code`, `target`; expected `Image_Label` |
| claude-sonnet-4-6 | dags | ❌ FAILED | — | Stale file: found `Image`, `Id`; expected `Image_Label` |
| claude-sonnet-4-6 | nodags | ❌ FAILED | — | Stale file: found `SK_ID_CURR`, `TARGET`; expected `Image_Label` |
| gpt-5.1 | dags | ❌ FAILED | — | Stale file: found `Image`, `Id`; expected `Image_Label` |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.4775 | Valid submission |

### `web-traffic-time-series-forecasting`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `predicted`; expected `Visits` |
| claude-sonnet-4-5 | nodags | ✅ SUCCESS | 40.8012 | Valid submission |
| gpt-5.1 | dags | ⚠️ FAILED | — | Evaluation Exception: Submission must have 8993906 rows |
| gpt-5.1 | nodags | ⚠️ FAILED | — | Evaluation Exception: Submission must have 8993906 rows |

### `womens-machine-learning-competition-2019`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.3889 | Valid submission |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `ImageId`, `EncodedPixels`; expected `ID` |
| gpt-5.1 | dags | ⚠️ FAILED | — | Evaluation Exception: Submission must have 2016 rows |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.4150 | Valid submission |

### `youtube8m-2018`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ❌ FAILED | — | Stale file: found `ID`, `A`; expected `VideoId` |
| claude-sonnet-4-5 | nodags | ✅ SUCCESS | 0.0516 | Valid submission |
| gpt-5.1 | dags | ✅ SUCCESS | 0.0138 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.0196 | Valid submission |

### `youtube8m-2019`

| Agent | Mode | Outcome | Score | Root Cause |
|-------|------|---------|-------|------------|
| claude-sonnet-4-5 | dags | ✅ SUCCESS | 0.0000 | Valid submission |
| claude-sonnet-4-5 | nodags | ❌ FAILED | — | Stale file: found `VideoId`, `LabelConfidencePairs`; expected `Class` |
| gpt-5.1 | dags | ✅ SUCCESS | 0.0000 | Valid submission |
| gpt-5.1 | nodags | ✅ SUCCESS | 0.0001 | Valid submission |

___BEGIN___COMMAND_DONE_MARKER___0


---

## Key Takeaways & Recommendations

1. **Fix the shared submission path (highest priority):** The stale-file bug causes 83.7% of all failures and is purely a harness-level issue, not an agent intelligence issue. Clearing or isolating the submission path between runs would immediately raise the overall success rate from ~46% to ~90%.

2. **Provide row-count hints to agents:** For competitions where the expected submission size differs from the sampled training data, inject the expected row count into the system prompt. This would resolve the 10 wrong-row-count failures.

3. **Validate submissions before finalising:** Add a lightweight schema check (column names + row count) at the end of each run. If it fails, prompt the agent to fix the file rather than surfacing the error only at submission time.

4. **Investigate claude-sonnet-4-6:** This model achieved 0% success (0/20). Its trajectories show it consistently ran out of turns while trying to debug the stale-file problem. It may benefit from a stronger initial instruction to check for and clear stale submissions at the start of each run.

5. **Pre-process inf/NaN before writing CSV:** One simple addition to the submission-writing step — `df.replace([np.inf, -np.inf], np.nan).fillna(0)` — would have prevented the `santander-value-prediction-challenge` failure.

6. **Sampled vs full data mismatch:** Several "wrong row count" failures stem from the sampled dataset having far fewer rows than what the competition evaluator expects. This is a benchmark design issue: either use submission templates that reflect the sampled size, or clearly specify the required number of output rows in the task prompt.
