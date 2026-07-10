# TRACE_ANALYSIS_V5 — DSPredict-Hard multi-model trace comparison

> Benchmark: **DSPredict-Hard** full benchmark for GPT models (53–54 runs) plus **DSPredict-sampled** for Claude Sonnet 4.5 (35 runs).  
> Compared modes: **gpt-4o-mini** (dags / nodags), **gpt-5.1** (dags / nodags), **claude-sonnet-4-5** (dags / nodags).

---

## 0. Executive summary

Three findings dominate the traces.

1. **gpt-4o-mini collapses under DAG prompting.** Its complete rate is cut in half by DAG mode: **9% vs 19%**. The central reason is not one bug but a stack of interacting failures: **25/54 dags runs hit server disconnects**, many also combine with bad path assumptions, and the model frequently drifts into formatting failures (`No python code found`, `400 Bad Request`, `Kernel not ready`).
2. **gpt-5.1 is much stronger overall, but DAGs still hurt it.** It drops from **78% complete in nodags** to **53% in dags**. The degradation is milder than 4o-mini because GPT-5.1 usually recovers from path errors and library/API issues, but DAG mode still induces context blowups and extra infrastructure failures.
3. **Claude Sonnet 4.5 shows the opposite pattern.** On the sampled 35-task subset, DAGs improve complete rate from **31% to 57%**. Claude appears to use DAGs as lightweight scaffolding; GPT models, especially 4o-mini, treat DAGs as overhead and frequently over-plan.

The strongest mechanistic contrast is this: **for gpt-4o-mini, more DAG revision is a bad sign** (complete runs average only **2.6** plan revisions, no-submit runs **6.5**); for **gpt-5.1**, revisions are mostly healthy adaptation (**5.0** for complete dags runs).

---

## 1. Summary table

| Model / mode | n | Complete | Error | No-submit | Avg turns | Avg exec (min) | Avg tokens | Avg public percentile (complete) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-4o-mini/dags | 54 | 5 (9%) | 4 (7%) | 45 (83%) | 12.0 | 84.7 | 62855 | 30.9 |
| gpt-4o-mini/nodags | 54 | 10 (19%) | 14 (26%) | 30 (56%) | 13.2 | 134.7 | 75280 | 37.9 |
| gpt-5.1/dags | 53 | 28 (53%) | 12 (23%) | 13 (25%) | 9.2 | 93.1 | n/a | 41.3 |
| gpt-5.1/nodags | 54 | 42 (78%) | 10 (19%) | 2 (4%) | 8.2 | 82.3 | n/a | 41.5 |
| claude-sonnet-4-5/dags | 35 | 20 (57%) | 5 (14%) | 10 (29%) | 13.1 | 69.1 | n/a | 39.9 |
| claude-sonnet-4-5/nodags | 35 | 11 (31%) | 4 (11%) | 20 (57%) | 14.5 | 22.9 | n/a | 32.2 |

Interpretation:
- **Best overall**: gpt-5.1/nodags.
- **Worst overall**: gpt-4o-mini/dags.
- **Claude** is sampled only, but on its own sample it is the only model here with a clear DAG benefit.
- Token logs are only available in the exported 4o-mini runs; GPT-5.1 and Claude runs report no token usage in these result files.

---

## 2. No-submit failure analysis (primary focus: gpt-4o-mini)

### 2.1 gpt-4o-mini / dags — failure mix

| Cause | Count | % of no-submit |
|---|---:|---:|
| server disconnect | 16 | 36% |
| server disconnect + FileNotFoundError | 7 | 16% |
| no submission written | 6 | 13% |
| read-only filesystem / archive tooling | 5 | 11% |
| timeout / long-running code | 4 | 9% |
| logic/runtime error | 3 | 7% |
| FileNotFoundError / wrong path assumption | 2 | 4% |
| context window exceeded | 1 | 2% |
| early stop / no recovery | 1 | 2% |

The dominant dags failure mode is the new one flagged in preliminary review: **server disconnects**. The characteristic trace quote is from `LANL-Earthquake-Prediction`:

> `Code execution error: Failed to execute code: Server disconnected without sending a response.`

That run starts with a perfectly formed DAG, immediately makes a wrong path assumption (`'/data/dspredict-hard/LANL-Earthquake-Prediction/test/test.csv'`), then spirals through three disconnects and finally `KernelError: Kernel not ready` without ever writing a submission.

### 2.2 gpt-4o-mini / nodags — failure mix

| Cause | Count | % of no-submit |
|---|---:|---:|
| 500 allocate error | 10 | 33% |
| timeout / long-running code | 8 | 27% |
| FileNotFoundError / wrong path assumption | 6 | 20% |
| read-only filesystem / archive tooling | 4 | 13% |
| no submission written | 2 | 7% |

Nodags is bad too, but in a different way. There are **zero server-disconnect traces**, yet there are many first-turn allocator failures instead:

> `Server error '500 Internal Server Error' for url 'http://localhost:5000/allocate'`

For example, `inclusive-images-challenge` hits that error on turn 1, retries the same load cell, gets the same 500 again, and ends by narrating a hypothetical plan rather than making progress.

### 2.3 Exhaustive enumeration — every gpt-4o-mini no-submit case

#### DAGS (45 no-submits)

| Competition | Turns | Exec min | Primary cause | Evidence |
|---|---:|---:|---|---|
| LANL-Earthquake-Prediction | 6 | 117.6 | server disconnect + FileNotFoundError | disconnect x3; FNF x3 |
| ashrae-energy-prediction | 15 | 123.8 | server disconnect + FileNotFoundError | disconnect x3; FNF x4; timeout x2; NameError x2 |
| champs-scalar-coupling | 9 | 116.3 | server disconnect | disconnect x2; timeout x1; NameError x2 |
| data-science-bowl-2018 | 15 | 49.6 | read-only filesystem / archive tooling | FNF x8; read-only FS |
| elo-merchant-category-recommendation | 15 | 47.4 | no submission written | none explicit |
| gendered-pronoun-resolution | 13 | 15.0 | logic/runtime error | NameError x4 |
| geolifeclef-2024 | 15 | 2.9 | FileNotFoundError / wrong path assumption | FNF x8 |
| google-smartphone-decimeter-challenge | 15 | 12.5 | no submission written | none explicit |
| humpback-whale-identification | 12 | 57.3 | context window exceeded | FNF x10; ctx overflow |
| ieee-fraud-detection | 15 | 172.1 | server disconnect | disconnect x3; NameError x2 |
| imaterialist-challenge-fashion-2018 | 15 | 132.4 | server disconnect + FileNotFoundError | disconnect x3; FNF x4; read-only FS; NameError x2 |
| imaterialist-challenge-furniture-2018 | 8 | 110.9 | server disconnect | disconnect x2; read-only FS; NameError x2 |
| inclusive-images-challenge | 4 | 117.5 | server disconnect | disconnect x3 |
| m5-forecasting-accuracy | 15 | 120.3 | server disconnect | disconnect x3; NameError x2 |
| m5-forecasting-uncertainty | 7 | 134.7 | server disconnect | disconnect x3 |
| march-machine-learning-mania-2023 | 15 | 237.6 | server disconnect | disconnect x3; timeout x1 |
| march-machine-learning-mania-2025 | 15 | 119.9 | server disconnect | disconnect x3; NameError x6 |
| mens-machine-learning-competition-2018 | 15 | 172.9 | server disconnect + FileNotFoundError | disconnect x3; FNF x10; NameError x2 |
| mens-machine-learning-competition-2019 | 15 | 172.6 | server disconnect + FileNotFoundError | disconnect x3; FNF x3; NameError x2 |
| mens-march-mania-2022 | 15 | 179.9 | server disconnect + FileNotFoundError | disconnect x3; FNF x4 |
| microsoft-malware-prediction | 7 | 113.5 | server disconnect | disconnect x2 |
| nlp-getting-started | 15 | 100.4 | server disconnect | disconnect x2 |
| novozymes-enzyme-stability-prediction | 6 | 110.0 | server disconnect | disconnect x3 |
| open-problems-single-cell-perturbations | 15 | 103.7 | server disconnect | disconnect x3 |
| otto-recommender-system | 7 | 106.6 | server disconnect | disconnect x3 |
| planttraits2024 | 7 | 73.9 | server disconnect | disconnect x2 |
| predict-ai-model-runtime | 15 | 63.9 | server disconnect | disconnect x2; NameError x24 |
| recruit-restaurant-visitor-forecasting | 8 | 114.7 | server disconnect + FileNotFoundError | disconnect x2; FNF x16; NameError x2 |
| rsna-pneumonia-detection-challenge | 13 | 11.9 | server disconnect | disconnect x1 |
| santander-customer-transaction-prediction | 13 | 70.6 | timeout / long-running code | timeout x2; NameError x2 |
| santander-value-prediction-challenge | 15 | 65.1 | logic/runtime error | NameError x2 |
| siim-acr-pneumothorax-segmentation | 14 | 65.1 | FileNotFoundError / wrong path assumption | FNF x4; NameError x4 |
| sp-society-camera-model-identification | 12 | 14.8 | no submission written | none explicit |
| spaceship-titanic | 13 | 136.1 | timeout / long-running code | timeout x1; NameError x2 |
| stanford-covid-vaccine | 15 | 15.0 | logic/runtime error | NameError x3 |
| statoil-iceberg-classifier-challenge | 15 | 10.9 | read-only filesystem / archive tooling | FNF x25; read-only FS |
| talkingdata-adtracking-fraud-detection | 8 | 73.4 | timeout / long-running code | timeout x2 |
| tensorflow-speech-recognition-challenge | 9 | 75.2 | read-only filesystem / archive tooling | FNF x5; read-only FS |
| tgs-salt-identification-challenge | 10 | 71.3 | read-only filesystem / archive tooling | FNF x8; read-only FS; NameError x2 |
| understanding_cloud_organization | 15 | 23.3 | no submission written | none explicit |
| ventilator-pressure-prediction | 10 | 67.5 | timeout / long-running code | timeout x1; NameError x2 |
| web-traffic-time-series-forecasting | 14 | 42.7 | read-only filesystem / archive tooling | FNF x3; read-only FS; NameError x4 |
| womens-machine-learning-competition-2019 | 15 | 41.8 | no submission written | none explicit |
| youtube8m-2018 | 15 | 40.7 | no submission written | none explicit |
| youtube8m-2019 | 3 | 39.3 | early stop / no recovery | none explicit |

#### NODAGS (30 no-submits)

| Competition | Turns | Exec min | Primary cause | Evidence |
|---|---:|---:|---|---|
| LANL-Earthquake-Prediction | 15 | 181.3 | timeout / long-running code | timeout x4; NameError x8 |
| ashrae-energy-prediction | 15 | 183.7 | FileNotFoundError / wrong path assumption | FNF x3; timeout x5; NameError x6 |
| champs-scalar-coupling | 15 | 177.5 | timeout / long-running code | timeout x3; NameError x6 |
| data-science-bowl-2018 | 15 | 2.1 | read-only filesystem / archive tooling | FNF x43; read-only FS |
| digit-recognizer | 13 | 130.1 | timeout / long-running code | timeout x2; NameError x4 |
| gendered-pronoun-resolution | 15 | 3.6 | FileNotFoundError / wrong path assumption | FNF x6 |
| geolifeclef-2024 | 15 | 359.7 | 500 allocate error | alloc500 x1; FNF x1; timeout x5; NameError x4 |
| ieee-fraud-detection | 15 | 60.7 | no submission written | none explicit |
| imaterialist-challenge-furniture-2018 | 3 | 180.7 | 500 allocate error | alloc500 x1 |
| inclusive-images-challenge | 4 | 241.1 | 500 allocate error | alloc500 x2 |
| liverpool-ion-switching | 5 | 240.8 | 500 allocate error | alloc500 x1 |
| m5-forecasting-uncertainty | 15 | 260.6 | 500 allocate error | alloc500 x1 |
| march-machine-learning-mania-2025 | 3 | 180.8 | 500 allocate error | alloc500 x1 |
| mens-machine-learning-competition-2019 | 4 | 241.0 | 500 allocate error | alloc500 x2 |
| mens-march-mania-2022 | 15 | 60.1 | FileNotFoundError / wrong path assumption | FNF x3 |
| microsoft-malware-prediction | 15 | 180.0 | timeout / long-running code | timeout x3 |
| otto-recommender-system | 15 | 422.0 | timeout / long-running code | timeout x8; NameError x4 |
| planttraits2024 | 15 | 110.5 | timeout / long-running code | timeout x2 |
| recruit-restaurant-visitor-forecasting | 15 | 39.8 | read-only filesystem / archive tooling | FNF x3; read-only FS |
| siim-acr-pneumothorax-segmentation | 15 | 102.8 | read-only filesystem / archive tooling | FNF x9; read-only FS; NameError x6 |
| sp-society-camera-model-identification | 15 | 100.1 | timeout / long-running code | timeout x2; NameError x4 |
| stanford-covid-vaccine | 15 | 160.4 | timeout / long-running code | timeout x2; NameError x2 |
| tensorflow-speech-recognition-challenge | 15 | 159.6 | 500 allocate error | alloc500 x1; FNF x5; read-only FS |
| tgs-salt-identification-challenge | 15 | 160.4 | 500 allocate error | alloc500 x1; FNF x11; read-only FS; NameError x4 |
| trec-covid-information-retrieval | 15 | 60.0 | FileNotFoundError / wrong path assumption | FNF x9 |
| understanding_cloud_organization | 15 | 65.1 | FileNotFoundError / wrong path assumption | FNF x4 |
| vsb-power-line-fault-detection | 5 | 152.3 | 500 allocate error | alloc500 x1 |
| web-traffic-time-series-forecasting | 15 | 43.8 | FileNotFoundError / wrong path assumption | FNF x3 |
| youtube8m-2018 | 15 | 26.1 | read-only filesystem / archive tooling | FNF x2; read-only FS |
| youtube8m-2019 | 15 | 29.5 | no submission written | none explicit |

### 2.4 Common no-submit motifs

Four recurring motifs are visible across those tables:

1. **Path hallucination / wrong file schema**. `geolifeclef-2024` assumes `PresenceAbsenceSurveys/GLC24_PA_metadata_train.csv`; `siim-acr-pneumothorax-segmentation` assumes `train.csv`; `LANL` assumes a monolithic `test/test.csv` instead of per-segment files.
2. **Environment/tooling mismatch**. `data-science-bowl-2018` and `statoil-iceberg-classifier-challenge` both run into archive/extraction problems and read-only storage, e.g. `OSError: [Errno 30] Read-only file system`.
3. **Timeout → state-loss loop**. In tasks like `ashrae`, `otto`, and `santander-customer-transaction-prediction`, a long training cell times out and the next revision fails with `NameError` because the notebook state the model expected is gone.
4. **Format drift**. `understanding_cloud_organization`, `youtube8m-2018`, and `youtube8m-2019` literally stop speaking the executor's dialect. The environment replies: `No python code found. You should either write python code within the <python>CODE</python> tag or write the answer in the <answer>ANSWER</answer> tag.` On `understanding_cloud_organization`, 4o-mini repeats raw fenced Python twice after being told this, never recovering.

---

## 3. Submission error analysis

### 3.1 gpt-4o-mini / dags errors (4)

| Error pattern | Count | Competitions |
|---|---:|---|
| key mismatch | 1 | liverpool-ion-switching |
| liverpool schema leak | 3 | pku-autonomous-driving, trec-covid-information-retrieval, vsb-power-line-fault-detection |

This is a striking contamination pattern. Three different competitions (`pku-autonomous-driving`, `trec-covid-information-retrieval`, `vsb-power-line-fault-detection`) were all rejected with the **same leaked Liverpool schema**:

> `Unexpected Column: 'time' ... Unexpected Column: 'open_channels' ... Required column 'ImageId' / 'topic-id' / 'signal_id' could not be found`

That is not a modeling mistake inside those competitions; it is almost certainly a stale submission artifact.

### 3.2 gpt-4o-mini / nodags errors (14)

| Error pattern | Count | Competitions |
|---|---:|---|
| google-smartphone schema leak | 7 | google-smartphone-decimeter-challenge, home-credit-default-risk, home-data-for-ml-course, humpback-whale-identification, imaterialist-challenge-fashion-2018, m5-forecasting-accuracy, mens-machine-learning-competition-2018 |
| row count mismatch | 3 | march-machine-learning-mania-2023, predict-ai-model-runtime, womens-machine-learning-competition-2019 |
| wrong columns | 1 | open-problems-single-cell-perturbations |
| open-problems schema leak | 2 | statoil-iceberg-classifier-challenge, store-sales-time-series-forecasting |
| ventilator schema leak | 1 | talkingdata-adtracking-fraud-detection |

Nodags has the same contamination story, but worse. Seven unrelated tasks were rejected with the **Google Smartphone** header (`seq_id`, `tm`). Two others leaked the **Open Problems gene-expression** schema (`A1BG`, `Gene_2`, ...), and `talkingdata-adtracking-fraud-detection` leaked the **Ventilator** schema (`id`, `pressure`).

The most damning example is `google-smartphone-decimeter-challenge` itself. The final assistant turn is still exploring `Pixel4_derived.csv`; there is **no submission-writing step at all**. Yet the run is scored as `SUBMISSIONSTATUS.ERROR` with:

> `Unexpected Column: 'seq_id' ... Required column 'phone' could not be found`

This is strong evidence of **cross-run stale submission pickup**, not an in-trace modeling error.

### 3.3 GPT-5.1 errors: mostly real evaluator rejections, plus a few stale-file leaks

#### gpt-5.1 / dags
| Error pattern | Count | Competitions |
|---|---:|---|
| row count mismatch | 6 | champs-scalar-coupling, gendered-pronoun-resolution, mens-machine-learning-competition-2018, planttraits2024, web-traffic-time-series-forecasting, womens-machine-learning-competition-2019 |
| ID mismatch | 1 | data-science-bowl-2018 |
| wrong columns | 1 | imaterialist-challenge-fashion-2018 |
| key mismatch | 1 | liverpool-ion-switching |
| champs schema leak | 1 | m5-forecasting-accuracy |
| segmentation schema leak | 1 | recruit-restaurant-visitor-forecasting |
| ventilator schema leak | 1 | ventilator-pressure-prediction |

#### gpt-5.1 / nodags
| Error pattern | Count | Competitions |
|---|---:|---|
| row count mismatch | 5 | champs-scalar-coupling, march-machine-learning-mania-2025, mens-machine-learning-competition-2018, planttraits2024, web-traffic-time-series-forecasting |
| ID mismatch | 1 | data-science-bowl-2018 |
| key mismatch | 1 | liverpool-ion-switching |
| champs schema leak | 1 | mens-machine-learning-competition-2019 |
| invalid inf values | 1 | santander-value-prediction-challenge |
| wrong columns | 1 | tensorflow-speech-recognition-challenge |

GPT-5.1's error profile is much cleaner. Most failures are genuine evaluator rejections: wrong row counts (`champs`, `planttraits2024`, `web-traffic-time-series-forecasting`), ID mismatch (`data-science-bowl-2018`), or the recurrent Liverpool key mismatch. But three dags errors still look stale-file-like: `m5-forecasting-accuracy` leaks `scalar_coupling_constant`, `recruit-restaurant-visitor-forecasting` leaks `rle_mask`, and `ventilator-pressure-prediction` leaks `Class,Segments`.

---

## 4. DAG format & instruction following

| Pattern | 4o/dags | 5.1/dags | Claude/dags | 4o/nodags | 5.1/nodags | Claude/nodags |
|---|---:|---:|---:|---:|---:|---:|
| <reasoning> | 54/54 | 53/53 | 35/35 | 54/54 | 54/54 | 35/35 |
| <python> | 54/54 | 52/53 | 35/35 | 54/54 | 54/54 | 35/35 |
| <answer> | 37/54 | 53/53 | 21/35 | 15/54 | 54/54 | 5/35 |
| DAG JSON | 54/54 | 53/53 | 35/35 | 0/54 | 0/54 | 0/35 |
| PLAN REVISION | 48/54 | 52/53 | 18/35 | 0/54 | 0/54 | 0/35 |
| Step N/M | 54/54 | 53/53 | 35/35 | 0/54 | 0/54 | 0/35 |
| Execution summary | 28/54 | 53/53 | 18/35 | 0/54 | 0/54 | 0/35 |

Key observations:

- **gpt-4o-mini does emit DAGs**: all **54/54** dags runs contain DAG JSON, Step N/M, and `<reasoning>` blocks.
- But its execution discipline is much weaker than GPT-5.1's. Only **37/54** 4o-mini dags runs produce `<answer>`, and only **28/54** produce an `Execution summary`. GPT-5.1 is almost perfectly compliant: **53/53** answers and summaries.
- **4o-mini's DAGs are verbose and generic.** The typical first turn is a two-object DAG followed by ambitious steps like “train baseline models”, “evaluate models and perform hyperparameter tuning”, and “validate submission format” before the model has correctly identified the data layout.
- **Plan revision is anti-correlated with success for 4o-mini**. Average revisions: **2.6** for complete runs, **6.5** for no-submits, **8.75** for error runs. For GPT-5.1 dags, the direction reverses: complete runs average **5.0** revisions vs **3.5** for no-submit.
- Claude is the lightest DAG user: only **0.77** average plan revisions. It seems to write the DAG once and then just execute.

One subtle formatting difference also matters: 4o-mini and Claude generally wrap DAGs in fenced ```json blocks, while GPT-5.1 usually emits raw JSON objects directly. GPT-5.1's formatting is actually *more* stable downstream; 4o-mini is the model that later falls into `No python code found`, repeated re-DAGging, and 400-session errors.

---

## 5. Server disconnect pattern (gpt-4o-mini specific)

The server-disconnect failure is genuinely dags-specific in these exports.

- **gpt-4o-mini/dags**: **25/54 runs** with at least one disconnect (**46%**)
- **gpt-4o-mini/nodags**: **0/54** runs with disconnects
- Avg disconnects per affected dags run: **2.6**
- Outcomes among affected dags runs: **23 no-submit, 2 error, 0 complete**
- First disconnect turn distribution: **20/25 on the first execution turn**, then one each at turns 2, 4, 5, 6, and 11.

This supports the resource-overhead hypothesis: most dags disconnects happen immediately after the initial DAG emission and first large “load everything” code cell. The best compact example is `LANL`, where the trace goes:

1. emit full DAG,
2. try `test/test.csv`,
3. get `FileNotFoundError`,
4. revise plan,
5. then hit `Server disconnected without sending a response.` three times,
6. finish at `KernelError: Kernel not ready`.

The later-turn disconnects look different. In `ashrae`, the first disconnect does not happen until turn 11, after a sequence of file mistakes, a one-hour timeout (`Execution timed out after 3600 seconds`), and then `NameError` caused by state loss. That is not a simple allocator failure; it looks like the environment collapsing under a longer, more complex run.

Important nuance: **nodags is not infrastructure-clean**. It avoids disconnects, but it has a different issue: **21/54 runs** show `/allocate` **500 Internal Server Error** messages on the first turn. So the right conclusion is not “only dags had infra problems”; it is “**dags had the much worse mid-run disconnect pathology**.”

All 25 disconnect-affected dags runs attempt recovery — the model always emits another `<python>` block after the first disconnect — but none recover to a complete submission. The recovery behavior is therefore mostly **looping**, not actual resilience.

---

## 6. FileNotFoundError pattern and recovery behavior

| Run | Runs with FileNotFoundError | Recovery to COMPLETE | Recovery rate |
|---|---:|---:|---:|
| gpt-4o-mini/dags | 16/54 | 1 | 6% |
| gpt-4o-mini/nodags | 24/54 | 3 | 12% |
| gpt-5.1/dags | 22/53 | 11 | 50% |
| gpt-5.1/nodags | 30/54 | 22 | 73% |

This is one of the clearest model-capability gaps in the whole dataset.

- 4o-mini hits FileNotFoundError a bit less often in dags than GPT-5.1, but it almost never converts those runs into success (**1/16** dags, **3/24** nodags).
- GPT-5.1 hits FileNotFoundError *more often* because it probes and validates more aggressively — but it recovers far better (**11/22** dags, **22/30** nodags).

The qualitative difference is visible in the traces.

**4o-mini failure style**:
- `LANL`: assumes `/test/test.csv`.
- `geolifeclef-2024`: assumes `PresenceAbsenceSurveys/GLC24_PA_metadata_train.csv`.
- `trec-covid-information-retrieval`: hardcodes a nonexistent paper JSON (`010vptx3.json`) instead of enumerating available files.

**GPT-5.1 recovery style**:
- In `gendered-pronoun-resolution`, GPT-5.1 actually writes `/submission/submission.csv`, then fails only because it validates against a bad sampled path: `FileNotFoundError: ... /data/dspredict-sampled/gendered-pronoun-resolution/sample_submission.csv`. The next reasoning turn explicitly says it should inspect `/data/dspredict-sampled` and locate the real sample file before overwriting the submission. That is corrective reasoning, not thrashing.

So the issue is not merely “path errors happen”; it is **what the model does next**. GPT-5.1 explores, verifies, and re-aligns. 4o-mini often just rewrites the same assumption with minor textual changes.

---

## 7. Strategy analysis — what each model actually does

### gpt-4o-mini / dags
Typical pattern: generic DAG → immediate bulk load → RandomForest/logistic-regression baseline → timeout/disconnect/revision loop. The first plan is often more ambitious than the model can execute. `LANL` opens with a 7-node DAG ending in “hyperparameter tuning” and “validate submission format” before it has even learned that test data are split into many files.

### gpt-4o-mini / nodags
Still weak, but slightly less self-sabotaging. Without the DAG bureaucracy it is more likely to directly inspect a dataset and occasionally finish (`elo`, `nlp-getting-started`, `spaceship-titanic`, `ventilator`). It still overuses RandomForest and often runs long monolithic cells that time out.

### gpt-5.1 / nodags
Best operational profile. It usually starts with **filesystem inspection + shape checks + memory-aware loading**, then chooses stronger tabular models (LightGBM/XGBoost) and more realistic validation schemes. Example from `m5-forecasting-accuracy`: it first inspects `calendar.csv`, `sales_train_validation.csv`, `sales_train_evaluation.csv`, `sell_prices.csv`, and `sample_submission.csv`, then explicitly reasons about memory and whether a full long-panel transform is feasible.

### gpt-5.1 / dags
The same technical instincts are visible, but DAG mode pushes it toward longer plans: more quantile-expansion steps, more explicit validation plans, more mid-run bookkeeping. When everything fits, quality is similar to nodags; when it does not, the extra scaffolding becomes context and execution overhead.

### Claude Sonnet 4.5 / dags
Claude uses DAGs as a concise execution scaffold. The `LANL` dags trace is a good example: small DAG, then direct feature engineering, LightGBM, regularization, and final submission. The DAG does not explode into repeated revisions.

### Claude Sonnet 4.5 / nodags
Claude nodags is often more meandering. On `LANL`, the run gets all the way to “Train final Random Forest model on all training data...” and then simply stops without producing predictions or a submission. DAG mode appears to help Claude keep a strict end-to-end structure.

---

## 8. DAG effect by model

### 8.1 gpt-4o-mini: DAGs are catastrophically harmful

- dags complete rate: **5/54 = 9%**
- nodags complete rate: **10/54 = 19%**

The trace evidence supports four hypotheses simultaneously:

1. **Over-ambitious planning.** DAGs encourage 4o-mini to promise full preprocessing, model selection, validation, tuning, and submission validation up front.
2. **Context and formatting overhead.** 4o-mini/dags has many more revision turns, more re-DAGging, and far more executor-format failures (`No python code found`, `400 Bad Request`, `Kernel not ready`).
3. **Poor divergence handling.** Once the initial plan is invalidated by a path error or timeout, the model revises the prose plan but often does not truly simplify the code path.
4. **Infrastructure amplification.** The 25 dags disconnects are exactly the kind of failure you would expect when a smaller model is pushed into long, verbose plan-execute-replan loops.

### 8.2 gpt-5.1: DAGs still hurt, but mainly by overhead

- dags: **28/53 = 53%**
- nodags: **42/54 = 78%**

GPT-5.1 clearly understands the competitions better than 4o-mini, so its DAG tax shows up less as confusion and more as **context blowups**, extra plan steps, and a few stale-file/infrastructure failures. `humpback-whale-identification` is the cleanest example: after ambitious image-feature extraction, the run starts failing with `ContextWindowExceededError ... 344810 tokens` and never gets past turn 2.

### 8.3 Claude Sonnet 4.5: DAGs help

- dags: **20/35 = 57%**
- nodags: **11/35 = 31%**

Claude's DAGs are shorter, less revision-heavy, and more execution-oriented. It seems to use the DAG as scaffolding rather than as a second task layered on top of the original one. That is why the same protocol that cripples 4o-mini improves Claude.

### 8.4 Counter-evidence

Not every 4o-mini dags failure is “too much planning”; some are plain infrastructure failures, and nodags has its own allocator problems. Likewise, GPT-5.1 dags still wins on four tasks (`elo`, `mens-machine-learning-competition-2019`, `santander-value-prediction-challenge`, `youtube8m-2019`). So the DAG protocol is not universally bad; it is **model-dependent**.

---

## 9. Per-competition cross-model comparison

Legend: **C** = complete, **E** = submission error, **NS** = no submit, **—** = missing run (`santander-customer-transaction-prediction` has no gpt-5.1/dags file in this folder).

| Competition | 4omini/dags | 4omini/nodags | gpt51/dags | gpt51/nodags |
|---|---:|---:|---:|---:|
| LANL-Earthquake-Prediction | NS | NS | C | C |
| ashrae-energy-prediction | NS | NS | C | C |
| career-con-2019 | C | C | C | C |
| champs-scalar-coupling | NS | NS | E | E |
| data-science-bowl-2018 | NS | NS | E | E |
| digit-recognizer | C | NS | C | C |
| elo-merchant-category-recommendation | NS | C | C | NS |
| gendered-pronoun-resolution | NS | NS | E | C |
| geolifeclef-2024 | NS | NS | C | C |
| google-smartphone-decimeter-challenge | NS | E | NS | C |
| home-credit-default-risk | C | E | C | C |
| home-data-for-ml-course | C | E | C | C |
| humpback-whale-identification | NS | E | NS | C |
| ieee-fraud-detection | NS | NS | NS | C |
| imaterialist-challenge-fashion-2018 | NS | E | E | C |
| imaterialist-challenge-furniture-2018 | NS | NS | C | C |
| inclusive-images-challenge | NS | NS | NS | C |
| liverpool-ion-switching | E | NS | E | E |
| m5-forecasting-accuracy | NS | E | E | C |
| m5-forecasting-uncertainty | NS | NS | NS | C |
| march-machine-learning-mania-2023 | NS | E | NS | C |
| march-machine-learning-mania-2025 | NS | NS | NS | E |
| mens-machine-learning-competition-2018 | NS | E | E | E |
| mens-machine-learning-competition-2019 | NS | NS | C | E |
| mens-march-mania-2022 | NS | NS | NS | C |
| microsoft-malware-prediction | NS | NS | NS | C |
| nlp-getting-started | NS | C | C | C |
| novozymes-enzyme-stability-prediction | NS | C | C | C |
| open-problems-single-cell-perturbations | NS | E | C | C |
| otto-recommender-system | NS | NS | C | C |
| pku-autonomous-driving | E | C | C | C |
| planttraits2024 | NS | NS | E | E |
| predict-ai-model-runtime | NS | E | C | C |
| recruit-restaurant-visitor-forecasting | NS | NS | E | C |
| rsna-pneumonia-detection-challenge | NS | C | C | C |
| santander-customer-transaction-prediction | NS | C | — | C |
| santander-value-prediction-challenge | NS | C | C | E |
| siim-acr-pneumothorax-segmentation | NS | NS | C | C |
| sp-society-camera-model-identification | NS | NS | NS | C |
| spaceship-titanic | NS | C | C | C |
| stanford-covid-vaccine | NS | NS | C | C |
| statoil-iceberg-classifier-challenge | NS | E | NS | C |
| store-sales-time-series-forecasting | C | E | C | C |
| talkingdata-adtracking-fraud-detection | NS | E | C | C |
| tensorflow-speech-recognition-challenge | NS | NS | NS | E |
| tgs-salt-identification-challenge | NS | NS | C | C |
| trec-covid-information-retrieval | E | NS | C | C |
| understanding_cloud_organization | NS | NS | NS | C |
| ventilator-pressure-prediction | NS | C | E | C |
| vsb-power-line-fault-detection | E | NS | C | C |
| web-traffic-time-series-forecasting | NS | NS | E | E |
| womens-machine-learning-competition-2019 | NS | E | E | C |
| youtube8m-2018 | NS | NS | C | C |
| youtube8m-2019 | NS | NS | C | NS |

The table has a strong asymmetry:
- There are **no outright small-over-large upsets** where 4o-mini completed a task that GPT-5.1 failed in both modes.
- There are **32 competitions** where GPT-5.1 completed at least one mode and 4o-mini completed neither.
- The nearest 4o-mini “wins” are mode-local only: `elo-merchant-category-recommendation` and `santander-value-prediction-challenge` are **4o-mini/nodags complete** while **gpt-5.1/nodags** is worse — but GPT-5.1/dags still completes both.

---

## 10. Token usage & efficiency

| Mode | Outcome | Avg input | Avg output | Avg total |
|---|---|---:|---:|---:|
| gpt-4o-mini/dags | complete | 90447 | 3959 | 94406 |
| gpt-4o-mini/dags | error | 56397 | 4785 | 61182 |
| gpt-4o-mini/dags | no-submit | 56380 | 3118 | 59498 |
| gpt-4o-mini/nodags | complete | 60166 | 3063 | 63229 |
| gpt-4o-mini/nodags | error | 43729 | 887 | 44616 |
| gpt-4o-mini/nodags | no-submit | 90605 | 3002 | 93607 |

Two token stories stand out.

1. **4o-mini/nodags is not cheaper in failure.** Its no-submit runs average **93.6k total tokens**, much higher than its complete runs (**63.2k**). The model often burns budget in long timeout/retry loops.
2. **4o-mini/dags spends less than nodags overall, but that is not a virtue.** Average total tokens are **62.9k** in dags vs **75.3k** in nodags, yet dags is much worse. It is “more efficient” only in the trivial sense that many dags runs die early.

The strongest efficiency-capability lesson is therefore not “small model = cheaper.” It is: **cheap tokens are worthless when the model cannot recover from environment friction.**

---

## 11. Qualitative failure-mode deep dives

### 11.1 `LANL-Earthquake-Prediction` — 4o-mini/dags: perfect DAG, immediate collapse
The run opens with a textbook DAG, including “Select and train baseline models” and “Evaluate models and perform hyperparameter tuning.” The very first code cell then assumes `'/data/dspredict-hard/LANL-Earthquake-Prediction/test/test.csv'`, causing `FileNotFoundError`. After a plan revision, execution degrades into three consecutive `Server disconnected without sending a response.` errors and ends at `KernelError: Kernel not ready`. This is the archetypal 4o-mini DAG failure: strong surface structure, weak environment-grounded execution.

### 11.2 `inclusive-images-challenge` — 4o-mini/nodags: allocator dead on arrival
This run never really starts. Turn 1 returns `Server error '500 Internal Server Error' for url 'http://localhost:5000/allocate'`; turn 2 retries the same dataset load; turn 3 gets the same 500; the assistant then switches into hypothetical narration. It is a pure infrastructure failure, but it matters because nodags accumulates many of these first-turn allocator deaths.

### 11.3 `understanding_cloud_organization` — 4o-mini/dags: format drift after partial progress
Late in the run the environment says, `No python code found. You should either write python code within the <python>CODE</python> tag...`. Instead of repairing the wrapper, 4o-mini replies with raw fenced Python twice in a row. This is one of the clearest pieces of evidence that 4o-mini's DAG discipline is brittle: once the trace goes off the rails, the model cannot stay within the executor protocol.

### 11.4 `google-smartphone-decimeter-challenge` — 4o-mini/nodags: stale submission contamination
The last assistant turn is still exploring `Pixel4_derived.csv` and preparing correlations; there is no sign of a finished submission. Yet the official error is `Unexpected Column: 'seq_id' ... Required column 'phone' could not be found`. This is almost certainly a stale submission file from another run being scored under the Google Smartphone run.

### 11.5 `open-problems-single-cell-perturbations` — 4o-mini/nodags: context blowup plus wrong-file submission
The trajectory accumulates repeated `ContextWindowExceededError` messages with absurd prompt sizes (`1,827,060 tokens`). Despite that, the run ends as a submission **error** with a gene-expression-shaped file. This looks like a two-layer failure: context explosion prevents the actual task from finishing, and then a stale or partial file is submitted anyway.

### 11.6 `humpback-whale-identification` — gpt-5.1/dags: ambitious plan meets context ceiling
GPT-5.1's DAG here is technically reasonable but long: metadata inspection, simple image features, validation design, boosting model, top-5 formatting. It gets through feature extraction, then starts throwing `ContextWindowExceededError ... 344810 tokens`. This is the best example of DAG overhead hurting a strong model: the reasoning is sound, but the trace becomes too large.

### 11.7 `m5-forecasting-uncertainty` — gpt-5.1/dags: technically capable, operationally unlucky
This run is a good example of “moderate degradation, not confusion.” GPT-5.1 correctly loads the M5 files, reshapes a 6M-row window, engineers sensible features, and writes `/submission/submission.csv`. It then hits an old-LightGBM API mismatch (`TypeError: train() got an unexpected keyword argument 'evals_result'`) and later a validation-path issue (`FileNotFoundError` on sampled `sample_submission.csv`). The model is doing the *right kind* of work; the run still dies because the DAG path is long and brittle.

### 11.8 `ieee-fraud-detection` — gpt-5.1/dags: robust data engineering, fatal library mismatch
The first DAG is exactly what you want: memory constraints, dtype optimization, time-aware validation, LightGBM baseline, feature refinement. The assistant even normalizes train/test column mismatches like `id-01` vs `id_01`. But the environment's LightGBM API is older than expected (`train() got an unexpected keyword argument 'early_stopping_rounds'`), and a later schema assertion kills the run. This is a strong failure trace, not a weak one.

### 11.9 `LANL-Earthquake-Prediction` — Claude dags vs nodags: DAG as real scaffold
Claude/dags cleanly builds a small DAG, engineers 23 statistical features, trains a regularized LightGBM, and completes. Claude/nodags gets surprisingly far — all the way to “Training final Random Forest model on all training data...” — then simply stops without generating predictions. This pair is the clearest evidence that Claude benefits from the DAG scaffold in a way GPT models do not.

---

## 12. Final hypotheses

### 12.1 gpt-4o-mini + DAGs = catastrophic degradation
Supported. The traces show that 4o-mini follows the DAG instructions literally enough to create long, ceremonious plans, but not strongly enough to keep execution grounded. The cost is visible in three places: **plan-revision inflation (6.3 avg), disconnect frequency (25/54), and protocol drift (24 `No python code found`, 20 `400 Bad Request`, 26 `Kernel not ready`).**

### 12.2 gpt-5.1 + DAGs = moderate degradation
Also supported. GPT-5.1 usually remains technically competent under DAG prompting, but the extra structure increases trace length, context pressure, and the number of places where infrastructure/library quirks can break the run. When a GPT-5.1 dags run fails, it usually fails **late and intelligently**, not early and confused.

### 12.3 Claude Sonnet 4.5 + DAGs = improvement
Supported on the sampled benchmark. Claude writes shorter DAGs, revises less, and appears to use the DAG as a control structure rather than as another content-generation objective. The `LANL` pair is especially persuasive here.

### 12.4 What is the most likely causal story?
The simplest story consistent with all three models is:

- **Claude** treats DAG prompting as *light scaffolding*.
- **GPT-5.1** treats it as *useful but expensive bookkeeping*.
- **gpt-4o-mini** treats it as *a second task that competes with the real task*.

That is why the same protocol helps Claude, mildly harms GPT-5.1, and catastrophically harms 4o-mini.

---

## 13. Server Disconnect Root Cause Analysis

**The short answer: server disconnects are container session timeouts caused by the long DAG generation time, not by heavy code execution.**

### 13.1 The mechanism

20 of 25 disconnects happen on **turn 1**, immediately after the model's first response. In 8 of those 25 cases the Python cell that triggered the disconnect was just `# No code needed for this step` — a placeholder. The container died before any real computation ran, ruling out resource exhaustion from an ML job.

The causal chain:

1. The model generates its full DAG response first — two JSON objects, step-by-step plan, implementation nodes — averaging **6,447 chars (~1,600 tokens)** for the first message.
2. That generation takes noticeably longer than a nodags first response (avg **5,218 chars**, and often much less since nodags skips the JSON preamble).
3. While the model is generating, the kernel/executor session that was allocated is sitting idle. If generation exceeds the session's keepalive timeout, the kernel drops — and the first `<python>` execution, even a trivial one, hits a dead connection.

This explains why nodags never hits disconnects: first messages are shorter and more variable (min 1,008 chars), so the model generates fast enough that the kernel is still alive when execution is requested.

### 13.2 Infrastructure pressure vs generation latency — two failure populations

Crossing the disconnect and 500-error competition sets reveals two distinct populations:

| Population | Count | Pattern |
|---|---:|---|
| Both modes hit infra problems (disconnect in dags, 500 in nodags) | **12** | Heavy-data competitions — imaterialist images, m5, march mania, open-problems gene expression. Infra overloaded regardless; DAG latency makes it worse. |
| Disconnect in dags only (nodags succeeds or hits different error) | **13** | Moderate-size competitions — LANL, ieee-fraud, ashrae, etc. These work fine in nodags. The disconnect is generation-latency-induced. |
| 500 in nodags only (no dags disconnect) | **9** | Competitions where nodags hits allocator saturation but dags avoids it (likely different container pool scheduling). |

The 12-competition overlap is the clearest proof that some competitions simply stress the infrastructure regardless of mode. For the 13 dags-only disconnect competitions, the generation overhead of the DAG preamble is the proximate cause.

### 13.3 Why gpt-4o-mini is especially vulnerable

gpt-4o-mini's DAG first messages are strikingly uniform: every single one is 5,947–6,790 chars with 15–18 nodes, essentially the same generic 7-step waterfall DAG repeated. This ceiling on message length (likely because the model hits a natural output-length budget) means the generation time is consistently high and consistent across all competitions, regardless of dataset size. GPT-5.1, by contrast, emits shorter and more targeted DAGs, giving it less session-idle time.

---

## 14. Claude/dags vs GPT-5.1/nodags — Path Exploration and the DAG as Constraint

These two configurations are the best performers and they succeed via opposite strategies.

### 14.1 GPT-5.1 nodags — reactive, inspection-first

Turn 1 is almost always an explicit filesystem probe before any data load:

```python
# LANL turn 1:
print("Contents of base path:", os.listdir(base_path))

# elo turn 1 reasoning: "I won't touch the big transaction tables yet; I want a quick overview"

# gendered-pronoun turn 1:
for root, dirs, files in os.walk(base_dir):
    ...
```

GPT-5.1 deliberately stages exploration: *inspect schema → sample small → decide memory strategy → commit to a model*. In `ashrae` it explicitly reasons *"The data is large, so I need to be careful with memory"* before deciding on LightGBM. In `elo` it loads only train/test first, then peeks at transaction tables with `nrows=1000` before committing to a join strategy.

**DAGs are not a constraint for GPT-5.1 because GPT-5.1 doesn't use them for navigation.** When DAG prompting is applied, GPT-5.1 writes the DAG and then behaves almost identically to nodags — except it carries ~300–600 extra tokens of JSON context per turn that accumulate into context overflow on long runs. The DAG is bureaucratic overhead on top of an already-working reactive strategy.

### 14.2 Claude/dags — plan-first, scaffold-driven

Claude's turn 1 starts with a brief `os.listdir` but the reasoning block already shows pre-formed domain knowledge: *"The training data is a continuous segment with acoustic_data"* (LANL). Its DAGs are shorter (5–7 nodes) and **functional** — they describe what the code will actually do, not a generic waterfall. And Claude almost never revises them (avg **0.77 revisions** vs 4.5 for GPT-5.1 dags and 6.3 for 4o-mini dags).

The DAG acts as a **commitment device** for Claude. In nodags mode, Claude frequently writes correct code all the way to "train final model on all training data" and then stops without writing a submission — it wanders off the end of the task. The DAG gives it a termination condition: the last node is "write submission.csv" and Claude checks that box. The `LANL` pair is the clearest illustration: nodags Claude reaches Random Forest training and stops; dags Claude trains LightGBM with regularization, submits, and completes.

### 14.3 gpt-4o-mini with DAGs — the plan as a trap

4o-mini forms **over-ambitious, generic DAGs** and then cannot deviate from them when the environment pushes back. When a path assumption fails, it revises the prose plan but reissues the same failing code pattern. The plan revision count is **anti-correlated with success** (complete runs: 2.6 avg revisions; no-submit runs: 6.5). More revisions signal mounting distress, not adaptation — the opposite of GPT-5.1's healthy plan-revision pattern.

### 14.4 Summary: three fundamentally different relationships with the DAG protocol

| Model | DAG relationship | Result |
|---|---|---|
| **Claude Sonnet 4.5** | Uses DAG as a lightweight termination scaffold; writes short, accurate plans and executes them linearly | DAGs **help** — provides structure nodags lacks |
| **GPT-5.1** | Builds plan from environment, not from DAG; treats DAG as bookkeeping layered on top of reactive exploration | DAGs **hurt mildly** — overhead with no benefit |
| **gpt-4o-mini** | DAG becomes a second competing task; over-plans, under-executes, loops on failures | DAGs **catastrophically hurt** — plan becomes a trap |

The simplest causal story: Claude needs the scaffold to finish; GPT-5.1 doesn't need it but can tolerate it; gpt-4o-mini is overwhelmed by it.

---

## 15. Bottom line

If the goal is reliable DSPredict-Hard execution, the ordering is clear:

1. **gpt-5.1 / nodags** — strongest and most reliable.
2. **Claude Sonnet 4.5 / dags** — strong on the sampled subset; DAGs genuinely help.
3. **gpt-5.1 / dags** — still usable, but clearly worse than nodags.
4. **gpt-4o-mini / nodags** — weak but occasionally salvageable.
5. **gpt-4o-mini / dags** — not just worse, but structurally unstable.

For future benchmark runs, the practical recommendation is simple: **do not use DAG prompting with gpt-4o-mini on DSPredict-Hard.** For GPT-5.1, default to **nodags** unless there is a separate reason to value structured reasoning traces over raw completion rate. For Claude Sonnet 4.5, keep the DAG prompt.
