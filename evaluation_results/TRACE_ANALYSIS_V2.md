# DSGym Trace Analysis v2 — Post Stale-File Fix

> Analysed from `evaluation_results/` conversation logs.  
> Benchmark: **DSPredict-Sampled** (35 Kaggle competitions).  
> Models: **claude-sonnet-4-5** · **gpt-5.1** · Modes: **dags** · **nodags**

---

## Quick Summary

| Run | n | Complete | Error | No-Submit | Submit Rate |
|-----|---|----------|-------|-----------|-------------|
| claude-sonnet-4-5 / dags   | 35 | 14 | 2 | 19 | **40%** |
| claude-sonnet-4-5 / nodags | 35 |  9 | 0 | 26 | **26%** |
| gpt-5.1 / dags             | 35 | 25 | 3 |  7 | **71%** |
| gpt-5.1 / nodags           | 35 | 29 | 3 |  3 | **83%** |

GPT-5.1 achieves a **2–3× higher submission rate** than Claude Sonnet 4.5.  
The stale-file contamination that drove 83% of the previous failures has been eliminated — submission errors are now a **small fraction** of outcomes and have a different character.

---

## 1 · Submission Errors (SUBMISSIONSTATUS.ERROR)

There are now **8 error cases total** (down from the previous majority-failure pattern). Every error case involves a submission that **passed local format validation** but was rejected by the Kaggle evaluator. These are not file-format bugs — they are content bugs.

### Recurring error competitions

| Competition | Models affected | Pattern |
|-------------|----------------|---------|
| `gendered-pronoun-resolution` | Claude dags, GPT dags, GPT nodags (3 models) | Stage-1 vs Stage-2 confusion — test data has no labels, agents incorrectly target Stage 1 sample (2,000 rows) instead of Stage 2 (12,359 rows) |
| `mens-march-mania-2022` | Claude dags | Submitted 11,390 rows; correct format but wrong season scope |
| `womens-machine-learning-competition-2019` | GPT dags | Stage-1 submission (10,080 rows) likely wrong vs expected |
| `web-traffic-time-series-forecasting` | GPT dags | Passes local check but Kaggle rejects — likely wrong page-key IDs |
| `liverpool-ion-switching` | GPT nodags | Format OK locally, Kaggle rejects — wrong row count or invalid values |
| `mens-machine-learning-competition-2018` | GPT nodags | Correct format, saved to `/submission/submission.csv`, Kaggle error |

### Key finding
The dominant error pattern is now **multi-stage competition confusion**: agents see both Stage 1 and Stage 2 sample submission files and pick the wrong one (usually Stage 1, which is smaller). `gendered-pronoun-resolution` hit 3 of 4 model/mode combos with this exact issue.

**Recommendation**: Inject the expected row count and column schema into the system prompt or as a pre-run validation step. Agents should explicitly verify submission dimensions against the *expected* output shape before saving.

---

## 2 · No-Submit Failure Modes

The no-submit category (agent ran out of turns/time without saving a valid submission) is now the dominant failure type, especially for Claude.

### Error breakdown (no-submit cases)

| Error type | Claude / dags | Claude / nodags | GPT / dags | GPT / nodags |
|-----------|:---:|:---:|:---:|:---:|
| **TimeoutError** (cell >300s) | 10 | 12 | 4 | 2 |
| **NoTraceback** (time/turn limit) |  1 |  7 | 2 | 0 |
| **ImportError** (missing package) |  3 |  4 | 0 | 1 |
| **FileNotFoundError** |  3 |  0 | 1 | 0 |
| **OtherTraceback** |  2 |  3 | 0 | 0 |

### Patterns

**Timeout is the #1 failure driver**, particularly for Claude (22/45 of Claude's no-submits). The 300-second per-cell execution limit forces agents to abandon training mid-run. Claude hits this significantly more than GPT because it writes larger, more monolithic code cells (avg 13.7 python blocks vs GPT's 20.6, but Claude's blocks are often more complex and training-heavy).

**Claude no-DAGs has the highest "no-traceback" rate** (7/26 no-submits with no error message): these are cases where the agent was still mid-computation at the turn limit, reasoning looked good in the final turn ("The model is training well...") but it simply never got to write the file.

**ImportError** hits Claude 3–4× more than GPT. Claude more aggressively attempts advanced libraries (`tensorflow`, `torch`, `transformers`) that are unavailable in the environment. GPT tends to fall back to `scikit-learn`/`lightgbm` earlier.

**FileNotFoundError** (3 cases in Claude dags) involves agents referencing file paths that exist in training data but not in the evaluation setup (e.g., extracting a compressed file that wasn't provided).

---

## 3 · DAG Format & Instruction Following

### Format compliance per instruction set

| Pattern | Claude / dags | Claude / nodags | GPT / dags | GPT / nodags |
|---------|:---:|:---:|:---:|:---:|
| `<reasoning>` tag | 35/35 | 35/35 | 35/35 | 35/35 |
| `<python>` tag | 35/35 | 35/35 | 33/35 | 35/35 |
| `<answer>` block | 15/35 | 3/35 | **35/35** | **35/35** |
| DAG (```json``` fenced) | **35/35** | 0/35 | 1/35 | 0/35 |
| DAG (raw JSON) | 0/35 | 0/35 | **34/35** | 0/35 |
| `PLAN REVISION` text | 25/35 | 0/35 | **35/35** | 1/35 |
| `Step N/M` format | 35/35 | 0/35 | **35/35** | 0/35 |
| `[PLAN REVISION Rn/M]` tag | 0/35 | 0/35 | **22/35** | 0/35 |
| `Execution summary` | 14/35 | 0/35 | **35/35** | 0/35 |

### Observations

**`<answer>` block**: GPT writes a structured summary block at the end of *every* run regardless of mode (35/35 dags, 35/35 nodags). Claude only does this in ~40% of dags runs and virtually never in nodags. The answer block is a useful post-hoc record of what the agent actually did; Claude's habit of skipping it makes trajectory debugging harder.

**DAG format**: Claude correctly wraps DAGs in ` ```json ``` ` fences as specified. GPT emits raw JSON objects (no fence). Both are parseable, but GPT's format diverges from the spec. This is likely because GPT interprets "emit a JSON DAG" as outputting the object directly rather than embedding it in markdown.

**Plan revisions**: In DAGs mode, GPT performs structured plan revisions in **100% of runs** (avg 3.7 revisions/run, max 9). Claude performs them in 71% of runs (avg 2.2 when counting all runs, avg 1.6 for non-submitting runs). GPT's revisions are more formally tagged with `[PLAN REVISION Rn/M — after Step N]` and include explicit diagnosis + alternative approach.

**Execution summary**: GPT closes *every* dags run with an `Execution summary:` paragraph in `<reasoning>` that lists steps taken, scores at each stage, and revisions. Claude does this only 40% of the time. This habit makes GPT trajectories significantly easier to interpret.

---

## 4 · DAG Consistency: Reasoning → Implementation → Code

> **Note on parsing**: GPT emits its two DAG objects as concatenated raw JSON in the first assistant turn (no markdown fence). A streaming `raw_decode` parser was used to extract them; the analysis below covers both models on equal footing.

### DAG structure

| Metric | Claude / dags | GPT / dags |
|--------|:---:|:---:|
| Runs with reasoning DAG | 35/35 | **35/35** |
| Runs with implementation DAG | 35/35 | **34/35** |
| Both present | 35/35 | **34/35** |
| Avg reasoning nodes | 5.9 (max 8) | 6.1 (max 9) |
| Avg impl nodes | 6.0 (max 8) | 5.9 (max 9) |
| Avg reasoning edges | 4.9 | 5.1 |
| Avg impl edges | 5.0 | 5.0 |
| DAG emitted at turn | always turn 0 | always turn 0 |

Both models produce exactly the right structure (reasoning DAG + implementation DAG, emitted before any code) in almost all runs. Graph density is nearly identical. GPT missed the implementation DAG in 1 run.

### Implementation DAG → code keyword coverage

Node label keywords from the implementation DAG were matched against the actual python code blocks in the same trajectory:

| Metric | Claude / dags | GPT / dags |
|--------|:---:|:---:|
| Avg coverage | **77.6%** | **75.6%** |
| Runs >80% coverage | 18/35 (51%) | 9/34 (26%) |
| Runs 50–80% coverage | 16/35 (46%) | 23/34 (68%) |
| Runs <50% coverage | 1/35 (3%) | 2/34 (6%) |
| [complete] avg | **81.4%** | 74.3% |
| [no_submit] avg | 74.8% | 77.8% |
| [error] avg | 77.6% | 80.5% |

Both models achieve ~76–78% coverage overall — strong evidence that the implementation DAG is an accurate forward plan, not a post-hoc rationalisation.

**Key divergence**: For Claude, coverage *predicts outcome* — successful runs show 81.4% vs 74.8% for non-submits. For GPT there is no such correlation (complete 74.3%, no-submit 77.8%, error 80.5% — if anything, reversed). This suggests Claude's code-writing is more tightly coupled to its plan, while GPT's execution diverges from the plan regardless of whether the run succeeds.

### Reasoning DAG → implementation DAG semantic overlap

| Metric | Claude / dags | GPT / dags |
|--------|:---:|:---:|
| Avg R→I keyword overlap | 46.4% | 45.9% |
| Jaccard similarity | 25.9% | 26.0% |
| Runs >70% overlap | 0/35 | 0/34 |
| Runs 40–70% overlap | 25/35 | 26/34 |
| Runs <40% overlap | 10/35 | 8/34 |
| [complete] avg | 42.7% | 45.5% |
| [no_submit] avg | 49.4% | 46.4% |

The R→I semantic overlap is virtually identical between models (~46%, Jaccard ~26%). Both models substantially rewrite the node vocabulary from reasoning to implementation — abstract planning language gets replaced with concrete technical steps.

### R → I label drift: what changes between DAGs?

| Drift metric | Claude | GPT |
|---|:---:|:---:|
| Avg new concepts added in impl DAG | 21.8 | **41.6** |
| Avg concepts dropped from reasoning DAG | 15.0 | **29.2** |
| Top dropped words (reasoning-only) | `data`, `baseline`, `build`, `explore`, `design`, `engineer` | `basic`, `based`, `boosting`, `build`, `approach`, `design` |
| Top added words (impl-only) | `create`, `check`, `analyze`, `classifier`, `columns` | `adjust`, `best`, `against`, `apply`, `compute` |

Both models drop abstract planning verbs (`explore`, `design`, `build`) and add operational verbs (`create`, `check`, `apply`, `compute`) when moving from reasoning to implementation. This is the expected and healthy behaviour — the implementation DAG specialises the plan into executable steps.

**GPT's implementation DAG is substantially more verbose** (avg 41.6 new concepts vs Claude's 21.8), reflecting GPT's habit of writing longer, more detailed node labels in the implementation graph. Claude's labels are terser but still operationally precise.

**Claude's `no_submit` runs show *higher* R→I overlap (49.4%) than successful ones (42.7%)**. Counter-intuitively, more overlap means less concretisation — the failing runs stayed too close to the abstract reasoning plan and didn't translate it into specific implementation steps. GPT shows no such pattern.

---

## 5 · Claude vs GPT — Key Behavioural Differences

### Completion rate gap

The most striking difference: GPT submits 2–3× more often. The DAG protocol helps Claude (+14pp) but helps GPT far less (+12pp → actually GPT nodags is already very high). The fundamental gap is that **GPT recovers better from failed code**: it catches errors faster, pivots to simpler models sooner (median 8.1–9.6 turns for GPT vs 12.6–15.2 for Claude), and doesn't over-invest in approaches that are hitting timeout walls.

### Execution style

| Behaviour | Claude | GPT |
|-----------|--------|-----|
| Avg code blocks per run (dags) | 13.7 | 20.6 |
| Avg turns to completion | 12.6 | 9.6 |
| Library preference | `tensorflow`, `torch`, `transformers` (ambition-first) | `sklearn`, `lightgbm`, `xgboost` (pragmatic-first) |
| On timeout → | Often re-tries same approach, smaller data | Immediately simplifies or baseline-predicts |
| Final answer summary | Optional | Always written |

Claude invests more deeply in each approach before pivoting. This pays off when the environment cooperates (Claude's successful runs average 20.8 min vs GPT's 20.0 — similar total work) but costs submissions when it doesn't. GPT's pattern of fewer turns but more code blocks suggests it writes smaller, more incremental cells that are less likely to hit the 300s timeout.

### Plan revision semantics

Claude's plan revisions in DAGs mode are **reactive** (triggered by actual errors or poor CV scores) and correlated with _not_ submitting (avg 2.2 revisions in no-submit vs 1.2 in complete). GPT's revisions are **proactive and structural** (every run, avg 3.7, no correlation with outcome) — GPT treats revisions as a routine part of the workflow rather than a distress signal.

### DAG format divergence

GPT emits raw JSON DAGs (no code fence) consistently. This is a formatting instruction miss — the spec requires ` ```json ``` ` blocks. Since GPT's raw objects are parseable, this is cosmetic for evaluation purposes, but it breaks any downstream tooling that relies on fenced extraction.

### `<answer>` discipline

GPT always writes a structured `<answer>` block summarising its method, key scores, and approach. Claude does this inconsistently. The answer block has no effect on scoring but makes the trajectory significantly more useful for post-hoc analysis and is a useful habit to reinforce in the prompt.

---

## 6 · Recommendations

### Immediate

1. **Multi-stage competition guard**: Inject expected row count + column names into the task prompt. Add a validation step that confirms `len(submission) == expected_rows` before saving. `gendered-pronoun-resolution` alone accounts for 3/8 errors (37.5%).

2. **Cell timeout recovery**: Add explicit timeout guidance in the prompt — e.g., "If a training cell exceeds 250 seconds, cancel it and immediately fall back to a simpler model (logistic regression, random forest with max_depth=5)." Claude in particular needs a clearer fallback heuristic.

3. **`<answer>` block enforcement**: Claude should be prompted to always write a `<answer>...</answer>` block summarising method and key metrics. GPT does this naturally; Claude needs explicit instruction.

4. **Library fallback ladder**: The prompt should specify a preferred library order (sklearn → lightgbm → xgboost → pytorch/tensorflow). ImportErrors from missing packages account for 3–4 no-submit cases per Claude run.

### For DAG quality

5. **GPT DAG formatting**: GPT should wrap DAG JSON in ` ```json ``` ` fences. It currently emits both DAG objects concatenated as raw JSON in a single turn, which works with a streaming parser but breaks standard fenced-block extractors and the analysis notebook.

6. **Node coverage metric**: The ~22–24% of implementation nodes not covered in code mostly represents nodes planned but abandoned before timeout. Prompting agents to mark nodes as `status: skipped` when abandoning them would make this trackable and could also serve as an early-exit signal.

7. **Revision correlation (Claude)**: Claude shows more plan revisions when _failing_ (avg 2.2 vs 1.2 for completions), and failing runs also have *higher* R→I semantic overlap (49.4% vs 42.7%). Both signals point to the same root cause: Claude's failing runs over-invest in abstract planning and under-invest in translating plans into concrete implementation steps. Prompt adjustment: "After writing the implementation DAG, immediately start executing step 1. Do not re-plan unless a step produces an error."

8. **GPT plan-execution decoupling**: GPT's code coverage does not correlate with outcome at all (74.3% complete, 77.8% no-submit). GPT revises its plan routinely whether or not it ultimately submits. This suggests GPT's revisions are largely cosmetic — the plan is updated but doesn't tightly constrain execution. To make DAGs more binding for GPT, consider requiring the agent to quote the current DAG node ID in each `<reasoning>` block (`executing node i3`).

---

## Appendix: Per-Competition No-Submit Hotspots

The following competitions had **all 4 runs** (both models × both modes) fail to submit:

| Competition | Likely cause |
|-------------|-------------|
| `stanford-covid-vaccine` | Complex JSON format, per-position prediction unusual structure |
| `ashrae-energy-prediction` | Very large dataset, memory/timeout consistent hit |
| `ieee-fraud-detection` | Large dataset + old LightGBM version incompatibility |
| `sp-society-camera-model-identification` | Image data, 300s timeout on iterating 2,750 images |
| `statoil-iceberg-classifier` | `.7z` compressed data, no `7z` binary in environment |

The last two are environment-level blockers (missing tools, strict per-cell timeout on large iteration loops) that no amount of agent improvement will resolve without environment fixes.
