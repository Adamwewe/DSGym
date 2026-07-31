# DAG Commitment Hypothesis — Why DAGs Hurt GPT 5.1 on DSPredict-Hard

> Written Jun 12 2026, based on trace analysis V4 (53 dags / 54 nodags runs).

---

## The Core Claim

The DAG protocol as currently written turns **exploratory planning into a pre-commitment contract**. For a model as capable as GPT 5.1, that contract is net-negative: it removes the ability to opportunistically pivot to better strategies mid-run, and charges a turn-budget tax for formal revision even when no failure has occurred. The result is a completion rate of 53% vs 78% for nodags, with zero quality uplift on head-to-head scores.

---

## What the Prompt Actually Says

`SYSTEM_PROMPT_DAG_DSPREDICT` has three clauses that together create the commitment problem:

### 1. Blind pre-commitment (Step 0)

> *"Build two DAGs before writing any code."*

Both DAGs — reasoning AND implementation — are written at turn 0, before the agent has seen a single row of data. The agent has only the competition name and description. This means:

- Model family is chosen on the basis of task description alone (e.g. "LightGBM" for tabular, "TF-IDF" for NLP)
- Data shape, memory constraints, class balance, correlation structure, label encoding: all unknown
- The implementation DAG binds specific steps to specific nodes with `reasoning_anchor` — a 1:1 mapping that structurally resists partial revision

This is fundamentally different from how nodags operates: the nodags agent loads the data in turn 0 and builds its plan from what it actually observes.

### 2. Reactive-only revision triggers

> *"Revise when: (a) an exception is raised, (b) overfitting (train >> val by >10 pp), (c) near-random performance, or (d) submission validation fails."*

The revision grammar is failure-conditioned. There is no trigger for:
- "I've seen the data and a better approach is now obvious"
- "This method works but I know a strictly better one for this task type"
- "My plan was designed for tabular data but the data is actually text/images"

This means a strategy can be **suboptimal but not triggering**. TF-IDF is not an exception, doesn't cause overfitting flags, doesn't produce near-random performance, and doesn't fail submission validation. So the dags agent stays on TF-IDF even after seeing data that clearly calls for BM25. The nodags agent has no revision protocol at all — it pivots whenever its reasoning says so.

### 3. Hard revision cap + "remaining steps only" semantics

> *"Up to 5 revisions total. After exhausting revisions, submit the best result obtained so far."*
>
> *"Revised steps (remaining only): …"*

Two compounding effects:
- The cap means later revisions get more expensive as the budget depletes — the agent becomes *increasingly reluctant* to revise in the second half of a run.
- "Remaining only" means a revision updates future nodes but doesn't reopen past ones. If node i3 was the model-selection step and it was wrong, a revision issued at node i6 can't go back to i3 — it can only adjust what follows. This is unlike the nodags agent which can rerun anything at any point.

---

## The Evidence

### TREC-COVID: the sharpest example

| Mode | Percentile | Turns | Approach |
|------|:----------:|:-----:|----------|
| dags | 27.5% | 14 | TF-IDF only — committed at DAG turn 0 |
| nodags | **85.0%** | 11 | BM25 — discovered after exploring the data |

The dags agent's implementation DAG node i1 said "build TF-IDF index over document text." BM25 doesn't appear anywhere in the trajectory. The nodags agent explored the file listing, identified this as a standard IR task, and chose BM25 without being asked to. TF-IDF never raised an exception, never showed near-random performance — it just scored worse. No revision trigger fired. The agent followed the plan to the end.

### Store-sales: over-engineering from a rigid plan

| Mode | Percentile | Turns | Code cells |
|------|:----------:|:-----:|:----------:|
| dags | 10.8% | 15 | **35** |
| nodags | **35.7%** | 8 | 13 |

The dags agent's plan specified LightGBM with lag features. When the first training attempt timed out (trigger: exception), it revised. When that revised approach also struggled, it revised again. Each revision deepened the complexity of the LightGBM pipeline rather than stepping back to a simpler baseline. 35 code cells, 5 revisions, 15 turns — all following the original model-family commitment. The nodags agent built a clean LightGBM baseline, it worked, and it stopped.

### Head-to-head across 24 shared completions

| Winner | Count |
|--------|:-----:|
| Nodags (>2 pp better) | 10 |
| Dags (>2 pp better) | 9 |
| Tie | 5 |
| Avg diff (dags − nodags) | **−3.6 pp** |

The coin-flip result means: on competitions where both modes complete, the DAG gives neither a consistent advantage nor a consistent disadvantage. It just changes *which* strategy the agent commits to. When the blind-commit strategy happens to be the best one, dags wins. When it isn't, nodags wins — and nodags has more turns to execute with because it didn't spend 41% of them on planning overhead.

### 41% of dags turns consumed by planning

In complete dags runs (avg 10.8 turns): 1 (DAG emission) + 3.4 (revisions) = **4.4 planning turns**, leaving **6.4 code turns**. Nodags gets **8.0 code turns**. Even if the DAG chose the perfect strategy every time, it would still produce fewer iterations, fewer CV evaluations, and less chance to recover from environment-level failures (timeouts, API bugs).

---

## Why the V2 Result Showed Slight Uplift

The earlier 35-problem run on a separate machine without torch showed a slight DAG advantage. This is consistent with the hypothesis:

1. **No torch = no over-commitment to deep learning.** Without torch available, the dags agent cannot commit to a CNN or transformer at turn 0. It is forced into tabular methods where a structured plan (load → merge → feature engineer → LightGBM → validate → submit) maps cleanly to a linear DAG. The commitment cost is low because the strategy space is narrow.

2. **Stable infrastructure.** No 500 container allocation errors. The 5 dags-only allocation failures in V4 are independent of the planning protocol — they're a batching artefact — but they inflate the completion gap artificially.

3. **Smaller problem set may sample easier competitions.** The 35-problem sample skews toward competitions where LightGBM with standard feature engineering is the dominant strategy. On those tasks, the DAG commits to exactly the right approach and the structure helps the agent execute it cleanly without wandering.

4. **The effect size was small.** "Slight uplift" is consistent with a noise effect on 35 samples where the committed strategy happened to be correct more often than not, rather than a reliable protocol benefit.

---

## Root Cause Summary

The DAG protocol has two distinct roles that are currently conflated:

| Role | What it does | Does GPT 5.1 need it? |
|------|-------------|----------------------|
| **Structural scaffold** | Forces the agent to articulate a plan before executing | No — 47/54 nodags runs self-structure natively |
| **Commitment contract** | Binds the agent to a pre-data strategy | No — and this is actively harmful |

GPT 5.1 benefits from the first role (it would produce similar plans anyway) but is harmed by the second. A well-designed protocol should provide structure without locking in a strategy before data is observed.

---

## What Would Actually Help

### Fix 1: Defer implementation DAG until after data exploration

Split Step 0 into two phases:
- **Step 0a** (before code): reasoning DAG only — high-level analytical questions, not model choices
- **Step 0b** (after first data exploration cell): implementation DAG — now written with actual shapes, dtypes, and distributions in hand

This preserves the structural benefit (agent articulates a plan) while eliminating the blind-commit cost (strategy is chosen on real data).

### Fix 2: Add proactive revision trigger

Add to the revision grammar:
> *(e) you discover during data exploration that your implementation DAG was designed for the wrong task type (e.g. you planned a tabular pipeline but the data is primarily image or text)*

This single trigger would have caught the TREC-COVID failure. The dags agent saw the document corpus in its first execution turn and could have pivoted to BM25 — but the protocol gave it no permission to revise without a failure signal.

### Fix 3: Revisions should be allowed to reopen any node

Change "Revised steps (remaining only)" to "Revised steps (including any that need revisiting)." This allows the agent to go back to model selection after discovering a constraint at training time, rather than only adjusting downstream steps.

### Fix 4: Remove or raise the 5-revision cap

The cap creates perverse incentives in the second half of long runs. An agent with 2 revisions remaining will be reluctant to use one on a minor adaptation, even when adapting is clearly correct. Either remove the cap or make it much higher (e.g. 10), since in practice well-functioning runs use 3–5 and the cap only bites when things go wrong late.

### Fix 5: Fix the validation path

`SYSTEM_PROMPT_DSPREDICT` (nodags) references `/data/dspredict-sampled/<challenge_name>/sample_submission.csv` but the actual dataset is `dspredict-hard`. Any agent that copies this validation snippet verbatim will get a `FileNotFoundError` on the hard dataset. This is a latent source of submission format errors across both prompts.

---

## Alternative Hypothesis: Is This Model-Specific?

The V2 analysis showed that Claude Sonnet 4.5 *does* benefit from DAGs (+14 pp submission rate, dags vs nodags). This suggests the commitment problem is model-specific, not protocol-specific. A plausible explanation:

| Model | Without DAG | With DAG | Interpretation |
|-------|:-----------:|:--------:|----------------|
| Claude Sonnet 4.5 | 26% | 40% | Needs external structure to maintain plan coherence across 15 turns |
| GPT 5.1 | **78%** | 53% | Already maintains plan coherence; DAG adds overhead and commitment cost |

The DAG protocol was likely tuned on models that benefit from scaffolding. For models that self-scaffold natively, the same protocol becomes a constraint. The right test: does a lighter-weight protocol (reasoning DAG only, no implementation DAG, no commitment) give GPT the same submission rate as nodags while recovering some of the quality upside? That would confirm the hypothesis and give a clear direction for protocol design.
