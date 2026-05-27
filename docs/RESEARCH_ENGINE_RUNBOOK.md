# Research Engine Runbook

This runbook operationalizes the DSGym README workflows and extends them into a reusable research and benchmarking pipeline.

## 1) Goals and Research Questions

This engine is designed to directly answer:

1. Which prompt/context/harness patterns improve agent quality and generalization?
2. Do DAG-guided workflows help or hurt relative to non-DAG baselines?
3. Does strict plan-following harness behavior reduce failure modes like skipping and looping while preserving quality and speed?

Primary outputs for direct answers:
- Quality metrics (exact match when available, dataset-adapted quality otherwise)
- DAG help delta from paired DAG vs non-DAG evaluations
- Strict harness compliance metrics (skip, loop, regression, stall)
- Execution-time and turn-efficiency statistics
- Pareto fronts across quality/speed/compliance

## 2) Prerequisites (from DSGym README, then extended)

From repository root:

1. Install dependencies
   - uv sync
   - uv sync --extra optimize

2. Configure API keys only for the parts that need them
   - export OPENAI_API_KEY=...
   - export TOGETHER_API_KEY=...

3. Start execution backend when running real DSGym datasets
   - Follow manager/executor setup in README and executors README

Notes on key efficiency:
- Use generation-model only when actively searching new variant text.
- Keep variant-bank cache enabled (default) to avoid regenerating candidate variants repeatedly.
- Use small limits and trials for smoke checks; scale only after pipeline validation.

## 3) Baseline DSGym Evaluation (README-compatible)

Use native DSGym evaluation first to establish baseline behavior.

Examples:
- dsgym eval --model gpt-4o --dataset dsbio --limit 10 --dags false
- dsgym eval --model gpt-4o --dataset dsbio --limit 10 --dags true --dag-model gpt-4o-mini

This validates environment, model wiring, and dataset metrics before optimization.

## 4) Research Optimization (DSGym-native)

Run optimization on a bounded subset first.

Example:
- uv run python examples/optimize_research_engine.py --model gpt-4o --dataset dsbio --backend litellm --limit 6 --trials 24 --population-size 8 --tune-dag-validation-mode --output-dir ./optimization_results/research

Recommended smoke-test first:
- uv run python examples/optimize_research_engine.py --model gpt-4o --dataset dsbio --backend litellm --limit 3 --trials 6 --population-size 4 --output-dir ./optimization_results/research_smoke

## 5) Analysis and Visualization

1. Structured report
- uv run python examples/analyze_research_optimization.py --input-dir ./optimization_results/research

2. Static plots and model comparison CSV
- uv run python examples/visualize_research_optimization.py --input-dirs ./optimization_results/research ./optimization_results/research_model_b --labels model_a model_b --output-dir ./optimization_results/research/figures

3. Consolidated HTML dashboard
- uv run python examples/build_research_dashboard.py --input-dirs ./optimization_results/research ./optimization_results/research_model_b --labels model_a model_b --output-html ./optimization_results/research/research_dashboard.html

## 6) Regression Gating and Statistical Confidence

1. CI-friendly threshold gate (pass/fail)
- uv run python examples/check_research_regression.py --baseline-dir ./optimization_results/research_baseline --candidate-dir ./optimization_results/research_candidate --selector top-balanced --top-k 10 --min-delta-quality -0.005 --min-delta-compliance -0.010 --max-delta-time 5.0 --max-delta-turns 0.5 --allow-missing-exact-match

Exit codes:
- 0: pass
- 1: regression detected
- 2: invalid run data

2. Bootstrap significance for research claims
- uv run python examples/bootstrap_research_significance.py --baseline-dir ./optimization_results/research_baseline --candidate-dir ./optimization_results/research_candidate --selector top-balanced --top-k 10 --n-bootstrap 5000 --confidence 0.95 --seed 42

Use the bootstrap CI to support claim strength beyond point estimates.

## 7) Generalization to Non-DSGym Agentic Contexts

The optimization core supports adapters so external frameworks can reuse the same NSGA-II loop.

Adapter interfaces:
- dsgym/optimization/interfaces.py
  - ResearchContextAdapter
  - TaskMutatorAdapter
  - ModeEvaluatorAdapter
  - QualityMetricAdapter
  - HarnessPolicyAdapter

End-to-end non-DSGym demonstration:
- uv run python examples/non_dsgym_adapter_demo.py --scenario baseline --output-dir ./optimization_results/non_dsgym_baseline
- uv run python examples/non_dsgym_adapter_demo.py --scenario candidate --output-dir ./optimization_results/non_dsgym_candidate

Then run the same dashboard/gating/significance scripts on those outputs.

## 8) Automated CI Workflow

A full pipeline workflow is provided:
- .github/workflows/research-optimization-ci.yml

A real DSGym smoke workflow for self-hosted runners is also provided:
- .github/workflows/research-optimization-dsbio-smoke-self-hosted.yml

Self-hosted smoke workflow guardrails:
- Triggered manually (`workflow_dispatch`) to avoid accidental spend.
- Uses protected environment-scoped secrets (`dsgym-smoke`).
- Fails early unless at least one provider key is present (`OPENAI_API_KEY`, `TOGETHER_API_KEY`, or `ANTHROPIC_API_KEY`).
- Keeps compute/token spend bounded by default (`limit=3`, `trials=6`, `population_size=4`, `max_workers=1`).
- Avoids extra variant-generation spend by default (no `--generation-model` in smoke mode).
- Runs invariant checks on optimization artifacts to verify objective wiring, DAG ablation presence, and recomputed objective math (`Q`, `T`, `R`, `C`, `Delta_DAG`) from per-mode records.

Pipeline stages:
1. Optimization subset run (baseline + candidate)
2. Regression gate
3. Bootstrap significance report
4. Consolidated dashboard generation
5. Artifact upload

Self-hosted smoke stages:
1. Guard secrets and install dependencies
2. Real `dsbio` optimization smoke run (minimal budget)
3. Structured analysis report + consolidated dashboard
4. Mathematical/reporting invariant validation
5. Artifact upload

## 9) Mathematical Objective Definitions

For each trial with modes M (typically {dags, nodags}):

- Q = (1/|M|) * sum(q_m), maximize
- T = (1/|M|) * sum(t_m), minimize
- R = (1/|M|) * sum(r_m), minimize
- C = (1/|M|) * sum(c_m), maximize
- Delta_DAG = q_dags - q_nodags, maximize

Where:
- q_m: mode quality metric
- t_m: mode mean execution time
- r_m: mode mean turns
- c_m: mode mean compliance

Optimization uses NSGA-II over these five objectives.

## 10) Validation Checklist (Senior/Principal Engineering Standard)

Before claiming results:

1. Reproducibility
- Fix seed
- Save config and variant bank
- Save trial records and Pareto summary

2. Correctness
- Verify dataset/task setup and manager connectivity
- Confirm quality metric alignment with task semantics
- Confirm DAG ablation actually runs both modes

3. Efficiency
- Use cached variant bank unless intentionally regenerating
- Keep trial limits bounded for iteration loops

4. Reliability
- Run regression gate against stable baseline
- Run bootstrap significance for top deltas
- Review strict harness violation breakdown

5. Reporting
- Publish HTML dashboard + CSV summaries
- Keep raw trial artifacts for audit

## 11) Interpreting Direct Answers

A candidate change is likely beneficial when all are true:
- mean quality non-inferior or improved
- compliance non-inferior or improved
- time/turns not materially regressed
- DAG help delta does not degrade unexpectedly
- bootstrap CI supports improvement direction for key metrics

If these conditions are not met, treat conclusions as inconclusive and iterate on variants/harness policy before scaling.
