# Prompt Optimization (NSGA-II)

This module performs evolutionary prompt optimization for DAG prompting using Optuna's NSGA-II sampler.

## Objective Design

Each trial evaluates a prompt profile on a fixed dataset slice and returns four objectives:

- maximize success rate
- minimize error rate
- minimize average execution time
- minimize average turns

Mathematically, with sample-level outcomes:

- $s_i \in \{0,1\}$ success flag
- $t_i$ execution time
- $r_i$ total turns

for $N$ samples:

$$
\text{success\_rate} = \frac{1}{N}\sum_{i=1}^{N} s_i
$$

$$
\text{error\_rate} = 1 - \text{success\_rate}
$$

$$
\overline{t} = \frac{1}{N}\sum_{i=1}^{N} t_i,
\quad
\overline{r} = \frac{1}{N}\sum_{i=1}^{N} r_i
$$

Optimization is multi-objective (Pareto), not a single weighted score:

$$
\max(\text{success\_rate}),
\min(\text{error\_rate}),
\min(\overline{t}),
\min(\overline{r})
$$

This avoids overfitting to one scalar and preserves trade-off frontiers.

## What Is Logged

Per trial:

- prompt profile JSON used for DAG prompting
- trace summary (`success_rate`, `error_rate`, `avg_execution_time`, `avg_turns`, p90 and medians)
- error breakdown by category
- aggregated evaluator metrics

Artifacts are written under:

- `optimization_results/trials/trial_XXXX/`
- `optimization_results/optimization_summary.json`
- `optimization_results/optuna_study.db`

## Usage

Install optimization extra:

```bash
uv sync --extra optimize
```

Run optimization:

```bash
uv run python examples/optimize_dag_prompts.py \
  --model claude-sonnet-4-5-20250929 \
  --dataset dsbio \
  --backend litellm \
  --dag-validation-mode default \
  --limit 10 \
  --trials 20 \
  --population-size 8 \
  --max-workers 2 \
  --output-dir ./optimization_results
```

Tune DAG validation mode jointly with prompts:

```bash
uv run python examples/optimize_dag_prompts.py \
  --model claude-sonnet-4-5-20250929 \
  --dataset dsbio \
  --backend litellm \
  --tune-dag-validation-mode \
  --trials 30
```

Supported DAG validation modes:

- `default`: preserve current behavior for reproducibility
- `relaxed`: same as default (acyclic + valid references)
- `strict`: enforce node bounds, topological list order, weak connectivity, and implementation/reasoning 1:1 structural mirroring

## Help/Hurt Analysis From Traces

After optimization, generate a mode-wise comparison report directly from trial traces:

```bash
uv run python examples/analyze_optimization_results.py \
  --input-dir ./optimization_results
```

This writes:

- `optimization_help_hurt_report.json`
- `optimization_help_hurt_report.md`

The report compares each mode against `default` using:

- mean deltas for success/error/time/turns
- standard error of the mean (SEM) per objective
- Pareto domination rate versus default trials

## Non-breaking Integration

The existing `dsgym eval ...` workflow remains unchanged.

Optimization uses an environment-driven prompt profile override (`DSGYM_DAG_PROMPT_PROFILE`) consumed by DAG prompt builders. If unset, default prompts are used exactly as before.

## Research Optimization Engine (Prompt + Context + Harness + DAG Ablation)

For a full operational playbook that builds directly on DSGym README commands,
see `docs/RESEARCH_ENGINE_RUNBOOK.md`.

To support broader 2026-style research workflows, DSGym also provides a research optimizer that:

- evolves prompt/context/harness directives (not only static DAG profile fields)
- optionally generates additional candidate variants using an LLM (`--generation-model`)
- runs DAG-vs-noDAG ablations per trial (`--no-dag-ablation` to disable)
- logs harness-compliance diagnostics (step skipping, looping, stalled runs)
- returns a Pareto front across quality, speed, and compliance

### Objective definitions

For each trial, let modes be `M` (`M={dags,nodags}` when ablation is enabled).
For each mode `m`, define:

- `q_m`: mode quality score (dataset-adapted; e.g., exact match mean or leaderboard metric)
- `t_m`: average execution time
- `r_m`: average turns
- `c_m`: average harness compliance

Then trial objectives are:

- maximize `Q = (1/|M|) * Σ q_m`
- minimize `T = (1/|M|) * Σ t_m`
- minimize `R = (1/|M|) * Σ r_m`
- maximize `C = (1/|M|) * Σ c_m`
- maximize `Δ_DAG = q_dags - q_nodags` (0 when no paired ablation exists)

This yields a 5-objective NSGA-II search over quality/performance/compliance trade-offs.

Run:

```bash
uv run python examples/optimize_research_engine.py \
  --model claude-sonnet-4-5-20250929 \
  --dataset dsbio \
  --backend litellm \
  --limit 6 \
  --trials 24 \
  --population-size 8 \
  --tune-dag-validation-mode \
  --generation-model gpt-4o-mini \
  --output-dir ./optimization_results/research
```

Main artifacts:

- `optimization_results/research/variant_bank.json`
- `optimization_results/research/trials/trial_XXXX/trial_record.json`
- `optimization_results/research/research_optimization_summary.json`
- `optimization_results/research/optuna_study.db`

Per-mode trial records include:

- quality metric summary (`quality_mean`, metric name, evaluable counts)
- exact-match extraction (`exact_match_correct`, `exact_match_accuracy`) when available
- strict harness compliance diagnostics (`violation_rates`, loop/skip/regression counters)
- execution-time and turn statistics

Analyze and summarize:

```bash
uv run python examples/analyze_research_optimization.py \
  --input-dir ./optimization_results/research
```

This writes:

- `research_report.json`
- `research_report.md`

## Visualization and Cross-Run Comparison

Generate run/model comparison plots and flat CSV tables:

```bash
uv run python examples/visualize_research_optimization.py \
  --input-dirs ./optimization_results/research ./optimization_results/research_model_b \
  --labels model_a model_b \
  --output-dir ./optimization_results/research/figures
```

Outputs:

- `research_trials_flat.csv` (per-trial flattened data)
- `model_comparison_summary.csv`
- `pareto_quality_time.png`
- `accuracy_vs_time.png`
- `run_metric_boxplots.png`

Build a single consolidated offline HTML dashboard (tables + plots + Pareto explorer):

```bash
uv run python examples/build_research_dashboard.py \
  --input-dirs ./optimization_results/research ./optimization_results/research_model_b \
  --labels baseline candidate \
  --output-html ./optimization_results/research/research_dashboard.html
```

The HTML dashboard includes:

- run/model summary tables
- quality/time and accuracy/compliance plots
- interactive Pareto table filters by run/model
- full trial table for auditability

## Real DSGym DSBio Smoke Workflow (Self-Hosted)

In addition to the synthetic CI workflow, a real-dataset smoke workflow is available for
self-hosted runners:

- `.github/workflows/research-optimization-dsbio-smoke-self-hosted.yml`

Design goals:

- guarded secret usage via protected workflow environment (`dsgym-smoke`)
- manual trigger only (`workflow_dispatch`) to prevent accidental API spend
- bounded default budget (`limit=3`, `trials=6`, `population_size=4`, `max_workers=1`)
- objective/invariant validation after the run

The workflow requires at least one provider secret in the environment:

- `OPENAI_API_KEY` or
- `TOGETHER_API_KEY` or
- `ANTHROPIC_API_KEY`

What it runs:

1. `examples/optimize_research_engine.py` on real `dsbio` tasks with DAG ablation enabled
2. `examples/analyze_research_optimization.py`
3. `examples/build_research_dashboard.py`
4. JSON invariant checks for objective direction wiring, per-mode DAG/noDAG records, and recomputed objective means/delta from raw mode outputs
5. artifact upload under `optimization_results/dsbio_smoke`

## CI Regression Gate

Use a CI-friendly pass/fail comparator between candidate and baseline runs:

```bash
uv run python examples/check_research_regression.py \
  --baseline-dir ./optimization_results/research_baseline \
  --candidate-dir ./optimization_results/research_candidate \
  --selector top-balanced \
  --top-k 10 \
  --min-delta-quality -0.005 \
  --min-delta-compliance -0.010 \
  --max-delta-time 5.0 \
  --max-delta-turns 0.5
```

Exit codes:

- `0`: passes thresholds
- `1`: regression detected
- `2`: invalid or missing run data

Artifacts:

- `regression_report.json`
- `regression_report.md`

## Bootstrap Significance Testing

Strengthen research claims with bootstrap confidence intervals for metric deltas:

```bash
uv run python examples/bootstrap_research_significance.py \
  --baseline-dir ./optimization_results/research_baseline \
  --candidate-dir ./optimization_results/research_candidate \
  --selector top-balanced \
  --top-k 10 \
  --n-bootstrap 5000 \
  --confidence 0.95
```

Outputs:

- `bootstrap_significance.json`
- `bootstrap_significance.md`

## Strict Harness Evaluator

Strict harness evaluation is enabled by default in the research optimizer.

- detects missing step labels, step skipping, and step regressions
- enforces revision-before-jump behavior
- detects repetitive looping (messages/code blocks)
- tracks max-turn stalls

CLI toggles:

- disable strict evaluator: `--no-strict-harness-evaluator`
- tune loop strictness:
  - `--harness-max-identical-message-repeats`
  - `--harness-max-identical-code-repeats`

## API/Token Efficiency Controls

The research optimizer reuses `variant_bank.json` by default to avoid redundant
LLM generation calls for prompt/context/harness candidates.

- disable reuse: `--no-reuse-variant-bank-cache`
- force regeneration: `--force-regenerate-variants`

This keeps API key/token usage focused on evaluation trials rather than repeated
candidate generation.

## Generalized Adapter Interface

The optimization core exposes pluggable adapters so non-DSGym contexts can reuse
the same NSGA-II loop with custom evaluators and policies:

- `ModeEvaluatorAdapter`
- `QualityMetricAdapter`
- `HarnessPolicyAdapter`

Interfaces are defined in:

- `dsgym/optimization/interfaces.py`

Default DSGym adapters are exported from `dsgym.optimization`:

- `DefaultDSGymModeEvaluatorAdapter`
- `DefaultQualityMetricAdapter`
- `DefaultHarnessPolicyAdapter`

Programmatic usage:

```python
from dsgym.optimization import (
  ResearchOptimizationConfig,
  run_research_nsga2,
  DefaultDSGymModeEvaluatorAdapter,
  DefaultQualityMetricAdapter,
  DefaultHarnessPolicyAdapter,
)

cfg = ResearchOptimizationConfig(model="gpt-4o", dataset="dsbio")
summary = run_research_nsga2(
  cfg,
  mode_evaluator_adapter=DefaultDSGymModeEvaluatorAdapter(),
  quality_adapter=DefaultQualityMetricAdapter(),
  harness_adapter=DefaultHarnessPolicyAdapter(cfg),
)
```

## Non-DSGym Adapter Demo

An end-to-end external-framework simulation is available at:

- `examples/non_dsgym_adapter_demo.py`

Run baseline and candidate synthetic studies:

```bash
uv run python examples/non_dsgym_adapter_demo.py \
  --scenario baseline \
  --output-dir ./optimization_results/non_dsgym_baseline

uv run python examples/non_dsgym_adapter_demo.py \
  --scenario candidate \
  --output-dir ./optimization_results/non_dsgym_candidate
```

These outputs can be consumed by the same regression, significance, visualization,
and dashboard tools used for DSGym-native runs.
