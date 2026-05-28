### 1 — Run the optimizer (covers all five goals in one study)

```bash
uv run python examples/optimize_research_engine.py \
  --model gpt-4o \
  --dataset dsbio \
  --backend litellm \
  --limit 10 \
  --trials 40 \
  --population-size 8 \
  --tune-dag-validation-mode \
  --generation-model gpt-4o-mini \
  --generated-variants-per-type 4 \
  --seed 42 \
  --output-dir ./optimization_results/dsbio_candidate
```

What each flag addresses:
- DAG prompts (reasoning + implementation) — always on; controlled by `_compose_prompt_profile` (15 parameters including `reasoning_framework`, `impl_specificity`, all domain-specific dsbio gates)
- DAG construction — `--tune-dag-validation-mode` makes NSGA-II also search over `default`/`relaxed`/`strict` and jointly optimise `max_nodes` and `max_turns`
- Context variants — built from the variant bank; `--generation-model gpt-4o-mini` generates domain-aware additional context variants
- Harness variants — same variant bank; `--generated-variants-per-type 4` generates additional harness compliance contract variants
- Harness strictness policy — controlled by the defaults (`--harness-max-identical-message-repeats 1`, `--harness-max-identical-code-repeats 1`); tighten if needed

---

### 2 — Establish a baseline (no generation model, fixed validation mode) to compare against

```bash
uv run python examples/optimize_research_engine.py \
  --model gpt-4o \
  --dataset dsbio \
  --backend litellm \
  --limit 10 \
  --trials 40 \
  --population-size 8 \
  --seed 42 \
  --output-dir ./optimization_results/dsbio_baseline
```

---

### 3 — Analyse each run individually

```bash
uv run python examples/analyze_research_optimization.py \
  --input-dir ./optimization_results/dsbio_baseline \
  --output-md ./optimization_results/dsbio_baseline/report.md

uv run python examples/analyze_research_optimization.py \
  --input-dir ./optimization_results/dsbio_candidate \
  --output-md ./optimization_results/dsbio_candidate/report.md
```

---

### 4 — Visualise both experiments together

```bash
uv run python examples/visualize_research_optimization.py \
  --input-dirs \
    ./optimization_results/dsbio_baseline \
    ./optimization_results/dsbio_candidate \
  --labels baseline candidate \
  --output-dir ./optimization_results/figures
```

Produces `pareto_quality_time.png`, `accuracy_vs_time.png`, `run_metric_boxplots.png`, `research_trials_flat.csv`, and `model_comparison_summary.csv` all overlaid with the two labels.

---

### 5 — HTML dashboard (interactive, both runs on one page)

```bash
uv run python examples/build_research_dashboard.py \
  --input-dirs \
    ./optimization_results/dsbio_baseline \
    ./optimization_results/dsbio_candidate \
  --labels baseline candidate \
  --output-html ./optimization_results/dsbio_dashboard.html
```

---

### 6 — Regression gate (CI-style pass/fail)

```bash
uv run python examples/check_research_regression.py \
  --baseline-dir ./optimization_results/dsbio_baseline \
  --candidate-dir ./optimization_results/dsbio_candidate \
  --selector top-balanced --top-k 10 \
  --min-delta-quality -0.005 \
  --min-delta-compliance -0.010 \
  --max-delta-time 5.0 \
  --max-delta-turns 0.5 \
  --allow-missing-exact-match
```

---

### 7 — Variant attribution (which DAG prompt variant group drove gains)

```bash
uv run python examples/attribute_research_variants.py \
  --input-dir ./optimization_results/dsbio_candidate \
  --selector top-balanced --top-k 10 \
  --n-bootstrap 5000 --seed 42
```

export MANAGER_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')

python examples/optimize_research_engine.py \
  --model gpt-4o \
  --dataset dsbio \
  --backend litellm \
  --manager-url http://localhost:5000 \
  --limit 6 \
  --trials 32 \
  --population-size 8 \
  --max-turns-min 10 \
  --max-turns-max 18 \
  --tune-dag-validation-mode \
  --study-name dsgym_dag_style_search \
  --output-dir ./optimization_results/dag_style \
  --force-regenerate-variants

