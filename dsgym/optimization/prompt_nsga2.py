from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dsgym.agents import DSPredictReActAgent, ReActDSAgent
from dsgym.datasets import DatasetRegistry
from dsgym.eval import EvaluationConfig, Evaluator


@dataclass
class PromptOptimizationConfig:
    model: str
    dataset: str = "dsbio"
    backend: str = "litellm"
    dag_model: Optional[str] = None
    manager_url: str = "http://localhost:5000"
    api_key: Optional[str] = None
    limit: int = 10
    sample_n: Optional[int] = None
    sample_seed: int = 42
    max_turns: int = 15
    temperature: float = 0.0
    max_workers: int = 4
    task_timeout_seconds: Optional[float] = 900.0
    dag_validation_mode: str = "default"
    tune_dag_validation_mode: bool = False
    n_trials: int = 20
    population_size: int = 8
    seed: int = 42
    study_name: str = "dsgym_prompt_nsga2"
    output_dir: str = "./optimization_results"


def _dataset_family(dataset: str) -> str:
    if dataset.startswith("dspredict"):
        return "dspredict"
    return "dsbio"


def _resolve_dspredict_split(dataset: str) -> str:
    if "-" in dataset:
        return dataset.split("-", 1)[1]
    return "hard"


def _load_tasks(config: PromptOptimizationConfig):
    family = _dataset_family(config.dataset)

    if family == "dsbio":
        dataset = DatasetRegistry.load("dsbio", virtual_data_root="/data")
        if config.sample_n is not None:
            dataset.create_sample(n=config.sample_n, seed=config.sample_seed)
            split = "sampled"
        elif config.dataset == "dsbio-sampled":
            split = "sampled"
        else:
            split = None

        kwargs: Dict[str, Any] = {"limit": config.limit, "generate_dags": True}
        if split is not None:
            kwargs["split"] = split
        tasks = dataset.load(**kwargs)
        return dataset, tasks

    dataset = DatasetRegistry.load("dspredict", virtual_data_root="/data")
    split = _resolve_dspredict_split(config.dataset)
    if config.sample_n is not None:
        source_split = split if split != "sampled" else "hard"
        dataset.create_sample(n=config.sample_n, split=source_split, seed=config.sample_seed)
        split = "sampled"

    tasks = dataset.load(limit=config.limit, split=split, generate_dags=True)
    return dataset, tasks


def _compose_prompt_profile(trial: Any, family: str) -> Dict[str, Any]:
    # ── Shared structural parameters ───────────────────────────────────────
    max_nodes = trial.suggest_int("max_nodes", 4, 8)
    mandate_json = trial.suggest_categorical("mandate_json", ["strict", "very_strict"])
    decomposition = trial.suggest_categorical("decomposition", ["coarse", "balanced", "fine"])
    require_risk_check = trial.suggest_categorical("require_risk_check", [True, False])
    require_validation_step = trial.suggest_categorical("require_validation_step", [True, False])
    include_budget = trial.suggest_categorical("include_budget", [True, False])

    # ── Reasoning DAG-specific parameters ──────────────────────────────────
    # Five distinct frameworks drive genuinely different graph topologies:
    #   top_down        – backwards decomposition from desired output
    #   risk_first      – defensive planning around highest-risk failure points
    #   validation_gated – each step is a conditional gate on its predecessor
    #   metric_anchored – every step references the evaluation metric
    #   assumption_audit – enumerate implicit assumptions and plan to test them
    reasoning_framework = trial.suggest_categorical(
        "reasoning_framework",
        ["top_down", "risk_first", "validation_gated", "metric_anchored", "assumption_audit"],
    )
    require_expected_output = trial.suggest_categorical("require_expected_output", [True, False])

    # ── Implementation DAG-specific parameters (independent search) ────────
    impl_specificity = trial.suggest_categorical(
        "impl_specificity",
        ["task_description", "library_agnostic_ops", "api_guided"],
    )
    impl_output_contract = trial.suggest_categorical("impl_output_contract", [True, False])

    # ── Domain-specific parameters ─────────────────────────────────────────
    if family == "dsbio":
        bio_qc_gate = trial.suggest_categorical("bio_qc_gate", [True, False])
        bio_stat_test_selection = trial.suggest_categorical("bio_stat_test_selection", [True, False])
        bio_interpretation = trial.suggest_categorical("bio_interpretation", [True, False])
    else:
        leakage_prevention = trial.suggest_categorical("leakage_prevention", [True, False])
        metric_alignment = trial.suggest_categorical("metric_alignment", [True, False])
        cv_strategy_explicit = trial.suggest_categorical("cv_strategy_explicit", [True, False])

    # ── Reasoning framework clause ─────────────────────────────────────────
    _framework_clauses: Dict[str, str] = {
        "top_down": (
            "- Decompose from the desired final output backwards: identify what must be true "
            "immediately before the final step, and before that, until you reach the inputs."
        ),
        "risk_first": (
            "- Identify the highest-risk failure points first (data quality, model assumptions, "
            "statistical validity), then structure the plan to mitigate them before each step."
        ),
        "validation_gated": (
            "- Treat each reasoning step as a conditional gate: state the condition that must hold "
            "before the next step proceeds (e.g., 'proceed only if QC passes n_cells > threshold')."
        ),
        "metric_anchored": (
            "- Anchor every reasoning step to the evaluation metric: each node must explain "
            "how it directly contributes to or protects the target metric."
        ),
        "assumption_audit": (
            "- Begin by enumerating the key assumptions implicit in the question "
            "(distributional, biological, domain-specific). Structure the plan to test or "
            "acknowledge each assumption before proceeding."
        ),
    }
    framework_clause = _framework_clauses[reasoning_framework]

    expected_output_clause = (
        "- Each reasoning node must specify the expected intermediate output or decision outcome "
        "(e.g., 'output: set of variable genes', 'decision: test type and significance threshold')."
        if require_expected_output
        else ""
    )

    # ── Implementation specificity clause ──────────────────────────────────
    _impl_specificity_clauses: Dict[str, str] = {
        "task_description": (
            "- Describe each step as a high-level task: what is accomplished, not which library call."
        ),
        "library_agnostic_ops": (
            "- Describe each step as a composable operation "
            "(e.g., 'filter rows where column > threshold', 'fit model on training folds') "
            "without naming specific library APIs."
        ),
        "api_guided": (
            "- Each node may name the primary library or function family as a hint "
            "(e.g., 'scanpy QC metrics', 'sklearn Pipeline + cross_val_score') "
            "to guide the executing agent."
        ),
    }
    impl_specificity_clause = _impl_specificity_clauses[impl_specificity]

    impl_output_contract_clause = (
        "- Each implementation node must state its output contract: shape, type, or path produced "
        "(e.g., 'produces filtered AnnData n_obs < original', 'writes submission.csv [id, target]')."
        if impl_output_contract
        else ""
    )

    # ── Shared structural clauses ──────────────────────────────────────────
    strict_json_clause = (
        "- JSON must parse with a strict RFC8259 parser; no trailing commentary."
        if mandate_json == "very_strict"
        else "- Ensure JSON is syntactically valid and self-contained."
    )
    decomposition_clause = {
        "coarse": "- Prefer broader strategic steps over micro-steps.",
        "fine": "- Prefer finer-grained steps only when they reduce execution risk.",
        "balanced": "- Use balanced granularity to preserve clarity and executability.",
    }[decomposition]
    risk_clause = (
        "- Include explicit failure-risk reasoning (data quality, leakage, unstable assumptions)."
        if require_risk_check
        else ""
    )
    validation_clause = (
        "- Include a validation or checkpoint step before the final answer or submission."
        if require_validation_step
        else ""
    )
    budget_clause = (
        "- Constrain the plan for efficient execution; avoid unnecessary expensive operations."
        if include_budget
        else ""
    )

    # ── Domain-specific clauses ────────────────────────────────────────────
    if family == "dsbio":
        domain_reasoning_extras = "\n".join(
            c for c in [
                (
                    "- Include a QC gate node: reasoning must explicitly decide whether the data "
                    "passes quality thresholds before proceeding to analysis."
                    if bio_qc_gate
                    else ""
                ),
                (
                    "- Include explicit statistical test selection: choose the appropriate test "
                    "(Wilcoxon, t-test, DESeq2, etc.) based on data distribution and sample size, "
                    "not by default."
                    if bio_stat_test_selection
                    else ""
                ),
                (
                    "- The final reasoning node must frame results in terms of biological "
                    "significance, not only statistical significance."
                    if bio_interpretation
                    else ""
                ),
            ]
            if c
        )
        domain_impl_extras = (
            "- Follow the biological analysis pipeline faithfully; do not skip QC or "
            "normalisation steps even if they appear optional."
        )
        base_reasoning = (
            "You are an expert computational biologist.\n"
            "Given a biological question and context, produce a concise JSON reasoning DAG\n"
            "capturing the ordered analytical decisions an expert makes before writing code.\n"
            "\n"
            "Rules:\n"
            '- Return ONLY valid JSON. No markdown or prose outside the JSON.\n'
            "- Each node: one analytical decision, not a code instruction.\n"
            '- Edges need only "source" and "target".\n'
            "- The graph must be acyclic (nodes in execution order).\n"
            "- Label at the decision level, not the library-call level."
        )
        base_impl = (
            "You are an expert computational biologist.\n"
            "Given a biological question, context, and a reasoning DAG, produce a JSON\n"
            "implementation DAG of concrete executable steps that realise each reasoning node.\n"
            "\n"
            "Rules:\n"
            '- Return ONLY valid JSON. No markdown or prose outside the JSON.\n'
            "- Produce exactly one implementation node per reasoning node (1:1 mapping).\n"
            "- Mirror the reasoning DAG structure exactly — same nodes and edges.\n"
            '- Edges need only "source" and "target".\n'
            "- The graph must be acyclic."
        )
    else:
        domain_reasoning_extras = "\n".join(
            c for c in [
                (
                    "- Include an explicit leakage-prevention node: identify features and "
                    "transformations that could introduce target leakage and specify how to prevent it."
                    if leakage_prevention
                    else ""
                ),
                (
                    "- Every reasoning step must reference the competition evaluation metric: "
                    "explain how it optimises or protects that metric."
                    if metric_alignment
                    else ""
                ),
                (
                    "- Include a cross-validation strategy node: choose fold count, "
                    "stratification, and grouping based on the competition data structure."
                    if cv_strategy_explicit
                    else ""
                ),
            ]
            if c
        )
        domain_impl_extras = (
            "- The final implementation node must produce /submission/submission.csv "
            "with the correct column names and format."
        )
        base_reasoning = (
            "You are an expert data scientist and machine learning engineer.\n"
            "Given a Kaggle-style competition description, produce a concise JSON reasoning DAG\n"
            "capturing the ordered analytical decisions an expert makes before writing code.\n"
            "\n"
            "Rules:\n"
            '- Return ONLY valid JSON. No markdown or prose outside the JSON.\n'
            "- Each node: one analytical decision, not a code instruction.\n"
            '- Edges need only "source" and "target".\n'
            "- The graph must be acyclic (nodes in execution order).\n"
            "- Label at the decision level, not the library-call level."
        )
        base_impl = (
            "You are an expert data scientist and machine learning engineer.\n"
            "Given a competition description, context, and a reasoning DAG, produce a JSON\n"
            "implementation DAG of concrete executable steps that realise each reasoning node.\n"
            "\n"
            "Rules:\n"
            '- Return ONLY valid JSON. No markdown or prose outside the JSON.\n'
            "- Produce exactly one implementation node per reasoning node (1:1 mapping).\n"
            "- Mirror the reasoning DAG structure exactly — same nodes and edges.\n"
            '- Edges need only "source" and "target".\n'
            "- The graph must be acyclic."
        )

    # ── Compose per-DAG prompt extensions ─────────────────────────────────
    reasoning_extension = "\n".join(
        x
        for x in [
            strict_json_clause,
            framework_clause,
            decomposition_clause,
            expected_output_clause,
            risk_clause,
            validation_clause,
            budget_clause,
            domain_reasoning_extras,
            f"- Use at most {max_nodes} nodes.",
        ]
        if x
    )
    implementation_extension = "\n".join(
        x
        for x in [
            strict_json_clause,
            decomposition_clause,
            impl_specificity_clause,
            impl_output_contract_clause,
            risk_clause,
            validation_clause,
            budget_clause,
            domain_impl_extras,
            f"- Use at most {max_nodes} nodes.",
        ]
        if x
    )

    profile_section = {
        "reasoning_system_prompt": f"{base_reasoning.strip()}\n{reasoning_extension}".strip(),
        "implementation_system_prompt": f"{base_impl.strip()}\n{implementation_extension}".strip(),
        "reasoning_user_suffix": "Return only JSON object output.",
        "implementation_user_suffix": "Return only JSON object output.",
    }

    return {"global": {}, family: profile_section}


def _compute_trace_summary(results) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total_samples": 0,
            "success_rate": 0.0,
            "error_rate": 1.0,
            "avg_execution_time": 0.0,
            "avg_turns": 0.0,
            "median_turns": 0.0,
            "p90_turns": 0.0,
            "median_execution_time": 0.0,
            "p90_execution_time": 0.0,
            "error_breakdown": {},
        }

    success = sum(1 for r in results if r.success)
    execution_times = [float(getattr(r, "execution_time", 0.0) or 0.0) for r in results]
    turns = [float(getattr(r, "total_turns", 0.0) or 0.0) for r in results]

    error_breakdown: Dict[str, int] = {}
    for result in results:
        error_info = getattr(result, "error_info", None) or {}
        category = error_info.get("error_category") if isinstance(error_info, dict) else None
        if category:
            error_breakdown[category] = error_breakdown.get(category, 0) + 1

    sorted_turns = sorted(turns)
    sorted_times = sorted(execution_times)
    p90_idx = max(0, min(total - 1, int(0.9 * (total - 1))))

    return {
        "total_samples": total,
        "success_rate": success / total,
        "error_rate": 1.0 - (success / total),
        "avg_execution_time": statistics.fmean(execution_times),
        "avg_turns": statistics.fmean(turns),
        "median_turns": statistics.median(sorted_turns),
        "p90_turns": sorted_turns[p90_idx],
        "median_execution_time": statistics.median(sorted_times),
        "p90_execution_time": sorted_times[p90_idx],
        "error_breakdown": error_breakdown,
    }


def _trial_output_dir(base: Path, trial_number: int) -> Path:
    path = base / "trials" / f"trial_{trial_number:04d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_prompt_nsga2(config: PromptOptimizationConfig) -> Dict[str, Any]:
    try:
        import optuna  # type: ignore[import-not-found]
        from optuna.samplers import NSGAIISampler  # type: ignore[import-not-found]
    except Exception as exc:
        raise ImportError("optuna is required for prompt optimization. Install with: uv sync --extra optimize") from exc

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, tasks = _load_tasks(config)
    family = _dataset_family(config.dataset)

    storage = f"sqlite:///{(out_dir / 'optuna_study.db').as_posix()}"
    sampler = NSGAIISampler(seed=config.seed, population_size=config.population_size)

    # Three independent objectives: success_rate is maximised; avg_execution_time and
    # avg_turns are minimised.  error_rate = 1 - success_rate is perfectly linearly
    # dependent on success_rate and therefore carries zero additional information for
    # Pareto dominance.  Including it would inflate the objective space dimensionality
    # from 3 to 4, waste NSGA-II's crowding-distance budget, and make the Pareto front
    # harder to interpret — with no benefit.
    study = optuna.create_study(
        study_name=config.study_name,
        directions=["maximize", "minimize", "minimize"],
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: Any) -> Tuple[float, float, float]:
        trial_dir = _trial_output_dir(out_dir, trial.number)
        profile = _compose_prompt_profile(trial, family)

        if config.tune_dag_validation_mode:
            dag_validation_mode = trial.suggest_categorical(
                "dag_validation_mode",
                ["default", "relaxed", "strict"],
            )
        else:
            dag_validation_mode = config.dag_validation_mode

        profile.setdefault("global", {})["dag_validation_mode"] = dag_validation_mode

        profile_path = trial_dir / "prompt_profile.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

        old_profile = os.environ.get("DSGYM_DAG_PROMPT_PROFILE")
        old_dag_validation_mode = os.environ.get("DSGYM_DAG_VALIDATION_MODE")
        os.environ["DSGYM_DAG_PROMPT_PROFILE"] = str(profile_path)
        os.environ["DSGYM_DAG_VALIDATION_MODE"] = dag_validation_mode

        try:
            agent_cfg: Dict[str, Any] = {
                "manager_url": config.manager_url,
                "max_turns": config.max_turns,
                "temperature": config.temperature,
                "output_dir": str(trial_dir / "agent_outputs"),
            }
            if config.api_key:
                agent_cfg["api_key"] = config.api_key

            if family == "dspredict":
                agent = DSPredictReActAgent(
                    backend=config.backend,
                    model=config.model,
                    submission_dir="./submissions",
                    **agent_cfg,
                )
            else:
                agent = ReActDSAgent(
                    backend=config.backend,
                    model=config.model,
                    **agent_cfg,
                )

            evaluator = Evaluator(protocol="multi_turn", dataset=dataset, parallel_workers=config.max_workers)
            eval_config = EvaluationConfig(
                model_name=config.model,
                backend_type=config.backend,
                dataset_name=config.dataset,
                output_dir=str(trial_dir / "eval"),
                run_name=f"trial_{trial.number}",
                max_turns=config.max_turns,
                temperature=config.temperature,
                max_workers=config.max_workers,
                extra_config={
                    "generate_dags": True,
                    "dag_model": config.dag_model,
                    "dag_validation_mode": dag_validation_mode,
                    "task_timeout_seconds": config.task_timeout_seconds,
                },
            )

            eval_result = evaluator.evaluate(
                agent=agent,
                tasks=tasks,
                config=eval_config,
                save_results=False,
                show_progress=False,
            )

            results = eval_result.get("results", [])
            aggregated_metrics = eval_result.get("metrics", {})
            trace_summary = _compute_trace_summary(results)

            # error_rate (= 1 - success_rate) is kept in trace_summary for
            # reporting/analysis but is NOT included as an Optuna objective because
            # it is perfectly linearly dependent on success_rate.
            objectives = (
                float(trace_summary["success_rate"]),       # index 0 — maximize
                float(trace_summary["avg_execution_time"]), # index 1 — minimize
                float(trace_summary["avg_turns"]),          # index 2 — minimize
            )

            trial_record = {
                "trial_number": trial.number,
                "params": trial.params,
                "objectives": {
                    "success_rate": objectives[0],
                    "avg_execution_time": objectives[1],
                    "avg_turns": objectives[2],
                },
                # error_rate stored separately for analysis; not an optimiser objective
                "error_rate": float(trace_summary["error_rate"]),
                "trace_summary": trace_summary,
                "aggregated_metrics": aggregated_metrics,
                "prompt_profile": profile,
                "dag_validation_mode": dag_validation_mode,
            }

            with open(trial_dir / "trial_record.json", "w", encoding="utf-8") as f:
                json.dump(trial_record, f, indent=2, ensure_ascii=False)

            trial.set_user_attr("trace_summary", trace_summary)
            trial.set_user_attr("prompt_profile_path", str(profile_path))
            trial.set_user_attr("dag_validation_mode", dag_validation_mode)
            return objectives
        finally:
            if old_profile is None:
                os.environ.pop("DSGYM_DAG_PROMPT_PROFILE", None)
            else:
                os.environ["DSGYM_DAG_PROMPT_PROFILE"] = old_profile
            if old_dag_validation_mode is None:
                os.environ.pop("DSGYM_DAG_VALIDATION_MODE", None)
            else:
                os.environ["DSGYM_DAG_VALIDATION_MODE"] = old_dag_validation_mode

    study.optimize(objective, n_trials=config.n_trials)

    pareto = []
    for trial in study.best_trials:
        pareto.append(
            {
                "trial_number": trial.number,
                "values": {
                    "success_rate": trial.values[0],
                    "avg_execution_time": trial.values[1],
                    "avg_turns": trial.values[2],
                },
                "error_rate": trial.user_attrs.get("trace_summary", {}).get("error_rate"),
                "params": trial.params,
                "dag_validation_mode": trial.user_attrs.get("dag_validation_mode", config.dag_validation_mode),
                "prompt_profile_path": trial.user_attrs.get("prompt_profile_path"),
                "trace_summary": trial.user_attrs.get("trace_summary", {}),
            }
        )

    summary = {
        "study_name": config.study_name,
        "dataset": config.dataset,
        "model": config.model,
        "backend": config.backend,
        "dag_validation_mode": config.dag_validation_mode,
        "tune_dag_validation_mode": config.tune_dag_validation_mode,
        "n_trials": config.n_trials,
        "population_size": config.population_size,
        "directions": ["maximize(success_rate)", "minimize(avg_execution_time)", "minimize(avg_turns)"],
        "pareto_front": pareto,
    }

    with open(out_dir / "optimization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
