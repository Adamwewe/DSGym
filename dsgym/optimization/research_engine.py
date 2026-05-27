from __future__ import annotations

import copy
import json
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dsgym.agents import DSPredictReActAgent, ReActDSAgent
from dsgym.datasets import DatasetRegistry
from dsgym.datasets.dag_generator import _supports_json_object_format
from dsgym.eval import EvaluationConfig, Evaluator
from dsgym.optimization.harness_evaluator import StrictHarnessEvaluator, StrictHarnessPolicy
from dsgym.optimization.interfaces import (
    HarnessPolicyAdapter,
    ModeEvaluationRecord,
    ModeEvaluatorAdapter,
    QualityMetricAdapter,
    ResearchContextAdapter,
    TaskMutatorAdapter,
)
from dsgym.optimization.prompt_nsga2 import _compose_prompt_profile


@dataclass
class TextVariant:
    name: str
    text: str


@dataclass
class ResearchOptimizationConfig:
    model: str
    dataset: str = "dsbio"
    backend: str = "litellm"
    dag_model: Optional[str] = None
    manager_url: str = "http://localhost:5000"
    api_key: Optional[str] = None
    limit: int = 6
    sample_n: Optional[int] = None
    sample_seed: int = 42
    temperature: float = 0.0
    max_workers: int = 2
    task_timeout_seconds: Optional[float] = 900.0
    dag_validation_mode: str = "default"
    tune_dag_validation_mode: bool = True
    ablate_dags: bool = True
    max_turns_min: int = 10
    max_turns_max: int = 18
    n_trials: int = 24
    population_size: int = 8
    seed: int = 42
    study_name: str = "dsgym_research_nsga2"
    output_dir: str = "./optimization_results/research"
    generation_model: Optional[str] = None
    generated_variants_per_type: int = 3
    reuse_variant_bank_cache: bool = True
    force_regenerate_variants: bool = False
    strict_harness_evaluator: bool = True
    harness_max_identical_message_repeats: int = 1
    harness_max_identical_code_repeats: int = 1


def _dataset_family(dataset: str) -> str:
    if dataset.startswith("dspredict"):
        return "dspredict"
    return "dsbio"


def _resolve_dspredict_split(dataset: str) -> str:
    if "-" in dataset:
        return dataset.split("-", 1)[1]
    return "hard"


def _prepare_task_bank(config: ResearchOptimizationConfig):
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

        kwargs: Dict[str, Any] = {"limit": config.limit}
        if split is not None:
            kwargs["split"] = split

        tasks_with_dags = dataset.load(generate_dags=True, **kwargs)
        tasks_without_dags = dataset.load(generate_dags=False, **kwargs)
        return dataset, family, {True: tasks_with_dags, False: tasks_without_dags}

    dataset = DatasetRegistry.load("dspredict", virtual_data_root="/data")
    split = _resolve_dspredict_split(config.dataset)
    if config.sample_n is not None:
        source_split = split if split != "sampled" else "hard"
        dataset.create_sample(n=config.sample_n, split=source_split, seed=config.sample_seed)
        split = "sampled"

    kwargs = {"limit": config.limit, "split": split}
    tasks_with_dags = dataset.load(generate_dags=True, **kwargs)
    tasks_without_dags = dataset.load(generate_dags=False, **kwargs)
    return dataset, family, {True: tasks_with_dags, False: tasks_without_dags}


def _template_prompt_variants(family: str) -> List[TextVariant]:
    if family == "dspredict":
        return [
            TextVariant(name="prompt_baseline", text=""),
            TextVariant(
                name="prompt_metric_first",
                text=(
                    "Before training, explicitly restate the evaluation metric and optimize every design "
                    "choice for that metric. Report one strong baseline and one improved variant."
                ),
            ),
            TextVariant(
                name="prompt_risk_budgeted",
                text=(
                    "In each step, include one concrete execution-risk check and one compute budget decision. "
                    "Prefer robust improvements over brittle leaderboard tricks."
                ),
            ),
            TextVariant(
                name="prompt_submission_guard",
                text=(
                    "Treat submission validity as a first-class objective: verify schema, row count, and nulls "
                    "before finishing."
                ),
            ),
        ]

    return [
        TextVariant(name="prompt_baseline", text=""),
        TextVariant(
            name="prompt_hypothesis_driven",
            text=(
                "Start from explicit biological hypotheses and tie each implementation step to one hypothesis "
                "or assumption check."
            ),
        ),
        TextVariant(
            name="prompt_qc_first",
            text=(
                "Treat data quality and confounder checks as gating conditions before downstream inference "
                "or interpretation."
            ),
        ),
        TextVariant(
            name="prompt_reproducibility_guard",
            text=(
                "Prioritize reproducible analysis choices: deterministic seeds, explicit thresholds, and concise "
                "justification for each key decision."
            ),
        ),
    ]


def _template_context_variants(family: str) -> List[TextVariant]:
    common = [
        TextVariant(name="context_baseline", text=""),
        TextVariant(
            name="context_compact_brief",
            text=(
                "RESEARCH BRIEF:\n"
                "- Restate the objective in one sentence.\n"
                "- List top 3 unknowns.\n"
                "- State one likely failure mode and mitigation."
            ),
        ),
        TextVariant(
            name="context_acceptance_checks",
            text=(
                "ACCEPTANCE CHECKS:\n"
                "- Intermediate outputs must be inspected before moving on.\n"
                "- If evidence contradicts assumptions, revise the remaining plan.\n"
                "- Final answer must cite the final validating evidence."
            ),
        ),
    ]

    if family == "dspredict":
        common.append(
            TextVariant(
                name="context_generalization_focus",
                text=(
                    "GENERALIZATION FOCUS:\n"
                    "- Prefer validation stability over single split gains.\n"
                    "- Track signs of leakage and overfitting every time a model is updated."
                ),
            )
        )
    else:
        common.append(
            TextVariant(
                name="context_bio_focus",
                text=(
                    "BIO FOCUS:\n"
                    "- Surface biological assumptions before statistical tests.\n"
                    "- Distinguish exploratory from confirmatory conclusions."
                ),
            )
        )

    return common


def _template_harness_variants() -> List[TextVariant]:
    return [
        TextVariant(name="harness_baseline", text=""),
        TextVariant(
            name="harness_strict_progress",
            text=(
                "HARNESS CONSTRAINTS:\n"
                "- Multi-step planning and revisions are allowed.\n"
                "- Skipping planned steps is forbidden unless a plan revision block is emitted first.\n"
                "- Repeating near-identical steps without new evidence is forbidden.\n"
                "- If blocked, revise the plan instead of looping."
            ),
        ),
        TextVariant(
            name="harness_obligation_contract",
            text=(
                "EXECUTION CONTRACT:\n"
                "1) Follow the current plan in order.\n"
                "2) For any deviation, emit a revision note and updated remaining plan.\n"
                "3) Do not execute the same failed tactic more than once without a changed hypothesis."
            ),
        ),
    ]


_KIND_LABELS: Dict[str, str] = {
    "prompt": (
        "reasoning and implementation DAG planning directives — analytical framework choice "
        "(risk-first, validation-gated, metric-anchored, assumption-audit), "
        "domain-specific reasoning gates (QC gates, leakage prevention, statistical test selection), "
        "and step decomposition strategy injected into the DAG generation system prompts"
    ),
    "context": (
        "task framing presented to the agent after DAGs are generated — "
        "how the problem statement, acceptance criteria, and intermediate output expectations "
        "are injected into the agent's user message"
    ),
    "harness": (
        "agent-DAG compliance contracts — plan adherence rules, "
        "loop and repetition prohibition, and deviation/revision protocol "
        "for when the agent must depart from the DAG plan"
    ),
}

_DOMAIN_DESCRIPTIONS: Dict[str, str] = {
    "dsbio": (
        "computational biology tasks: single-cell RNA-seq analysis, differential expression, "
        "pathway enrichment, statistical hypothesis testing on biological datasets. "
        "Agents must handle biological assumptions explicitly, gate on QC results, "
        "and produce reproducible analytical choices."
    ),
    "dspredict": (
        "Kaggle-style ML competition tasks: feature engineering, model selection, "
        "hyperparameter tuning, leakage prevention, and valid submission generation. "
        "Agents must respect evaluation metrics and avoid overfitting to the public leaderboard."
    ),
}


def _first_json_object(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        return {}

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}


def _llm_generate_variants(
    kind: str,
    family: str,
    model: str,
    count: int,
    *,
    existing_names: Optional[List[str]] = None,
) -> List[TextVariant]:
    try:
        import litellm
    except Exception:
        return []

    if count <= 0:
        return []

    domain_description = _DOMAIN_DESCRIPTIONS.get(family, f"{family} data science tasks")
    kind_label = _KIND_LABELS.get(kind, kind)
    existing_str = (
        ", ".join(existing_names) if existing_names else "none yet — all angles are available"
    )

    system_prompt = (
        "You are an expert AI research engineer designing precise, domain-specific directives "
        "for data-science ReAct agents. "
        "The agent operates in a multi-turn loop using <reasoning>, <python>, and <information> "
        "blocks. Each variant you generate is injected as a short addendum to the agent's existing "
        "system or user prompt — it does NOT replace the full prompt. "
        "Variants must be immediately actionable, specific to the target dimension and domain, "
        "and clearly distinct from any existing variants. Return only JSON."
    )
    user_prompt = (
        f"Generate {count} distinct '{kind}' variants for a data-science agent.\n\n"
        f"Target dimension: {kind_label}\n"
        f"Domain: {domain_description}\n\n"
        f"Existing variant names (your variants MUST take a different angle from these): {existing_str}\n\n"
        "Requirements:\n"
        "- Each variant text: 15–80 words, concrete and specific (no generic advice).\n"
        "- Each variant must address the target dimension from a clearly distinct perspective.\n"
        "- Names must be concise unique snake_case identifiers.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "variants": [\n'
        '    {"name": "short_snake_case", "text": "directive text"}\n'
        "  ]\n"
        "}"
    )

    call_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8,
    }
    if _supports_json_object_format(model):
        call_kwargs["response_format"] = {"type": "json_object"}

    try:
        response = litellm.completion(**call_kwargs)
        content = response.choices[0].message.content
    except Exception:
        return []

    payload = _first_json_object(content)
    variants = payload.get("variants", []) if isinstance(payload, dict) else []
    generated: List[TextVariant] = []

    if not isinstance(variants, list):
        return []

    for idx, item in enumerate(variants):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip().lower()
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if not name:
            name = f"{kind}_gen_{idx + 1}"
        generated.append(TextVariant(name=name, text=text))

    return generated


def _dedupe_variants(variants: List[TextVariant]) -> List[TextVariant]:
    seen: set[str] = set()
    deduped: List[TextVariant] = []
    for variant in variants:
        signature = f"{variant.name.strip().lower()}::{variant.text.strip()}"
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(variant)
    return deduped


def _variants_from_payload(payload: Dict[str, Any], key: str) -> List[TextVariant]:
    values = payload.get(key, []) if isinstance(payload.get(key, []), list) else []
    variants: List[TextVariant] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        text = str(item.get("text", "")).strip()
        if not name:
            continue
        variants.append(TextVariant(name=name, text=text))
    return variants


def _build_variant_bank(config: ResearchOptimizationConfig, family: str, out_dir: Path) -> Dict[str, List[TextVariant]]:
    cache_path = out_dir / "variant_bank.json"

    if config.reuse_variant_bank_cache and not config.force_regenerate_variants and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                cached = {
                    "prompt": _variants_from_payload(payload, "prompt"),
                    "context": _variants_from_payload(payload, "context"),
                    "harness": _variants_from_payload(payload, "harness"),
                }
                if all(cached[key] for key in ("prompt", "context", "harness")):
                    return cached
        except Exception:
            pass

    prompt_variants = _template_prompt_variants(family)
    context_variants = _template_context_variants(family)
    harness_variants = _template_harness_variants()

    if config.generation_model:
        prompt_variants.extend(
            _llm_generate_variants(
                kind="prompt",
                family=family,
                model=config.generation_model,
                count=config.generated_variants_per_type,
                existing_names=[v.name for v in prompt_variants],
            )
        )
        context_variants.extend(
            _llm_generate_variants(
                kind="context",
                family=family,
                model=config.generation_model,
                count=config.generated_variants_per_type,
                existing_names=[v.name for v in context_variants],
            )
        )
        harness_variants.extend(
            _llm_generate_variants(
                kind="harness",
                family=family,
                model=config.generation_model,
                count=config.generated_variants_per_type,
                existing_names=[v.name for v in harness_variants],
            )
        )

    bank = {
        "prompt": _dedupe_variants(prompt_variants),
        "context": _dedupe_variants(context_variants),
        "harness": _dedupe_variants(harness_variants),
    }

    serializable = {
        "_meta": {
            "family": family,
            "generation_model": config.generation_model,
            "generated_variants_per_type": config.generated_variants_per_type,
        },
        **{key: [asdict(variant) for variant in variants] for key, variants in bank.items()},
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    return bank


def _mutate_tasks(
    base_tasks: List[Dict[str, Any]],
    prompt_variant: TextVariant,
    context_variant: TextVariant,
    harness_variant: TextVariant,
) -> List[Dict[str, Any]]:
    tasks = copy.deepcopy(base_tasks)

    for task in tasks:
        conversation = task.get("prompt", [])
        if not isinstance(conversation, list) or len(conversation) < 2:
            continue

        system_msg = conversation[0]
        user_msg = conversation[1]
        if not isinstance(system_msg, dict) or not isinstance(user_msg, dict):
            continue

        system_content = str(system_msg.get("content", ""))
        user_content = str(user_msg.get("content", ""))

        system_chunks = [system_content]
        if prompt_variant.text:
            system_chunks.append(prompt_variant.text)
        if harness_variant.text:
            system_chunks.append(harness_variant.text)
        system_msg["content"] = "\n\n".join(chunk.strip() for chunk in system_chunks if chunk.strip())

        user_chunks = [user_content]
        if context_variant.text:
            user_chunks.append(context_variant.text)
        if harness_variant.text:
            user_chunks.append(
                "Compliance reminder: follow the current plan order; revise before deviating; avoid repeated loops."
            )
        user_msg["content"] = "\n\n".join(chunk.strip() for chunk in user_chunks if chunk.strip())

        extra_info = task.get("extra_info", {})
        if isinstance(extra_info, dict):
            extra_info["research_variants"] = {
                "prompt": prompt_variant.name,
                "context": context_variant.name,
                "harness": harness_variant.name,
            }

    return tasks


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _metric_score(metric_payload: Any) -> Optional[float]:
    if isinstance(metric_payload, (int, float)):
        return float(metric_payload)
    if isinstance(metric_payload, dict):
        score = metric_payload.get("score")
        try:
            return float(score)
        except Exception:
            return None
    return None


def _compute_quality_summary(results: List[Any], aggregated_metrics: Dict[str, Any]) -> Dict[str, Any]:
    total_samples = len(results)

    exact_scores: List[float] = []
    for result in results:
        metric_payload = getattr(result, "metrics", {}).get("exact_match")
        score = _metric_score(metric_payload)
        if score is not None:
            exact_scores.append(score)

    exact_match_correct = float(sum(1 for score in exact_scores if score >= 0.999999))
    exact_match_evaluable = float(len(exact_scores))
    # Denominator is evaluable count, not total_samples — tasks without exact-match
    # metrics must not dilute the accuracy of those that do have them.
    exact_match_accuracy = (
        exact_match_correct / exact_match_evaluable
        if exact_match_evaluable > 0
        else None
    )

    quality_metric_candidates = [
        "exact_match",
        "fuzzy_exact_match",
        "semantic_similarity",
        "kaggle_submission",
        "mlebench_submission",
    ]

    quality_metric_name = "success_rate"
    quality_mean = _safe_float(aggregated_metrics.get("success_rate", 0.0))

    for metric_name in quality_metric_candidates:
        key = f"{metric_name}_mean"
        if key in aggregated_metrics:
            quality_metric_name = metric_name
            quality_mean = _safe_float(aggregated_metrics.get(key, 0.0))
            break

    return {
        "total_samples": float(total_samples),
        "quality_metric": quality_metric_name,
        "quality_mean": quality_mean,
        "exact_match_evaluable": exact_match_evaluable,
        "exact_match_correct": exact_match_correct,
        "exact_match_accuracy": exact_match_accuracy,
        "successful_samples": _safe_float(aggregated_metrics.get("successful_samples", 0.0)),
        "error_samples": _safe_float(aggregated_metrics.get("error_samples", 0.0)),
    }


class DefaultQualityMetricAdapter:
    def summarize(self, results: List[Any], aggregated_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return _compute_quality_summary(results, aggregated_metrics=aggregated_metrics)


class DefaultHarnessPolicyAdapter:
    def __init__(self, config: ResearchOptimizationConfig):
        if config.strict_harness_evaluator:
            self._evaluator: Optional[StrictHarnessEvaluator] = StrictHarnessEvaluator(
                policy=StrictHarnessPolicy(
                    max_identical_message_repeats=config.harness_max_identical_message_repeats,
                    max_identical_code_repeats=config.harness_max_identical_code_repeats,
                )
            )
        else:
            self._evaluator = None

    def summarize(self, results: List[Any], max_turns: int) -> Dict[str, Any]:
        if self._evaluator is None:
            return {
                "policy": {},
                "avg_compliance": 0.0,
                "violation_rates": {},
                "counters": {},
            }
        return self._evaluator.summarize(results, max_turns=max_turns)


class DefaultDSGymContextAdapter:
    def prepare_context(self, config: ResearchOptimizationConfig):
        return _prepare_task_bank(config)


class DefaultTaskMutatorAdapter:
    def mutate(
        self,
        base_tasks: List[Dict[str, Any]],
        prompt_variant: TextVariant,
        context_variant: TextVariant,
        harness_variant: TextVariant,
    ) -> List[Dict[str, Any]]:
        return _mutate_tasks(
            base_tasks=base_tasks,
            prompt_variant=prompt_variant,
            context_variant=context_variant,
            harness_variant=harness_variant,
        )


def _trial_output_dir(base: Path, trial_number: int) -> Path:
    path = base / "trials" / f"trial_{trial_number:04d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _evaluate_mode(
    *,
    config: ResearchOptimizationConfig,
    family: str,
    dataset: Any,
    tasks: List[Dict[str, Any]],
    use_dags: bool,
    max_turns: int,
    dag_validation_mode: str,
    trial_dir: Path,
    mode_tag: str,
    quality_adapter: QualityMetricAdapter,
    harness_adapter: HarnessPolicyAdapter,
) -> ModeEvaluationRecord:
    agent_cfg: Dict[str, Any] = {
        "manager_url": config.manager_url,
        "max_turns": max_turns,
        "temperature": config.temperature,
        "output_dir": str(trial_dir / f"agent_outputs_{mode_tag}"),
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
        output_dir=str(trial_dir / f"eval_{mode_tag}"),
        run_name=f"{mode_tag}",
        max_turns=max_turns,
        temperature=config.temperature,
        max_workers=config.max_workers,
        extra_config={
            "generate_dags": use_dags,
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
    quality_summary = quality_adapter.summarize(results, aggregated_metrics=aggregated_metrics)
    harness_summary = harness_adapter.summarize(results, max_turns=max_turns)

    return ModeEvaluationRecord(
        use_dags=use_dags,
        trace_summary=trace_summary,
        quality_summary=quality_summary,
        harness_summary=harness_summary,
        aggregated_metrics=aggregated_metrics,
    )


class DefaultDSGymModeEvaluatorAdapter:
    def evaluate(
        self,
        *,
        config: Any,
        family: str,
        dataset: Any,
        tasks: List[Dict[str, Any]],
        use_dags: bool,
        max_turns: int,
        dag_validation_mode: str,
        trial_dir: Path,
        mode_tag: str,
        quality_adapter: QualityMetricAdapter,
        harness_adapter: HarnessPolicyAdapter,
    ) -> ModeEvaluationRecord:
        return _evaluate_mode(
            config=config,
            family=family,
            dataset=dataset,
            tasks=tasks,
            use_dags=use_dags,
            max_turns=max_turns,
            dag_validation_mode=dag_validation_mode,
            trial_dir=trial_dir,
            mode_tag=mode_tag,
            quality_adapter=quality_adapter,
            harness_adapter=harness_adapter,
        )


def run_research_nsga2(
    config: ResearchOptimizationConfig,
    *,
    context_adapter: Optional[ResearchContextAdapter] = None,
    task_mutator_adapter: Optional[TaskMutatorAdapter] = None,
    mode_evaluator_adapter: Optional[ModeEvaluatorAdapter] = None,
    quality_adapter: Optional[QualityMetricAdapter] = None,
    harness_adapter: Optional[HarnessPolicyAdapter] = None,
) -> Dict[str, Any]:
    try:
        import optuna  # type: ignore[import-not-found]
        from optuna.samplers import NSGAIISampler  # type: ignore[import-not-found]
    except Exception as exc:
        raise ImportError(
            "optuna is required for research optimization. Install with: uv sync --extra optimize"
        ) from exc

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context_adapter_impl = context_adapter or DefaultDSGymContextAdapter()
    task_mutator_impl = task_mutator_adapter or DefaultTaskMutatorAdapter()

    dataset, family, task_bank = context_adapter_impl.prepare_context(config)
    variant_bank = _build_variant_bank(config, family, out_dir)

    storage = f"sqlite:///{(out_dir / 'optuna_study.db').as_posix()}"
    sampler = NSGAIISampler(seed=config.seed, population_size=config.population_size)
    mode_evaluator = mode_evaluator_adapter or DefaultDSGymModeEvaluatorAdapter()
    quality_adapter_impl = quality_adapter or DefaultQualityMetricAdapter()
    harness_adapter_impl = harness_adapter or DefaultHarnessPolicyAdapter(config)

    study = optuna.create_study(
        study_name=config.study_name,
        directions=[
            "maximize",  # mean quality score (exact-match/primary metric)
            "minimize",  # mean execution time
            "minimize",  # mean turns
            "maximize",  # harness compliance
            "maximize",  # dag help delta (dags - no_dags)
        ],
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    prompt_variants = variant_bank["prompt"]
    context_variants = variant_bank["context"]
    harness_variants = variant_bank["harness"]

    def objective(trial: Any) -> Tuple[float, float, float, float, float]:
        trial_dir = _trial_output_dir(out_dir, trial.number)

        prompt_idx = trial.suggest_int("prompt_variant_idx", 0, len(prompt_variants) - 1)
        context_idx = trial.suggest_int("context_variant_idx", 0, len(context_variants) - 1)
        harness_idx = trial.suggest_int("harness_variant_idx", 0, len(harness_variants) - 1)

        prompt_variant = prompt_variants[prompt_idx]
        context_variant = context_variants[context_idx]
        harness_variant = harness_variants[harness_idx]

        max_turns = trial.suggest_int("max_turns", config.max_turns_min, config.max_turns_max)

        if config.tune_dag_validation_mode:
            dag_validation_mode = trial.suggest_categorical(
                "dag_validation_mode",
                ["default", "relaxed", "strict"],
            )
        else:
            dag_validation_mode = config.dag_validation_mode

        if config.ablate_dags:
            modes = [True, False]
        else:
            use_dags = trial.suggest_categorical("use_dags", [True, False])
            modes = [use_dags]

        profile = _compose_prompt_profile(trial, family)
        profile.setdefault("global", {})["dag_validation_mode"] = dag_validation_mode
        profile.setdefault("global", {})["selected_prompt_variant"] = prompt_variant.name
        profile.setdefault("global", {})["selected_context_variant"] = context_variant.name
        profile.setdefault("global", {})["selected_harness_variant"] = harness_variant.name

        if prompt_variant.text:
            section = profile.setdefault(family, {})
            if isinstance(section, dict):
                for key in ("reasoning_system_prompt", "implementation_system_prompt"):
                    value = section.get(key)
                    if isinstance(value, str) and value.strip():
                        section[key] = f"{value.strip()}\n{prompt_variant.text}"

        profile_path = trial_dir / "prompt_profile.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

        old_profile = os.environ.get("DSGYM_DAG_PROMPT_PROFILE")
        old_dag_validation_mode = os.environ.get("DSGYM_DAG_VALIDATION_MODE")
        os.environ["DSGYM_DAG_PROMPT_PROFILE"] = str(profile_path)
        os.environ["DSGYM_DAG_VALIDATION_MODE"] = dag_validation_mode

        try:
            per_mode: Dict[str, Dict[str, Any]] = {}
            for mode in modes:
                base_tasks = task_bank[mode]
                mode_tag = "dags" if mode else "nodags"
                trial_tasks = task_mutator_impl.mutate(
                    base_tasks=base_tasks,
                    prompt_variant=prompt_variant,
                    context_variant=context_variant,
                    harness_variant=harness_variant,
                )
                mode_record = mode_evaluator.evaluate(
                    config=config,
                    family=family,
                    dataset=dataset,
                    tasks=trial_tasks,
                    use_dags=mode,
                    max_turns=max_turns,
                    dag_validation_mode=dag_validation_mode,
                    trial_dir=trial_dir,
                    mode_tag=mode_tag,
                    quality_adapter=quality_adapter_impl,
                    harness_adapter=harness_adapter_impl,
                )
                per_mode[mode_tag] = asdict(mode_record)

            quality_values = [
                float(mode_record["quality_summary"].get("quality_mean", 0.0))
                for mode_record in per_mode.values()
            ]
            success_values = [
                float(mode_record["trace_summary"].get("success_rate", 0.0))
                for mode_record in per_mode.values()
            ]
            time_values = [
                float(mode_record["trace_summary"].get("avg_execution_time", 0.0))
                for mode_record in per_mode.values()
            ]
            turn_values = [
                float(mode_record["trace_summary"].get("avg_turns", 0.0))
                for mode_record in per_mode.values()
            ]
            compliance_values = [
                float(mode_record["harness_summary"].get("avg_compliance", 0.0))
                for mode_record in per_mode.values()
            ]

            mean_quality = statistics.fmean(quality_values) if quality_values else 0.0
            mean_success = statistics.fmean(success_values) if success_values else 0.0
            mean_time = statistics.fmean(time_values) if time_values else 0.0
            mean_turns = statistics.fmean(turn_values) if turn_values else 0.0
            mean_compliance = statistics.fmean(compliance_values) if compliance_values else 0.0

            exact_accuracy_values = [
                float(mode_record["quality_summary"]["exact_match_accuracy"])
                for mode_record in per_mode.values()
                if mode_record["quality_summary"].get("exact_match_accuracy") is not None
            ]
            mean_exact_match_accuracy = (
                statistics.fmean(exact_accuracy_values) if exact_accuracy_values else None
            )

            quality_dags = per_mode.get("dags", {}).get("quality_summary", {}).get("quality_mean")
            quality_nodags = per_mode.get("nodags", {}).get("quality_summary", {}).get("quality_mean")
            if quality_dags is not None and quality_nodags is not None:
                dag_help_delta = float(quality_dags) - float(quality_nodags)
            else:
                dag_help_delta = 0.0

            objectives = (
                mean_quality,
                mean_time,
                mean_turns,
                mean_compliance,
                dag_help_delta,
            )

            trial_record = {
                "trial_number": trial.number,
                "params": trial.params,
                "selected_variants": {
                    "prompt": asdict(prompt_variant),
                    "context": asdict(context_variant),
                    "harness": asdict(harness_variant),
                },
                "max_turns": max_turns,
                "dag_validation_mode": dag_validation_mode,
                "per_mode": per_mode,
                "objectives": {
                    "mean_quality_score": objectives[0],
                    "mean_success_rate": mean_success,
                    "mean_execution_time": objectives[1],
                    "mean_turns": objectives[2],
                    "mean_harness_compliance": objectives[3],
                    "dag_help_delta": objectives[4],
                    "mean_exact_match_accuracy": mean_exact_match_accuracy,
                },
            }

            with open(trial_dir / "trial_record.json", "w", encoding="utf-8") as f:
                json.dump(trial_record, f, indent=2, ensure_ascii=False)

            trial.set_user_attr("trial_record_path", str(trial_dir / "trial_record.json"))
            trial.set_user_attr("selected_variants", trial_record["selected_variants"])
            trial.set_user_attr("per_mode", per_mode)
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

    pareto: List[Dict[str, Any]] = []
    for trial in study.best_trials:
        per_mode = trial.user_attrs.get("per_mode", {}) if isinstance(trial.user_attrs, dict) else {}
        pareto.append(
            {
                "trial_number": trial.number,
                "values": {
                    "mean_quality_score": trial.values[0],
                    "mean_execution_time": trial.values[1],
                    "mean_turns": trial.values[2],
                    "mean_harness_compliance": trial.values[3],
                    "dag_help_delta": trial.values[4],
                },
                "params": trial.params,
                "selected_variants": trial.user_attrs.get("selected_variants", {}),
                "per_mode": per_mode,
                "trial_record_path": trial.user_attrs.get("trial_record_path"),
            }
        )

    summary = {
        "study_name": config.study_name,
        "dataset": config.dataset,
        "model": config.model,
        "backend": config.backend,
        "ablate_dags": config.ablate_dags,
        "strict_harness_evaluator": config.strict_harness_evaluator,
        "harness_max_identical_message_repeats": config.harness_max_identical_message_repeats,
        "harness_max_identical_code_repeats": config.harness_max_identical_code_repeats,
        "reuse_variant_bank_cache": config.reuse_variant_bank_cache,
        "force_regenerate_variants": config.force_regenerate_variants,
        "mode_evaluator_adapter": type(mode_evaluator).__name__,
        "context_adapter": type(context_adapter_impl).__name__,
        "task_mutator_adapter": type(task_mutator_impl).__name__,
        "quality_adapter": type(quality_adapter_impl).__name__,
        "harness_adapter": type(harness_adapter_impl).__name__,
        "n_trials": config.n_trials,
        "population_size": config.population_size,
        "objectives": [
            "maximize(mean_quality_score)",
            "minimize(mean_execution_time)",
            "minimize(mean_turns)",
            "maximize(mean_harness_compliance)",
            "maximize(dag_help_delta)",
        ],
        "variant_bank_path": str(out_dir / "variant_bank.json"),
        "pareto_front": pareto,
    }

    with open(out_dir / "research_optimization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
