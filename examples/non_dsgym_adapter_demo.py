#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dsgym.optimization import (
    DefaultHarnessPolicyAdapter,
    DefaultQualityMetricAdapter,
    ModeEvaluationRecord,
    ResearchOptimizationConfig,
    run_research_nsga2,
)


def _unit_hash(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class SyntheticExternalContextAdapter:
    """Context adapter for a non-DSGym agent framework schema.

    Tasks are represented as:
    - messages: list of role/content messages (not DSGym prompt field)
    - metadata: domain-specific fields
    """

    def prepare_context(self, config: Any) -> Tuple[Any, str, Dict[bool, List[Dict[str, Any]]]]:
        base_tasks = [
            {
                "messages": [
                    {"role": "system", "content": "External framework system prompt."},
                    {"role": "user", "content": "Solve synthetic task A with careful planning."},
                ],
                "metadata": {"task_id": "A", "domain": "external_sim"},
            },
            {
                "messages": [
                    {"role": "system", "content": "External framework system prompt."},
                    {"role": "user", "content": "Solve synthetic task B with reproducible steps."},
                ],
                "metadata": {"task_id": "B", "domain": "external_sim"},
            },
        ]

        # The adapter controls DAG/non-DAG banks; here we keep separate copies.
        task_bank = {
            True: json.loads(json.dumps(base_tasks)),
            False: json.loads(json.dumps(base_tasks)),
        }
        dataset_handle = {"name": "synthetic_external_dataset"}
        family = "synthetic_external"
        return dataset_handle, family, task_bank


class SyntheticExternalTaskMutator:
    """Mutates non-DSGym task schema using optimizer variants."""

    def mutate(
        self,
        base_tasks: List[Dict[str, Any]],
        prompt_variant: Any,
        context_variant: Any,
        harness_variant: Any,
    ) -> List[Dict[str, Any]]:
        tasks = json.loads(json.dumps(base_tasks))
        for task in tasks:
            messages = task.get("messages", [])
            if len(messages) < 2:
                continue

            system = messages[0].get("content", "")
            user = messages[1].get("content", "")

            system_additions = [str(system)]
            if getattr(prompt_variant, "text", ""):
                system_additions.append(str(prompt_variant.text))
            if getattr(harness_variant, "text", ""):
                system_additions.append(str(harness_variant.text))
            messages[0]["content"] = "\n\n".join(chunk for chunk in system_additions if chunk)

            user_additions = [str(user)]
            if getattr(context_variant, "text", ""):
                user_additions.append(str(context_variant.text))
            user_additions.append("External execution contract: plan first, revise before deviation.")
            messages[1]["content"] = "\n\n".join(chunk for chunk in user_additions if chunk)

            task.setdefault("metadata", {})["variants"] = {
                "prompt": getattr(prompt_variant, "name", "unknown_prompt"),
                "context": getattr(context_variant, "name", "unknown_context"),
                "harness": getattr(harness_variant, "name", "unknown_harness"),
            }

        return tasks


class SyntheticExternalModeEvaluator:
    """Evaluates synthetic external tasks and returns optimizer-compatible summaries.

    This simulates a non-DSGym framework evaluator while preserving deterministic
    behavior for CI reproducibility.
    """

    def __init__(self, scenario: str = "baseline"):
        if scenario not in {"baseline", "candidate"}:
            raise ValueError("scenario must be baseline or candidate")
        self.scenario = scenario

    def _scenario_shift(self) -> Dict[str, float]:
        if self.scenario == "candidate":
            return {
                "quality": 0.035,
                "success": 0.030,
                "compliance": 0.025,
                "time": -0.9,
                "turns": -0.5,
                "dag_help": 0.02,
            }
        return {
            "quality": 0.0,
            "success": 0.0,
            "compliance": 0.0,
            "time": 0.0,
            "turns": 0.0,
            "dag_help": 0.0,
        }

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
        quality_adapter: Any,
        harness_adapter: Any,
    ) -> ModeEvaluationRecord:
        if not tasks:
            raise ValueError("Synthetic evaluator requires non-empty tasks")

        variants = tasks[0].get("metadata", {}).get("variants", {})
        prompt_name = str(variants.get("prompt", "prompt_baseline"))
        context_name = str(variants.get("context", "context_baseline"))
        harness_name = str(variants.get("harness", "harness_baseline"))

        prompt_u = _unit_hash(prompt_name)
        context_u = _unit_hash(context_name)
        harness_u = _unit_hash(harness_name)
        validation_u = _unit_hash(dag_validation_mode)
        shifts = self._scenario_shift()

        dag_bonus = 0.02 if use_dags else -0.01

        quality = _clamp(
            0.52
            + 0.12 * prompt_u
            + 0.08 * context_u
            + 0.06 * harness_u
            + 0.03 * validation_u
            + dag_bonus
            + shifts["quality"]
        )

        success_rate = _clamp(quality - 0.04 + shifts["success"])
        compliance = _clamp(0.60 + 0.20 * harness_u + 0.05 * context_u + shifts["compliance"])

        avg_time = max(0.1, 11.0 - 1.6 * prompt_u - 1.1 * context_u + (0.2 if use_dags else 0.0) + shifts["time"])
        avg_turns = max(1.0, 8.5 - 1.1 * harness_u - 0.7 * prompt_u + (0.3 if use_dags else 0.0) + shifts["turns"])

        exact_match_accuracy = _clamp(quality - 0.03)
        total_samples = float(len(tasks))
        exact_match_correct = round(exact_match_accuracy * total_samples, 6)

        dag_delta_shift = shifts["dag_help"] if use_dags else -0.5 * shifts["dag_help"]

        trace_summary = {
            "total_samples": len(tasks),
            "success_rate": success_rate,
            "error_rate": 1.0 - success_rate,
            "avg_execution_time": avg_time,
            "avg_turns": avg_turns,
            "median_turns": avg_turns,
            "p90_turns": avg_turns + 0.4,
            "median_execution_time": avg_time,
            "p90_execution_time": avg_time + 0.5,
            "error_breakdown": {},
        }

        quality_summary = {
            "total_samples": total_samples,
            "quality_metric": "synthetic_external_quality",
            "quality_mean": _clamp(quality + dag_delta_shift),
            "exact_match_evaluable": total_samples,
            "exact_match_correct": exact_match_correct,
            "exact_match_accuracy": exact_match_accuracy,
            "successful_samples": round(success_rate * total_samples, 6),
            "error_samples": round((1.0 - success_rate) * total_samples, 6),
        }

        harness_summary = {
            "policy": {
                "type": "synthetic_external_policy",
                "max_turns": max_turns,
                "mode_tag": mode_tag,
            },
            "avg_compliance": compliance,
            "violation_rates": {
                "missing_step_labels": _clamp(0.12 - 0.10 * harness_u),
                "skipped_step": _clamp(0.10 - 0.08 * context_u),
                "skipped_step_without_revision": _clamp(0.08 - 0.06 * harness_u),
                "step_regression": _clamp(0.06 - 0.04 * prompt_u),
                "looping": _clamp(0.14 - 0.10 * harness_u),
                "max_turn_stall": _clamp(0.11 - 0.07 * context_u),
            },
            "counters": {
                "assistant_messages_mean": avg_turns,
                "labeled_steps_mean": avg_turns - 0.4,
                "repeated_message_pairs_mean": 0.6 * (1.0 - harness_u),
                "repeated_code_pairs_mean": 0.45 * (1.0 - harness_u),
            },
        }

        aggregated_metrics = {
            "total_samples": len(tasks),
            "successful_samples": int(round(success_rate * len(tasks))),
            "error_samples": int(round((1.0 - success_rate) * len(tasks))),
            "success_rate": success_rate,
            "average_execution_time": avg_time,
            "average_turns": avg_turns,
            "exact_match_mean": exact_match_accuracy,
        }

        return ModeEvaluationRecord(
            use_dags=use_dags,
            trace_summary=trace_summary,
            quality_summary=quality_summary,
            harness_summary=harness_summary,
            aggregated_metrics=aggregated_metrics,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-DSGym adapter demo for research optimizer")
    parser.add_argument("--scenario", choices=["baseline", "candidate"], default="baseline")
    parser.add_argument("--output-dir", type=str, default="./optimization_results/non_dsgym_demo")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ResearchOptimizationConfig(
        model=f"synthetic-{args.scenario}",
        dataset="external-synthetic",
        backend="litellm",
        limit=2,
        n_trials=args.trials,
        population_size=args.population_size,
        seed=args.seed,
        study_name=f"external_demo_{args.scenario}",
        output_dir=str(output_dir),
        tune_dag_validation_mode=False,
        dag_validation_mode="default",
        generation_model=None,
        strict_harness_evaluator=False,
        reuse_variant_bank_cache=True,
    )

    summary = run_research_nsga2(
        config,
        context_adapter=SyntheticExternalContextAdapter(),
        task_mutator_adapter=SyntheticExternalTaskMutator(),
        mode_evaluator_adapter=SyntheticExternalModeEvaluator(scenario=args.scenario),
        quality_adapter=DefaultQualityMetricAdapter(),
        harness_adapter=DefaultHarnessPolicyAdapter(config),
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
