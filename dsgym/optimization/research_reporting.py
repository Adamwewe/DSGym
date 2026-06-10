from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class ResearchTrialPoint:
    input_dir: str
    study_name: str
    model: str
    dataset: str
    trial_number: int
    prompt_variant: str
    context_variant: str
    harness_variant: str
    dag_system_variant: str
    mean_quality: float
    mean_success: float
    mean_time: float
    mean_turns: float
    mean_compliance: float
    dag_help_delta: float
    exact_match_accuracy: float
    exact_match_correct: float
    total_samples: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_summary(input_dir: Path) -> Dict[str, Any]:
    summary_path = input_dir / "research_optimization_summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _finite(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def load_research_trial_points(input_dirs: List[Path]) -> List[ResearchTrialPoint]:
    points: List[ResearchTrialPoint] = []

    for input_dir in input_dirs:
        summary = _load_summary(input_dir)
        study_name = str(summary.get("study_name", input_dir.name))
        model = str(summary.get("model", "unknown_model"))
        dataset = str(summary.get("dataset", "unknown_dataset"))

        for path in sorted((input_dir / "trials").glob("trial_*/trial_record.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            objectives = payload.get("objectives", {}) if isinstance(payload.get("objectives"), dict) else {}
            per_mode = payload.get("per_mode", {}) if isinstance(payload.get("per_mode"), dict) else {}
            variants = payload.get("selected_variants", {}) if isinstance(payload.get("selected_variants"), dict) else {}

            quality_values = []
            success_values = []
            time_values = []
            turn_values = []
            compliance_values = []
            accuracy_values = []
            correct_values = []
            total_sample_values = []

            for mode_payload in per_mode.values():
                if not isinstance(mode_payload, dict):
                    continue
                trace = mode_payload.get("trace_summary", {}) if isinstance(mode_payload.get("trace_summary"), dict) else {}
                quality = mode_payload.get("quality_summary", {}) if isinstance(mode_payload.get("quality_summary"), dict) else {}
                harness = mode_payload.get("harness_summary", {}) if isinstance(mode_payload.get("harness_summary"), dict) else {}

                quality_values.append(_safe_float(quality.get("quality_mean", 0.0)))
                success_values.append(_safe_float(trace.get("success_rate", 0.0)))
                time_values.append(_safe_float(trace.get("avg_execution_time", 0.0)))
                turn_values.append(_safe_float(trace.get("avg_turns", 0.0)))
                compliance_values.append(_safe_float(harness.get("avg_compliance", 0.0)))

                exact_acc = quality.get("exact_match_accuracy")
                if exact_acc is not None:
                    accuracy_values.append(_safe_float(exact_acc, 0.0))
                exact_correct = quality.get("exact_match_correct")
                if exact_correct is not None:
                    correct_values.append(_safe_float(exact_correct, 0.0))
                total_samples = quality.get("total_samples")
                if total_samples is not None:
                    total_sample_values.append(_safe_float(total_samples, 0.0))

            point = ResearchTrialPoint(
                input_dir=str(input_dir),
                study_name=study_name,
                model=model,
                dataset=dataset,
                trial_number=int(payload.get("trial_number", -1)),
                prompt_variant=str((variants.get("prompt") or {}).get("name", "unknown_prompt")),
                context_variant=str((variants.get("context") or {}).get("name", "unknown_context")),
                harness_variant=str((variants.get("harness") or {}).get("name", "unknown_harness")),
                dag_system_variant=str((variants.get("dag_system") or {}).get("name", "dag_style_baseline")),
                mean_quality=_safe_float(objectives.get("mean_quality_score", _mean(quality_values))),
                mean_success=_safe_float(objectives.get("mean_success_rate", _mean(success_values))),
                mean_time=_safe_float(objectives.get("mean_execution_time", _mean(time_values))),
                mean_turns=_safe_float(objectives.get("mean_turns", _mean(turn_values))),
                mean_compliance=_safe_float(objectives.get("mean_harness_compliance", _mean(compliance_values))),
                dag_help_delta=_safe_float(objectives.get("dag_help_delta", 0.0)),
                exact_match_accuracy=_safe_float(
                    objectives.get("mean_exact_match_accuracy", _mean_or_nan(accuracy_values)),
                    float("nan"),
                ),
                exact_match_correct=_mean_or_nan(correct_values),
                total_samples=_mean_or_nan(total_sample_values),
            )
            points.append(point)

    return points


def group_stats(points: List[ResearchTrialPoint], field: str) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[ResearchTrialPoint]] = {}
    for point in points:
        key = str(getattr(point, field))
        grouped.setdefault(key, []).append(point)

    stats: Dict[str, Dict[str, float]] = {}
    for key, group in grouped.items():
        n = float(len(group))
        stats[key] = {
            "n": n,
            "mean_quality": _mean([p.mean_quality for p in group]),
            "mean_success": _mean([p.mean_success for p in group]),
            "mean_time": _mean([p.mean_time for p in group]),
            "mean_turns": _mean([p.mean_turns for p in group]),
            "mean_compliance": _mean([p.mean_compliance for p in group]),
            "mean_dag_help_delta": _mean([p.dag_help_delta for p in group]),
            "mean_exact_match_accuracy": _mean_or_nan(_finite([p.exact_match_accuracy for p in group])),
            "mean_exact_match_correct": _mean_or_nan(_finite([p.exact_match_correct for p in group])),
            "mean_total_samples": _mean_or_nan(_finite([p.total_samples for p in group])),
        }
    return stats


def pareto_front_indices(points: List[ResearchTrialPoint]) -> List[int]:
    """Return indices for non-dominated points.

    Objective convention:
    - maximize quality
    - minimize execution time
    - minimize turns
    - maximize compliance
    - maximize dag_help_delta
    """

    if not points:
        return []

    def dominates(a: ResearchTrialPoint, b: ResearchTrialPoint) -> bool:
        no_worse = (
            a.mean_quality >= b.mean_quality
            and a.mean_time <= b.mean_time
            and a.mean_turns <= b.mean_turns
            and a.mean_compliance >= b.mean_compliance
            and a.dag_help_delta >= b.dag_help_delta
        )
        strictly_better = (
            a.mean_quality > b.mean_quality
            or a.mean_time < b.mean_time
            or a.mean_turns < b.mean_turns
            or a.mean_compliance > b.mean_compliance
            or a.dag_help_delta > b.dag_help_delta
        )
        return no_worse and strictly_better

    indices: List[int] = []
    for i, point_i in enumerate(points):
        dominated = False
        for j, point_j in enumerate(points):
            if i == j:
                continue
            if dominates(point_j, point_i):
                dominated = True
                break
        if not dominated:
            indices.append(i)

    return indices
