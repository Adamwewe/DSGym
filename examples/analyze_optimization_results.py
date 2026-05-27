#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Dict, List


@dataclass
class TrialPoint:
    trial_number: int
    mode: str
    success_rate: float
    error_rate: float
    avg_execution_time: float
    avg_turns: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_trial_record(path: Path) -> TrialPoint | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    mode = str(payload.get("dag_validation_mode", "default") or "default").strip().lower()
    trial_number = int(payload.get("trial_number", -1))

    objectives = payload.get("objectives", {}) if isinstance(payload.get("objectives"), dict) else {}
    trace_summary = payload.get("trace_summary", {}) if isinstance(payload.get("trace_summary"), dict) else {}

    success_rate = _safe_float(objectives.get("success_rate", trace_summary.get("success_rate", 0.0)))
    error_rate = _safe_float(objectives.get("error_rate", trace_summary.get("error_rate", 1.0)))
    avg_execution_time = _safe_float(
        objectives.get("avg_execution_time", trace_summary.get("avg_execution_time", 0.0))
    )
    avg_turns = _safe_float(objectives.get("avg_turns", trace_summary.get("avg_turns", 0.0)))

    return TrialPoint(
        trial_number=trial_number,
        mode=mode,
        success_rate=success_rate,
        error_rate=error_rate,
        avg_execution_time=avg_execution_time,
        avg_turns=avg_turns,
    )


def _sem(values: List[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = fmean(values)
    variance = sum((x - mu) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance) / math.sqrt(n)


def _dominates(a: TrialPoint, b: TrialPoint) -> bool:
    no_worse = (
        a.success_rate >= b.success_rate
        and a.error_rate <= b.error_rate
        and a.avg_execution_time <= b.avg_execution_time
        and a.avg_turns <= b.avg_turns
    )
    strictly_better = (
        a.success_rate > b.success_rate
        or a.error_rate < b.error_rate
        or a.avg_execution_time < b.avg_execution_time
        or a.avg_turns < b.avg_turns
    )
    return no_worse and strictly_better


def _aggregate(points: List[TrialPoint]) -> Dict[str, float]:
    return {
        "n": float(len(points)),
        "success_rate_mean": fmean([p.success_rate for p in points]) if points else 0.0,
        "error_rate_mean": fmean([p.error_rate for p in points]) if points else 0.0,
        "avg_execution_time_mean": fmean([p.avg_execution_time for p in points]) if points else 0.0,
        "avg_turns_mean": fmean([p.avg_turns for p in points]) if points else 0.0,
        "success_rate_median": median([p.success_rate for p in points]) if points else 0.0,
        "error_rate_median": median([p.error_rate for p in points]) if points else 0.0,
        "avg_execution_time_median": median([p.avg_execution_time for p in points]) if points else 0.0,
        "avg_turns_median": median([p.avg_turns for p in points]) if points else 0.0,
        "success_rate_sem": _sem([p.success_rate for p in points]) if points else 0.0,
        "error_rate_sem": _sem([p.error_rate for p in points]) if points else 0.0,
        "avg_execution_time_sem": _sem([p.avg_execution_time for p in points]) if points else 0.0,
        "avg_turns_sem": _sem([p.avg_turns for p in points]) if points else 0.0,
    }


def _classify_effect(delta_success: float, delta_error_improvement: float, delta_time_improvement: float, delta_turns_improvement: float) -> str:
    improvements = [
        delta_success >= 0.0,
        delta_error_improvement >= 0.0,
        delta_time_improvement >= 0.0,
        delta_turns_improvement >= 0.0,
    ]
    positives = sum(1 for x in improvements if x)

    if delta_success > 0 and positives >= 3:
        return "helps"
    if delta_success < 0 and positives <= 1:
        return "hurts"
    return "mixed"


def analyze(input_dir: Path) -> Dict[str, Any]:
    trials_dir = input_dir / "trials"
    records = sorted(trials_dir.glob("trial_*/trial_record.json"))

    points: List[TrialPoint] = []
    for record_path in records:
        point = _read_trial_record(record_path)
        if point is not None:
            points.append(point)

    by_mode: Dict[str, List[TrialPoint]] = {}
    for point in points:
        by_mode.setdefault(point.mode, []).append(point)

    mode_stats: Dict[str, Dict[str, float]] = {mode: _aggregate(mode_points) for mode, mode_points in by_mode.items()}

    comparisons: Dict[str, Dict[str, Any]] = {}
    baseline = by_mode.get("default", [])

    if baseline:
        baseline_stats = _aggregate(baseline)
        for mode, mode_points in by_mode.items():
            stats = _aggregate(mode_points)
            if mode == "default":
                comparisons[mode] = {
                    "classification": "baseline",
                    "delta_vs_default": {
                        "success_rate": 0.0,
                        "error_rate_improvement": 0.0,
                        "avg_execution_time_improvement": 0.0,
                        "avg_turns_improvement": 0.0,
                    },
                    "pareto_domination_rate_vs_default": 0.0,
                }
                continue

            delta_success = stats["success_rate_mean"] - baseline_stats["success_rate_mean"]
            delta_error_improvement = baseline_stats["error_rate_mean"] - stats["error_rate_mean"]
            delta_time_improvement = baseline_stats["avg_execution_time_mean"] - stats["avg_execution_time_mean"]
            delta_turns_improvement = baseline_stats["avg_turns_mean"] - stats["avg_turns_mean"]

            domination_rate = 0.0
            if mode_points and baseline:
                total_pairs = len(mode_points) * len(baseline)
                dominates_count = 0
                for a in mode_points:
                    for b in baseline:
                        if _dominates(a, b):
                            dominates_count += 1
                domination_rate = dominates_count / total_pairs

            comparisons[mode] = {
                "classification": _classify_effect(
                    delta_success,
                    delta_error_improvement,
                    delta_time_improvement,
                    delta_turns_improvement,
                ),
                "delta_vs_default": {
                    "success_rate": delta_success,
                    "error_rate_improvement": delta_error_improvement,
                    "avg_execution_time_improvement": delta_time_improvement,
                    "avg_turns_improvement": delta_turns_improvement,
                },
                "pareto_domination_rate_vs_default": domination_rate,
            }
    else:
        for mode in by_mode:
            comparisons[mode] = {
                "classification": "no_default_baseline",
                "delta_vs_default": None,
                "pareto_domination_rate_vs_default": None,
            }

    return {
        "input_dir": str(input_dir),
        "total_trials": len(points),
        "available_modes": sorted(by_mode.keys()),
        "equations": {
            "success_rate": "(1/N) * sum_i s_i",
            "error_rate": "1 - success_rate",
            "avg_execution_time": "(1/N) * sum_i t_i",
            "avg_turns": "(1/N) * sum_i r_i",
            "pareto_domination_rate": "count(mode_trial dominates default_trial) / (n_mode * n_default)",
        },
        "mode_stats": mode_stats,
        "comparisons_vs_default": comparisons,
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Optimization Help/Hurt Report")
    lines.append("")
    lines.append(f"- Input directory: `{report['input_dir']}`")
    lines.append(f"- Total trial records: `{report['total_trials']}`")
    lines.append(f"- Modes: `{', '.join(report['available_modes']) if report['available_modes'] else 'none'}`")
    lines.append("")
    lines.append("## Objective Equations")
    lines.append("")
    lines.append("- $\\text{success\\_rate} = \\frac{1}{N}\\sum_i s_i$")
    lines.append("- $\\text{error\\_rate} = 1 - \\text{success\\_rate}$")
    lines.append("- $\\overline{t} = \\frac{1}{N}\\sum_i t_i$")
    lines.append("- $\\overline{r} = \\frac{1}{N}\\sum_i r_i$")
    lines.append("- $\\text{domination\\_rate}(m,\\text{default}) = \\frac{\\#\\{(a,b): a\\in m, b\\in default, a \\succ b\\}}{|m|\\cdot|default|}$")
    lines.append("")
    lines.append("## Mode Summary")
    lines.append("")
    lines.append("| mode | n | success(mean±sem) | error(mean±sem) | time(mean±sem) | turns(mean±sem) |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    mode_stats = report.get("mode_stats", {})
    for mode in sorted(mode_stats.keys()):
        stats = mode_stats[mode]
        lines.append(
            "| {mode} | {n:.0f} | {s:.4f}±{s_sem:.4f} | {e:.4f}±{e_sem:.4f} | {t:.2f}±{t_sem:.2f} | {r:.2f}±{r_sem:.2f} |".format(
                mode=mode,
                n=stats["n"],
                s=stats["success_rate_mean"],
                s_sem=stats["success_rate_sem"],
                e=stats["error_rate_mean"],
                e_sem=stats["error_rate_sem"],
                t=stats["avg_execution_time_mean"],
                t_sem=stats["avg_execution_time_sem"],
                r=stats["avg_turns_mean"],
                r_sem=stats["avg_turns_sem"],
            )
        )

    lines.append("")
    lines.append("## Help/Hurt vs default")
    lines.append("")
    lines.append("| mode | classification | Δsuccess | Δerror improvement | Δtime improvement | Δturns improvement | domination rate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    comparisons = report.get("comparisons_vs_default", {})
    for mode in sorted(comparisons.keys()):
        row = comparisons[mode]
        delta = row.get("delta_vs_default")
        if delta is None:
            lines.append(
                f"| {mode} | {row.get('classification', 'n/a')} | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            "| {mode} | {cls} | {ds:+.4f} | {de:+.4f} | {dt:+.2f} | {dr:+.2f} | {dom:.4f} |".format(
                mode=mode,
                cls=row.get("classification", "mixed"),
                ds=delta.get("success_rate", 0.0),
                de=delta.get("error_rate_improvement", 0.0),
                dt=delta.get("avg_execution_time_improvement", 0.0),
                dr=delta.get("avg_turns_improvement", 0.0),
                dom=_safe_float(row.get("pareto_domination_rate_vs_default", 0.0)),
            )
        )

    lines.append("")
    lines.append("Classification rule: `helps` if success increases and at least 3/4 objectives improve; `hurts` if success decreases and at most 1/4 objectives improve; otherwise `mixed`.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze DSGym optimization trial traces and report help/hurt by DAG mode")
    parser.add_argument("--input-dir", type=str, default="./optimization_results", help="Directory containing optimization outputs")
    parser.add_argument("--output-json", type=str, default=None, help="Optional explicit path for JSON report")
    parser.add_argument("--output-md", type=str, default=None, help="Optional explicit path for markdown report")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    report = analyze(input_dir)

    output_json = Path(args.output_json) if args.output_json else input_dir / "optimization_help_hurt_report.json"
    output_md = Path(args.output_md) if args.output_md else input_dir / "optimization_help_hurt_report.md"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote JSON report: {output_json}")
    print(f"Wrote markdown report: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
