#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from dsgym.optimization import group_stats, load_research_trial_points, pareto_front_indices


def _classify_dag_effect(mean_delta: float) -> str:
    if mean_delta > 0.02:
        return "helps"
    if mean_delta < -0.02:
        return "hurts"
    return "mixed"


def analyze(input_dir: Path) -> Dict[str, Any]:
    records = load_research_trial_points([input_dir])
    if not records:
        return {
            "input_dir": str(input_dir),
            "total_trials": 0,
            "message": "No trial records found.",
        }

    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    overall = {
        "mean_quality": _mean([r.mean_quality for r in records]),
        "mean_success": _mean([r.mean_success for r in records]),
        "mean_time": _mean([r.mean_time for r in records]),
        "mean_turns": _mean([r.mean_turns for r in records]),
        "mean_compliance": _mean([r.mean_compliance for r in records]),
        "mean_dag_help_delta": _mean([r.dag_help_delta for r in records]),
        "mean_exact_match_accuracy": _mean(
            [r.exact_match_accuracy for r in records if math.isfinite(r.exact_match_accuracy)]
        ),
        "dag_effect": _classify_dag_effect(_mean([r.dag_help_delta for r in records])),
    }

    best_by_quality = sorted(records, key=lambda x: (x.mean_quality, x.mean_compliance), reverse=True)[:10]
    best_balanced = sorted(
        records,
        key=lambda x: (
            x.mean_quality,
            x.mean_compliance,
            -x.mean_time,
            -x.mean_turns,
            x.dag_help_delta,
        ),
        reverse=True,
    )[:10]

    pareto_idx = set(pareto_front_indices(records))
    pareto_trials = [record.__dict__ for i, record in enumerate(records) if i in pareto_idx]

    return {
        "input_dir": str(input_dir),
        "total_trials": len(records),
        "overall": overall,
        "prompt_variant_stats": group_stats(records, "prompt_variant"),
        "context_variant_stats": group_stats(records, "context_variant"),
        "harness_variant_stats": group_stats(records, "harness_variant"),
        "top_by_quality": [record.__dict__ for record in best_by_quality],
        "top_balanced": [record.__dict__ for record in best_balanced],
        "pareto_trials": pareto_trials,
    }


def _fmt_exact(value: Any) -> str:
    """Format an exact-match accuracy value; return 'n/a' when unavailable or NaN."""
    try:
        f = float(value)
        if math.isfinite(f):
            return f"{f:.4f}"
    except (TypeError, ValueError):
        pass
    return "n/a"


def _render_markdown(report: Dict[str, Any]) -> str:
    if report.get("total_trials", 0) == 0:
        return "# Research Optimization Report\n\nNo trial records found."

    overall = report["overall"]
    lines: List[str] = []
    lines.append("# Research Optimization Report")
    lines.append("")
    lines.append(f"- Input directory: `{report['input_dir']}`")
    lines.append(f"- Total trials: `{report['total_trials']}`")
    lines.append(f"- Mean quality score: `{overall['mean_quality']:.4f}`")
    lines.append(f"- Mean success rate: `{overall['mean_success']:.4f}`")
    lines.append(f"- Mean exact-match accuracy: `{overall['mean_exact_match_accuracy']:.4f}`")
    lines.append(f"- Mean execution time: `{overall['mean_time']:.2f}`")
    lines.append(f"- Mean turns: `{overall['mean_turns']:.2f}`")
    lines.append(f"- Mean harness compliance: `{overall['mean_compliance']:.4f}`")
    lines.append(f"- Mean DAG help delta: `{overall['mean_dag_help_delta']:+.4f}` ({overall['dag_effect']})")
    lines.append("")

    lines.append("## Top Balanced Trials")
    lines.append("")
    lines.append("| trial | prompt | context | harness | quality | success | exact acc | compliance | time | turns | dag delta |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in report.get("top_balanced", [])[:10]:
        lines.append(
            "| {trial_number} | {prompt_variant} | {context_variant} | {harness_variant} |"
            " {mean_quality:.4f} | {mean_success:.4f} | {exact_acc} |"
            " {mean_compliance:.4f} | {mean_time:.2f} | {mean_turns:.2f} | {dag_help_delta:+.4f} |".format(
                exact_acc=_fmt_exact(row.get("exact_match_accuracy")),
                **row,
            )
        )

    lines.append("")
    lines.append("## Harness Variant Summary")
    lines.append("")
    lines.append("| harness variant | n | quality | success | exact acc | compliance | time | turns | dag delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for variant, stats in sorted(report.get("harness_variant_stats", {}).items()):
        lines.append(
            "| {variant} | {n:.0f} | {mean_quality:.4f} | {mean_success:.4f} | {exact_acc} |"
            " {mean_compliance:.4f} | {mean_time:.2f} | {mean_turns:.2f} | {mean_dag_help_delta:+.4f} |".format(
                variant=variant,
                exact_acc=_fmt_exact(stats.get("mean_exact_match_accuracy")),
                **stats,
            )
        )

    lines.append("")
    lines.append("## Pareto Trials")
    lines.append("")
    lines.append("| trial | quality | time | turns | compliance | dag delta |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in report.get("pareto_trials", [])[:20]:
        lines.append(
            "| {trial_number} | {mean_quality:.4f} | {mean_time:.2f} | {mean_turns:.2f} | {mean_compliance:.4f} | {dag_help_delta:+.4f} |".format(
                **row
            )
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze research optimization trial records")
    parser.add_argument("--input-dir", type=str, default="./optimization_results/research")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    report = analyze(input_dir)

    output_json = Path(args.output_json) if args.output_json else input_dir / "research_report.json"
    output_md = Path(args.output_md) if args.output_md else input_dir / "research_report.md"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote JSON report: {output_json}")
    print(f"Wrote markdown report: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
