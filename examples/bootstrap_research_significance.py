#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from dsgym.optimization import ResearchTrialPoint, load_research_trial_points
from dsgym.optimization.statistics import bootstrap_mean_delta, summarize_direction


def _select_points(points: List[ResearchTrialPoint], selector: str, top_k: int) -> List[ResearchTrialPoint]:
    if not points:
        return []

    if selector == "all-mean":
        return points

    if selector == "pareto-mean":
        from dsgym.optimization import pareto_front_indices

        idx = pareto_front_indices(points)
        return [points[i] for i in idx] if idx else points

    ranked = sorted(
        points,
        key=lambda p: (
            p.mean_quality,
            p.mean_compliance,
            p.exact_match_accuracy if math.isfinite(p.exact_match_accuracy) else -1.0,
            -p.mean_time,
            -p.mean_turns,
            p.dag_help_delta,
        ),
        reverse=True,
    )
    return ranked[: max(1, min(top_k, len(ranked)))]


def _metric_vectors(points: List[ResearchTrialPoint], field: str) -> List[float]:
    values = []
    for point in points:
        value = getattr(point, field)
        if value is None:
            continue
        value_f = float(value)
        if not math.isfinite(value_f):
            continue
        values.append(value_f)
    return values


def _build_metric_result(
    baseline: List[float],
    candidate: List[float],
    *,
    objective: str,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    paired: bool,
) -> Dict[str, Any]:
    result = bootstrap_mean_delta(
        baseline,
        candidate,
        n_resamples=n_bootstrap,
        confidence=confidence,
        seed=seed,
        paired=paired,
    )
    direction = summarize_direction(result, objective=objective)

    return {
        "n_baseline": len(baseline),
        "n_candidate": len(candidate),
        "objective": objective,
        "point_estimate": result.point_estimate,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "p_value_two_sided": result.p_value_two_sided,
        "prob_delta_gt_zero": result.prob_delta_gt_zero,
        "prob_delta_lt_zero": result.prob_delta_lt_zero,
        "n_resamples": result.n_resamples,
        **direction,
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Bootstrap Significance Report")
    lines.append("")
    lines.append(f"- Baseline: {report['baseline_dir']}")
    lines.append(f"- Candidate: {report['candidate_dir']}")
    lines.append(f"- Selector: {report['selector']}")
    lines.append(f"- Bootstrap samples: {report['n_bootstrap']}")
    lines.append(f"- Confidence: {report['confidence']}")
    lines.append(f"- Paired: {report['paired']}")
    lines.append("")

    lines.append("| metric | objective | delta | CI low | CI high | p-value | sig improve | sig regress |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")

    for metric_name, payload in report["metrics"].items():
        lines.append(
            "| {name} | {objective} | {delta:+.6f} | {lo:+.6f} | {hi:+.6f} | {p:.6f} | {imp} | {reg} |".format(
                name=metric_name,
                objective=payload["objective"],
                delta=payload["point_estimate"],
                lo=payload["ci_lower"],
                hi=payload["ci_upper"],
                p=payload["p_value_two_sided"],
                imp=str(payload["significant_improvement"]),
                reg=str(payload["significant_regression"]),
            )
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap significance testing for research optimization comparisons")
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--candidate-dir", type=str, required=True)
    parser.add_argument(
        "--selector",
        type=str,
        default="top-balanced",
        choices=["top-balanced", "pareto-mean", "all-mean"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)

    baseline_points = _select_points(
        load_research_trial_points([baseline_dir]), selector=args.selector, top_k=args.top_k
    )
    candidate_points = _select_points(
        load_research_trial_points([candidate_dir]), selector=args.selector, top_k=args.top_k
    )

    if not baseline_points:
        raise SystemExit("No baseline trial records found")
    if not candidate_points:
        raise SystemExit("No candidate trial records found")

    metrics_def = {
        "quality": ("mean_quality", "maximize"),
        "compliance": ("mean_compliance", "maximize"),
        "exact_match_accuracy": ("exact_match_accuracy", "maximize"),
        "time": ("mean_time", "minimize"),
        "turns": ("mean_turns", "minimize"),
        "dag_help_delta": ("dag_help_delta", "maximize"),
    }

    metrics: Dict[str, Any] = {}
    for metric_name, (field, objective) in metrics_def.items():
        b_values = _metric_vectors(baseline_points, field)
        c_values = _metric_vectors(candidate_points, field)

        if not b_values or not c_values:
            metrics[metric_name] = {
                "objective": objective,
                "skipped": True,
                "reason": "missing_values",
                "n_baseline": len(b_values),
                "n_candidate": len(c_values),
            }
            continue

        try:
            metrics[metric_name] = _build_metric_result(
                b_values,
                c_values,
                objective=objective,
                n_bootstrap=args.n_bootstrap,
                confidence=args.confidence,
                seed=args.seed,
                paired=args.paired,
            )
        except ValueError as exc:
            metrics[metric_name] = {
                "objective": objective,
                "skipped": True,
                "reason": str(exc),
                "n_baseline": len(b_values),
                "n_candidate": len(c_values),
            }

    report = {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "selector": args.selector,
        "top_k": args.top_k,
        "n_bootstrap": args.n_bootstrap,
        "confidence": args.confidence,
        "seed": args.seed,
        "paired": args.paired,
        "metrics": metrics,
    }

    output_json = Path(args.output_json) if args.output_json else candidate_dir / "bootstrap_significance.json"
    output_md = Path(args.output_md) if args.output_md else candidate_dir / "bootstrap_significance.md"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote bootstrap JSON: {output_json}")
    print(f"Wrote bootstrap markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
