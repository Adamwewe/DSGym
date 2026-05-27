#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dsgym.optimization import ResearchTrialPoint, load_research_trial_points


@dataclass
class AggregateMetrics:
    trials: int
    mean_quality: float
    mean_success: float
    mean_time: float
    mean_turns: float
    mean_compliance: float
    mean_exact_match_accuracy: Optional[float]
    mean_dag_help_delta: float


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _select_points(points: List[ResearchTrialPoint], selector: str, top_k: int) -> List[ResearchTrialPoint]:
    if not points:
        return []

    if selector == "all-mean":
        return points

    if selector == "pareto-mean":
        from dsgym.optimization import pareto_front_indices

        idx = pareto_front_indices(points)
        return [points[i] for i in idx] if idx else points

    # selector == top-balanced
    ranked = sorted(
        points,
        key=lambda p: (
            p.mean_quality,
            p.mean_compliance,
            p.exact_match_accuracy,
            -p.mean_time,
            -p.mean_turns,
            p.dag_help_delta,
        ),
        reverse=True,
    )
    return ranked[: max(1, min(top_k, len(ranked)))]


def _aggregate(points: List[ResearchTrialPoint]) -> AggregateMetrics:
    if not points:
        return AggregateMetrics(
            trials=0,
            mean_quality=0.0,
            mean_success=0.0,
            mean_time=0.0,
            mean_turns=0.0,
            mean_compliance=0.0,
            mean_exact_match_accuracy=None,
            mean_dag_help_delta=0.0,
        )

    exact_values = [p.exact_match_accuracy for p in points if p.exact_match_accuracy is not None]

    return AggregateMetrics(
        trials=len(points),
        mean_quality=_mean([p.mean_quality for p in points]),
        mean_success=_mean([p.mean_success for p in points]),
        mean_time=_mean([p.mean_time for p in points]),
        mean_turns=_mean([p.mean_turns for p in points]),
        mean_compliance=_mean([p.mean_compliance for p in points]),
        mean_exact_match_accuracy=_mean(exact_values) if exact_values else None,
        mean_dag_help_delta=_mean([p.dag_help_delta for p in points]),
    )


def _render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Research Regression Report")
    lines.append("")
    lines.append(f"- Baseline: `{report['baseline_dir']}`")
    lines.append(f"- Candidate: `{report['candidate_dir']}`")
    lines.append(f"- Selector: `{report['selector']}`")
    lines.append(f"- Passed: `{report['passed']}`")
    lines.append("")

    lines.append("## Aggregates")
    lines.append("")
    lines.append("| side | trials | quality | success | exact acc | compliance | time | turns | dag delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for side in ("baseline", "candidate"):
        row = report[side]
        exact_acc = row["mean_exact_match_accuracy"]
        exact_text = "n/a" if exact_acc is None else f"{exact_acc:.4f}"
        lines.append(
            "| {side} | {trials} | {quality:.4f} | {success:.4f} | {exact_acc} | {compliance:.4f} | {time:.2f} | {turns:.2f} | {dag:+.4f} |".format(
                side=side,
                trials=row["trials"],
                quality=row["mean_quality"],
                success=row["mean_success"],
                exact_acc=exact_text,
                compliance=row["mean_compliance"],
                time=row["mean_time"],
                turns=row["mean_turns"],
                dag=row["mean_dag_help_delta"],
            )
        )

    lines.append("")
    lines.append("## Deltas (candidate - baseline)")
    lines.append("")
    for key, value in report["deltas"].items():
        if value is None:
            lines.append(f"- {key}: n/a")
        else:
            lines.append(f"- {key}: {value:+.6f}")

    lines.append("")
    if report["regressions"]:
        lines.append("## Regressions")
        lines.append("")
        for item in report["regressions"]:
            lines.append(f"- {item}")
    else:
        lines.append("## Regressions")
        lines.append("")
        lines.append("- None")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="CI-friendly regression check for research optimization outputs")
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--candidate-dir", type=str, required=True)
    parser.add_argument(
        "--selector",
        type=str,
        default="top-balanced",
        choices=["top-balanced", "pareto-mean", "all-mean"],
        help="How to select trial subset before aggregation",
    )
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument("--min-delta-quality", type=float, default=-0.005)
    parser.add_argument("--min-delta-compliance", type=float, default=-0.010)
    parser.add_argument("--max-delta-time", type=float, default=5.0)
    parser.add_argument("--max-delta-turns", type=float, default=0.5)
    parser.add_argument("--min-delta-exact-match-accuracy", type=float, default=-0.010)
    parser.add_argument("--min-delta-dag-help", type=float, default=-0.020)
    parser.add_argument("--require-same-dataset", action="store_true")
    parser.add_argument("--allow-missing-exact-match", action="store_true")

    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)

    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)

    baseline_points = load_research_trial_points([baseline_dir])
    candidate_points = load_research_trial_points([candidate_dir])

    if not baseline_points:
        print("Baseline has no trial records.")
        return 2
    if not candidate_points:
        print("Candidate has no trial records.")
        return 2

    if args.require_same_dataset:
        baseline_datasets = {p.dataset for p in baseline_points}
        candidate_datasets = {p.dataset for p in candidate_points}
        if baseline_datasets != candidate_datasets:
            print(f"Dataset mismatch: baseline={baseline_datasets}, candidate={candidate_datasets}")
            return 2

    selected_baseline = _select_points(baseline_points, selector=args.selector, top_k=args.top_k)
    selected_candidate = _select_points(candidate_points, selector=args.selector, top_k=args.top_k)

    baseline = _aggregate(selected_baseline)
    candidate = _aggregate(selected_candidate)

    deltas: Dict[str, Optional[float]] = {
        "quality": candidate.mean_quality - baseline.mean_quality,
        "compliance": candidate.mean_compliance - baseline.mean_compliance,
        "time": candidate.mean_time - baseline.mean_time,
        "turns": candidate.mean_turns - baseline.mean_turns,
        "exact_match_accuracy": (
            None
            if (baseline.mean_exact_match_accuracy is None or candidate.mean_exact_match_accuracy is None)
            else candidate.mean_exact_match_accuracy - baseline.mean_exact_match_accuracy
        ),
        "dag_help_delta": candidate.mean_dag_help_delta - baseline.mean_dag_help_delta,
    }

    regressions: List[str] = []

    if deltas["quality"] is not None and deltas["quality"] < args.min_delta_quality:
        regressions.append(
            f"Quality regression: delta={deltas['quality']:+.6f} < threshold={args.min_delta_quality:+.6f}"
        )

    if deltas["compliance"] is not None and deltas["compliance"] < args.min_delta_compliance:
        regressions.append(
            f"Compliance regression: delta={deltas['compliance']:+.6f} < threshold={args.min_delta_compliance:+.6f}"
        )

    if deltas["time"] is not None and deltas["time"] > args.max_delta_time:
        regressions.append(
            f"Time regression: delta={deltas['time']:+.6f} > threshold={args.max_delta_time:+.6f}"
        )

    if deltas["turns"] is not None and deltas["turns"] > args.max_delta_turns:
        regressions.append(
            f"Turns regression: delta={deltas['turns']:+.6f} > threshold={args.max_delta_turns:+.6f}"
        )

    if deltas["exact_match_accuracy"] is None:
        if not args.allow_missing_exact_match:
            regressions.append("Exact-match accuracy unavailable; use --allow-missing-exact-match to ignore")
    elif deltas["exact_match_accuracy"] < args.min_delta_exact_match_accuracy:
        regressions.append(
            "Exact-match regression: "
            f"delta={deltas['exact_match_accuracy']:+.6f} < threshold={args.min_delta_exact_match_accuracy:+.6f}"
        )

    if deltas["dag_help_delta"] is not None and deltas["dag_help_delta"] < args.min_delta_dag_help:
        regressions.append(
            f"DAG-help regression: delta={deltas['dag_help_delta']:+.6f} < threshold={args.min_delta_dag_help:+.6f}"
        )

    passed = len(regressions) == 0

    report = {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "selector": args.selector,
        "thresholds": {
            "min_delta_quality": args.min_delta_quality,
            "min_delta_compliance": args.min_delta_compliance,
            "max_delta_time": args.max_delta_time,
            "max_delta_turns": args.max_delta_turns,
            "min_delta_exact_match_accuracy": args.min_delta_exact_match_accuracy,
            "min_delta_dag_help": args.min_delta_dag_help,
            "allow_missing_exact_match": args.allow_missing_exact_match,
        },
        "baseline": asdict(baseline),
        "candidate": asdict(candidate),
        "deltas": deltas,
        "regressions": regressions,
        "passed": passed,
    }

    output_json = Path(args.output_json) if args.output_json else candidate_dir / "regression_report.json"
    output_md = Path(args.output_md) if args.output_md else candidate_dir / "regression_report.md"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote regression JSON: {output_json}")
    print(f"Wrote regression markdown: {output_md}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
