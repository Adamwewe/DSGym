#!/usr/bin/env python3
"""Variant attribution analysis for DSGym research optimization runs.

Groups trial records by each variant dimension (prompt_variant, context_variant,
harness_variant) and runs bootstrap significance tests comparing each non-baseline
variant to the *_baseline group for quality, compliance, and DAG-help delta.

Usage:
    uv run python examples/attribute_research_variants.py \\
        --input-dir ./optimization_results/research \\
        [--confidence 0.90] \\
        [--n-resamples 2000] \\
        [--output-json ./optimization_results/research/variant_attribution.json] \\
        [--output-md  ./optimization_results/research/variant_attribution.md]

Statistical caveats:
  - NSGA-II adaptively favors better-performing variants in later trials. Samples per
    variant are therefore NOT uniformly distributed — groups receiving fewer trials
    may genuinely be worse performers rather than under-sampled. Bootstrap p-values
    should be interpreted as directional indicators, not unbiased hypothesis tests.
  - Groups with fewer than 3 trials produce very wide confidence intervals. Results
    for n < 5 are labeled exploratory-only in the output.
  - Exact-match attribution is only computed when both the baseline and the candidate
    group have at least 2 finite exact-match measurements.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from dsgym.optimization import (
    ResearchTrialPoint,
    bootstrap_mean_delta,
    load_research_trial_points,
    summarize_direction,
)

_EXPLORATORY_N_THRESHOLD = 5


def _finite_field(points: List[ResearchTrialPoint], field: str) -> List[float]:
    """Return finite float values for `field` across `points`."""
    out: List[float] = []
    for p in points:
        v = getattr(p, field, None)
        if v is not None:
            try:
                fv = float(v)
                if math.isfinite(fv):
                    out.append(fv)
            except (TypeError, ValueError):
                pass
    return out


def _bootstrap_compare(
    baseline: List[float],
    candidate: List[float],
    objective: str,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap significance test for a single metric comparison.

    Returns a dict with point_estimate, CI bounds, p-value, and direction flags.
    Returns a stub with note="insufficient samples" when either group has < 2 values.
    """
    if len(baseline) < 2 or len(candidate) < 2:
        return {
            "point_estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "significant_improvement": None,
            "significant_regression": None,
            "note": "insufficient samples for bootstrap (need >= 2 per group)",
        }

    result = bootstrap_mean_delta(
        baseline,
        candidate,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
        paired=False,
    )
    direction = summarize_direction(result, objective=objective)
    return {
        "point_estimate": result.point_estimate,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "p_value": result.p_value_two_sided,
        "prob_delta_gt_zero": result.prob_delta_gt_zero,
        **direction,
    }


def _attribute_dimension(
    points: List[ResearchTrialPoint],
    field: str,
    baseline_suffix: str = "_baseline",
    n_resamples: int = 2000,
    confidence: float = 0.90,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute attribution statistics for one variant dimension.

    Identifies the baseline group (variant name ending with `baseline_suffix`),
    then bootstrap-tests every other group against it for quality, compliance,
    dag_help_delta, and (where available) exact_match_accuracy.
    """
    groups: Dict[str, List[ResearchTrialPoint]] = {}
    for point in points:
        key = str(getattr(point, field))
        groups.setdefault(key, []).append(point)

    if len(groups) < 2:
        return {
            "field": field,
            "baseline_variant": None,
            "n_baseline_trials": 0,
            "n_other_variants": 0,
            "note": "fewer than 2 distinct variants; attribution not possible",
            "comparisons": [],
        }

    # Prefer an explicit baseline; fall back to the largest group.
    baseline_key = next((k for k in groups if k.endswith(baseline_suffix)), None)
    if baseline_key is None:
        baseline_key = max(groups, key=lambda k: len(groups[k]))

    baseline_pts = groups[baseline_key]
    b_quality = _finite_field(baseline_pts, "mean_quality")
    b_compliance = _finite_field(baseline_pts, "mean_compliance")
    b_dag = _finite_field(baseline_pts, "dag_help_delta")
    b_exact = _finite_field(baseline_pts, "exact_match_accuracy")

    comparisons: List[Dict[str, Any]] = []
    for key in sorted(groups):
        if key == baseline_key:
            continue

        cand_pts = groups[key]
        c_quality = _finite_field(cand_pts, "mean_quality")
        c_compliance = _finite_field(cand_pts, "mean_compliance")
        c_dag = _finite_field(cand_pts, "dag_help_delta")
        c_exact = _finite_field(cand_pts, "exact_match_accuracy")

        entry: Dict[str, Any] = {
            "variant": key,
            "n_trials": len(cand_pts),
            "exploratory": len(cand_pts) < _EXPLORATORY_N_THRESHOLD,
            "mean_quality": (sum(c_quality) / len(c_quality)) if c_quality else None,
            "mean_compliance": (sum(c_compliance) / len(c_compliance)) if c_compliance else None,
            "mean_dag_help_delta": (sum(c_dag) / len(c_dag)) if c_dag else None,
            "quality_vs_baseline": _bootstrap_compare(
                b_quality, c_quality, "maximize", n_resamples, confidence, seed,
            ),
            "compliance_vs_baseline": _bootstrap_compare(
                b_compliance, c_compliance, "maximize", n_resamples, confidence, seed + 1,
            ),
            "dag_help_delta_vs_baseline": _bootstrap_compare(
                b_dag, c_dag, "maximize", n_resamples, confidence, seed + 2,
            ),
        }
        if len(b_exact) >= 2 and len(c_exact) >= 2:
            entry["exact_match_vs_baseline"] = _bootstrap_compare(
                b_exact, c_exact, "maximize", n_resamples, confidence, seed + 3,
            )
        comparisons.append(entry)

    return {
        "field": field,
        "baseline_variant": baseline_key,
        "n_baseline_trials": len(baseline_pts),
        "n_other_variants": len(comparisons),
        "confidence": confidence,
        "adaptive_sampling_caveat": (
            "NSGA-II adaptively favors better-performing variants in later trials. "
            "Bootstrap p-values are directional indicators, not unbiased hypothesis tests. "
            f"Results with n < {_EXPLORATORY_N_THRESHOLD} trials are marked exploratory."
        ),
        "comparisons": comparisons,
    }


def attribute_variants(
    input_dir: Path,
    n_resamples: int = 2000,
    confidence: float = 0.90,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full variant attribution analysis for a research optimization output dir."""
    points = load_research_trial_points([input_dir])
    if not points:
        return {
            "error": "no trial records found in input directory",
            "input_dir": str(input_dir),
        }

    dimensions = ["prompt_variant", "context_variant", "harness_variant"]
    attribution: Dict[str, Any] = {}
    for dim in dimensions:
        attribution[dim] = _attribute_dimension(
            points, dim, n_resamples=n_resamples, confidence=confidence, seed=seed
        )

    # Aggregate significant wins and losses for a top-level summary.
    significant_wins: List[Dict[str, str]] = []
    significant_losses: List[Dict[str, str]] = []
    for dim, dim_result in attribution.items():
        for comp in dim_result.get("comparisons", []):
            for metric_key in (
                "quality_vs_baseline",
                "compliance_vs_baseline",
                "dag_help_delta_vs_baseline",
                "exact_match_vs_baseline",
            ):
                metric = comp.get(metric_key)
                if not metric:
                    continue
                metric_label = metric_key.replace("_vs_baseline", "")
                if metric.get("significant_improvement"):
                    significant_wins.append(
                        {"dimension": dim, "variant": comp["variant"], "metric": metric_label}
                    )
                elif metric.get("significant_regression"):
                    significant_losses.append(
                        {"dimension": dim, "variant": comp["variant"], "metric": metric_label}
                    )

    return {
        "input_dir": str(input_dir),
        "total_trials": len(points),
        "n_resamples": n_resamples,
        "confidence": confidence,
        "significant_wins": significant_wins,
        "significant_losses": significant_losses,
        "attribution": attribution,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _fmt_delta(comp: Dict[str, Any], metric_key: str) -> str:
    m = comp.get(metric_key) or {}
    pe = m.get("point_estimate")
    if pe is None:
        return "n/a"
    return f"{pe:+.4f}"


def _fmt_sig(comp: Dict[str, Any], metric_key: str) -> str:
    m = comp.get(metric_key) or {}
    if m.get("significant_improvement"):
        return "✓+"
    if m.get("significant_regression"):
        return "✗-"
    if m.get("point_estimate") is None:
        return "n/a"
    return "—"


def _render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Variant Attribution Analysis")
    lines.append("")
    lines.append(f"- Input: `{report.get('input_dir', 'unknown')}`")
    lines.append(f"- Total trials loaded: **{report.get('total_trials', 0)}**")
    lines.append(
        f"- Bootstrap resamples: {report.get('n_resamples', 0)}, "
        f"confidence: {report.get('confidence', 0):.0%}"
    )
    lines.append("")

    wins = report.get("significant_wins", [])
    losses = report.get("significant_losses", [])

    lines.append("## Summary")
    lines.append("")
    if wins:
        lines.append("### Variants significantly better than baseline")
        lines.append("")
        for w in wins:
            lines.append(f"- **{w['dimension']}** / `{w['variant']}` — {w['metric']}")
    else:
        lines.append("No variants significantly outperformed their respective baseline "
                     "at the given confidence level.")
    lines.append("")

    if losses:
        lines.append("### Variants significantly worse than baseline")
        lines.append("")
        for item in losses:
            lines.append(f"- **{item['dimension']}** / `{item['variant']}` — {item['metric']}")
        lines.append("")

    for dim, dim_result in report.get("attribution", {}).items():
        lines.append(f"## Dimension: `{dim}`")
        lines.append("")
        baseline = dim_result.get("baseline_variant", "n/a")
        n_base = dim_result.get("n_baseline_trials", 0)
        lines.append(f"Baseline variant: `{baseline}` (n={n_base} trials)")
        caveat = dim_result.get("adaptive_sampling_caveat", "")
        if caveat:
            lines.append(f"> *{caveat}*")
        lines.append("")

        note = dim_result.get("note")
        if note:
            lines.append(f"*{note}*")
            lines.append("")
            continue

        lines.append(
            "| variant | n | exploratory | Δquality | Δcompliance | Δdag_delta | "
            "q_sig | c_sig | d_sig |"
        )
        lines.append("|---|---:|:---:|---:|---:|---:|:---:|:---:|:---:|")

        for comp in dim_result.get("comparisons", []):
            lines.append(
                "| {v} | {n} | {exp} | {dq} | {dc} | {dd} | {sq} | {sc} | {sd} |".format(
                    v=comp["variant"],
                    n=comp["n_trials"],
                    exp="yes" if comp.get("exploratory") else "no",
                    dq=_fmt_delta(comp, "quality_vs_baseline"),
                    dc=_fmt_delta(comp, "compliance_vs_baseline"),
                    dd=_fmt_delta(comp, "dag_help_delta_vs_baseline"),
                    sq=_fmt_sig(comp, "quality_vs_baseline"),
                    sc=_fmt_sig(comp, "compliance_vs_baseline"),
                    sd=_fmt_sig(comp, "dag_help_delta_vs_baseline"),
                )
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Variant attribution analysis: bootstrap significance per variant group"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to a research optimization output directory (contains trials/)",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples per comparison (default: 2000)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.90,
        help="Confidence level for bootstrap CI (default: 0.90; use lower values for small n)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output path for JSON report (default: <input-dir>/variant_attribution.json)",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Output path for markdown report (default: <input-dir>/variant_attribution.md)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    report = attribute_variants(
        input_dir,
        n_resamples=args.n_resamples,
        confidence=args.confidence,
        seed=args.seed,
    )

    output_json = (
        Path(args.output_json) if args.output_json else input_dir / "variant_attribution.json"
    )
    output_md = (
        Path(args.output_md) if args.output_md else input_dir / "variant_attribution.md"
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote attribution JSON: {output_json}")
    print(f"Wrote attribution markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
