#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from dsgym.optimization import load_research_trial_points, pareto_front_indices


def _build_dataframe(input_dirs: List[Path], labels: List[str] | None = None) -> pd.DataFrame:
    points = load_research_trial_points(input_dirs)
    if not points:
        return pd.DataFrame()

    rows = []
    label_map: Dict[str, str] = {}
    if labels and len(labels) == len(input_dirs):
        for path, label in zip(input_dirs, labels):
            label_map[str(path)] = label

    for point in points:
        run_label = label_map.get(point.input_dir, Path(point.input_dir).name)
        rows.append(
            {
                "run_label": run_label,
                "input_dir": point.input_dir,
                "study_name": point.study_name,
                "model": point.model,
                "dataset": point.dataset,
                "trial_number": point.trial_number,
                "prompt_variant": point.prompt_variant,
                "context_variant": point.context_variant,
                "harness_variant": point.harness_variant,
                "dag_system_variant": point.dag_system_variant,
                "mean_quality": point.mean_quality,
                "mean_success": point.mean_success,
                "mean_time": point.mean_time,
                "mean_turns": point.mean_turns,
                "mean_compliance": point.mean_compliance,
                "dag_help_delta": point.dag_help_delta,
                "exact_match_accuracy": point.exact_match_accuracy,
                "exact_match_correct": point.exact_match_correct,
                "total_samples": point.total_samples,
            }
        )

    return pd.DataFrame(rows)


def _plot_quality_time_pareto(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(11, 7))

    for run_label, group in df.groupby("run_label"):
        plt.scatter(
            group["mean_time"],
            group["mean_quality"],
            alpha=0.6,
            label=run_label,
        )

        group_points = group.to_dict("records")
        idx = pareto_front_indices([
            type("Point", (), {
                "mean_quality": row["mean_quality"],
                "mean_time": row["mean_time"],
                "mean_turns": row["mean_turns"],
                "mean_compliance": row["mean_compliance"],
                "dag_help_delta": row["dag_help_delta"],
            })()
            for row in group_points
        ])
        if idx:
            frontier = group.iloc[idx].sort_values("mean_time")
            plt.plot(frontier["mean_time"], frontier["mean_quality"], linewidth=2)

    plt.xlabel("Mean execution time (lower is better)")
    plt.ylabel("Mean quality score (higher is better)")
    plt.title("Pareto View: Quality vs Time")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_accuracy_time(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(11, 7))
    for run_label, group in df.groupby("run_label"):
        plt.scatter(
            group["mean_time"],
            group["exact_match_accuracy"],
            alpha=0.7,
            label=run_label,
        )
    plt.xlabel("Mean execution time")
    plt.ylabel("Exact-match accuracy")
    plt.title("Accuracy vs Execution Time")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_metric_boxplots(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics: List[Tuple[str, str, int, int]] = [
        ("mean_quality", "Quality", 0, 0),
        ("mean_time", "Execution Time", 0, 1),
        ("mean_compliance", "Harness Compliance", 1, 0),
        ("exact_match_accuracy", "Exact-Match Accuracy", 1, 1),
    ]

    for column, title, row, col in metrics:
        pivot = [group[column].dropna().values for _, group in df.groupby("run_label")]
        labels = [name for name, _ in df.groupby("run_label")]
        axes[row][col].boxplot(pivot, labels=labels)
        axes[row][col].set_title(title)
        axes[row][col].tick_params(axis="x", rotation=20)
        axes[row][col].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_model_comparison(df: pd.DataFrame, out_path: Path) -> None:
    summary = (
        df.groupby(["model", "dataset", "run_label"], as_index=False)
        .agg(
            trials=("trial_number", "count"),
            mean_quality=("mean_quality", "mean"),
            mean_success=("mean_success", "mean"),
            mean_time=("mean_time", "mean"),
            mean_turns=("mean_turns", "mean"),
            mean_compliance=("mean_compliance", "mean"),
            mean_exact_match_accuracy=("exact_match_accuracy", "mean"),
            mean_dag_help_delta=("dag_help_delta", "mean"),
        )
        .sort_values(["mean_quality", "mean_compliance", "mean_exact_match_accuracy"], ascending=False)
    )
    summary.to_csv(out_path, index=False)


def _plot_dag_style_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Box-plot each key metric broken down by dag_system_variant.

    This is the primary plot for understanding which DAG instruction style
    (baseline full-format, reasoning-only, generic-plan, or minimal) drives
    the best agent performance.
    """
    if "dag_system_variant" not in df.columns or df["dag_system_variant"].nunique() < 2:
        return

    style_order = [
        "dag_style_baseline",
        "dag_style_reasoning_only",
        "dag_style_generic_plan",
        "dag_style_minimal",
    ]
    present = [s for s in style_order if s in df["dag_system_variant"].values]
    present += [s for s in df["dag_system_variant"].unique() if s not in style_order]

    metrics: List[Tuple[str, str]] = [
        ("mean_quality", "Quality score (↑)"),
        ("dag_help_delta", "DAG help delta (↑)"),
        ("mean_compliance", "Harness compliance (↑)"),
        ("mean_turns", "Turns (↓)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, metrics):
        data = [df[df["dag_system_variant"] == style][col].dropna().values for style in present]
        bp = ax.boxplot(data, patch_artist=True, labels=[s.replace("dag_style_", "") for s in present])
        colours = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("DAG instruction style comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize research optimization outputs across runs/models")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["./optimization_results/research"],
        help="One or more research optimization output directories",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels (same count/order as --input-dirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./optimization_results/research/figures",
        help="Directory for output figures and CSV summaries",
    )
    args = parser.parse_args()

    input_dirs = [Path(path) for path in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _build_dataframe(input_dirs=input_dirs, labels=args.labels)
    if df.empty:
        print("No trial records found in the provided input directories.")
        return 0

    csv_path = output_dir / "research_trials_flat.csv"
    df.to_csv(csv_path, index=False)

    _write_model_comparison(df, output_dir / "model_comparison_summary.csv")
    _plot_quality_time_pareto(df, output_dir / "pareto_quality_time.png")
    _plot_accuracy_time(df, output_dir / "accuracy_vs_time.png")
    _plot_metric_boxplots(df, output_dir / "run_metric_boxplots.png")
    _plot_dag_style_comparison(df, output_dir / "dag_style_comparison.png")

    print(f"Wrote flattened trials: {csv_path}")
    print(f"Wrote figures to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
