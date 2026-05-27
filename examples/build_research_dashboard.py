#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from dsgym.optimization import load_research_trial_points, pareto_front_indices


def _build_dataframe(input_dirs: List[Path], labels: List[str] | None = None) -> pd.DataFrame:
    points = load_research_trial_points(input_dirs)
    if not points:
        return pd.DataFrame()

    label_map: Dict[str, str] = {}
    if labels and len(labels) == len(input_dirs):
        for path, label in zip(input_dirs, labels):
            label_map[str(path)] = label

    rows = []
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

    df = pd.DataFrame(rows)
    return df


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_quality_time(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_label, group in df.groupby("run_label"):
        ax.scatter(group["mean_time"], group["mean_quality"], alpha=0.7, label=run_label)

        pseudo_points = [
            type(
                "Point",
                (),
                {
                    "mean_quality": float(row["mean_quality"]),
                    "mean_time": float(row["mean_time"]),
                    "mean_turns": float(row["mean_turns"]),
                    "mean_compliance": float(row["mean_compliance"]),
                    "dag_help_delta": float(row["dag_help_delta"]),
                },
            )()
            for _, row in group.iterrows()
        ]
        idx = pareto_front_indices(pseudo_points)
        if idx:
            frontier = group.iloc[idx].sort_values("mean_time")
            ax.plot(frontier["mean_time"], frontier["mean_quality"], linewidth=2)

    ax.set_xlabel("Mean execution time (lower is better)")
    ax.set_ylabel("Mean quality score (higher is better)")
    ax.set_title("Pareto View: Quality vs Time")
    ax.grid(alpha=0.25)
    ax.legend()
    return _fig_to_base64(fig)


def _plot_accuracy_compliance(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_label, group in df.groupby("run_label"):
    eval_group = group[group["exact_match_accuracy"].notna()]
    if eval_group.empty:
      continue
    sizes = 40 + 120 * eval_group["mean_quality"].clip(lower=0.0)
        ax.scatter(
      eval_group["exact_match_accuracy"],
      eval_group["mean_compliance"],
            s=sizes,
            alpha=0.65,
            label=run_label,
        )

    ax.set_xlabel("Exact-match accuracy")
    ax.set_ylabel("Harness compliance")
    ax.set_title("Accuracy vs Harness Compliance (bubble size = quality)")
    ax.grid(alpha=0.25)
    if ax.has_data():
      ax.legend()
    return _fig_to_base64(fig)


def _plot_run_summary(df: pd.DataFrame) -> str:
    summary = (
        df.groupby("run_label", as_index=False)
        .agg(
            mean_quality=("mean_quality", "mean"),
            mean_time=("mean_time", "mean"),
            mean_compliance=("mean_compliance", "mean"),
            mean_exact_match_accuracy=("exact_match_accuracy", "mean"),
        )
        .sort_values("mean_quality", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].bar(summary["run_label"], summary["mean_quality"], alpha=0.8, label="quality")
    axes[0].bar(summary["run_label"], summary["mean_compliance"], alpha=0.65, label="compliance")
    axes[0].set_title("Quality and Compliance by Run")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(summary["run_label"], summary["mean_time"], alpha=0.8, color="#d97706")
    axes[1].set_title("Mean Execution Time by Run")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.2)

    return _fig_to_base64(fig)


def _pareto_rows(df: pd.DataFrame) -> List[Dict[str, object]]:
    pseudo_points = [
        type(
            "Point",
            (),
            {
                "mean_quality": float(row.mean_quality),
                "mean_time": float(row.mean_time),
                "mean_turns": float(row.mean_turns),
                "mean_compliance": float(row.mean_compliance),
                "dag_help_delta": float(row.dag_help_delta),
            },
        )()
        for row in df.itertuples(index=False)
    ]
    global_idx = set(pareto_front_indices(pseudo_points))

    rows: List[Dict[str, object]] = []
    for i, row in enumerate(df.itertuples(index=False)):
        if i not in global_idx:
            continue
        rows.append(
            {
                "run_label": row.run_label,
                "model": row.model,
                "dataset": row.dataset,
                "trial_number": int(row.trial_number),
                "quality": float(row.mean_quality),
                "success": float(row.mean_success),
                "exact_match_accuracy": float(row.exact_match_accuracy),
                "compliance": float(row.mean_compliance),
                "time": float(row.mean_time),
                "turns": float(row.mean_turns),
                "dag_help_delta": float(row.dag_help_delta),
                "prompt_variant": row.prompt_variant,
                "context_variant": row.context_variant,
                "harness_variant": row.harness_variant,
            }
        )

    rows.sort(key=lambda item: (-item["quality"], item["time"], item["turns"]))
    return rows


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["run_label", "model", "dataset"], as_index=False)
        .agg(
            trials=("trial_number", "count"),
            mean_quality=("mean_quality", "mean"),
            mean_success=("mean_success", "mean"),
            mean_exact_match_accuracy=("exact_match_accuracy", "mean"),
            mean_compliance=("mean_compliance", "mean"),
            mean_time=("mean_time", "mean"),
            mean_turns=("mean_turns", "mean"),
            mean_dag_help_delta=("dag_help_delta", "mean"),
        )
        .sort_values(["mean_quality", "mean_compliance", "mean_exact_match_accuracy"], ascending=False)
    )


def _html_table_from_df(df: pd.DataFrame, table_id: str) -> str:
    header = "".join(f"<th>{html.escape(col)}</th>" for col in df.columns)
    body_rows = []
    for row in df.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
              if math.isnan(value):
                text = ""
              elif abs(value) >= 1000:
                text = f"{value:,.2f}"
              else:
                text = f"{value:.4f}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        f"<table id=\"{table_id}\">"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _render_dashboard_html(title: str, df: pd.DataFrame) -> str:
    summary_df = _summary_table(df)
    pareto = _pareto_rows(df)

    quality_img = _plot_quality_time(df)
    acc_img = _plot_accuracy_compliance(df)
    summary_img = _plot_run_summary(df)

    total_trials = int(len(df))
    total_runs = int(df["run_label"].nunique())
    total_models = int(df["model"].nunique())
    mean_quality = float(df["mean_quality"].mean()) if total_trials else 0.0

    pareto_json = json.dumps(pareto, ensure_ascii=False)
    run_labels = sorted(df["run_label"].unique().tolist())
    models = sorted(df["model"].unique().tolist())

    summary_table_html = _html_table_from_df(summary_df, "summary-table")
    trials_table_html = _html_table_from_df(df.sort_values(["mean_quality", "mean_compliance"], ascending=False), "trials-table")

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --card: #ffffff;
      --ink: #101828;
      --muted: #475467;
      --line: #d0d5dd;
      --accent: #175cd3;
      --accent2: #ea580c;
    }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
    header {{ padding: 26px 32px; background: linear-gradient(135deg, #dbeafe, #fff7ed); border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0; font-size: 28px; }}
    .subtitle {{ margin-top: 8px; color: var(--muted); }}
    .container {{ padding: 20px 24px 36px; display: grid; gap: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }}
    .k {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
    .v {{ margin-top: 6px; font-size: 24px; font-weight: 700; }}
    .panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }}
    .panel h2 {{ margin: 4px 0 10px; font-size: 18px; }}
    img.plot {{ width: 100%; border: 1px solid var(--line); border-radius: 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e4e7ec; padding: 8px 6px; text-align: left; }}
    th {{ background: #f9fafb; position: sticky; top: 0; }}
    .table-wrap {{ max-height: 340px; overflow: auto; border: 1px solid var(--line); border-radius: 10px; }}
    .filters {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
    select {{ padding: 6px 8px; border-radius: 8px; border: 1px solid var(--line); background: #fff; }}
    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eff8ff; color: #175cd3; font-size: 12px; }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class=\"subtitle\">Consolidated research optimization dashboard: quality, exact-match, speed, compliance, and Pareto frontier.</div>
  </header>

  <div class=\"container\">
    <div class=\"cards\">
      <div class=\"card\"><div class=\"k\">Total Trials</div><div class=\"v\">{total_trials}</div></div>
      <div class=\"card\"><div class=\"k\">Run Groups</div><div class=\"v\">{total_runs}</div></div>
      <div class=\"card\"><div class=\"k\">Models</div><div class=\"v\">{total_models}</div></div>
      <div class=\"card\"><div class=\"k\">Mean Quality</div><div class=\"v\">{mean_quality:.4f}</div></div>
    </div>

    <section class=\"panel\">
      <h2>Run and Model Summary</h2>
      <div class=\"table-wrap\">{summary_table_html}</div>
    </section>

    <section class=\"panel\">
      <h2>Pareto View: Quality vs Time</h2>
      <img class=\"plot\" src=\"data:image/png;base64,{quality_img}\" alt=\"Quality vs time plot\" />
    </section>

    <section class=\"panel\">
      <h2>Accuracy vs Compliance</h2>
      <img class=\"plot\" src=\"data:image/png;base64,{acc_img}\" alt=\"Accuracy vs compliance plot\" />
    </section>

    <section class=\"panel\">
      <h2>Run-Level Aggregates</h2>
      <img class=\"plot\" src=\"data:image/png;base64,{summary_img}\" alt=\"Run summary plot\" />
    </section>

    <section class=\"panel\">
      <h2>Pareto Explorer <span class=\"tag\">interactive filters</span></h2>
      <div class=\"filters\">
        <label>Run:
          <select id=\"run-filter\">
            <option value=\"ALL\">ALL</option>
            {''.join(f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in run_labels)}
          </select>
        </label>
        <label>Model:
          <select id=\"model-filter\">
            <option value=\"ALL\">ALL</option>
            {''.join(f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in models)}
          </select>
        </label>
      </div>
      <div class=\"table-wrap\">
        <table id=\"pareto-table\">
          <thead>
            <tr>
              <th>run</th><th>model</th><th>dataset</th><th>trial</th><th>quality</th><th>success</th>
              <th>exact acc</th><th>compliance</th><th>time</th><th>turns</th><th>dag delta</th>
              <th>prompt</th><th>context</th><th>harness</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </section>

    <section class=\"panel\">
      <h2>All Trials</h2>
      <div class=\"table-wrap\">{trials_table_html}</div>
    </section>
  </div>

  <script>
    const paretoRows = {pareto_json};
    const tbody = document.querySelector('#pareto-table tbody');
    const runFilter = document.getElementById('run-filter');
    const modelFilter = document.getElementById('model-filter');

    function fmt(num, digits=4) {{
      if (num === null || num === undefined || Number.isNaN(num)) return '';
      return Number(num).toFixed(digits);
    }}

    function renderPareto() {{
      const runVal = runFilter.value;
      const modelVal = modelFilter.value;
      const filtered = paretoRows.filter((row) =>
        (runVal === 'ALL' || row.run_label === runVal) &&
        (modelVal === 'ALL' || row.model === modelVal)
      );

      tbody.innerHTML = filtered.map((row) => `
        <tr>
          <td>${{row.run_label}}</td>
          <td>${{row.model}}</td>
          <td>${{row.dataset}}</td>
          <td>${{row.trial_number}}</td>
          <td>${{fmt(row.quality)}}</td>
          <td>${{fmt(row.success)}}</td>
          <td>${{fmt(row.exact_match_accuracy)}}</td>
          <td>${{fmt(row.compliance)}}</td>
          <td>${{fmt(row.time, 2)}}</td>
          <td>${{fmt(row.turns, 2)}}</td>
          <td>${{fmt(row.dag_help_delta)}}</td>
          <td>${{row.prompt_variant}}</td>
          <td>${{row.context_variant}}</td>
          <td>${{row.harness_variant}}</td>
        </tr>
      `).join('');
    }}

    runFilter.addEventListener('change', renderPareto);
    modelFilter.addEventListener('change', renderPareto);
    renderPareto();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build consolidated HTML dashboard for research optimization")
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
        help="Optional run labels (same order as input dirs)",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="./optimization_results/research/research_dashboard.html",
        help="Dashboard HTML output path",
    )
    parser.add_argument("--title", type=str, default="DSGym Research Benchmark Dashboard")
    args = parser.parse_args()

    input_dirs = [Path(path) for path in args.input_dirs]
    df = _build_dataframe(input_dirs=input_dirs, labels=args.labels)

    if df.empty:
        print("No research trial records found in the provided directories.")
        return 0

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    dashboard_html = _render_dashboard_html(args.title, df)
    output_html.write_text(dashboard_html, encoding="utf-8")
    print(f"Wrote dashboard: {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
