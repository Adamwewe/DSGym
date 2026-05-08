#!/usr/bin/env python3
"""
DSGym resubmit command.

Reads per-task intermediate checkpoint JSON files produced by a previous
evaluation run, submits the saved submission.csv files to Kaggle, and writes
the metric results back into each checkpoint file in-place.

This lets you recover Kaggle scores from a partial or completed run without
re-running the agent.

Usage example:
  dsgym resubmit --input-dir ./evaluation_results --pattern "*_claude_sonnet_4_5_dags.json"
"""

import argparse
import json
import os
import sys
import time
from glob import glob
from pathlib import Path
from typing import Optional


# Suffixes used by the aggregate results files — skip them
_AGGREGATE_SUFFIXES = ("_results.json", "_summary.json", "_config.json", "_metrics.json")


def add_resubmit_parser(subparsers):
    """Add resubmit command parser."""
    parser = subparsers.add_parser(
        "resubmit",
        help="Re-score saved submission files against Kaggle without re-running agents",
        description=(
            "Re-submit Kaggle CSV files from a previous evaluation run.\n\n"
            "Each matching checkpoint JSON (produced by the evaluator's per-task "
            "intermediate save) is read, its 'prediction' field is re-submitted to "
            "Kaggle, and the resulting metrics are written back into the file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-score all claude-sonnet-4-5 DAG checkpoints
  dsgym resubmit --pattern "*_claude_sonnet_4_5_dags.json"

  # Re-score from a specific directory with a timeout of 20 min per submission
  dsgym resubmit --input-dir ./evaluation_results --pattern "*_dags.json" --timeout 20

  # Dry-run: print what would be submitted without actually uploading
  dsgym resubmit --pattern "*_dags.json" --dry-run
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="./evaluation_results",
        help="Directory containing checkpoint JSON files (default: ./evaluation_results)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help=(
            "Glob pattern for checkpoint files relative to --input-dir "
            "(default: '*.json').  Use a more specific pattern to avoid "
            "accidentally picking up aggregate result files, e.g. "
            "'*_claude_sonnet_4_5_dags.json'."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Minutes to wait for each Kaggle submission to be scored (default: 10)",
    )
    parser.add_argument(
        "--skip-scored",
        action="store_true",
        default=True,
        help=(
            "Skip checkpoints that already have a non-null kaggle_submission score "
            "(default: True — avoids wasting Kaggle submission quota)"
        ),
    )
    parser.add_argument(
        "--no-skip-scored",
        dest="skip_scored",
        action="store_false",
        help="Re-submit even checkpoints that already have a score",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be submitted without uploading to Kaggle",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel submissions (default: 1 — sequential).  "
            "Sequential is recommended on slow connections so each upload gets "
            "full bandwidth.  Increase only on fast connections (100+ Mbps)."
        ),
    )

    return parser


def _is_aggregate_file(path: Path) -> bool:
    """Return True if the file is an aggregate results / config / metrics file."""
    name = path.name
    return any(name.endswith(s) for s in _AGGREGATE_SUFFIXES)


def _already_scored(checkpoint: dict) -> bool:
    """Return True if this checkpoint already has a non-null kaggle_submission score."""
    ks = checkpoint.get("metrics", {}).get("kaggle_submission", {})
    return ks.get("score") is not None


def _resubmit_one(filepath: Path, timeout_minutes: int, dry_run: bool) -> dict:
    """
    Read a single checkpoint, submit its prediction to Kaggle, update the file.

    Returns a summary dict with keys: filepath, challenge, status, score, error.
    """
    summary = {"filepath": str(filepath), "challenge": None, "status": None, "score": None, "error": None}

    try:
        with open(filepath, encoding="utf-8") as f:
            checkpoint = json.load(f)
    except Exception as e:
        summary["error"] = f"Failed to read file: {e}"
        return summary

    extra_info = checkpoint.get("extra_info", {})
    challenge_name = (
        extra_info.get("challenge_name")
        or extra_info.get("metadata_id")
        or ""
    )
    submission_path = (checkpoint.get("prediction") or "").strip()
    summary["challenge"] = challenge_name or filepath.stem

    if not challenge_name:
        summary["error"] = "No challenge_name found in checkpoint"
        return summary

    if not submission_path or not os.path.exists(submission_path):
        summary["error"] = f"Submission file not found: {submission_path!r}"
        return summary

    if dry_run:
        print(f"  [dry-run] Would submit {submission_path} → {challenge_name}")
        summary["status"] = "DRY_RUN"
        return summary

    # Import metric here to avoid heavy imports at module level
    from dsgym.eval.metrics.dspredict.dspredict_metric import KaggleSubmissionMetric

    metric = KaggleSubmissionMetric(timeout_minutes=timeout_minutes, online=True)

    result = metric.evaluate(
        prediction=submission_path,
        ground_truth=None,
        query=None,
        extra_info={"challenge_name": challenge_name},
    )

    result_dict = result.to_dict() if hasattr(result, "to_dict") else {
        "metric_name": result.metric_name,
        "score": result.score,
        "details": result.details,
        "error": result.error,
    }

    # Write metric result back into the checkpoint
    if "metrics" not in checkpoint:
        checkpoint["metrics"] = {}
    checkpoint["metrics"]["kaggle_submission"] = result_dict

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    except Exception as e:
        summary["error"] = f"Failed to write updated checkpoint: {e}"
        return summary

    details = result_dict.get("details", {})
    summary["status"] = details.get("status", "UNKNOWN")
    summary["score"] = result_dict.get("score")
    return summary


def run_resubmit(args) -> int:
    """Run the resubmit command."""
    import concurrent.futures

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        return 1

    # Collect matching checkpoint files
    all_files = sorted(input_dir.glob(args.pattern))
    checkpoint_files = [p for p in all_files if p.is_file() and not _is_aggregate_file(p)]

    if not checkpoint_files:
        print(f"❌ No checkpoint files found matching '{args.pattern}' in {input_dir}")
        return 1

    print(f"📂 Found {len(checkpoint_files)} checkpoint file(s) in {input_dir}")

    # Optionally skip already-scored files
    to_process = []
    skipped = 0
    for fp in checkpoint_files:
        try:
            with open(fp, encoding="utf-8") as f:
                chk = json.load(f)
            if args.skip_scored and _already_scored(chk):
                skipped += 1
                continue
        except Exception:
            pass
        to_process.append(fp)

    if skipped:
        print(f"⏭️  Skipping {skipped} already-scored checkpoint(s) (use --no-skip-scored to override)")

    if not to_process:
        print("✅ Nothing to resubmit.")
        return 0

    print(f"🚀 Resubmitting {len(to_process)} checkpoint(s) with {args.max_workers} worker(s)…")
    if args.dry_run:
        print("⚠️  DRY RUN — no actual Kaggle submissions will be made\n")

    summaries = []
    start = time.time()

    def _do(fp):
        return _resubmit_one(fp, timeout_minutes=args.timeout, dry_run=args.dry_run)

    if args.max_workers <= 1:
        for fp in to_process:
            s = _do(fp)
            summaries.append(s)
            _print_summary_row(s)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(_do, fp): fp for fp in to_process}
            for fut in concurrent.futures.as_completed(futures):
                s = fut.result()
                summaries.append(s)
                _print_summary_row(s)

    elapsed = time.time() - start

    # Final overview
    ok = [s for s in summaries if s["score"] is not None]
    errors = [s for s in summaries if s["error"]]
    print("\n" + "=" * 50)
    print(f"✅ Scored     : {len(ok)} / {len(summaries)}")
    print(f"❌ Errors     : {len(errors)}")
    print(f"⏱  Total time : {elapsed / 60:.1f} min")
    if ok:
        scores = [s["score"] for s in ok]
        print(f"📊 Avg public score: {sum(scores)/len(scores):.6f}")
    if errors:
        print("\nFailed submissions:")
        for s in errors:
            print(f"  {Path(s['filepath']).name}: {s['error']}")

    # Print the full DSPredict leaderboard summary (same as eval run output)
    _print_dspredict_overview(to_process)

    return 0


def _print_summary_row(s: dict) -> None:
    name = Path(s["filepath"]).name
    if s["error"]:
        print(f"  ❌ {name}: {s['error']}")
    elif s["status"] == "DRY_RUN":
        print(f"  🔍 {name}: dry-run ok")
    else:
        score_str = f"{s['score']:.6f}" if s["score"] is not None else "no score"
        print(f"  ✅ {name}: {s['status']} | score={score_str}")


def _print_dspredict_overview(checkpoint_files: list) -> None:
    """Load the (now-scored) checkpoint files and print the same leaderboard
    summary that ``print_dspredict_results_overview`` produces at the end of
    a normal eval run."""
    samples = []
    for fp in checkpoint_files:
        try:
            with open(fp, encoding="utf-8") as f:
                samples.append(json.load(f))
        except Exception:
            pass

    if not samples:
        return

    try:
        from dsgym.datasets.loaders.dspredict import DSPredictDataset
        # DSPredictDataset is instantiated without arguments for this purpose
        ds = object.__new__(DSPredictDataset)
        ds.print_dspredict_results_overview(samples)
    except Exception as e:
        print(f"⚠️  Could not print DSPredict overview: {e}")
