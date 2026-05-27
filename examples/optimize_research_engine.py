#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from dsgym.optimization import ResearchOptimizationConfig, run_research_nsga2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run research-grade DAG/prompt/context/harness optimization with NSGA-II"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name for agent evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dsbio",
        choices=["dsbio", "dsbio-sampled", "dspredict-easy", "dspredict-hard", "dspredict-sampled"],
        help="Dataset for optimization",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="litellm",
        choices=["litellm", "vllm", "sglang"],
        help="Backend",
    )
    parser.add_argument("--dag-model", type=str, default=None, help="Optional DAG model metadata")
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--task-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--dag-validation-mode",
        type=str,
        default="default",
        choices=["default", "relaxed", "strict"],
    )
    parser.add_argument(
        "--tune-dag-validation-mode",
        action="store_true",
        help="Jointly optimize DAG validation mode",
    )
    parser.add_argument(
        "--no-dag-ablation",
        action="store_true",
        help="Disable DAG-vs-noDAG ablation inside each trial",
    )
    parser.add_argument("--max-turns-min", type=int, default=10)
    parser.add_argument("--max-turns-max", type=int, default=18)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default="dsgym_research_nsga2")
    parser.add_argument("--output-dir", type=str, default="./optimization_results/research")
    parser.add_argument(
        "--generation-model",
        type=str,
        default=None,
        help="Optional model used to generate additional prompt/context/harness variants",
    )
    parser.add_argument("--generated-variants-per-type", type=int, default=3)
    parser.add_argument(
        "--no-reuse-variant-bank-cache",
        action="store_true",
        help="Do not reuse existing variant_bank.json; regenerate from templates/LLM",
    )
    parser.add_argument(
        "--force-regenerate-variants",
        action="store_true",
        help="Force regeneration of variant bank and overwrite cache",
    )
    parser.add_argument(
        "--no-strict-harness-evaluator",
        action="store_true",
        help="Disable strict harness compliance evaluation",
    )
    parser.add_argument(
        "--harness-max-identical-message-repeats",
        type=int,
        default=1,
        help="Max allowed consecutive repeated assistant messages before loop violation",
    )
    parser.add_argument(
        "--harness-max-identical-code-repeats",
        type=int,
        default=1,
        help="Max allowed consecutive repeated Python blocks before loop violation",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = ResearchOptimizationConfig(
        model=args.model,
        dataset=args.dataset,
        backend=args.backend,
        dag_model=args.dag_model,
        manager_url=args.manager_url,
        api_key=args.api_key,
        limit=args.limit,
        sample_n=args.sample_n,
        sample_seed=args.sample_seed,
        temperature=args.temperature,
        max_workers=args.max_workers,
        task_timeout_seconds=args.task_timeout_seconds,
        dag_validation_mode=args.dag_validation_mode,
        tune_dag_validation_mode=args.tune_dag_validation_mode,
        ablate_dags=not args.no_dag_ablation,
        max_turns_min=args.max_turns_min,
        max_turns_max=args.max_turns_max,
        n_trials=args.trials,
        population_size=args.population_size,
        seed=args.seed,
        study_name=args.study_name,
        output_dir=args.output_dir,
        generation_model=args.generation_model,
        generated_variants_per_type=args.generated_variants_per_type,
        reuse_variant_bank_cache=not args.no_reuse_variant_bank_cache,
        force_regenerate_variants=args.force_regenerate_variants,
        strict_harness_evaluator=not args.no_strict_harness_evaluator,
        harness_max_identical_message_repeats=args.harness_max_identical_message_repeats,
        harness_max_identical_code_repeats=args.harness_max_identical_code_repeats,
    )

    summary = run_research_nsga2(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
