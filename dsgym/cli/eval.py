#!/usr/bin/env python3
"""
DSGym evaluation command.

This command provides a unified interface for evaluating agents on datasets,
similar to the example scripts but with a standardized CLI.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add DSGym to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dsgym.datasets import DatasetRegistry
from dsgym.agents import ReActDSAgent, DSPredictReActAgent
from dsgym.eval import Evaluator
from dsgym.eval.utils import EvaluationConfig


def add_eval_parser(subparsers):
    """Add eval command parser."""
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate agent on dataset",
        description="Evaluate LLM agents on data science datasets"
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 'gpt-4', 'together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput')")
    parser.add_argument("--backend", type=str, default="litellm",
                       choices=["litellm", "vllm", "sglang"],
                       help="Backend to use for model inference")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["daeval", "discoverybench", "qrdata", "dabstep", "dspredict-easy", "dspredict-hard", "dspredict-sampled", "dsbio", "dsbio-sampled"],
                       help="Dataset to evaluate on. Use 'dsbio-sampled' or 'dspredict-sampled' to evaluate on a previously created sample.")
    parser.add_argument("--limit", type=int, default=None,
                       help="Number of samples to evaluate")
    parser.add_argument("--dags", type=lambda x: x.lower() not in ("false", "0", "no"),
                       default=False, metavar="BOOL",
                       help="Enable iterative DAG-guided reasoning for DSBio and DSPredict. "
                            "When true, the agent self-generates and revises its analysis plan. "
                            "When false (default), uses the standard system prompt for baseline comparison. "
                            "Accepts true/false.")
    parser.add_argument("--prompt-variant", type=str, default="structured",
                       choices=["lightweight", "structured", "reasoning-only", "reasoning-focused",
                               "reasoning-implementation-strict", "detailed", "adaptive-revision", "expert-agentic"],
                       help="Prompt variant to use for system prompts. "
                            "lightweight: minimal guidance; structured: standard format; "
                            "reasoning-only: strict reasoning DAG; reasoning-focused: reasoning primary; "
                            "reasoning-implementation-strict: both DAGs required; detailed: rich guidance; "
                            "adaptive-revision: adaptive planning with up to 5 revisions; "
                            "expert-agentic: 2026 best practices with agentic loop.")
    parser.add_argument("--sample-n", type=int, default=None, metavar="N",
                       help="(DSBio/DSPredict) Randomly sample N tasks from the source split and save "
                            "as dsbio_sampled.json / dspredict_sampled.json before evaluating. "
                            "The saved file can be reused via --dataset dsbio-sampled / dspredict-sampled.")
    parser.add_argument("--sample-seed", type=int, default=42,
                       help="Random seed for --sample-n (default: 42)")
    parser.add_argument("--dag-validation-mode", "--dag_type", type=str, default="default",
                       choices=["default", "relaxed", "strict"],
                       help="DAG validation mode for generation: default|relaxed|strict. "
                            "Default preserves current behavior.")
    parser.add_argument("--synthetic-path", type=str, default=None,
                       help="Path to synthetic dataset (optional)")
    
    # Evaluation configuration
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Custom run name for output files")
    parser.add_argument("--max-turns", type=int, default=15,
                       help="Maximum turns per sample")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    
    # Infrastructure
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000",
                       help="Code sandbox manager URL")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers (auto-set based on backend if not provided)")
    parser.add_argument("--allocation-timeout", type=float, default=None,
                       help="Timeout in seconds for container allocation requests. "
                            "If omitted, dsbio defaults to 120s; other datasets block until available.")
    
    # API keys
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (uses environment variable if not provided)")
    
    return parser


def run_eval(args) -> int:
    """Run evaluation command."""
    print("🚀 Starting DSGym Evaluation")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.limit if args.limit is not None else 'all'}")
    print(f"Prompt variant: {args.prompt_variant}")

    is_dsbio = args.dataset in ("dsbio", "dsbio-sampled")
    is_dspredict = "dspredict" in args.dataset
    if is_dsbio or is_dspredict:
        mode = "iterative DAG" if args.dags else "standard (no DAG)"
        print(f"System prompt mode: {mode}")

    # Set default max_workers based on backend/provider
    if args.max_workers is None:
        if args.backend == "litellm":
            model_lower = args.model.lower()
            if model_lower.startswith("claude-") or model_lower.startswith("anthropic/"):
                args.max_workers = 4
            elif model_lower.startswith("together_ai/"):
                args.max_workers = 8
            else:
                args.max_workers = 24
        else:  # vllm or sglang
            args.max_workers = 1

    print(f"Max workers: {args.max_workers}")
    print(f"DAG validation mode: {args.dag_validation_mode}")
    print("-" * 50)

    old_dag_validation_mode = os.environ.get("DSGYM_DAG_VALIDATION_MODE")
    os.environ["DSGYM_DAG_VALIDATION_MODE"] = args.dag_validation_mode
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize agent
        print("🤖 Initializing agent...")
        agent_config = {
            "manager_url": args.manager_url,
            "max_turns": args.max_turns,
            "temperature": args.temperature,
            "output_dir": args.output_dir,
        }

        # DSBio requires no GPU workloads, so skip Apple Silicon / Darwin platform
        # detection and let the manager allocate any available container.
        if is_dsbio:
            agent_config["container_type"] = None
            # Fast-fail instead of waiting indefinitely for a container in dsbio runs.
            # This prevents apparent hangs when manager/container state is unhealthy.
            agent_config["allocation_timeout"] = 120.0 if args.allocation_timeout is None else args.allocation_timeout
        elif args.allocation_timeout is not None:
            agent_config["allocation_timeout"] = args.allocation_timeout

        if args.backend == "litellm" and args.api_key:
            agent_config["api_key"] = args.api_key

        try:
            if "dspredict" in args.dataset:
                agent = DSPredictReActAgent(
                    backend=args.backend,
                    model=args.model,
                    submission_dir="./submissions",
                    **agent_config
                )
            else:
                agent = ReActDSAgent(
                    backend=args.backend,
                    model=args.model,
                    **agent_config
                )
            print("✅ Agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            return 1

        # Load dataset
        print(f"📊 Loading {args.dataset} dataset...")
        try:
            dataset_config = {}
            if args.synthetic_path:
                dataset_config["synthetic_path"] = args.synthetic_path

            load_config = {"limit": args.limit}
            dataset_name = args.dataset

            if "dspredict" in dataset_name:
                # Parse split: dspredict-easy → "easy", dspredict-sampled → "sampled"
                split = dataset_name.split("-", 1)[-1]  # "easy", "hard", "sampled", …
                dataset_config["virtual_data_root"] = "/data"
                load_config["split"] = split
                load_config["generate_dags"] = args.dags
                dataset_name = "dspredict"
                dataset = DatasetRegistry.load(dataset_name, **dataset_config)

                # --sample-n: create dspredict_sampled.json from the specified split, then evaluate on it
                if args.sample_n is not None:
                    source_split = split if split != "sampled" else "hard"
                    dataset.create_sample(n=args.sample_n, split=source_split, seed=args.sample_seed)
                    load_config["split"] = "sampled"

            elif dataset_name in ("dsbio", "dsbio-sampled"):
                dataset_config["virtual_data_root"] = "/data"
                dataset_name = "dsbio"

                # --sample-n: create dsbio_sampled.json from the hard split, then evaluate on it
                if args.sample_n is not None:
                    _dataset = DatasetRegistry.load(dataset_name, **dataset_config)
                    _dataset.create_sample(n=args.sample_n, seed=args.sample_seed)
                    dataset = _dataset
                    load_config["split"] = "sampled"
                else:
                    dataset = DatasetRegistry.load(dataset_name, **dataset_config)
                    if args.dataset == "dsbio-sampled":
                        load_config["split"] = "sampled"

                load_config["generate_dags"] = args.dags

            else:
                dataset = DatasetRegistry.load(dataset_name, **dataset_config)

            samples = dataset.load(**load_config)
            print(f"✅ Loaded {len(samples)} samples from {args.dataset}")

        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return 1

        # Create evaluator with dataset-specific metrics
        print("⚖️ Setting up evaluator...")
        evaluator = Evaluator(
            protocol="multi_turn",
            dataset=dataset,
            parallel_workers=args.max_workers
        )

        # Create evaluation config
        run_name = args.run_name or f"{args.dataset}_{args.backend}_{args.model.replace('/', '_')}"
        if (is_dsbio or is_dspredict) and not args.run_name:
            run_name += "_dags" if args.dags else "_nodags"
        config = EvaluationConfig(
            model_name=args.model,
            backend_type=args.backend,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            run_name=run_name,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_workers=args.max_workers,
            extra_config={"generate_dags": args.dags} if is_dsbio or is_dspredict else {}
        )

        # Run evaluation
        print("🏃 Starting evaluation...")
        try:
            results = evaluator.evaluate(
                agent=agent,
                tasks=samples,
                config=config,
                save_results=True
            )

            print("✅ Evaluation completed!")
            print(f"📁 Results saved to: {args.output_dir}")

            # Print summary statistics
            if "metrics" in results:
                print("\n📈 Results Summary:")
                print("-" * 30)
                metrics = results["metrics"]

                # Show key metrics
                key_metrics = [
                    "success_rate", "total_samples", "successful_samples",
                    "average_execution_time", "average_turns"
                ]

                for metric_name in key_metrics:
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        if isinstance(value, float):
                            print(f"{metric_name}: {value:.3f}")
                        else:
                            print(f"{metric_name}: {value}")

                # Show dataset-specific metrics
                metric_scores = {}
                for key, value in metrics.items():
                    if key.endswith("_mean") and not key.startswith("error"):
                        metric_name = key.replace("_mean", "")
                        metric_scores[metric_name] = value

                if metric_scores:
                    print("\n📊 Dataset Metrics:")
                    print("-" * 20)
                    for metric_name, score in metric_scores.items():
                        print(f"{metric_name}: {score:.3f}")
            if is_dspredict:
                dataset.print_dspredict_results_overview(results["results"])

            return 0

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            print("Traceback:")
            traceback.print_exc()
            return 1
    finally:
        if old_dag_validation_mode is None:
            os.environ.pop("DSGYM_DAG_VALIDATION_MODE", None)
        else:
            os.environ["DSGYM_DAG_VALIDATION_MODE"] = old_dag_validation_mode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSGym evaluation")
    add_eval_parser(parser._subparsers_action.add_parser if hasattr(parser, '_subparsers_action') else parser.add_subparsers())
    args = parser.parse_args()
    sys.exit(run_eval(args))
