"""
DSPredict dataset loader.
"""

import os
import json
from typing import List, Dict, Any, Optional
from collections import Counter
from statistics import mean

from ..base import BaseDataset
from ..registry import register_dataset
from ..utils import apply_limit_and_start, validate_file_exists, create_standard_task
from ..prompts import SYSTEM_PROMPT_DSPREDICT, SYSTEM_PROMPT_DAG_DSPREDICT, get_variant, list_variants
from dsgym.datasets.config import get_task_path, RAW_DATA_DIR
from .kaggle_downloader import KaggleChallengeDownloader

DSPREDICT_INSTRUCTIONS = """**INSTRUCTIONS:**
1. Load and explore the **training** and **test** datasets using Python (use the dataset folder location provided).
2. Perform **data preprocessing** (handling missing values, encoding, scaling, feature engineering) and **exploratory analysis** to understand distributions, correlations, and relationships between variables.
3. Where simple preprocessing and baseline models are insufficient, attempt more advanced approaches such as:
   * Model selection (e.g., tree-based models, linear models, neural networks)
   * Cross-validation and hyperparameter tuning
   * Dimensionality reduction, feature selection, or ensembling
   * Robustness checks or combining datasets if useful
4. Use the training data to build a model, evaluate it with proper validation, and then generate **predictions for the test data**.
5. Do one step at a time. Explore and validate thoroughly before moving on to model training and submission.
6. When doing exploration and data analysis, print the results in a clear and concise way.
7. Do not use plotting libraries (assume you cannot view plots). Use text-based summaries and statistics instead.
8. When workflow tags or competition-specific guidelines are provided, you should follow them closely.
9. Only produce the **final submission and answer** when you have enough evidence and validation to support your approach.

When you finished training the best model, you should generate the final submission:

1. Use the best model to generate predictions for the test data located at the path shown above.
2. Save predictions in the required **`submission.csv` format** for the competition at /submission/submission.csv.
3. **Validate the submission format** by comparing your file against the sample submission:
   ```python
   import pandas as pd
    sample = pd.read_csv('__SAMPLE_SUBMISSION_PATH__')
   submission = pd.read_csv('/submission/submission.csv')
   assert list(submission.columns) == list(sample.columns), f"Column mismatch: expected {list(sample.columns)}, got {list(submission.columns)}"
   assert len(submission) == len(sample), f"Row count mismatch: expected {len(sample)}, got {len(submission)}"
   print("Submission format OK")
   ```
   Fix any column name, column order, or row count mismatches before finishing.
4. Provide a concise summary of your approach in the format: <answer>your final summary</answer>
"""


def create_dspredict_prompt(
    challenge_name: str,
    description: str,
    data_paths: Dict[str, List[str]],
    sample_submission_path: str,
) -> str:
    dataset_path = data_paths['virtual'][0]
    instructions = DSPREDICT_INSTRUCTIONS.replace('__SAMPLE_SUBMISSION_PATH__', sample_submission_path)
    
    user_content = f"""**CHALLENGE NAME: {challenge_name}**

{description}

**DATASET LOCATIONS (use full paths):**
{dataset_path}

**SAMPLE SUBMISSION PATH (use this exact file for format checks):**
{sample_submission_path}

{instructions}
"""
    return user_content


@register_dataset("dspredict")
class DSPredictDataset(BaseDataset):
    """DSPredict dataset loader."""
    
    def __init__(self, virtual_data_root: Optional[str] = None, **kwargs):
        """
        Initialize DSPredict dataset.
        
        Args:
            virtual_data_root: Root path for virtual/docker paths (default: "/data")
            **kwargs: Additional configuration
        """
        super().__init__(virtual_data_root=virtual_data_root, **kwargs)
    
    def load(
        self, 
        limit: Optional[int] = None, 
        split: str = "easy", 
        start_index: int = 0,
        generate_dags: bool = False,
        prompt_variant: str = "structured",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load DSPredict dataset.
        
        Args:
            limit: Maximum number of samples to load
            split: Dataset split ("easy", "hard", "lite", or "sampled")
            start_index: Starting index for data selection
            generate_dags: If True, switch to the iterative DAG system prompt so the
                agent self-generates and self-revises its reasoning and implementation
                DAGs inline during solving. No upfront API calls are made.
            **kwargs: Additional loading parameters
            
        Returns:
            List of dataset samples
        """
        if split == "easy":
            dataset_path = str(get_task_path("dspredict") / "easy.json")
        elif split == "hard":
            dataset_path = str(get_task_path("dspredict") / "hard.json")
        elif split == "lite":
            dataset_path = str(get_task_path("dspredict") / "lite.json")
        elif split == "sampled":
            dataset_path = str(get_task_path("dspredict") / "dspredict_sampled.json")
        else:
            dataset_path = str(get_task_path("dspredict") / "hard.json")
        
        validate_file_exists(dataset_path, f"DSPredict {split} dataset")
        
        # Load JSON file (not JSONL)
        with open(dataset_path, 'r', encoding='utf-8') as f:
            items = json.load(f)

        # Apply start_index and limit
        items = apply_limit_and_start(
            items, limit, start_index, random_sample=False,
            random_seed=self.config.get('random_seed', 42)
        )

        samples = []
        downloader = None
        kaggle_data_dir = RAW_DATA_DIR / f"dspredict-{split}"

        selected_variant = get_variant(prompt_variant)
        if selected_variant is None:
            available = ", ".join(sorted(list_variants().keys()))
            raise ValueError(
                f"Unknown prompt variant: {prompt_variant}. Available variants: {available}"
            )

        for idx, item in enumerate(items):
            docker_path = item['docker_challenge_path']
            
            if docker_path.startswith('/data/'):
                challenge_dir = docker_path.replace('/data/', '')
            else:
                challenge_dir = docker_path.split('/')[-1]
            
            competition_path = kaggle_data_dir / challenge_dir
            if not competition_path.exists():
                print(f"Competition data not found: {challenge_dir}")
                print(f"Attempting to download from Kaggle...")
                
                if downloader is None:
                    downloader = KaggleChallengeDownloader(download_dir=str(kaggle_data_dir))
                
                competition_name = item['challenge_name']
                result = downloader.download_competition_data(competition_name)
                
                if result['status'] == 'failed':
                    print(f"Failed to download {competition_name}, skipping sample")
                    continue
                
                print(f"Successfully downloaded {competition_name}")

            from ..utils import construct_data_paths
            data_paths = construct_data_paths(
                relative_paths=[challenge_dir],
                dataset_name=f"dspredict-{split}",
                data_root=RAW_DATA_DIR,
                virtual_data_root=self.virtual_data_root
            )
            sample_submission_path = f"{data_paths['virtual'][0]}/sample_submission.csv"
            
            user_content = create_dspredict_prompt(
                challenge_name=item['challenge_name'],
                description=item['description'],
                data_paths=data_paths
                , sample_submission_path=sample_submission_path
            )
            
            extra_info = {
                "challenge_name": item['challenge_name'],
                "docker_challenge_path": item['docker_challenge_path'],
                "data_files": data_paths,
                "question": user_content,
                'index': start_index + idx,
                'source': 'dspredict',
                'metadata_id': item['challenge_name'],
                'query_id': item['challenge_name'],
                'id': item['challenge_name']
            }
            
            # Create standardized sample
            base_system_prompt = SYSTEM_PROMPT_DAG_DSPREDICT if generate_dags else SYSTEM_PROMPT_DSPREDICT
            active_system_prompt = (
                f"{base_system_prompt}\n\n"
                f"# Prompt Variant ({selected_variant.name})\n"
                f"{selected_variant.system_prompt_template}\n\n"
                f"# Response Format Requirements\n"
                f"{selected_variant.format_instructions}"
            )
            standard_sample = create_standard_task(
                prompt_content=user_content,
                ground_truth="",  # DSPredict competitions don't have ground truth
                extra_info=extra_info,
                system_prompt=active_system_prompt
            )
            
            samples.append(standard_sample)
        
        self._samples = samples
        return samples
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a single sample by index."""
        if self._samples is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        if index < 0 or index >= len(self._samples):
            raise IndexError(f"Sample index {index} out of range [0, {len(self._samples)})")
        
        return self._samples[index]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get DSPredict dataset metadata."""
        if self._metadata is None:
            self._metadata = {
                'name': 'DSPredict',
                'description': 'DSPredict competition challenges for machine learning',
                'format': 'json',
                'splits': ['easy', 'hard', 'lite', 'sampled'],
                'fields': ['challenge_name', 'description', 'docker_challenge_path'],
                'source': 'dspredict'
            }
        
        return self._metadata

    def create_sample(
        self,
        n: int,
        split: str = "hard",
        seed: int = 42,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Randomly sample ``n`` tasks from a DSPredict split and save them as
        ``dspredict_sampled.json`` for reproducible repeated experiments.

        The saved file uses the same raw JSON format as the source dataset so it
        can be loaded via ``load(split="sampled")``.

        Args:
            n:           Number of tasks to sample.
            split:       Source split to sample from ("easy", "hard", or "lite"). Default: "hard".
            seed:        Random seed for reproducible sampling (default: 42).
            output_path: Where to write the file. Defaults to
                         ``{task_dir}/dspredict_sampled.json``.

        Returns:
            Absolute path to the written file.
        """
        import random

        split_file_map = {"easy": "easy.json", "hard": "hard.json", "lite": "lite.json"}
        if split not in split_file_map:
            raise ValueError(
                f"Invalid split '{split}'. Choose from: {list(split_file_map.keys())}"
            )

        source_path = str(get_task_path("dspredict") / split_file_map[split])
        if not os.path.exists(source_path):
            raise FileNotFoundError(
                f"Split file not found: {source_path}"
            )

        with open(source_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        if n > len(all_data):
            raise ValueError(
                f"Requested sample size {n} exceeds dataset size {len(all_data)} for split '{split}'."
            )

        rng = random.Random(seed)
        sampled = rng.sample(all_data, n)

        if output_path is None:
            output_path = str(get_task_path("dspredict") / "dspredict_sampled.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sampled, f, indent=2, ensure_ascii=False)

        print(
            f"✅ Sampled {n} tasks (seed={seed}) from {split}.json "
            f"→ {output_path}"
        )
        return output_path

    # ------------------------------------------------------------------
    # DAG generation (post-hoc, API-based)
    # ------------------------------------------------------------------

    def generate_dag(
        self,
        sample: Dict[str, Any],
        dag_type: str = "reasoning",
        reasoning_dag=None,
        model: str = "gpt-4o",
        dag_validation_mode: Optional[str] = None,
    ):
        """
        Generate a reasoning or implementation DAG for a single DSPredict sample.

        Args:
            sample:        A DSPredict sample dict (from :meth:`load`).
            dag_type:      ``"reasoning"`` or ``"implementation"``.
            reasoning_dag: Required when ``dag_type="implementation"``; the
                           previously generated ReasoningDAG (or equivalent dict).
            model:         LiteLLM-compatible model name (default: ``"gpt-4o"``).

        Returns:
            A validated ReasoningDAG or ImplementationDAG.
        """
        from dsgym.datasets.dag import validate_implementation_dag, validate_reasoning_dag
        from dsgym.datasets.dag_generator import resolve_dag_validation_mode
        from dsgym.datasets.prompts.dag_prompts import (
            build_reasoning_prompt_dspredict,
            build_implementation_prompt_dspredict,
        )
        import litellm
        import json as _json
        import re

        extra = sample.get("extra_info", {})
        challenge_name = extra.get("challenge_name", "")
        # description is embedded in the full prompt; extract from extra_info if available
        description = extra.get("description", extra.get("question", ""))

        if not challenge_name:
            raise ValueError("Sample must contain a non-empty 'challenge_name' in extra_info")

        def _call(system_prompt, user_prompt):
            from dsgym.datasets.dag_generator import _supports_json_object_format, _extract_json
            from dsgym.datasets.dag import ReasoningDAG, ImplementationDAG
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if _supports_json_object_format(model):
                kwargs["response_format"] = {"type": "json_object"}
            response = litellm.completion(**kwargs)
            raw = response.choices[0].message.content
            if not raw:
                raise RuntimeError("Empty response from model")
            return _extract_json(raw)

        mode = (dag_validation_mode or resolve_dag_validation_mode("dspredict")).strip().lower()

        if dag_type == "reasoning":
            from dsgym.datasets.dag import ReasoningDAG
            sys_p, usr_p = build_reasoning_prompt_dspredict(challenge_name, description)
            raw_json = _call(sys_p, usr_p)
            reasoning = ReasoningDAG.model_validate_json(raw_json)
            validate_reasoning_dag(reasoning, mode=mode)
            return reasoning

        if dag_type == "implementation":
            if reasoning_dag is None:
                raise ValueError(
                    "reasoning_dag is required when dag_type='implementation'. "
                    "Generate a reasoning DAG first with dag_type='reasoning'."
                )
            from dsgym.datasets.dag import ImplementationDAG, ReasoningDAG
            import json as _json
            typed_reasoning_dag = (
                reasoning_dag if isinstance(reasoning_dag, ReasoningDAG) else ReasoningDAG.model_validate(reasoning_dag)
            )
            reasoning_dict = _json.loads(typed_reasoning_dag.model_dump_json())
            sys_p, usr_p = build_implementation_prompt_dspredict(challenge_name, description, reasoning_dict)
            raw_json = _call(sys_p, usr_p)
            implementation = ImplementationDAG.model_validate_json(raw_json)
            validate_implementation_dag(implementation, reasoning_dag=typed_reasoning_dag, mode=mode)
            return implementation

        raise ValueError(f"Unknown dag_type '{dag_type}'. Choose 'reasoning' or 'implementation'.")

    def generate_dags(
        self,
        sample: Dict[str, Any],
        model: str = "gpt-4o",
        dag_validation_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method: generate both reasoning and implementation DAGs
        for a single DSPredict sample in one call.

        Args:
            sample: A DSPredict sample dict (from :meth:`load`).
            model:  LiteLLM-compatible model name (default: ``"gpt-4o"``).

        Returns:
            Dict with keys ``"reasoning"`` (ReasoningDAG) and
            ``"implementation"`` (ImplementationDAG).
        """
        reasoning = self.generate_dag(
            sample,
            dag_type="reasoning",
            model=model,
            dag_validation_mode=dag_validation_mode,
        )
        implementation = self.generate_dag(
            sample,
            dag_type="implementation",
            reasoning_dag=reasoning,
            model=model,
            dag_validation_mode=dag_validation_mode,
        )
        return {"reasoning": reasoning, "implementation": implementation}

    def get_metrics(self) -> List[str]:
        """Get metrics for DSPredict dataset."""
        return ["dspredict_submission"]


    def print_dspredict_results_overview(self, sample_results):
        """Aggregate Kaggle metrics across sample results and print a summary."""
        def _get_metric_dict(sample, metric_name):
            """Extract metric details as a dictionary."""
            metrics = getattr(sample, "metrics", None)
            if metrics is None and isinstance(sample, dict):
                metrics = sample.get("metrics", {})
            metrics = metrics or {}
            metric = metrics.get(metric_name)
            if metric is None:
                return None
            if hasattr(metric, "to_dict"):
                return metric.to_dict()
            return metric
        def _safe_float(value):
            """Convert value to float when possible."""
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.strip()
                if not value or value.lower() in {"nan", "n/a"}:
                    return None
                try:
                    return float(value)
                except ValueError:
                    return None
            return None
        def _format_pct(numerator, denominator):
            if denominator == 0:
                return "0.0%"
            return f"{(numerator / denominator) * 100:.1f}%"

        total_competitions = len(sample_results)
        total_valid_submissions = 0
        total_above_median = 0
        medal_counts: Counter[str] = Counter()
        leaderboard_data = {
            "public": {
                "competitions": 0,
                "valid_submissions": 0,
                "scores": [],
                "percentiles": [],
                "above_median_total": 0,
                "above_median_count": 0,
                "medals": Counter(),
            },
            "private": {
                "competitions": 0,
                "valid_submissions": 0,
                "scores": [],
                "percentiles": [],
                "above_median_total": 0,
                "above_median_count": 0,
                "medals": Counter(),
            },
        }
        status_counts: Counter[str] = Counter()

        for sample in sample_results:
            metric_info = _get_metric_dict(sample, "kaggle_submission")
            if not metric_info:
                continue

            details = metric_info.get("details") or {}
            status = details.get("status")
            if status:
                status_counts[status] += 1

            leaderboard_stats = details.get("leaderboard_stats") or {}
            for board in ("public", "private"):
                scores_snapshot = leaderboard_stats.get(f"{board}_scores")
                if not scores_snapshot:
                    continue

                board_data = leaderboard_data[board]
                board_data["competitions"] += 1

                score_value = _safe_float(details.get(f"{board}_score"))
                if score_value is not None:
                    board_data["scores"].append(score_value)
                    board_data["valid_submissions"] += 1

                percentile_value = _safe_float(details.get(f"{board}_percentile"))
                if percentile_value is not None:
                    board_data["percentiles"].append(percentile_value)

                above_median_value = details.get(f"{board}_above_median")
                if isinstance(above_median_value, bool):
                    board_data["above_median_total"] += 1
                    if above_median_value:
                        board_data["above_median_count"] += 1

                medal_value = details.get(f"{board}_medal")
                if isinstance(medal_value, str):
                    medal = medal_value.strip().lower()
                    if medal in {"gold", "silver", "bronze"}:
                        medal_counts[medal] += 1
                        board_data["medals"][medal] += 1

        print("\n📝 Kaggle Competition Overview")
        print("-" * 40)
        print(f"Total competitions evaluated : {total_competitions}")
        if status_counts:
            print("Submission status breakdown  :")
            for status, count in sorted(status_counts.items()):
                print(f"  - {status}: {count}")

        for board, label in (("public", "Public"), ("private", "Private")):
            board_data = leaderboard_data[board]
            competitions = board_data["competitions"]
            print(f"\n=== {label} Leaderboard ===")
            if competitions == 0:
                print("  No leaderboard data available.")
                continue

            print(
                f"📊 Coverage               : {competitions} / {total_competitions}  "
                f"({_format_pct(competitions, total_competitions)})"
            )
            print(
                f"✅ Valid submissions       : {board_data['valid_submissions']} / {competitions}  "
                f"({_format_pct(board_data['valid_submissions'], competitions)})"
            )

            if board_data["percentiles"]:
                percentiles = board_data["percentiles"]
                print(f"🎯 Avg percentile          : {mean(percentiles):.1f}")

            if board_data["above_median_total"]:
                above = board_data["above_median_count"]
                total = board_data["above_median_total"]
                print(
                    f"⬆️ Above median            : {above} / {competitions}  "
                    f"({_format_pct(above, competitions)})"
                )

            total_board_medals = sum(board_data["medals"].values())
            if competitions:
                print(
                    f"🏅 Medals                 : {total_board_medals} / {competitions}  "
                    f"({_format_pct(total_board_medals, competitions)})"
                )
                print(
                    f"  🥇 Gold                 : {board_data['medals'].get('gold', 0)}"
                )
                print(
                    f"  🥈 Silver               : {board_data['medals'].get('silver', 0)}"
                )
                print(
                    f"  🥉 Bronze               : {board_data['medals'].get('bronze', 0)}"
                )