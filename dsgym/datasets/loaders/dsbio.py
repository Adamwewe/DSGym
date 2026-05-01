"""
Bio dataset loader for single-cell transcriptomics tasks.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union

from ..base import BaseDataset
from ..registry import register_dataset
from ..utils import apply_limit_and_start, validate_file_exists, create_standard_task, construct_data_paths
from ..prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_DAG
from dsgym.datasets.config import get_task_path, RAW_DATA_DIR


DSBIO_INSTRUCTIONS = """
1. Load and explore the provided biological datasets using appropriate Python libraries, especially their metadata information.
2. You must be very careful with printing large objects (e.g., dataframes, columns, etc). The dsbio dataset is super large and you must estimate the output size before printing. Prefer `nunique()`, `head()`, `value_counts().head()` to print the size of the object first. If headers/values contain \\t (suggesting TSV got loaded as one column), fix parsing with sep or split the column) before any further exploration/printing.
3. Understand the biological context, experimental design, and research objectives from the context information.
4. Pay attention to domain-specific terminology and biological concepts mentioned in the question.
5. Use appropriate statistical methods and techniques for biological data analysis.
6. Provide your final answer based on the analysis results.
7. Pay attention to the answer guidelines and only give answer in correct required format. 
8. In the environment, these python packages are already installed: numpy, pandas, scikit-learn, networkx, scipy, sklearn, scanpy, anndata, h5py, statsmodels, gseapy, squidpy, scrublet, harmonypy, geneconverter (converting ensembl id to gene name), numcodecs, zarr, etc. You can use `!pip install <package_name>` to install any other required python package.
"""


def create_dsbio_prompt(
    question: str,
    context: str,
    data_paths: Dict[str, List[str]],
    metadata: Dict[str, Any],
) -> str:
    virtual_paths = data_paths['virtual']
    dataset_locations = '\n'.join(virtual_paths) if virtual_paths else 'No dataset provided'

    domain = metadata.get('domain', 'computational biology')
    guidelines = metadata.get('guidelines', '')

    metadata_lines = [f"DOMAIN: {domain}"]
    if guidelines:
        metadata_lines.append(f"ANSWER GUIDELINES: {guidelines}")
    metadata_section = '\n'.join(metadata_lines)

    prompt = f"""QUESTION: {question}

CONTEXT:
{context}

{metadata_section}

DATASET LOCATIONS (use full paths):
{dataset_locations}

INSTRUCTIONS:
{DSBIO_INSTRUCTIONS}

Please analyze the data and provide your answer based on the specific question asked."""

    return prompt


@register_dataset("dsbio")
class DSBioDataset(BaseDataset):
    """DSBio dataset loader for single-cell transcriptomics tasks."""
    
    def __init__(self, data_dir: Optional[str] = None, virtual_data_root: Optional[str] = None, **kwargs):
        """
        Initialize DSBio dataset.
        
        Args:
            data_dir: Directory containing dsbio dataset files
            virtual_data_root: Root path for virtual/docker paths (default: "/data")
            **kwargs: Additional configuration
        """
        if data_dir is None:
            data_dir = str(get_task_path("dsbio"))
        super().__init__(data_dir=data_dir, virtual_data_root=virtual_data_root, **kwargs)
    
    def load(
        self,
        split: str = "hard",
        limit: Optional[int] = None,
        start_index: int = 0,
        random_sample: bool = False,
        generate_dags: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load DSBio dataset.

        Args:
            split: Dataset split ("hard" or "easy")
            limit: Maximum number of samples to load
            start_index: Starting index for data selection
            random_sample: If True, randomly sample from dataset
            generate_dags: If True, switch to the iterative DAG system prompt so the
                agent self-generates and self-revises its reasoning and implementation
                DAGs inline during solving. No upfront API calls are made.
            **kwargs: Additional loading parameters

        Returns:
            List of dataset samples
        """
        # Determine file path based on split
        if split == "hard":
            for candidate in ("dsbio-hard.json", "dsbio.json"):
                candidate_path = os.path.join(self.data_dir, candidate)
                if os.path.exists(candidate_path):
                    file_path = candidate_path
                    break
            else:
                file_path = os.path.join(self.data_dir, "dsbio-hard.json")  # will trigger FileNotFoundError below
        elif split == "sampled":
            file_path = os.path.join(self.data_dir, "dsbio_sampled.json")
        elif split == "easy":
            raise NotImplementedError("Easy split is not implemented yet")
        else:
            raise ValueError(f"Invalid split: {split}. Available splits: ['hard', 'sampled', 'easy']")

        # Validate file exists
        validate_file_exists(file_path, f"DSBio {split} dataset")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        # Apply start_index and limit
        raw_data = apply_limit_and_start(
            original_data, limit, start_index, random_sample,
            self.config.get('random_seed', 42)
        )

        samples = []

        # Convert to standard format
        for idx, item in enumerate(raw_data):
            original_index = start_index + idx
            
            context = item['context']
            question = item['question']
            raw_paths = item['data']
            metadata = item.get('metadata', {})
            guidelines = item.get('answer_guideline', '')
            metadata['guidelines'] = guidelines

            relative_paths = []
            for path in raw_paths:
                # Paths in the JSON are Docker-style /data/<subdir>/file paths.
                # Strip the leading /data/ root so construct_data_paths can
                # reconstruct both the local absolute path and the Docker virtual
                # path correctly (e.g. 'bio/file.h5ad').
                if path.startswith('/data/'):
                    relative_paths.append(path[len('/data/'):])
                else:
                    relative_paths.append(path)

            data_paths = construct_data_paths(
                relative_paths=relative_paths,
                dataset_name='',       # subdir is already encoded in relative_path
                data_root=RAW_DATA_DIR,
                virtual_data_root=self.virtual_data_root
            )

            # Create prompt using dsbio format
            prompt_content = create_dsbio_prompt(
                question,
                context,
                data_paths,
                metadata,
            )

            ground_truth = str(item['answer'])
            
            extra_info = {
                'question': question,
                'answer': ground_truth,
                'context': context,
                'data_files': data_paths,
                'metadata': metadata,
                'index': original_index,
                'source': 'dsbio',
                'split': split,
                'id': str(original_index),
            }
            
            # Select system prompt: iterative DAG prompt when generate_dags is set.
            active_system_prompt = SYSTEM_PROMPT_DAG if generate_dags else SYSTEM_PROMPT

            # Create standardized sample
            standard_sample = create_standard_task(
                prompt_content=prompt_content,
                ground_truth=ground_truth,
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
        """Get DSBio dataset metadata."""
        if self._metadata is None:
            self._metadata = {
                'name': 'DSBio',
                'description': 'Dataset for single-cell transcriptomics and computational biology tasks',
                'data_dir': self.data_dir,
                'format': 'json',
                'splits': ['hard', 'sampled', 'easy'],
                'fields': ['context', 'question', 'answer', 'data', 'metadata'],
                'source': 'bio'
            }
        return self._metadata

    def create_sample(
        self,
        n: int,
        seed: int = 42,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Randomly sample ``n`` tasks from the hard split and save them as
        ``dsbio_sampled.json`` for reproducible repeated experiments.

        The saved file uses the same raw JSON format as the source dataset so it
        can be loaded via ``load(split="sampled")``.

        Args:
            n:           Number of tasks to sample.
            seed:        Random seed for reproducible sampling (default: 42).
            output_path: Where to write the file. Defaults to
                         ``{data_dir}/dsbio_sampled.json``.

        Returns:
            Absolute path to the written file.
        """
        import random

        # Locate the hard split source file
        for candidate in ("dsbio-hard.json", "dsbio.json"):
            candidate_path = os.path.join(self.data_dir, candidate)
            if os.path.exists(candidate_path):
                source_path = candidate_path
                break
        else:
            raise FileNotFoundError(
                f"Hard split not found in {self.data_dir}. "
                "Expected dsbio-hard.json or dsbio.json."
            )

        with open(source_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        if n > len(all_data):
            raise ValueError(
                f"Requested sample size {n} exceeds dataset size {len(all_data)}."
            )

        rng = random.Random(seed)
        sampled = rng.sample(all_data, n)

        if output_path is None:
            output_path = os.path.join(self.data_dir, "dsbio_sampled.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sampled, f, indent=2, ensure_ascii=False)

        print(
            f"✅ Sampled {n} tasks (seed={seed}) from {os.path.basename(source_path)} "
            f"→ {output_path}"
        )
        return output_path
    
    def get_metrics(self) -> List[str]:
        """Get metrics for DSBio dataset."""
        return ["exact_match"]
    
    def get_metric_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get metric configurations for DSBio dataset."""
        return {
            "exact_match": {
                "normalize": True,
                "case_sensitive": False,
                "numeric_tolerance": 0.0
            }
        }

    # ------------------------------------------------------------------
    # DAG generation
    # ------------------------------------------------------------------

    def generate_dag(
        self,
        sample: Dict[str, Any],
        dag_type: str = "reasoning",
        reasoning_dag: Optional[Union["ReasoningDAG", Dict[str, Any]]] = None,
        model: str = "gpt-4o",
    ) -> Union["ReasoningDAG", "ImplementationDAG"]:
        """
        Generate a reasoning or implementation DAG for a single DSBio sample.

        Uses litellm to produce a validated DAG grounded in
        the biological question and context of the sample.

        Args:
            sample:        A DSBio sample dict (from :meth:`load`).
            dag_type:      ``"reasoning"`` or ``"implementation"``.
            reasoning_dag: Required when ``dag_type="implementation"``; the
                           previously generated :class:`ReasoningDAG` (or
                           equivalent dict) to use as a mental model.
            model:         LiteLLM-compatible model name (default: ``"gpt-4o"``).

        Returns:
            A validated :class:`ReasoningDAG` or :class:`ImplementationDAG`.

        Example::

            dataset = DSBioDataset()
            samples = dataset.load(limit=1)
            reasoning = dataset.generate_dag(samples[0], dag_type="reasoning")
            impl = dataset.generate_dag(
                samples[0], dag_type="implementation", reasoning_dag=reasoning
            )
        """
        from dsgym.datasets.dag_generator import (
            generate_implementation_dag,
            generate_reasoning_dag,
        )

        extra = sample.get("extra_info", {})
        question = extra.get("question", "")
        context = extra.get("context", "")

        if not question:
            raise ValueError("Sample must contain a non-empty 'question' in extra_info")

        if dag_type == "reasoning":
            return generate_reasoning_dag(question, context, model=model)

        if dag_type == "implementation":
            if reasoning_dag is None:
                raise ValueError(
                    "reasoning_dag is required when dag_type='implementation'. "
                    "Generate a reasoning DAG first with dag_type='reasoning'."
                )
            return generate_implementation_dag(question, context, reasoning_dag, model=model)

        raise ValueError(f"Unknown dag_type '{dag_type}'. Choose 'reasoning' or 'implementation'.")

    def generate_dags(
        self,
        sample: Dict[str, Any],
        model: str = "gpt-4o",
    ) -> Dict[str, Any]:
        """
        Convenience method: generate both reasoning and implementation DAGs
        for a single DSBio sample in one call.

        Args:
            sample: A DSBio sample dict (from :meth:`load`).
            model:  LiteLLM-compatible model name (default: ``"gpt-4o"``).

        Returns:
            Dict with keys ``"reasoning"`` (:class:`ReasoningDAG`) and
            ``"implementation"`` (:class:`ImplementationDAG`).

        Example::

            dataset = DSBioDataset()
            samples = dataset.load(limit=1)
            dags = dataset.generate_dags(samples[0])
            print(dags["reasoning"].model_dump_json(indent=2))
            print(dags["implementation"].model_dump_json(indent=2))
        """
        from dsgym.datasets.dag_generator import generate_dags_for_sample

        return generate_dags_for_sample(sample, model=model)
