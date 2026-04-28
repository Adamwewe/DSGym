# DSGym: A Holistic Framework for Advancing Data Science Agents

DSGym is a comprehensive framework for evaluating and training Large Language Model (LLM) agents on real-world data science tasks. Building upon the research presented in our [paper](https://arxiv.org/pdf/2601.16344), DSGym provides a unified evaluation platform with isolated execution environments, multiple datasets spanning diverse data science domains, and robust evaluation metrics.

## Overview

DSGym addresses the critical need for systematic evaluation of AI agents' data science capabilities. Unlike existing benchmarks that focus on narrow tasks, DSGym provides a holistic assessment across the full data science workflow - from data exploration and preprocessing to modeling and interpretation.

## Features

- **Unified Benchmark Framework**: Evaluate LLMs across multiple data science datasets
- **Isolated Execution**: Docker-based container system for safe code execution
- **Multiple Backend Support**: LiteLLM (API-based), vLLM, and SGLang inference backends
- **Comprehensive Metrics**: Various evaluation metrics including exact match, semantic similarity, and domain-specific scoring
- **Multi-Dataset Support**: DAEval, DiscoveryBench, DABStep, QRData, DSBio, and DSPredict integration
- **Trajectory Generation**: Generate multiple trajectories per sample for synthetic data creation and pass@k evaluation

## Project Structure

```
DSGym/
├── dsgym/                    # Core framework
│   ├── agents/              # LLM agents and backends
│   ├── datasets/            # Dataset loaders and prompts
│   │   ├── dag.py           # ReasoningDAG & ImplementationDAG Pydantic schemas
│   │   ├── dag_generator.py # OpenAI structured-output DAG generation
│   │   ├── loaders/         # Per-dataset loaders (DSBio, QRData, …)
│   │   └── prompts/         # System prompts + DAG prompt builders
│   ├── eval/                # Evaluation system and metrics
│   ├── synth/               # Synthetic data generation and trajectory tools
│   ├── train/               # Model training and fine-tuning (coming soon)
│   └── cli/                 # Command-line interface
├── executors/               # Docker-based execution system
├── examples/                # Ready-to-use evaluation scripts
└── data/                    # Dataset storage
```

## Quick Start

### 1. Installation

```bash
# Install main dependencies (includes litellm by default)
uv sync

# Install with optional extras
uv sync --extra dev        # Development tools
uv sync --extra vllm       # vLLM inference backend
uv sync --extra sglang     # SGLang inference backend
uv sync --extra metrics    # Additional evaluation metrics
uv sync --extra synth      # Synthetic data generation
```

### 2. Setup Docker Execution Environment

```bash
cd executors

# Standard DS env
python generate_compose.py -n 64 --types "executor-prebuilt:64" -m ../data/data

sudo docker build -t executor-prebuilt ./container_images/instance
sudo docker build -t manager-prebuilt ./manager
sudo docker compose -f docker-compose.yml up -d --build

sudo docker compose -f docker-compose.yml down

# Bio-specific DS env
python generate_compose.py -n 64 --types "executor-bio:64" -m ../data/data --output docker-compose-bio.yml -c container_config_bio.json

sudo docker build -t executor-bio ./container_images/dsbio_image
sudo docker build -t manager-prebuilt ./manager
sudo docker compose -f docker-compose-bio.yml up -d --build

sudo docker compose -f docker-compose-bio.yml down

# DSPredict-specific env
python generate_compose.py -n 8 --types "executor-kaggle:8" -m ../data/data --output docker-dspredict-hard.yml -c container_config_dspredict_hard.json \
-g 0,1,2,3,4,5,6,7 -s ../submissions -e "EXECUTION_TIMEOUT=3600,MEM_LIMIT=24G,CPUS=8,MEM_RESERVATION=100G" \

sudo docker build -t executor-kaggle ./container_images/kaggle_image
sudo docker build -t manager-prebuilt ./manager
sudo docker compose -f docker-dspredict-hard.yml up -d --build

sudo docker compose -f docker-dspredict-hard.yml down
```

### 3. Run Evaluations

#### Using CLI

```bash
# Set API key
export OPENAI_API_KEY=<your-api-key>
# or
export TOGETHER_API_KEY=<your-api-key>

# Evaluate on DSBio (without DAGs, baseline)
dsgym eval --model gpt-4o --dataset dsbio --limit 10

# Evaluate on DSBio with DAG-guided prompts
dsgym eval --model gpt-4o --dataset dsbio --limit 10 --dags true

# Evaluate on DSBio with DAGs using a specific DAG model
dsgym eval --model gpt-4o --dataset dsbio --limit 10 --dags true --dag-model gpt-4o-mini

# Evaluate on DAEval
dsgym eval --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput --dataset daeval --backend litellm --limit 10


# Get help
dsgym --help
dsgym eval --help
```
To run DSPredict, ensure your Kaggle API credentials are correctly configured. Generate a kaggle api key and:

```bash
export KAGGLE_API_TOKEN=$YOUR_KEY
```

See the [Kaggle API documentation](https://www.kaggle.com/docs/api) for more details. Note that for online assessment, you will have to manually enroll in the competitions (on your account) in order to submit. This will take a bit of time at the beginning. Alternatively, you can use the offline leaderboard for evaluation, which compares submissions against cached leaderboard data. However, this will still require manual enrollment into the competition. (you will see a 403 error if you don't enroll).


#### Using Example Scripts

Alternatively, use the example scripts for different datasets:

#### DSBio
```bash
export TOGETHER_API_KEY=<your-api-key>

cd examples

# Baseline (no DAGs)
python evaluate.py \
    --dataset dsbio \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --backend litellm \
    --limit 10 \
    --output-dir ./results/dsbio_baseline

# With DAG-guided prompts (requires OPENAI_API_KEY for DAG generation)
python evaluate.py \
    --dataset dsbio \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --backend litellm \
    --limit 10 \
    --dags true \
    --dag-model gpt-4o \
    --output-dir ./results/dsbio_dags
```

> **Note:** DSBio requires the bio Docker environment (`executor-bio`). Raw `.h5ad`
> data files must be placed in `data/data/bio/` — they are automatically mounted
> into containers at `/data/bio/` at runtime.

#### Downloading DSBio data files

DSBio tasks reference 25 data files (`.h5ad`, compressed matrices, metadata) sourced
from CellxGENE, Zenodo, and EBI BioStudies. A download script is provided at the
repo root to fetch all of them:

```bash
# Run from the DSGym repo root
bash download_bio_data.sh
```

The script will:
- Create `data/data/bio/` if it does not exist
- Download 24 of the 25 required files (~6 GB total) from their public sources
- **Skip files that are already present** (safe to re-run / resume after interruption)
- Print the final file listing with sizes when done

| Source | Files |
|--------|-------|
| [CellxGENE](https://cellxgene.cziscience.com) | `Colon.h5ad`, `Ileum.h5ad`, `Rectum.h5ad`, `Single_cell_transcriptomic_profiling_identifies_molecular_phenotypes_of_newborn_human_lung_cells.h5ad`, `scRNA-seq_of_adult_gamma_delta_T_cells.h5ad`, `scRNA-seq_of_vascular_tissue.h5ad`, `slide-seqV2_analysis_of_aorta.h5ad`, `slide-seqV2_analysis_of_right_coronary_artery.h5ad`, `snRNA-seq_analyses_of_breast_tissues_of_healthy_women_of_diverse_genetic_ancestry.h5ad` |
| [Zenodo 259661](https://zenodo.org/records/259661) | `RNA_cqn_matrix.txt.gz`, `RNA_gene_metadata.txt.gz`, `RNA_sample_metadata.txt.gz`, `ATAC_cqn_matrix.txt.gz`, `ATAC_peak_metadata.txt.gz`, `ATAC_sample_metadata.txt.gz` |
| [EBI BioStudies S-BSST16](https://www.ebi.ac.uk/biostudies/studies/S-BSST16) | `all_basic_counts.open.txt.gz`, `atac_consensus_peaks.txt.gz`, `atac_qtl.rasqual.1k.txt.gz`, `eqtl.fastqtl.500k.nominals.txt.gz`, `eqtl.fastqtl.500k.permutations.txt.gz`, `eqtl.rasqual.500k.txt.gz`, `metadata.open.txt`, `sqtl.fastqtl.nominals.txt.gz`, `sqtl.fastqtl.permutations.txt.gz` |

> **Missing file:** `raw_adult_vd_cells_post_qc.h5ad` has no public download URL.
> It is a derived artifact from `scRNA-seq_of_adult_gamma_delta_T_cells.h5ad`
> (which is downloaded). Generate it by running `bio_code/pediatric_gdt_pipeline.ipynb`,
> or contact the study authors. Tasks that depend on this file will be skipped
> until it is present.

#### QRData
```bash
cd examples
python evaluate.py \
    --dataset qrdata \
    --model gpt-4o \
    --backend litellm \
    --limit 10 \
    --output-dir ./results/qrdata_test
```

### 4. CLI Commands

DSGym provides a unified CLI interface:

#### Available Commands

- **`dsgym eval`**: Evaluate models on datasets
- **`dsgym generate`**: Generate trajectories for synthetic data (coming soon)
- **`dsgym train`**: Train models (coming soon)

#### CLI Examples

```bash
# Evaluate with custom output directory
dsgym eval \
    --model gpt-4o \
    --dataset qrdata \
    --limit 10 \
    --output-dir ./my_results \
    --temperature 0.1


# Use different backend
dsgym eval \
    --model gpt-4o \
    --dataset daeval \
    --backend vllm \
    --max-workers 1
```

#### Output Structure

```
{output_dir}/
├── {run_name}_results.json       # Full per-sample results (with trajectories)
├── {run_name}_summary.json       # Lightweight per-sample summary
├── {run_name}_summary.csv        # CSV for easy analysis
├── {run_name}_metrics.json       # Aggregated metrics
├── {run_name}_config.json        # Evaluation configuration
└── dags/                         # DAG artifacts (DSBio --dags true only)
    ├── 0_dags.json               # reasoning + implementation DAGs for sample 0
    ├── 1_dags.json               # reasoning + implementation DAGs for sample 1
    └── ...
```

## Supported Datasets

- **DAEval**: Data analysis evaluation with list matching metrics
- **DiscoveryBench**: Scientific discovery tasks with LLM and HMS scoring  
- **DABStep**: Step-by-step data analysis tasks
- **QRData**: Question-reasoning evaluation
- **DSBio**: Bioinformatics and computational biology tasks (90 tasks from academic literature)
- **DSPredict**: Competition integration and submission validation

All datasets support trajectory generation for synthetic data creation and pass@k evaluation.

## DAG Generation for DSBio

DSBio tasks include structured DAG generation support powered by OpenAI structured
outputs. Given any loaded DSBio sample, the framework can produce:

- **ReasoningDAG** — abstract analytical steps a biologist should reason through
  *before* implementation (QC strategy, normalisation choice, clustering approach, …)
- **ImplementationDAG** — concrete executable steps derived from the reasoning DAG,
  each node annotated with `test_type` (`deterministic` / `heuristic`) and a valid
  Python snake_case `function_id`

Both graphs are validated as proper directed acyclic graphs (topological order
enforced, no cycles, no duplicate node ids).

### How DAGs become agent tools

When DAGs are generated they are injected directly into the sample's prompt in two
sections that follow the same tool interface as every other dataset in the repo
(Python code execution via `<python>` tags):

1. **`ANALYSIS PLAN`** — the reasoning DAG rendered as a numbered checklist the
   agent works through before writing any code.
2. **`IMPLEMENTATION TOOLS`** — the implementation DAG rendered as Python function
   stubs with docstrings and an explicit **execution order**, e.g.:

```
IMPLEMENTATION TOOLS:
The following Python functions represent each implementation step. Implement and
call them in sequence using <python> blocks, following the order shown below.

def load_anndata_and_inspect_shape():
    """Load AnnData object and inspect shape  [test_type: deterministic]"""
    pass

def compute_per_cell_qc_metrics():
    """Compute per-cell QC metrics  [test_type: deterministic]"""
    pass

Execution order: load_anndata_and_inspect_shape → compute_per_cell_qc_metrics → …
```

The agent implements each stub and calls them in order, treating the DAG nodes as
the callable tools it must exercise — identical to how other datasets expose their
analysis steps.

### Inline DAG generation during `load()`

The primary way to use DAGs is to generate them as part of loading, either via
the CLI or the Python API.

**CLI flags** (DSBio only):

| Flag | Default | Description |
|------|---------|-------------|
| `--dags true\|false` | `false` | Enable DAG-guided prompts per sample |
| `--dag-model MODEL` | `gpt-4o` | OpenAI model used for DAG generation |

When `--dags true` is set, DAG artifacts are automatically saved as individual
JSON files under `{output_dir}/dags/` (e.g. `results/dags/0_dags.json`,
`results/dags/1_dags.json`, …). Each file contains the sample index, the
question text, and the serialised `reasoning_dag` and `implementation_dag`.

```bash
# Compare with and without DAGs
dsgym eval --model gpt-4o --dataset dsbio --limit 10 --dags false
dsgym eval --model gpt-4o --dataset dsbio --limit 10 --dags true --dag-model gpt-4o-mini
# → DAG artifacts written to {output_dir}/dags/
```

**Python API:**

```python
from dsgym.datasets import DatasetRegistry

dataset = DatasetRegistry.load("dsbio")

# generate_dags=True calls the OpenAI API once per sample and injects the
# resulting plan + function stubs into every sample's prompt automatically.
# dag_output_dir saves each sample's DAGs as {index}_dags.json files.
samples = dataset.load(
    limit=10,
    generate_dags=True,
    model="gpt-4o",
    dag_output_dir="./results/dags",   # omit to skip saving artifacts
)

# The raw DAG dicts are also stored in extra_info for downstream use.
sample = samples[0]
reasoning = sample["extra_info"]["reasoning_dag"]   # plain dict
impl      = sample["extra_info"]["implementation_dag"]
```

DAG generation failures (e.g. API errors) are caught per-sample and logged as
warnings; the sample is still returned without a DAG section so evaluation can
continue uninterrupted.

### Post-hoc DAG generation

You can also generate DAGs after loading without re-running `load()`:

```python
# Both DAGs in one call
dags = dataset.generate_dags(samples[0], model="gpt-4o")

# Or separately, to inspect the reasoning DAG first
reasoning = dataset.generate_dag(samples[0], dag_type="reasoning", model="gpt-4o")
impl      = dataset.generate_dag(
    samples[0],
    dag_type="implementation",
    reasoning_dag=reasoning,
    model="gpt-4o",
)
```

Or call the generator functions directly without a dataset instance:

```python
from dsgym.datasets.dag_generator import generate_reasoning_dag, generate_implementation_dag

reasoning = generate_reasoning_dag(question, context, model="gpt-4o")
impl      = generate_implementation_dag(question, context, reasoning, model="gpt-4o")
```

### Task types

`DSBioTaskType` covers the single-cell transcriptomics workflow:

| value | meaning |
|-------|---------|
| `data_exploration` | Initial data loading & shape inspection |
| `quality_control` | Per-cell / per-gene QC metrics |
| `qc_validation` | Threshold selection & cell filtering |
| `preprocessing` | Normalisation, log-transform, HVG selection |
| `dimensionality_reduction` | PCA, UMAP, t-SNE |
| `clustering` | Leiden / Louvain / k-means clustering |
| `differential_expression` | DEG analysis |
| `pathway_analysis` | GSEA, pathway enrichment |
| `cell_type_annotation` | Marker-based cell type labelling |
| `statistical_analysis` | Statistical tests & correlation |
| `result_validation` | Sanity checks on final outputs |

### Relevant files

| File | Purpose |
|------|---------|
| `dsgym/datasets/dag.py` | `ReasoningDAG`, `ImplementationDAG`, `DSBioTaskType`, shared cycle-detection |
| `dsgym/datasets/dag_generator.py` | `generate_reasoning_dag`, `generate_implementation_dag`, `generate_dags_for_sample` |
| `dsgym/datasets/prompts/dag_prompts.py` | System prompts, reference examples, `build_reasoning_prompt` / `build_implementation_prompt` |
| `dsgym/datasets/loaders/dsbio.py` | `DSBioDataset.load(generate_dags=…, dag_output_dir=…)`, `generate_dag()` / `generate_dags()`, `_format_dag_section()` |

> **Requires** `OPENAI_API_KEY` and the `openai` package (already listed in
> `pyproject.toml` dependencies).

## Key Innovations

### Holistic Evaluation Framework
- **Real-World Tasks**: Derived from actual data science competitions and academic research
- **End-to-End Workflows**: Evaluates entire data science pipelines, not isolated components

### Robust Execution Environment
- **Isolated Containers**: Docker-based execution prevents interference and ensures reproducibility
- **Scalable Architecture**: Supports parallel evaluation across multiple containers
- **Safety First**: Sandboxed execution environment for secure code evaluation

### Dataset Coverage
- **Diverse Domains**: From general/applied ds to scientific ds tasks, covering various data science applications
- **Varied Complexity**: Tasks ranging from basic data analysis to complex scientific discovery

## Resources

- 📄 **Paper**: [DSGym: A Holistic Framework for Evaluating and Training Data Science Agents](https://arxiv.org/pdf/2601.16344)
- 🤗 **Hugging Face**: [DSGym Repository](https://huggingface.co/DSGym)
- 💻 **GitHub**: [Source Code](https://github.com/fannie1208/DSGym)

## Development

### Linting and Formatting
```bash
uv run ruff check .              # Lint check
uv run ruff check --fix .        # Auto-fix linting issues
uv run black .                   # Format code
```

### Testing
```bash
# Executor system tests
cd executors/tests
pytest

# Main system tests
uv run pytest
```

## Configuration

### Environment Variables
- `TOGETHER_API_KEY`: For Together AI models
- `OPENAI_API_KEY`: For OpenAI models
- `ANTHROPIC_API_KEY`: For Anthropic models

### Docker Configuration
- Container specifications in `executors/container_config.json`
- Docker compose generation via `executors/generate_compose.py`

## Adding New Datasets

1. Create dataset loader in `dsgym/datasets/loaders/your_dataset.py`
2. Implement `get_metrics()` and `get_metric_configs()` methods
3. Register dataset using `@register_dataset("your_dataset")` decorator
4. Create example evaluation script in `examples/evaluate_your_dataset.py`
5. Dataset automatically supports trajectory generation

## Adding New Metrics

1. Create metric class in `dsgym/eval/metrics/`
2. Inherit from `BaseMetric` and implement required methods
3. Register metric in `dsgym/eval/metric_registry.py`
4. Update dataset loaders to use the new metric
5. Metrics automatically support pass@k and average calculations

## Trajectory Generation

Generate multiple trajectories per sample for synthetic data creation and pass@k evaluation. Supports configurable temperature, parallel processing, and automatic metric computation including both pass@k (maximum) and average scores across trajectories.

## Requirements

- Python 3.12+
- Docker and Docker Compose
- GPU support recommended for local inference backends

## Citation

If you use DSGym in your research, please cite our paper:

```bibtex
@misc{nie2026dsgym,
      title={DSGym: A Holistic Framework for Evaluating and Training Data Science Agents}, 
      author={Fan Nie and Junlin Wang and Harper Hua and Federico Bianchi and Yongchan Kwon and Zhenting Qi and Owen Queen and Shang Zhu and James Zou},
      year={2026},
      eprint={2601.16344},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.16344}, 
}
```


## Contributing

We welcome contributions to DSGym! Please see our contributing guidelines for more information on how to:
- Add new datasets
- Add new tasks
- Add new agent scaffolds
- Implement new evaluation metrics
- Improve the framework
- Report bugs and suggest features

Detailed guidelines will be posted soon!
