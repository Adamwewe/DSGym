# DSGym: A Holistic Framework for Advancing Data Science Agents

DSGym is a comprehensive framework for evaluating and training Large Language Model (LLM) agents on real-world data science tasks. Building upon the research presented in our [paper](https://arxiv.org/pdf/2601.16344), DSGym provides a unified evaluation platform with isolated execution environments, multiple datasets spanning diverse data science domains, and robust evaluation metrics.

## Overview

DSGym addresses the critical need for systematic evaluation of AI agents' data science capabilities. Unlike existing benchmarks that focus on narrow tasks, DSGym provides a holistic assessment across the full data science workflow - from data exploration and preprocessing to modeling and interpretation.

## Features

- **Unified Benchmark Framework**: Evaluate LLMs across multiple data science datasets
- **Isolated Execution**: Docker-based container system for safe code execution
- **Multiple Backend Support**: AI Coding Gateway and LiteLLM (API-based), vLLM, and SGLang inference backends
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

### Use the AI Coding Gateway

Select `gateway` as the backend to route an OpenAI-compatible chat completion
through the AI Coding Gateway. Set the gateway credentials in the environment;
the gateway backend sends the API key as a bearer token and the project code as
the required `X-Project` header:

```bash
export AI_CODING_GATEWAY_API_KEY="..."
export AI_CODING_GATEWAY_PROJECT="..."
dsgym eval --backend gateway --model claude-opus-4-8 --dataset daeval
```

Bare model names are routed to the `anthropic/` model family. Provider-qualified
names such as `anthropic/claude-opus-4-8` are sent unchanged. `litellm` remains
available for direct provider integrations.

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

# DSPredict on Apple Silicon (M-series, no GPU)
sudo docker build -t executor-kaggle-apple -f ./container_images/kaggle_image/Dockerfile.apple ./container_images/kaggle_image
sudo docker build -t manager-prebuilt ./manager

python generate_compose.py -n 8 --types "executor-kaggle-apple:8" -m ../data/data --output docker-dspredict-apple.yml \
  -c container_config_dspredict_hard.json -s ../submissions \
  -e "EXECUTION_TIMEOUT=3600,MEM_LIMIT=24G,CPUS=8" --apple-silicon

sudo docker compose -f docker-dspredict-apple.yml up -d
sudo docker compose -f docker-dspredict-apple.yml down
```

> ⚠️ **Gotcha: Data mount path resolution**
>
> `generate_compose.py` embeds the `-m` path **literally** into the generated compose file.
> Docker Compose then resolves relative paths **relative to the compose file's location**, not
> relative to wherever `generate_compose.py` was run.
>
> The instructions above run `generate_compose.py` from `executors/`, so `-m ../data/data`
> resolves correctly at generation time. But when Docker Compose reads the generated file
> (which lives in the project root), it re-resolves `../data/data` relative to the project
> root — pointing **one level too high** to an empty directory. Agents will start but see no
> data, producing 35/35 no-submit runs with `FileNotFoundError`.
>
> **Fix:** always pass an absolute path to `-m`:
> ```bash
> python generate_compose.py -n 8 --types "executor-kaggle-apple:8" \
>   -m /absolute/path/to/DSGym/data/data \
>   --output ../docker-dspredict-apple.yml ...
> ```
>
> **Check:** after `docker compose up`, verify data is visible inside a container:
> ```bash
> docker exec $(docker ps -q --filter "name=executor" | head -1) ls /data/
> # Should list: dspredict-sampled  bio  dabstep  ...
> # Empty output = wrong mount path
> ```

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

# Create a 20-task subsample and evaluate with DAGs (first time)
dsgym eval --dataset dsbio --sample-n 20 --dags true --model gpt-4o

# Baseline comparison on the same saved subsample (no resampling)
dsgym eval --dataset dsbio-sampled --dags false --model gpt-4o

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

> ⚠️ **Gotcha: Run DSPredict evaluations from the repo root**
>
> The agent uses `submission_dir="./submissions"` (a relative path). Docker containers
> write their `submission.csv` to `<repo-root>/submissions/container_XXX/` (the absolute
> path embedded in the generated compose file). If you run `dsgym eval` from a
> subdirectory (e.g. `executors/`), `./submissions` resolves to
> `executors/submissions/` — which is empty — and every run will silently fail with
> `"submission file not found"` even though the agent printed `✓ Submission format is valid!`.
>
> **Fix:** always run evaluations from the repo root:
> ```bash
> # ✅ Correct — run from repo root
> cd /path/to/DSGym
> dsgym eval --dataset dspredict-hard --model gpt-4o
>
> # ❌ Wrong — ./submissions resolves to executors/submissions/ (empty)
> cd /path/to/DSGym/executors
> dsgym eval --dataset dspredict-hard --model gpt-4o
> ```
>
> **Symptom:** `success: false`, `prediction: ""`, metric reason `"submission file not found"`
> in the result JSON, despite the trajectory showing the agent completed successfully.


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
- **`dsgym resubmit`**: Re-score saved DSPredict submissions against Kaggle without re-running agents
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
├── {challenge}_{index}_{model}_{dags|nodags}.json  # Per-task checkpoint (see below)
└── dags/                         # DAG artifacts (DSBio --dags true only)
    ├── 0_dags.json               # reasoning + implementation DAGs for sample 0
    ├── 1_dags.json               # reasoning + implementation DAGs for sample 1
    └── ...
```

#### Per-task Checkpoints (Intermediate Results)

Every time a task finishes the evaluator immediately writes a per-task checkpoint
file in addition to the aggregate results files:

```
{output_dir}/{challenge_name}_{index}_{model_slug}_{dags|nodags}.json
# e.g. mens-machine-learning-competition-2018_31_claude_sonnet_4_5_dags.json
```

Each file contains the full `EvaluationResult` payload — prediction path,
metrics, trajectory, execution time — so you can recover partial results even
if the run does not complete.

To load partial results in `analysis.ipynb` use the helper added to that
notebook:

```python
dp_data[("claude-sonnet-4-6", "dags")] = load_dspredict_from_intermediate(
    "*_claude_sonnet_4_6_dags.json"
)
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

## Subsampling DSBio

DSBio ships with 90 tasks on the hard split. When you need a smaller, reproducible
subset — for rapid iteration, cost control, or controlled ablations — use the
built-in sampling helpers.

### Creating a subsample

**CLI** (`--sample-n` / `--sample-seed`):

```bash
# Sample 20 tasks, save to data/dsbio_sampled.json, then evaluate with DAGs
dsgym eval --dataset dsbio --sample-n 20 --dags true --model claude-sonnet-4-6

# Reproducible with a different seed
dsgym eval --dataset dsbio --sample-n 20 --sample-seed 7 --dags true --model claude-sonnet-4-6
```

| Flag | Default | Description |
|------|---------|-------------|
| `--sample-n N` | — | Sample *N* tasks from the hard split and save as `dsbio_sampled.json` |
| `--sample-seed SEED` | `42` | Random seed for fully deterministic sampling |

**Python API:**

```python
from dsgym.datasets import DatasetRegistry

dataset = DatasetRegistry.load("dsbio")

# Creates data/dsbio_sampled.json (same raw format as the source file)
dataset.create_sample(n=20, seed=42)

# Or write to a custom path
dataset.create_sample(n=20, seed=42, output_path="./my_sample.json")
```

`create_sample` raises a clear error if `n` exceeds the dataset size or the
source file is missing.

### Evaluating on an existing subsample

Once `dsbio_sampled.json` exists, load it directly without resampling using the
`dsbio-sampled` dataset name. It supports `--dags true/false` exactly like the
hard split:

```bash
# Baseline (no DAGs) — same 20 tasks, no resampling
dsgym eval --dataset dsbio-sampled --dags false --model claude-sonnet-4-6

# DAG-guided run with a different model on the same sample
dsgym eval --dataset dsbio-sampled --dags true --model gpt-4o
```

**Python API:**

```python
samples = DatasetRegistry.load("dsbio").load(split="sampled", limit=None)
```

### Typical workflows

```bash
# 1. Create sample + DAG run (first time)
dsgym eval --dataset dsbio --sample-n 20 --dags true --model claude-sonnet-4-6

# 2. Baseline comparison on the same 20 tasks
dsgym eval --dataset dsbio-sampled --dags false --model claude-sonnet-4-6

# 3. DAG run on existing sample with a different model
dsgym eval --dataset dsbio-sampled --dags true --model gpt-4o

# 4. Recreate sample with a different seed
dsgym eval --dataset dsbio --sample-n 20 --sample-seed 7 --dags true --model claude-sonnet-4-6
```

## DAG Generation for DSPredict

DSPredict tasks support the same iterative DAG-guided reasoning as DSBio. Before
writing any code the agent emits two self-contained DAGs — a high-level
**ReasoningDAG** (competition analysis decisions) and a concrete **ImplementationDAG**
(step-by-step ML pipeline) — and follows them, revising inline on failures.

### CLI flags (DSPredict)

| Flag | Default | Description |
|------|---------|-------------|
| `--dags true\|false` | `false` | Enable DAG-guided prompts for DSPredict |
| `--prompt VARIANT` | `default` | System prompt variant to use when `--dags true` (see below) |
| `--dag-model MODEL` | `gpt-4o` | Model used for post-hoc DAG generation (Python API only) |

```bash
# Baseline (no DAGs)
dsgym eval --dataset dspredict-hard --dags false --model gpt-4o

# DAG-guided run with default prompt (SYSTEM_PROMPT_DAG_DSPREDICT)
dsgym eval --dataset dspredict-hard --dags true --model gpt-4o

# DAG-guided run with a prompt variant
dsgym eval --dataset dspredict-hard --dags true --prompt adaptive-revision --model gpt-4o

# DAG-guided run on a subsample (see below)
dsgym eval --dataset dspredict-hard --sample-n 5 --dags true --model gpt-4o
```

### Prompt variants (`--dags true` only)

When `--dags true` is set, `--prompt` selects which system prompt the agent receives.
`--prompt default` (the default) uses `SYSTEM_PROMPT_DAG_DSPREDICT` from
`dsgym/datasets/prompts/system_prompts.py` — the same prompt used in all previous
DAG runs. All other variants come from `dsgym/datasets/prompt_variants.py`.

| Variant name | Characteristics |
|---|---|
| `default` | `SYSTEM_PROMPT_DAG_DSPREDICT` — full reasoning + implementation DAG, 5 revisions, submission validation |
| `adaptive-revision` | Extended revision grammar with proactive triggers; designed for models that need less structural scaffolding |
| `lightweight` | Minimal guidance, no explicit DAG requirement; best for fast prototyping |
| `structured` | Clear step-by-step format, optional implementation DAG |
| `reasoning-only` | Strict reasoning DAG (5–8 nodes), no implementation DAG |
| `reasoning-focused` | Reasoning DAG with moderate implementation guidance |
| `reasoning-implementation-strict` | Both DAGs strictly enforced, rigid step ordering |
| `detailed` | Long, expert-level prompt with extensive methodology guidance |
| `expert-agentic` | Full agentic loop with explicit self-evaluation gates |
| `problem-decomposition` | Emphasis on problem decomposition before any ML decisions |
| `data-quality-methods` | Prioritises data quality assessment and validation methodology |

`--prompt` is silently ignored when `--dags false` — the standard `SYSTEM_PROMPT_DSPREDICT` is always used in that case.

Output files are automatically suffixed with the variant name when it differs from `default`:

```
# --prompt default  →  ..._dags.json
# --prompt adaptive-revision  →  ..._dags_adaptive-revision.json
```

**Python API:**

```python
samples = dataset.load(split="hard", limit=10, generate_dags=True, prompt_variant="adaptive-revision")
```

### Python API

```python
from dsgym.datasets import DatasetRegistry

dataset = DatasetRegistry.load("dspredict")

# DAG system prompt injected automatically — no upfront API calls
samples = dataset.load(split="hard", limit=10, generate_dags=True)

# Use a prompt variant
samples = dataset.load(split="hard", limit=10, generate_dags=True, prompt_variant="adaptive-revision")

# Post-hoc DAG generation for a loaded sample
reasoning = dataset.generate_dag(samples[0], dag_type="reasoning", model="gpt-4o")
impl = dataset.generate_dag(
    samples[0], dag_type="implementation", reasoning_dag=reasoning, model="gpt-4o"
)

# Or both in one call
dags = dataset.generate_dags(samples[0], model="gpt-4o")
print(dags["reasoning"].model_dump_json(indent=2))
print(dags["implementation"].model_dump_json(indent=2))
```

## Subsampling DSPredict

DSPredict has three splits (`easy`, `hard`, `lite`). The same subsampling helpers
available for DSBio are also available here.

### Creating a subsample

**CLI** (`--sample-n` / `--sample-seed`):

```bash
# Sample 5 tasks from the hard split, save as dspredict_sampled.json, evaluate with DAGs
dsgym eval --dataset dspredict-hard --sample-n 5 --dags true --model gpt-4o

# Reproducible with a different seed
dsgym eval --dataset dspredict-hard --sample-n 5 --sample-seed 7 --dags true --model gpt-4o
```

**Python API:**

```python
from dsgym.datasets import DatasetRegistry

dataset = DatasetRegistry.load("dspredict")

# Creates data/task/dspredict/dspredict_sampled.json
dataset.create_sample(n=5, split="hard", seed=42)

# Or sample from the easy split
dataset.create_sample(n=10, split="easy", seed=42)
```

### Evaluating on an existing subsample

```bash
# Baseline (no DAGs) — same 5 tasks, no resampling
dsgym eval --dataset dspredict-sampled --dags false --model gpt-4o

# DAG-guided run with a different model on the same sample
dsgym eval --dataset dspredict-sampled --dags true --model claude-sonnet-4-6
```

**Python API:**

```python
samples = DatasetRegistry.load("dspredict").load(split="sampled")
```

### Typical workflows

```bash
# 1. Create sample + DAG run (first time)
dsgym eval --dataset dspredict-hard --sample-n 5 --dags true --model gpt-4o

# 2. Baseline comparison on the same tasks
dsgym eval --dataset dspredict-sampled --dags false --model gpt-4o

# 3. DAG run with a different model on the same sample
dsgym eval --dataset dspredict-sampled --dags true --model claude-sonnet-4-6

# 4. Recreate sample with a different seed
dsgym eval --dataset dspredict-hard --sample-n 5 --sample-seed 7 --dags true --model gpt-4o
```

## Re-submitting DSPredict Results (`dsgym resubmit`)

If a DSPredict evaluation run times out or is interrupted, the per-task
checkpoint files are still on disk. `dsgym resubmit` reads those files,
re-submits each saved `submission.csv` to Kaggle, and writes the scored
metrics back into the checkpoint files — without re-running any agents.

### Basic usage

```bash
# Re-score all claude-sonnet-4-5 DAG checkpoints in the default output directory
dsgym resubmit --pattern "*_claude_sonnet_4_5_dags.json"

# Use a specific directory and a longer per-submission timeout
dsgym resubmit \
    --input-dir ./evaluation_results \
    --pattern "*_dags.json" \
    --timeout 20

# Force re-score even checkpoints that already have a score
dsgym resubmit --pattern "*_dags.json" --no-skip-scored

# Dry-run: show which files would be submitted without uploading
dsgym resubmit --pattern "*_dags.json" --dry-run
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | `./evaluation_results` | Directory containing checkpoint JSON files |
| `--pattern` | `*.json` | Glob pattern for checkpoint files (use a specific model/run pattern) |
| `--timeout` | `10` | Minutes to wait for each Kaggle submission to be scored |
| `--max-workers` | `1` | Parallel submissions. Default is 1 (sequential) — see note below |
| `--skip-scored` | `True` | Skip files that already have a non-null score |
| `--no-skip-scored` | — | Re-submit even already-scored checkpoints |
| `--dry-run` | `False` | Print what would be submitted without uploading |

### Upload speed and parallelism

DSPredict submission files can be large (50–400 MB). At a typical upload speed
of 8 Mbps (1 MB/s), a 400 MB file takes ~6–7 minutes to upload.

`--max-workers` defaults to **1** (sequential). This is intentional: running
multiple uploads in parallel splits your available upload bandwidth between them,
making each one slower without any real throughput gain. Only increase
`--max-workers` on fast connections (100+ Mbps).

Upload progress is printed as the file transfers:

```
⬆️  Uploading ashrae-energy-prediction_submission.csv (51.0 MB) to ashrae-energy-prediction…
  … 10.2 / 51.0 MB  (20%)
  … 20.5 / 51.0 MB  (40%)
  …
✅ Upload complete: 51.0 MB in 51s (1.0 MB/s)
```

### Retry behaviour

The submission pipeline automatically retries on transient failures
(broken connections, rate limits, timeouts). Each retry creates a **fresh
Kaggle API client** with a new connection pool, which is the root cause fix for
`BrokenPipeError` — stale HTTP connections in `KaggleClient` were silently
failing on subsequent API calls after the first connection dropped.

Uploads use `requests` (not `httpx`) with a file-like body so that urllib3
honours the `Content-Length` header and never adds `Transfer-Encoding: chunked`.
Sending both headers simultaneously caused GCS to reset the connection near the
end of large uploads. TCP keepalive (30s probe interval) is also enabled on the
upload socket to prevent NAT/firewall table expiry during slow transfers.

| Stage | Retries | Behaviour |
|---|---|---|
| File upload (PUT to GCS presigned URL) | 5 (outer) | New `KaggleClient` per attempt; 20s–30s jittered backoff |
| Score polling (`GetSubmission`) | Continuous until timeout | Jittered 30 ± 10s interval; transient errors logged and retried |

Parallel workers use a random initial delay (5–20s) before their first poll so
they don't all hit the Kaggle API simultaneously.

### Submission format validation

Before a file is uploaded to Kaggle the agent automatically validates the
generated `submission.csv` against the local `sample_submission.csv`:

- **Missing required columns** — e.g. column expected by Kaggle is absent
- **Extra/unexpected columns** — columns not present in the sample
- **Wrong row count** — number of rows doesn't match the sample

If errors are found the framework injects a detailed error message back into the
agent's conversation and gives it up to **5 extra turns** to fix the file.
After each fix turn the file is re-validated. If all fix turns are exhausted the
file is uploaded anyway (rather than losing the result entirely) and the
`error_description` returned by Kaggle is printed and stored in the checkpoint.

The validation snippet is also included in both agent system prompts so the
agent validates its own output before finishing:

```python
import pandas as pd
sample = pd.read_csv('/data/dspredict-sampled/<challenge>/sample_submission.csv')
sub    = pd.read_csv('/submission/submission.csv')
assert list(sub.columns) == list(sample.columns), f"Column mismatch: {list(sample.columns)}"
assert len(sub) == len(sample), f"Row count mismatch: expected {len(sample)}, got {len(sub)}"
print("Format OK")
```

### Typical recovery workflow

```bash
# 1. A DSPredict run times out part-way through — checkpoints are on disk
dsgym eval --dataset dspredict-sampled --dags true --model claude-sonnet-4-6
#    → times out after 20/35 tasks; 20 checkpoint files are saved

# 2. Re-score the checkpoints that completed (skips already-scored ones)
dsgym resubmit \
    --input-dir ./evaluation_results \
    --pattern "*_claude_sonnet_4_6_dags.json"

# 3. Load partial results in analysis.ipynb
#    Use load_dspredict_from_intermediate() (already in the notebook):
#    dp_data[("claude-sonnet-4-6", "dags")] = load_dspredict_from_intermediate(
#        "*_claude_sonnet_4_6_dags.json"
#    )
```

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

## DAG Prompting: Research Findings

Our trace analysis on DSPredict-Hard (54 Kaggle competitions) reveals important insights about DAG-guided reasoning for coding agents.

### Key Finding: DAGs Guide Sophisticated Solutions

| Metric | DAGs | NoDags | Interpretation |
|--------|------|--------|----------------|
| GPT-5.1 Success Rate | 62.5% | **75.3%** | NoDags wins overall |
| Timeouts | **36.4%** | 31.5% | DAGs hit compute limits more |
| Score when both succeed | **18 wins** | 17 wins | Quality is equivalent |

**The paradox explained**: DAGs don't produce *worse* solutions—they produce *more sophisticated* solutions that timeout on limited compute (8GB GPU). When both approaches complete successfully, DAG-guided runs win as often as non-DAG runs.

### Evidence

```
When both DAGs and NoDags succeed on same competition:
├── DAGs score higher: 18 competitions
├── NoDags score higher: 17 competitions  
└── Essentially tied: 5 competitions

Timeout rates:
├── DAGs: 36.4% (guided toward CV, ensembles, hyperparameter tuning)
└── NoDags: 31.5% (simpler, faster approaches)
```

### Semantic Analysis

Implementation DAGs provide **40-60% better code grounding** than Reasoning DAGs alone:

| Model | R→Code Similarity | I→Code Similarity | Improvement |
|-------|-------------------|-------------------|-------------|
| GPT-5.1 | 0.067 | 0.094 | +40% |
| Claude | 0.050 | 0.080 | +60% |
| DeepSeek | 0.081 | 0.113 | +39% |

### Recommendations

| Scenario | Recommendation |
|----------|----------------|
| **Limited compute** (≤8GB GPU) | Use `nodags` mode for higher success rate |
| **Adequate compute** (≥24GB GPU) | Use `dags` mode for potentially higher quality |
| **Claude models** | Always use `dags` (provides termination structure) |
| **Research/cluster** | DAGs are worth pursuing with sufficient timeout |

### Compute Requirements for DAG Mode

DAG-guided agents attempt more sophisticated approaches (cross-validation, ensembles, hyperparameter search). Recommended minimums:

- **GPU Memory**: 24GB+ (vs 8GB for nodags)
- **Timeout**: 7200s+ (vs 3600s for nodags)
- **Containers**: 3-5 with higher memory allocation

See `evaluation_results/analysis_deepseek.md` for detailed trace analysis.

## Resources

- 📄 **Paper**: [DSGym: A Holistic Framework for Evaluating and Training Data Science Agents](https://arxiv.org/pdf/2601.16344)
- 🤗 **Hugging Face**: [DSGym Repository](https://huggingface.co/DSGym)
- 💻 **GitHub**: [Source Code](https://github.com/fannie1208/DSGym)

## Gotchas

### Container pool vs. worker count mismatch
**Symptom**: `⚠️ Warning: Failed to allocate container: Server error '500'/'503' for url 'http://localhost:5000/allocate'`

The default litellm worker count is 24, but a typical dspredict-hard setup only has 5 containers. Having more workers than containers is harmless — excess workers queue up and get a container as soon as one is freed. However, if workers hold containers for longer than `ALLOCATION_TIMEOUT` (default 1 hour), queued workers will time out.

`eval.py` now **automatically caps `max_workers` to the container pool size** by querying `/status` at startup, so this is handled for you. If you override `--max-workers` manually, keep it ≤ the number of containers in your compose file.

### Ghost allocations after a crash
If the eval process crashes or is killed without cleanly deallocating containers, the manager retains those containers in the "allocated" state permanently — leaving **0 containers available** for new runs. Two ways to recover:

```bash
# Option 1: soft restart — restarts all currently allocated containers and returns them to the pool
curl -X POST http://localhost:5000/restart

# Option 2: full reset — abandons stale queue waiters, restarts all containers, rebuilds the pool
curl -X POST http://localhost:5000/reset
```

> **Note**: `/reset` calls `/restart` on each executor with a 15-second timeout. If executors are genuinely down it will be slow but will still re-populate the queue so new runs can start.

The manager also performs **automatic zombie recovery**: if a waiting worker hits `ALLOCATION_TIMEOUT`, the manager health-checks every allocated container and restarts any that are unresponsive, returning them to the pool before giving up with 503.

### ALLOCATION_TIMEOUT too short for long-running tasks
The default `ALLOCATION_TIMEOUT` is **3600 seconds (1 hour)**. DSPredict tasks can run for several hours, so if a worker has been queued for more than an hour it will be dropped.

When generating a compose file for dspredict, pass a longer timeout:
```bash
python executors/generate_compose.py ... --allocation-timeout 86400   # 24 hours
```

### GPU device IDs on a single-GPU machine
When running multiple executor containers on a host with only one GPU, all containers must use `device_ids: ['0']`. Using IDs `1`–`7` on a single-GPU machine causes a `device error: 3: unknown device` and the container will fail to start. The `--gpu-ids` argument to `generate_compose.py` controls this:
```bash
# 5 containers all sharing GPU 0
python executors/generate_compose.py -n 5 --gpu-ids 0,0,0,0,0 ...
```

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
