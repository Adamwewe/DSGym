"""
DAG generation prompts for DSBio tasks.

Provides system prompts, reference examples, and helpers that construct
prompts dynamically from a DSBio sample's question and context —
rather than hardcoding a single problem as in sandbox-agents.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Reference examples (bio domain) — kept deliberately small so models
# learn the format without copying the content.
# ---------------------------------------------------------------------------

REFERENCE_REASONING_DAG: Dict[str, Any] = {
    "graph_id": "example_reasoning",
    "nodes": [
        {"id": "n1", "label": "Assess raw count matrix quality: decide whether n_genes and n_counts distributions indicate high-quality cells"},
        {"id": "n2", "label": "Gate on QC: set mitochondrial fraction and minimum-genes thresholds; halt if fewer than 200 cells pass"},
        {"id": "n3", "label": "Choose normalisation strategy and highly-variable gene selection criteria given library-size distribution"},
        {"id": "n4", "label": "Select dimensionality-reduction approach (PCA components) and clustering resolution based on expected cell-type diversity"},
        {"id": "n5", "label": "Choose DE test (Wilcoxon vs t-test) based on group sizes; interpret top markers in biological context"},
    ],
    "edges": [
        {"source": "n1", "target": "n2"},
        {"source": "n2", "target": "n3"},
        {"source": "n3", "target": "n4"},
        {"source": "n4", "target": "n5"},
    ],
}


REFERENCE_IMPLEMENTATION_DAG: Dict[str, Any] = {
    "graph_id": "example_implementation",
    "nodes": [
        {"id": "n1", "label": "Load AnnData; inspect .shape, .obs, .var; print n_cells × n_genes"},
        {"id": "n2", "label": "Compute sc.pp.calculate_qc_metrics; filter cells where pct_counts_mt > 20 or n_genes_by_counts < 200"},
        {"id": "n3", "label": "sc.pp.normalize_total(1e4); sc.pp.log1p; sc.pp.highly_variable_genes(n_top_genes=2000)"},
        {"id": "n4", "label": "sc.pp.pca(n_comps=50); sc.pp.neighbors; sc.tl.leiden(resolution=0.5)"},
        {"id": "n5", "label": "sc.tl.rank_genes_groups(method='wilcoxon'); extract top-5 markers per cluster; save result CSV"},
    ],
    "edges": [
        {"source": "n1", "target": "n2"},
        {"source": "n2", "target": "n3"},
        {"source": "n3", "target": "n4"},
        {"source": "n4", "target": "n5"},
    ],
}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_REASONING_SYSTEM_PROMPT = """You are an expert computational biologist.
Given a biological question and context, produce a concise JSON reasoning DAG
capturing the ordered analytical decisions an expert makes before writing any code.

Rules:
- Return ONLY valid JSON. No markdown or prose outside the JSON.
- Use 3 to 6 nodes. Do not over-engineer; each node must carry genuine decision weight.
- Each node: one analytical decision, not a code instruction.
  Good: "Determine whether library-size variation warrants scran-style pooling normalisation"
  Bad: "Run sc.pp.normalize_total"
- Edges need only "source" and "target".
- The graph must be acyclic (nodes listed in execution order).
- Include a QC gate node when data quality is uncertain: the gate explicitly decides
  whether analysis should proceed or stop.
- If the question involves differential expression, include a node that selects the
  statistical test based on group sizes and data distribution.
- The final node should frame the expected biological interpretation, not just the
  statistical result."""

_IMPLEMENTATION_SYSTEM_PROMPT = """You are an expert computational biologist.
Given a biological question, context, and a reasoning DAG, produce a JSON implementation
DAG of concrete executable steps that realise each reasoning node exactly.

Rules:
- Return ONLY valid JSON. No markdown or prose outside the JSON.
- Produce exactly one implementation node per reasoning node (1:1 mapping).
- Mirror the reasoning DAG structure exactly — same number of nodes and edges.
- Each node: one concrete executable step. Name the primary scanpy/anndata/scipy
  function or operation involved so the agent can translate it directly to code.
- Edges need only "source" and "target".
- The graph must be acyclic (nodes listed in execution order).
- Do not skip QC or normalisation steps even if they appear optional; a failed QC
  gate implementation node should raise an explicit error with a diagnostic message."""

_PROMPT_PROFILE_CACHE: Dict[str, Any] = {
    "path": None,
    "mtime": None,
    "profile": {},
}


def _load_prompt_profile() -> Dict[str, Any]:
    profile_path = os.getenv("DSGYM_DAG_PROMPT_PROFILE", "").strip()
    if not profile_path:
        _PROMPT_PROFILE_CACHE.update({"path": None, "mtime": None, "profile": {}})
        return {}

    try:
        mtime = os.path.getmtime(profile_path)
    except OSError:
        return {}

    if (
        _PROMPT_PROFILE_CACHE["path"] == profile_path
        and _PROMPT_PROFILE_CACHE["mtime"] == mtime
    ):
        return _PROMPT_PROFILE_CACHE["profile"]

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
            if not isinstance(profile, dict):
                profile = {}
    except Exception:
        profile = {}

    _PROMPT_PROFILE_CACHE.update({"path": profile_path, "mtime": mtime, "profile": profile})
    return profile


def _profile_value(section: str, key: str, default: str) -> str:
    profile = _load_prompt_profile()
    scoped = profile.get(section, {}) if isinstance(profile.get(section, {}), dict) else {}
    global_values = profile.get("global", {}) if isinstance(profile.get("global", {}), dict) else {}

    value = scoped.get(key, global_values.get(key, default))
    return value if isinstance(value, str) else default


def _with_prefix_suffix(section: str, kind: str, body: str) -> str:
    prefix = _profile_value(section, f"{kind}_user_prefix", "").strip()
    suffix = _profile_value(section, f"{kind}_user_suffix", "").strip()

    chunks = []
    if prefix:
        chunks.append(prefix)
    chunks.append(body)
    if suffix:
        chunks.append(suffix)

    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_reasoning_prompt(question: str, context: str) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for generating a reasoning DAG.

    Args:
        question: The DSBio question text.
        context:  The biological context / background for the task.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    user_prompt = f"""BIOLOGICAL QUESTION:
{question}

CONTEXT:
{context}

EXAMPLE OUTPUT (follow the schema only — do NOT copy this graph for your answer):
{json.dumps(REFERENCE_REASONING_DAG, indent=2)}

Produce the reasoning DAG for the question above.
"""
    system_prompt = _profile_value("dsbio", "reasoning_system_prompt", _REASONING_SYSTEM_PROMPT)
    return system_prompt, _with_prefix_suffix("dsbio", "reasoning", user_prompt)


def build_implementation_prompt(
    question: str,
    context: str,
    reasoning_dag: Dict[str, Any],
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for generating an implementation DAG.

    Args:
        question:      The DSBio question text.
        context:       The biological context / background.
        reasoning_dag: The previously generated reasoning DAG (as a dict).

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    user_prompt = f"""BIOLOGICAL QUESTION:
{question}

CONTEXT:
{context}

REASONING DAG:
{json.dumps(reasoning_dag, indent=2)}

EXAMPLE OUTPUT (follow schema only — do NOT copy this graph for your answer):
{json.dumps(REFERENCE_IMPLEMENTATION_DAG, indent=2)}

Produce the implementation DAG for the question above.
"""
    system_prompt = _profile_value("dsbio", "implementation_system_prompt", _IMPLEMENTATION_SYSTEM_PROMPT)
    return system_prompt, _with_prefix_suffix("dsbio", "implementation", user_prompt)


# ---------------------------------------------------------------------------
# DSPredict reference examples and prompt builders
# ---------------------------------------------------------------------------

REFERENCE_REASONING_DAG_DSPREDICT: Dict[str, Any] = {
    "graph_id": "example_reasoning_dspredict",
    "nodes": [
        {"id": "n1", "label": "Anchor to evaluation metric and submission format: understand what is optimised and how predictions are scored"},
        {"id": "n2", "label": "Audit data for leakage: identify time-based, ID-based, or target-encoded features before any transformation"},
        {"id": "n3", "label": "Assess feature distributions, missing-value patterns, and target imbalance to choose preprocessing strategy"},
        {"id": "n4", "label": "Select model family and cross-validation strategy: choose fold count, stratification, and grouping for the data structure"},
        {"id": "n5", "label": "Decide on post-processing or threshold tuning required to convert model output to the submission format"},
    ],
    "edges": [
        {"source": "n1", "target": "n2"},
        {"source": "n2", "target": "n3"},
        {"source": "n3", "target": "n4"},
        {"source": "n4", "target": "n5"},
    ],
}

REFERENCE_IMPLEMENTATION_DAG_DSPREDICT: Dict[str, Any] = {
    "graph_id": "example_implementation_dspredict",
    "nodes": [
        {"id": "n1", "label": "Load train.csv and test.csv; print shapes, dtypes, target distribution, and unique ID column"},
        {"id": "n2", "label": "Assert no target-leaking columns in test set; drop columns with > 50% missing; encode categoricals"},
        {"id": "n3", "label": "Compute missing value fractions, correlation matrix, and per-feature variance; drop near-zero-variance features"},
        {"id": "n4", "label": "StratifiedKFold(n_splits=5); fit LightGBM; report OOF metric after each fold; select best hyperparameter via early stopping"},
        {"id": "n5", "label": "Generate predictions on test set using all-fold averaged model; format as required; write /submission/submission.csv"},
    ],
    "edges": [
        {"source": "n1", "target": "n2"},
        {"source": "n2", "target": "n3"},
        {"source": "n3", "target": "n4"},
        {"source": "n4", "target": "n5"},
    ],
}

_REASONING_SYSTEM_PROMPT_DSPREDICT = """You are an expert data scientist and machine learning engineer.
Given a Kaggle-style competition description, produce a concise JSON reasoning DAG
capturing the ordered analytical decisions an expert makes before writing any code.

Rules:
- Return ONLY valid JSON. No markdown or prose outside the JSON.
- Use 3 to 6 nodes. Each node must carry genuine decision weight.
- Each node: one analytical decision, not a code instruction.
  Good: "Determine whether class imbalance requires stratified CV or SMOTE before model fitting"
  Bad: "Run XGBClassifier"
- Edges need only "source" and "target".
- The graph must be acyclic (nodes listed in execution order).
- The first reasoning node must anchor the entire plan to the evaluation metric and
  submission format — every subsequent decision is judged against that anchor.
- Include an explicit leakage-prevention node when the data contains time, ID, or
  aggregate features that could encode the target.
- Include a cross-validation strategy node that selects fold count, stratification,
  and grouping appropriate for the competition data structure."""

_IMPLEMENTATION_SYSTEM_PROMPT_DSPREDICT = """You are an expert data scientist and machine learning engineer.
Given a competition description, context, and a reasoning DAG, produce a JSON implementation
DAG of concrete executable steps that realise each reasoning node exactly.

Rules:
- Return ONLY valid JSON. No markdown or prose outside the JSON.
- Produce exactly one implementation node per reasoning node (1:1 mapping).
- Mirror the reasoning DAG structure exactly — same number of nodes and edges.
- Each node: one concrete executable step. Name the primary library, sklearn API,
  or pandas/numpy operation so the agent can translate it directly to code.
- Edges need only "source" and "target".
- The graph must be acyclic (nodes listed in execution order).
- The leakage-prevention implementation node must assert no test-set columns encode
  the target before any downstream transformation.
- The final node must produce /submission/submission.csv with the correct column names."""


def build_reasoning_prompt_dspredict(challenge_name: str, description: str) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for generating a reasoning DAG for a DSPredict task.

    Args:
        challenge_name: The Kaggle competition name.
        description:    The competition description / objective.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    user_prompt = f"""COMPETITION NAME:
{challenge_name}

DESCRIPTION:
{description}

EXAMPLE OUTPUT (follow the schema only — do NOT copy this graph for your answer):
{json.dumps(REFERENCE_REASONING_DAG_DSPREDICT, indent=2)}

Produce the reasoning DAG for the competition above.
"""
    system_prompt = _profile_value("dspredict", "reasoning_system_prompt", _REASONING_SYSTEM_PROMPT_DSPREDICT)
    return system_prompt, _with_prefix_suffix("dspredict", "reasoning", user_prompt)


def build_implementation_prompt_dspredict(
    challenge_name: str,
    description: str,
    reasoning_dag: Dict[str, Any],
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for generating an implementation DAG for a DSPredict task.

    Args:
        challenge_name: The Kaggle competition name.
        description:    The competition description / objective.
        reasoning_dag:  The previously generated reasoning DAG (as a dict).

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    user_prompt = f"""COMPETITION NAME:
{challenge_name}

DESCRIPTION:
{description}

REASONING DAG:
{json.dumps(reasoning_dag, indent=2)}

EXAMPLE OUTPUT (follow schema only — do NOT copy this graph for your answer):
{json.dumps(REFERENCE_IMPLEMENTATION_DAG_DSPREDICT, indent=2)}

Produce the implementation DAG for the competition above.
"""
    system_prompt = _profile_value("dspredict", "implementation_system_prompt", _IMPLEMENTATION_SYSTEM_PROMPT_DSPREDICT)
    return system_prompt, _with_prefix_suffix("dspredict", "implementation", user_prompt)
