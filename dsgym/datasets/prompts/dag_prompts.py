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
        {"id": "n1", "label": "Assess raw count matrix quality"},
        {"id": "n2", "label": "Determine QC thresholds and filter cells"},
        {"id": "n3", "label": "Normalise, select variable genes, and cluster cells"},
    ],
    "edges": [
        {"source": "n1", "target": "n2"},
        {"source": "n2", "target": "n3"},
    ],
}


REFERENCE_IMPLEMENTATION_DAG: Dict[str, Any] = {
    "graph_id": "example_implementation",
    "nodes": [
        {"id": "n1", "label": "Load AnnData and inspect shape"},
        {"id": "n2", "label": "Compute QC metrics and filter low-quality cells"},
        {"id": "n3", "label": "Normalise, find variable genes, run PCA and Leiden clustering"},
    ],
    "edges": [
        {"source": "n1", "target": "n2"},
        {"source": "n2", "target": "n3"},
    ],
}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_REASONING_SYSTEM_PROMPT = """You are an expert computational biologist.
Given a biological question and context, produce a concise JSON reasoning DAG
capturing the ordered analytical steps an expert would think through before writing code.

Rules:
- Return ONLY valid JSON. No markdown or prose outside the JSON.
- Use 3 to 6 nodes maximum. Do not over-engineer.
- Each node: one clear, high-level reasoning step (not a code instruction).
- Edges need only "source" and "target".
- The graph must be acyclic (nodes listed in execution order).
- Label each node at the level of an analytical decision, not a library call
  (e.g. "Determine appropriate DE test given data distribution", not "Run sc.tl.rank_genes_groups").
"""

_IMPLEMENTATION_SYSTEM_PROMPT = """You are an expert computational biologist.
Given a biological question, context, and a reasoning DAG, produce a JSON implementation
DAG of concrete, executable steps that follow the provided reasoning DAG.

Rules:
- Return ONLY valid JSON. No markdown or prose outside the JSON.
- Produce exactly one implementation node per reasoning node (1:1 mapping).
- Mirror the reasoning DAG structure exactly — same number of nodes and edges.
- Each node: one concrete, executable step with a descriptive label.
- Edges need only "source" and "target".
- The graph must be acyclic (nodes listed in execution order).
"""

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
        {"id": "n1", "label": "Understand evaluation metric and submission format"},
        {"id": "n2", "label": "Assess data quality, distributions, and feature relationships"},
        {"id": "n3", "label": "Select model family and validation strategy given data size and task type"},
    ],
    "edges": [
        {"source": "n1", "target": "n2"},
        {"source": "n2", "target": "n3"},
    ],
}

REFERENCE_IMPLEMENTATION_DAG_DSPREDICT: Dict[str, Any] = {
    "graph_id": "example_implementation_dspredict",
    "nodes": [
        {"id": "n1", "label": "Load training and test data, inspect shapes, dtypes, and target distribution"},
        {"id": "n2", "label": "Compute missing value counts, correlation matrix, and per-feature statistics"},
        {"id": "n3", "label": "Train LightGBM with 5-fold CV, tune one key hyperparameter, generate submission.csv"},
    ],
    "edges": [
        {"source": "n1", "target": "n2"},
        {"source": "n2", "target": "n3"},
    ],
}

_REASONING_SYSTEM_PROMPT_DSPREDICT = """You are an expert data scientist and machine learning engineer.
Given a Kaggle-style competition description, produce a concise JSON reasoning DAG
capturing the ordered analytical steps an expert would think through before writing code.

Rules:
- Return ONLY valid JSON. No markdown or prose outside the JSON.
- Use 3 to 6 nodes maximum. Do not over-engineer.
- Each node: one clear, high-level reasoning step (not a code instruction).
- Edges need only "source" and "target".
- The graph must be acyclic (nodes listed in execution order).
- Label each node at the level of an analytical decision, not a library call
  (e.g. "Determine appropriate model given imbalanced classes", not "Run XGBClassifier").
"""

_IMPLEMENTATION_SYSTEM_PROMPT_DSPREDICT = """You are an expert data scientist and machine learning engineer.
Given a competition description, context, and a reasoning DAG, produce a JSON implementation
DAG of concrete, executable steps that follow the provided reasoning DAG.

Rules:
- Return ONLY valid JSON. No markdown or prose outside the JSON.
- Produce exactly one implementation node per reasoning node (1:1 mapping).
- Mirror the reasoning DAG structure exactly — same number of nodes and edges.
- Each node: one concrete, executable step with a descriptive label.
- Edges need only "source" and "target".
- The graph must be acyclic (nodes listed in execution order).
- The final node must produce /submission/submission.csv in the required format.
"""


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
