"""
DAG generation prompts for DSBio tasks.

Provides system prompts, reference examples, and helpers that construct
prompts dynamically from a DSBio sample's question and context —
rather than hardcoding a single problem as in sandbox-agents.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Reference examples (bio domain)
# ---------------------------------------------------------------------------

REFERENCE_REASONING_DAG: Dict[str, Any] = {
    "graph_id": "cell_clustering_reasoning",
    "nodes": [
        {
            "id": "n1",
            "label": "Assess raw count matrix quality",
            "task_type": "quality_control",
            "metadata": {},
        },
        {
            "id": "n2",
            "label": "Determine QC thresholds (min genes, max MT%)",
            "task_type": "qc_validation",
            "metadata": {},
        },
        {
            "id": "n3",
            "label": "Evaluate normalization strategy",
            "task_type": "preprocessing",
            "metadata": {},
        },
        {
            "id": "n4",
            "label": "Select dimensionality reduction approach",
            "task_type": "dimensionality_reduction",
            "metadata": {},
        },
        {
            "id": "n5",
            "label": "Determine clustering resolution",
            "task_type": "clustering",
            "metadata": {},
        },
        {
            "id": "n6",
            "label": "Validate cluster biological significance",
            "task_type": "result_validation",
            "metadata": {},
        },
    ],
    "edges": [
        {"source": "n1", "target": "n2", "weight": None, "label": None, "condition": None},
        {"source": "n2", "target": "n3", "weight": None, "label": "cells filtered", "condition": None},
        {"source": "n3", "target": "n4", "weight": None, "label": "data normalised", "condition": None},
        {"source": "n4", "target": "n5", "weight": None, "label": "PCA computed", "condition": None},
        {"source": "n5", "target": "n6", "weight": None, "label": "clusters identified", "condition": None},
    ],
}


REFERENCE_IMPLEMENTATION_DAG: Dict[str, Any] = {
    "graph_id": "cell_clustering_implementation",
    "nodes": [
        {
            "id": "n1",
            "label": "Load AnnData object and inspect shape",
            "task_type": "data_exploration",
            "metadata": {"test_type": "deterministic", "function_id": "PLACEHOLDER"},
        },
        {
            "id": "n2",
            "label": "Compute per-cell QC metrics (n_genes_by_counts, pct_counts_mt)",
            "task_type": "quality_control",
            "metadata": {"test_type": "deterministic", "function_id": "PLACEHOLDER"},
        },
        {
            "id": "n3",
            "label": "Filter cells below min_genes threshold and above max MT%",
            "task_type": "qc_validation",
            "metadata": {"test_type": "heuristic", "function_id": "PLACEHOLDER"},
        },
        {
            "id": "n4",
            "label": "Normalise total counts per cell and log1p transform",
            "task_type": "preprocessing",
            "metadata": {"test_type": "deterministic", "function_id": "PLACEHOLDER"},
        },
        {
            "id": "n5",
            "label": "Select highly variable genes and compute PCA",
            "task_type": "dimensionality_reduction",
            "metadata": {"test_type": "deterministic", "function_id": "PLACEHOLDER"},
        },
        {
            "id": "n6",
            "label": "Build neighbour graph and run Leiden clustering",
            "task_type": "clustering",
            "metadata": {"test_type": "heuristic", "function_id": "PLACEHOLDER"},
        },
        {
            "id": "n7",
            "label": "Assert cluster count matches expected biological groups",
            "task_type": "result_validation",
            "metadata": {"test_type": "deterministic", "function_id": "PLACEHOLDER"},
        },
    ],
    "edges": [
        {"source": "n1", "target": "n2", "weight": None, "label": "data loaded", "condition": None},
        {"source": "n2", "target": "n3", "weight": None, "label": "metrics computed", "condition": None},
        {"source": "n3", "target": "n4", "weight": None, "label": "cells filtered", "condition": "pass"},
        {"source": "n4", "target": "n5", "weight": None, "label": "normalised", "condition": None},
        {"source": "n5", "target": "n6", "weight": None, "label": "PCA done", "condition": None},
        {"source": "n6", "target": "n7", "weight": None, "label": "clusters ready", "condition": None},
    ],
}


# ---------------------------------------------------------------------------
# Allowed task_type values (kept in sync with DSBioTaskType enum)
# ---------------------------------------------------------------------------

_ALLOWED_TASK_TYPES = (
    "data_exploration | quality_control | qc_validation | preprocessing | "
    "dimensionality_reduction | clustering | differential_expression | "
    "pathway_analysis | cell_type_annotation | statistical_analysis | result_validation"
)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_REASONING_SYSTEM_PROMPT = f"""You are an expert computational biologist and data scientist.
Your job is to reason about how a single-cell transcriptomics (or related bio) problem should be
solved BEFORE any implementation.

Convert the user's biological question into a directed acyclic graph (DAG) of reasoning steps.

Rules:
- Return ONLY data matching the required schema. No markdown, no prose outside the schema.
- The graph must be directed and acyclic (a valid topological ordering must exist).
- Every node represents ONE reasoning or validation step — NOT a code step.
- No duplicate node ids.
- Keep node labels concise and action-oriented (e.g., "Assess batch-effect severity").
- task_type must be one of: {_ALLOWED_TASK_TYPES}
- Edges carry optional weight (float), label (str), and condition (str) fields.
- The DAG must be topologically sorted (like networkx.DiGraph).
- If uncertain, produce the smallest valid DAG that captures the main analytical reasoning.
"""

_IMPLEMENTATION_SYSTEM_PROMPT = f"""You are an expert computational biologist and data scientist.
Your job is to plan the IMPLEMENTATION of a solution for a single-cell transcriptomics problem
based on a reasoning DAG. Do NOT execute — only plan in detail.

Convert the reasoning DAG into an implementation DAG of concrete, executable steps.

Rules:
- Return ONLY data matching the required schema. No markdown, no prose outside the schema.
- The graph must be directed and acyclic (a valid topological ordering must exist).
- Every node represents ONE concrete implementation step.
- No duplicate node ids.
- Node labels must be action-oriented and implementation-ready
  (e.g., "Compute PCA with n_comps=50", "Run Leiden clustering at resolution=0.5").
- task_type must be one of: {_ALLOWED_TASK_TYPES}
- Each node metadata must include:
    - test_type: "deterministic" or "heuristic"
    - function_id: a placeholder string for the function (use "PLACEHOLDER" if unknown)
- Branching edges should carry a condition field (e.g., "pass", "fail", "imbalanced").
- Stay close to the reasoning DAG — do NOT deviate from its reasoning steps.
- The DAG must be topologically sorted (like networkx.DiGraph).
"""


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

REFERENCE EXAMPLE (schema only — do NOT copy this graph):
{json.dumps(REFERENCE_REASONING_DAG, indent=2)}

Task:
Construct the reasoning DAG a computational biologist should follow before implementing
a solution to the question above.
"""
    return _REASONING_SYSTEM_PROMPT, user_prompt


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

REASONING DAG (use this as your mental model — do NOT deviate from its steps):
{json.dumps(reasoning_dag, indent=2)}

REFERENCE EXAMPLE (schema only — do NOT copy this graph):
{json.dumps(REFERENCE_IMPLEMENTATION_DAG, indent=2)}

Task:
Construct the implementation DAG for the question above.
Each node must include test_type and function_id in its metadata.
"""
    return _IMPLEMENTATION_SYSTEM_PROMPT, user_prompt
