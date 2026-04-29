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

REASONING DAG:
{json.dumps(reasoning_dag, indent=2)}

EXAMPLE OUTPUT (follow schema only — do NOT copy this graph for your answer):
{json.dumps(REFERENCE_IMPLEMENTATION_DAG, indent=2)}

Produce the implementation DAG for the question above.
"""
    return _IMPLEMENTATION_SYSTEM_PROMPT, user_prompt
