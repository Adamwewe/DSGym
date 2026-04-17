"""
DAG generator for DSGym datasets.

Uses OpenAI's structured-output (responses.parse) API to produce validated
ReasoningDAG and ImplementationDAG objects from a DSBio sample.

Key improvements over sandbox-agents:
  - Takes reasoning DAG as a parameter instead of reading from disk.
  - Works with any sample, not a single hardcoded problem.
  - Uses two proper schemas so test_type / function_id are actually captured.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Union

from openai import OpenAI

from dsgym.datasets.dag import ImplementationDAG, ReasoningDAG
from dsgym.datasets.prompts.dag_prompts import (
    build_implementation_prompt,
    build_reasoning_prompt,
)


def generate_reasoning_dag(
    question: str,
    context: str,
    model: str = "gpt-4o",
    client: OpenAI | None = None,
) -> ReasoningDAG:
    """
    Generate a reasoning DAG for a DSBio question.

    Args:
        question: The DSBio question text.
        context:  Biological background / context for the task.
        model:    OpenAI model name.
        client:   Optional pre-built OpenAI client (created if not provided).

    Returns:
        Validated ReasoningDAG instance.
    """
    client = client or OpenAI()
    system_prompt, user_prompt = build_reasoning_prompt(question, context)

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=ReasoningDAG,
    )

    if response.output_parsed is None:
        raise RuntimeError("No parsed ReasoningDAG returned by model")

    return response.output_parsed


def generate_implementation_dag(
    question: str,
    context: str,
    reasoning_dag: Union[ReasoningDAG, Dict[str, Any]],
    model: str = "gpt-4o",
    client: OpenAI | None = None,
) -> ImplementationDAG:
    """
    Generate an implementation DAG grounded in a reasoning DAG.

    Args:
        question:      The DSBio question text.
        context:       Biological background / context.
        reasoning_dag: Completed ReasoningDAG (or plain dict) to use as mental model.
        model:         OpenAI model name.
        client:        Optional pre-built OpenAI client.

    Returns:
        Validated ImplementationDAG instance.
    """
    client = client or OpenAI()

    # Accept either a ReasoningDAG model or a plain dict
    if isinstance(reasoning_dag, ReasoningDAG):
        reasoning_dict = json.loads(reasoning_dag.model_dump_json())
    else:
        reasoning_dict = reasoning_dag

    system_prompt, user_prompt = build_implementation_prompt(
        question, context, reasoning_dict
    )

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=ImplementationDAG,
    )

    if response.output_parsed is None:
        raise RuntimeError("No parsed ImplementationDAG returned by model")

    return response.output_parsed


def generate_dags_for_sample(
    sample: Dict[str, Any],
    model: str = "gpt-4o",
    client: OpenAI | None = None,
) -> Dict[str, Any]:
    """
    Convenience function: generate both reasoning and implementation DAGs
    for a single DSBio sample dict (as returned by DSBioDataset.load()).

    Args:
        sample: A DSBio sample dict (must contain 'extra_info' with 'question'
                and 'context' keys).
        model:  OpenAI model name.
        client: Optional pre-built OpenAI client.

    Returns:
        Dict with keys 'reasoning' (ReasoningDAG) and 'implementation' (ImplementationDAG).
    """
    extra = sample.get("extra_info", {})
    question = extra.get("question", "")
    context = extra.get("context", "")

    if not question:
        raise ValueError("Sample 'extra_info' must contain a non-empty 'question' field")

    client = client or OpenAI()

    reasoning = generate_reasoning_dag(question, context, model=model, client=client)
    implementation = generate_implementation_dag(
        question, context, reasoning, model=model, client=client
    )

    return {"reasoning": reasoning, "implementation": implementation}
