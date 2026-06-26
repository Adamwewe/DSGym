"""
DAG generator for DSGym datasets.

Uses litellm (consistent with the rest of the repo) to call the model and produce
validated ReasoningDAG and ImplementationDAG objects from a DSBio sample.

Key improvements over sandbox-agents:
  - Takes reasoning DAG as a parameter instead of reading from disk.
  - Works with any sample, not a single hardcoded problem.
  - Uses two proper schemas so test_type / function_id are actually captured.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Union

import litellm

from dsgym.datasets.dag import ImplementationDAG, ReasoningDAG
from dsgym.datasets.prompts.dag_prompts import (
    build_implementation_prompt,
    build_reasoning_prompt,
)


def _supports_json_object_format(model: str) -> bool:
    """
    Return True if the model/provider accepts response_format={"type":"json_object"}.

    litellm's supports_response_schema() covers structured-output (json_schema) but
    json_object mode has its own compatibility surface.  We use a conservative
    allow-list: OpenAI and Azure OpenAI GPT models support it; everything else
    (Anthropic, Together, Mistral, …) relies on the system-prompt instruction alone.
    """
    model_lower = model.lower()
    # OpenAI direct or via litellm "openai/" prefix
    if model_lower.startswith(("gpt-", "openai/", "o1", "o3", "o4")):
        return True
    # Azure OpenAI deployments
    if model_lower.startswith("azure/"):
        return True
    return False


def _extract_json(raw: str) -> str:
    """
    Return the bare JSON string from a model response.

    Tries to parse the response directly first. If that fails, looks for a
    markdown code fence (```json ... ``` or ``` ... ```) and extracts its
    contents. Raises ValueError if neither attempt yields valid JSON.
    """
    raw = raw.strip()
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from model response: {raw[:200]!r}")


def generate_reasoning_dag(
    question: str,
    context: str,
    model: str = "gpt-4o",
) -> ReasoningDAG:
    """
    Generate a reasoning DAG for a DSBio question.

    Args:
        question: The DSBio question text.
        context:  Biological background / context for the task.
        model:    LiteLLM-compatible model name.

    Returns:
        Validated ReasoningDAG instance.
    """
    system_prompt, user_prompt = build_reasoning_prompt(question, context)

    kwargs: Dict[str, Any] = dict(
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
        raise RuntimeError("Empty response from model when generating ReasoningDAG")
    return ReasoningDAG.model_validate_json(_extract_json(raw))


def generate_implementation_dag(
    question: str,
    context: str,
    reasoning_dag: Union[ReasoningDAG, Dict[str, Any]],
    model: str = "gpt-4o",
) -> ImplementationDAG:
    """
    Generate an implementation DAG grounded in a reasoning DAG.

    Args:
        question:      The DSBio question text.
        context:       Biological background / context.
        reasoning_dag: Completed ReasoningDAG (or plain dict) to use as mental model.
        model:         LiteLLM-compatible model name.

    Returns:
        Validated ImplementationDAG instance.
    """
    # Accept either a ReasoningDAG model or a plain dict
    if isinstance(reasoning_dag, ReasoningDAG):
        reasoning_dict = json.loads(reasoning_dag.model_dump_json())
    else:
        reasoning_dict = reasoning_dag

    system_prompt, user_prompt = build_implementation_prompt(
        question, context, reasoning_dict
    )

    kwargs: Dict[str, Any] = dict(
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
        raise RuntimeError("Empty response from model when generating ImplementationDAG")

    return ImplementationDAG.model_validate_json(_extract_json(raw))


def generate_dags_for_sample(
    sample: Dict[str, Any],
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Convenience function: generate both reasoning and implementation DAGs
    for a single DSBio sample dict (as returned by DSBioDataset.load()).

    Args:
        sample: A DSBio sample dict (must contain 'extra_info' with 'question'
                and 'context' keys).
        model:  LiteLLM-compatible model name.

    Returns:
        Dict with keys 'reasoning' (ReasoningDAG) and 'implementation' (ImplementationDAG).
    """
    extra = sample.get("extra_info", {})
    question = extra.get("question", "")
    context = extra.get("context", "")

    if not question:
        raise ValueError("Sample 'extra_info' must contain a non-empty 'question' field")
    reasoning = generate_reasoning_dag(question, context, model=model)
    implementation = generate_implementation_dag(
        question, context, reasoning, model=model
    )

    return {"reasoning": reasoning, "implementation": implementation}
