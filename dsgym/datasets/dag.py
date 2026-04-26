"""
DAG schema for reasoning and implementation graphs in DSGym.

Adapted from sandbox-agents with two distinct schemas:
  - ReasoningDAG  : abstract reasoning steps before implementation
  - ImplementationDAG : concrete execution steps with test_type / function_id
"""
from __future__ import annotations

from collections import defaultdict, deque
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Task types for DSBio (single-cell transcriptomics) workflows
# ---------------------------------------------------------------------------

class DSBioTaskType(str, Enum):
    data_exploration       = "data_exploration"
    quality_control        = "quality_control"
    qc_validation          = "qc_validation"
    preprocessing          = "preprocessing"
    dimensionality_reduction = "dimensionality_reduction"
    clustering             = "clustering"
    differential_expression = "differential_expression"
    pathway_analysis       = "pathway_analysis"
    cell_type_annotation   = "cell_type_annotation"
    statistical_analysis   = "statistical_analysis"
    result_validation      = "result_validation"


# ---------------------------------------------------------------------------
# Shared edge schema
# ---------------------------------------------------------------------------

class Edge(BaseModel):
    model_config = ConfigDict(extra="ignore")  #relaxed from extra="forbit"

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    weight: Optional[float] = None
    label: Optional[str] = None
    condition: Optional[str] = None


# ---------------------------------------------------------------------------
# Reasoning DAG  (abstract reasoning / validation steps)
# ---------------------------------------------------------------------------

class ReasoningNodeMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")  #relaxed from extra="forbit"


class ReasoningNode(BaseModel):
    model_config = ConfigDict(extra="ignore")  #relaxed from extra="forbit"

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    label: str = Field(min_length=1)
    task_type: DSBioTaskType
    metadata: ReasoningNodeMetadata = Field(default_factory=ReasoningNodeMetadata)


class ReasoningDAG(BaseModel):
    model_config = ConfigDict(extra="ignore")  #relaxed from extra="forbit"

    graph_id: str = Field(min_length=1)
    nodes: list[ReasoningNode] = Field(min_length=1)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_semantics(self) -> "ReasoningDAG":
        _validate_dag_structure(self.nodes, self.edges)
        return self


# ---------------------------------------------------------------------------
# Implementation DAG  (executable steps with test_type & function_id)
# ---------------------------------------------------------------------------

class ImplementationNodeMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")  #relaxed from extra="forbit"

    test_type: Optional[Literal["deterministic", "heuristic"]] = None
    function_id: Optional[str] = None


class ImplementationNode(BaseModel):
    model_config = ConfigDict(extra="ignore")  #relaxed from extra="forbit"

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    label: str = Field(min_length=1)
    task_type: DSBioTaskType
    metadata: ImplementationNodeMetadata = Field(default_factory=ImplementationNodeMetadata)


class ImplementationDAG(BaseModel):
    model_config = ConfigDict(extra="ignore")  #relaxed from extra="forbit"

    graph_id: str = Field(min_length=1)
    nodes: list[ImplementationNode] = Field(min_length=1)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_semantics(self) -> "ImplementationDAG":
        _validate_dag_structure(self.nodes, self.edges)
        return self


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _validate_dag_structure(nodes: list, edges: list[Edge]) -> None:
    """Validate that nodes are unique, edges reference valid nodes, and there are no cycles."""
    node_ids = [n.id for n in nodes]
    node_id_set = set(node_ids)

    if len(node_ids) != len(node_id_set):
        raise ValueError("Duplicate node ids are not allowed")

    for edge in edges:
        if edge.source not in node_id_set:
            raise ValueError(f"Unknown edge source: {edge.source}")
        if edge.target not in node_id_set:
            raise ValueError(f"Unknown edge target: {edge.target}")
        if edge.source == edge.target:
            raise ValueError(f"Self-loop detected on node: {edge.source}")

    if _has_cycle(node_id_set, edges):
        raise ValueError("Graph contains a cycle; expected a DAG")


def _has_cycle(node_ids: set[str], edges: list[Edge]) -> bool:
    """Kahn's algorithm to detect cycles in a directed graph."""
    indegree: dict[str, int] = {n: 0 for n in node_ids}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1

    queue: deque[str] = deque(n for n, d in indegree.items() if d == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for neighbor in adjacency[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return visited != len(node_ids)
