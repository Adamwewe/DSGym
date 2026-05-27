"""
DAG schema for reasoning and implementation graphs in DSGym.

Two distinct schemas:
  - ReasoningDAG  : abstract reasoning steps before implementation
  - ImplementationDAG : concrete execution steps
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Shared edge schema
# ---------------------------------------------------------------------------

class Edge(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Reasoning DAG  (abstract reasoning / validation steps)
# ---------------------------------------------------------------------------

class ReasoningNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    label: str = Field(min_length=1)


class ReasoningDAG(BaseModel):
    model_config = ConfigDict(extra="ignore")

    graph_id: str = Field(min_length=1)
    nodes: list[ReasoningNode] = Field(min_length=1)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_semantics(self) -> "ReasoningDAG":
        _validate_dag_structure(self.nodes, self.edges)
        return self


# ---------------------------------------------------------------------------
# Implementation DAG  (executable steps)
# ---------------------------------------------------------------------------

class ImplementationNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    label: str = Field(min_length=1)


class ImplementationDAG(BaseModel):
    model_config = ConfigDict(extra="ignore")

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


def _is_weakly_connected(node_ids: set[str], edges: list[Edge]) -> bool:
    """Check weak connectivity by treating edges as undirected."""
    if not node_ids:
        return True

    undirected: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in edges:
        undirected[edge.source].add(edge.target)
        undirected[edge.target].add(edge.source)

    start = next(iter(node_ids))
    seen = {start}
    queue: deque[str] = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in undirected[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    return len(seen) == len(node_ids)


def _is_topologically_listed(nodes: list, edges: list[Edge]) -> bool:
    """Return True if every edge source appears before its target in node list order."""
    order = {node.id: index for index, node in enumerate(nodes)}
    return all(order[edge.source] < order[edge.target] for edge in edges)


def _edge_index_set(nodes: list, edges: list[Edge]) -> set[tuple[int, int]]:
    """Convert edges into index-pair representation based on node list order."""
    order = {node.id: index for index, node in enumerate(nodes)}
    return {(order[edge.source], order[edge.target]) for edge in edges}


def validate_reasoning_dag(
    dag: ReasoningDAG,
    mode: str = "default",
    min_nodes: int = 3,
    max_nodes: int = 6,
) -> None:
    """Optional stricter validation for reasoning DAGs.

    Modes:
      - default / relaxed: preserve current behavior (acyclic + valid refs)
      - strict: additionally enforce node-count bounds, list-topological order,
        and weak connectivity.
    """
    normalized = (mode or "default").strip().lower()
    if normalized in {"default", "relaxed"}:
        return
    if normalized != "strict":
        raise ValueError(f"Unknown DAG validation mode: {mode}")

    node_count = len(dag.nodes)
    if node_count < min_nodes or node_count > max_nodes:
        raise ValueError(
            f"Strict DAG validation failed: node count must be in [{min_nodes}, {max_nodes}], got {node_count}"
        )

    if not _is_topologically_listed(dag.nodes, dag.edges):
        raise ValueError("Strict DAG validation failed: nodes must be listed in topological execution order")

    node_ids = {node.id for node in dag.nodes}
    if not _is_weakly_connected(node_ids, dag.edges):
        raise ValueError("Strict DAG validation failed: graph must be weakly connected")


def validate_implementation_dag(
    implementation_dag: ImplementationDAG,
    reasoning_dag: Optional[ReasoningDAG] = None,
    mode: str = "default",
    min_nodes: int = 3,
    max_nodes: int = 6,
) -> None:
    """Optional stricter validation for implementation DAGs.

    strict mode adds node bounds, list-topological order, weak connectivity,
    and (if reasoning_dag is provided) 1:1 structural mirroring checks.
    """
    normalized = (mode or "default").strip().lower()
    if normalized in {"default", "relaxed"}:
        return
    if normalized != "strict":
        raise ValueError(f"Unknown DAG validation mode: {mode}")

    node_count = len(implementation_dag.nodes)
    if node_count < min_nodes or node_count > max_nodes:
        raise ValueError(
            f"Strict DAG validation failed: node count must be in [{min_nodes}, {max_nodes}], got {node_count}"
        )

    if not _is_topologically_listed(implementation_dag.nodes, implementation_dag.edges):
        raise ValueError("Strict DAG validation failed: nodes must be listed in topological execution order")

    node_ids = {node.id for node in implementation_dag.nodes}
    if not _is_weakly_connected(node_ids, implementation_dag.edges):
        raise ValueError("Strict DAG validation failed: graph must be weakly connected")

    if reasoning_dag is None:
        return

    if len(implementation_dag.nodes) != len(reasoning_dag.nodes):
        raise ValueError(
            "Strict DAG validation failed: implementation/reasoning node counts must match (1:1 mapping)"
        )

    if len(implementation_dag.edges) != len(reasoning_dag.edges):
        raise ValueError(
            "Strict DAG validation failed: implementation/reasoning edge counts must match (structure mirror)"
        )

    impl_edges = _edge_index_set(implementation_dag.nodes, implementation_dag.edges)
    reason_edges = _edge_index_set(reasoning_dag.nodes, reasoning_dag.edges)
    if impl_edges != reason_edges:
        raise ValueError(
            "Strict DAG validation failed: implementation graph must mirror reasoning structure in node-list index space"
        )
