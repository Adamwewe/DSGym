"""
DSGym Evaluation Metrics

A collection of metrics for evaluating data science task performance.
"""

from .base import BaseMetric, MetricResult
from .exact_match import ExactMatchMetric
from .code_execution import CodeExecutionMetric
from .equivalence_by_llm import EquivalenceByLLMMetric
from .semantic_similarity import SemanticSimilarityMetric
from .domain_specific import (
    DABStepMetric, 
    LLMScoreMetric,
    HMSScoreMetric
)
from .dag_consistency import (
    DAGConsistencyMetric,
    DAGEmbedder,
    DAGSequence,
    DAG,
    analyze_dag_consistency,
    extract_dags_from_trajectory,
    compute_dag_pair_metrics,
    compute_code_coverage,
)

__all__ = [
    "BaseMetric",
    "MetricResult", 
    "ExactMatchMetric",
    "CodeExecutionMetric",
    "EquivalenceByLLMMetric", 
    "SemanticSimilarityMetric",
    "DABStepMetric",
    "LLMScoreMetric",
    "HMSScoreMetric",
    # DAG consistency analysis
    "DAGConsistencyMetric",
    "DAGEmbedder",
    "DAGSequence",
    "DAG",
    "analyze_dag_consistency",
    "extract_dags_from_trajectory",
    "compute_dag_pair_metrics",
    "compute_code_coverage",
]