from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple


@dataclass
class ModeEvaluationRecord:
    use_dags: bool
    trace_summary: Dict[str, Any]
    quality_summary: Dict[str, Any]
    harness_summary: Dict[str, Any]
    aggregated_metrics: Dict[str, Any]


class QualityMetricAdapter(Protocol):
    """Adapter that maps raw evaluator outputs to quality metrics.

    This allows external agentic frameworks to define domain-specific quality
    semantics (exact-match, reward score, simulator score, human preference, etc.)
    without changing the optimizer core.
    """

    def summarize(self, results: List[Any], aggregated_metrics: Dict[str, Any]) -> Dict[str, Any]:
        ...


class HarnessPolicyAdapter(Protocol):
    """Adapter that enforces and summarizes harness/policy compliance."""

    def summarize(self, results: List[Any], max_turns: int) -> Dict[str, Any]:
        ...


class ModeEvaluatorAdapter(Protocol):
    """Adapter that evaluates one mode (e.g., DAG on/off) for one trial.

    Non-DSGym integrations can implement this protocol to use the same
    multi-objective optimization loop while swapping agent/task/evaluator stacks.
    """

    def evaluate(
        self,
        *,
        config: Any,
        family: str,
        dataset: Any,
        tasks: List[Dict[str, Any]],
        use_dags: bool,
        max_turns: int,
        dag_validation_mode: str,
        trial_dir: Path,
        mode_tag: str,
        quality_adapter: QualityMetricAdapter,
        harness_adapter: HarnessPolicyAdapter,
    ) -> ModeEvaluationRecord:
        ...


class ResearchContextAdapter(Protocol):
    """Adapter that supplies dataset context and task banks for optimization.

    Returns:
    - dataset: dataset handle/object used by downstream evaluator(s)
    - family: dataset family name used for prompt-profile composition
    - task_bank: mapping {use_dags_bool: tasks}
    """

    def prepare_context(self, config: Any) -> Tuple[Any, str, Dict[bool, List[Dict[str, Any]]]]:
        ...


class TaskMutatorAdapter(Protocol):
    """Adapter that applies prompt/context/harness variants to task payloads."""

    def mutate(
        self,
        base_tasks: List[Dict[str, Any]],
        prompt_variant: Any,
        context_variant: Any,
        harness_variant: Any,
    ) -> List[Dict[str, Any]]:
        ...
