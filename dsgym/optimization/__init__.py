"""Prompt optimization utilities for DSGym."""

from .harness_evaluator import StrictHarnessEvaluator, StrictHarnessPolicy
from .interfaces import (
    HarnessPolicyAdapter,
    ModeEvaluationRecord,
    ModeEvaluatorAdapter,
    QualityMetricAdapter,
    ResearchContextAdapter,
    TaskMutatorAdapter,
)
from .prompt_nsga2 import PromptOptimizationConfig, run_prompt_nsga2
from .research_engine import (
    DefaultDSGymModeEvaluatorAdapter,
    DefaultDSGymContextAdapter,
    DefaultHarnessPolicyAdapter,
    DefaultQualityMetricAdapter,
    DefaultTaskMutatorAdapter,
    ResearchOptimizationConfig,
    run_research_nsga2,
)
from .research_reporting import ResearchTrialPoint, load_research_trial_points, group_stats, pareto_front_indices
from .statistics import BootstrapDeltaResult, bootstrap_mean_delta, summarize_direction

__all__ = [
    "StrictHarnessEvaluator",
    "StrictHarnessPolicy",
    "QualityMetricAdapter",
    "HarnessPolicyAdapter",
    "ModeEvaluatorAdapter",
    "ModeEvaluationRecord",
    "ResearchContextAdapter",
    "TaskMutatorAdapter",
    "PromptOptimizationConfig",
    "run_prompt_nsga2",
    "ResearchOptimizationConfig",
    "run_research_nsga2",
    "DefaultQualityMetricAdapter",
    "DefaultHarnessPolicyAdapter",
    "DefaultDSGymModeEvaluatorAdapter",
    "DefaultDSGymContextAdapter",
    "DefaultTaskMutatorAdapter",
    "ResearchTrialPoint",
    "load_research_trial_points",
    "group_stats",
    "pareto_front_indices",
    "BootstrapDeltaResult",
    "bootstrap_mean_delta",
    "summarize_direction",
]
