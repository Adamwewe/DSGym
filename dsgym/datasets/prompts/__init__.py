"""
Prompts for DSGym datasets.

Centralized location for system prompts.
"""

from .system_prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_DSPREDICT,
    SYSTEM_PROMPT_MLEBENCH_TEMPLATE,
    SYSTEM_PROMPT_DATAMIND,
    SYSTEM_PROMPT_DEEPANALYZE,
    get_system_prompt
)
from .dag_prompts import (
    build_reasoning_prompt,
    build_implementation_prompt,
    REFERENCE_REASONING_DAG,
    REFERENCE_IMPLEMENTATION_DAG,
)

__all__ = [
    'SYSTEM_PROMPT',
    'SYSTEM_PROMPT_DSPREDICT',
    'SYSTEM_PROMPT_MLEBENCH_TEMPLATE',
    'SYSTEM_PROMPT_DATAMIND',
    'SYSTEM_PROMPT_DEEPANALYZE',
    'get_system_prompt',
    'build_reasoning_prompt',
    'build_implementation_prompt',
    'REFERENCE_REASONING_DAG',
    'REFERENCE_IMPLEMENTATION_DAG',
]
