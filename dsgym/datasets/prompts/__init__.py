"""
Prompts for DSGym datasets.

Centralized location for system prompts and prompt variants.
"""

from .system_prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_DAG,
    SYSTEM_PROMPT_DAG_DSPREDICT,
    SYSTEM_PROMPT_DSPREDICT,
    SYSTEM_PROMPT_MLEBENCH_TEMPLATE,
    SYSTEM_PROMPT_DATAMIND,
    SYSTEM_PROMPT_DEEPANALYZE,
    get_system_prompt
)
from .dag_prompts import (
    build_reasoning_prompt,
    build_implementation_prompt,
    build_reasoning_prompt_dspredict,
    build_implementation_prompt_dspredict,
    REFERENCE_REASONING_DAG,
    REFERENCE_IMPLEMENTATION_DAG,
    REFERENCE_REASONING_DAG_DSPREDICT,
    REFERENCE_IMPLEMENTATION_DAG_DSPREDICT,
)
from .prompt_variants import (
    PromptVariant,
    VARIANT_LIGHTWEIGHT,
    VARIANT_STRUCTURED,
    VARIANT_REASONING_ONLY,
    VARIANT_REASONING_FOCUSED,
    VARIANT_REASONING_IMPLEMENTATION_STRICT,
    VARIANT_DETAILED,
    VARIANT_ADAPTIVE_REVISION,
    VARIANT_EXPERT_AGENTIC,
    VARIANTS_REGISTRY,
    get_variant,
    list_variants,
    get_variant_metadata,
    build_prompt_for_variant,
)

__all__ = [
    'SYSTEM_PROMPT',
    'SYSTEM_PROMPT_DAG',
    'SYSTEM_PROMPT_DAG_DSPREDICT',
    'SYSTEM_PROMPT_DSPREDICT',
    'SYSTEM_PROMPT_MLEBENCH_TEMPLATE',
    'SYSTEM_PROMPT_DATAMIND',
    'SYSTEM_PROMPT_DEEPANALYZE',
    'get_system_prompt',
    'build_reasoning_prompt',
    'build_implementation_prompt',
    'build_reasoning_prompt_dspredict',
    'build_implementation_prompt_dspredict',
    'REFERENCE_REASONING_DAG',
    'REFERENCE_IMPLEMENTATION_DAG',
    'REFERENCE_REASONING_DAG_DSPREDICT',
    'REFERENCE_IMPLEMENTATION_DAG_DSPREDICT',
    # Prompt variants
    'PromptVariant',
    'VARIANT_LIGHTWEIGHT',
    'VARIANT_STRUCTURED',
    'VARIANT_REASONING_ONLY',
    'VARIANT_REASONING_FOCUSED',
    'VARIANT_REASONING_IMPLEMENTATION_STRICT',
    'VARIANT_DETAILED',
    'VARIANT_ADAPTIVE_REVISION',
    'VARIANT_EXPERT_AGENTIC',
    'VARIANTS_REGISTRY',
    'get_variant',
    'list_variants',
    'get_variant_metadata',
    'build_prompt_for_variant',
]
