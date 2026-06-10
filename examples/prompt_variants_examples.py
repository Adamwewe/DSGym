#!/usr/bin/env python3
"""
Examples of using DSGym prompt variants.

This file demonstrates how to use the 8 prompt variants for different scenarios:
- Quick iteration (lightweight)
- Standard evaluation (structured)
- Analytical planning (reasoning-only)
- Research optimization (expert-agentic)
"""

import sys
import importlib.util
from pathlib import Path

# Add DSGym to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import prompt variants directly to avoid dsgym.__init__ dependencies
spec = importlib.util.spec_from_file_location(
    "prompt_variants",
    Path(__file__).parent.parent / "dsgym/datasets/prompts/prompt_variants.py"
)
prompt_variants_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt_variants_module)

get_variant = prompt_variants_module.get_variant
list_variants = prompt_variants_module.list_variants
build_prompt_for_variant = prompt_variants_module.build_prompt_for_variant
get_variant_metadata = prompt_variants_module.get_variant_metadata


def example_1_list_variants():
    """Example 1: List all available prompt variants."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: List All Available Variants")
    print("=" * 80)
    
    variants = list_variants()
    print(f"\n{len(variants)} variants available:\n")
    
    for name, description in sorted(variants.items()):
        metadata = get_variant_metadata(name)
        print(f"  {name.upper()}")
        print(f"    Description: {description}")
        print(f"    Complexity:  {metadata['complexity_level']}/10")
        print(f"    DAG Strategy: {metadata['dag_strategy']}")
        print()


def example_2_variant_metadata():
    """Example 2: Get detailed metadata about a variant."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Get Detailed Metadata")
    print("=" * 80)
    
    variant_name = "expert-agentic"
    print(f"\nMetadata for '{variant_name}':\n")
    
    metadata = get_variant_metadata(variant_name)
    for key, value in metadata.items():
        if key == "system_prompt_template":
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")


def example_3_quick_iteration():
    """Example 3: Use lightweight variant for quick iteration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Quick Iteration (lightweight variant)")
    print("=" * 80)
    
    print("\nScenario: Testing a new model on QRData with minimal overhead")
    print("Command: dsgym eval --dataset qrdata --model gpt-4 --prompt-variant lightweight\n")
    
    # Build prompts programmatically
    system_prompt, user_prompt = build_prompt_for_variant(
        "lightweight",
        task_description="Analyze this dataset and answer the question: What are the top 3 features?",
        data_description="Dataset: 1000 rows, 50 numerical features, target variable 'score'"
    )
    
    print(f"System Prompt ({len(system_prompt)} chars):")
    print("-" * 80)
    print(system_prompt[:400] + "...\n")
    
    print(f"User Prompt ({len(user_prompt)} chars):")
    print("-" * 80)
    print(user_prompt[:400] + "...\n")


def example_4_analytical_planning():
    """Example 4: Use reasoning-only for analytical planning emphasis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Analytical Planning (reasoning-only variant)")
    print("=" * 80)
    
    print("\nScenario: DSBio task requiring careful analytical decomposition")
    print("Command: dsgym eval --dataset dsbio --model gpt-4 --prompt-variant reasoning-only\n")
    
    variant = get_variant("reasoning-only")
    print(f"Variant characteristics:")
    print(f"  - Enforces 5-8 node reasoning DAG BEFORE any code")
    print(f"  - Each node captures an analytical decision")
    print(f"  - No implementation DAG")
    print(f"  - Best for: Complex analytical problems\n")
    
    system_prompt, user_prompt = build_prompt_for_variant(
        "reasoning-only",
        task_description="Analyze single-cell RNA-seq data to identify cell types",
        data_description="AnnData object: 5000 cells × 2000 genes, gene expression matrix"
    )
    
    print("Reasoning DAG nodes will be generated as:")
    print("""
    [r1] Assess data quality: n_cells, n_genes, read counts distribution
    [r2] Determine QC thresholds: mitochondrial fraction, minimum genes
    [r3] Choose normalization: library-size, log, etc.
    [r4] Select dimensionality reduction: PCA components
    [r5] Choose clustering: Leiden resolution based on expected cell types
    [r6] Interpret markers and assign cell types
    [r7] Validate against known marker genes
    """)


def example_5_dual_dag():
    """Example 5: Use reasoning-implementation-strict for dual DAG approach."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Dual DAG (reasoning-implementation-strict)")
    print("=" * 80)
    
    print("\nScenario: Multi-step Kaggle competition with strict execution plan")
    print("Command: dsgym eval --dataset kaggle --model gpt-4 --prompt-variant reasoning-implementation-strict\n")
    
    metadata = get_variant_metadata("reasoning-implementation-strict")
    print(f"Variant characteristics:")
    print(f"  - Reasoning DAG: 5-15 analytical decision nodes")
    print(f"  - Implementation DAG: 1:1 mapping with reasoning nodes")
    print(f"  - Up to {metadata['revision_limit']} diagnostic revisions")
    print(f"  - Enforces concrete executable steps\n")
    
    print("Example DAG mapping:")
    print("""
    REASONING                              IMPLEMENTATION
    [r1] Assess feature distributions  → [i1] load data; compute mean/std/quantiles
    [r2] Detect missing values         → [i2] pd.isna(); determine imputation strategy
    [r3] Choose validation strategy    → [i3] StratifiedKFold or TimeSeriesSplit
    [r4] Select model family           → [i4] sklearn.ensemble.RandomForest or XGBoost
    [r5] Tune hyperparameters          → [i5] GridSearchCV with scoring metric
    [r6] Generate predictions          → [i6] predict() and format submission.csv
    """)


def example_6_adaptive_iteration():
    """Example 6: Use adaptive-revision for exploratory work."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Adaptive Iteration (adaptive-revision variant)")
    print("=" * 80)
    
    print("\nScenario: Novel research task with uncertain approach")
    print("Command: dsgym eval --dataset discoverybench --model gpt-4 --prompt-variant adaptive-revision\n")
    
    metadata = get_variant_metadata("adaptive-revision")
    print(f"Variant characteristics:")
    print(f"  - Initial reasoning DAG: 5-12 nodes (not fixed)")
    print(f"  - Plan can be revised after each execution step")
    print(f"  - Up to {metadata['revision_limit']} total revisions")
    print(f"  - Each revision captures learning\n")
    
    print("Example execution flow:")
    print("""
    INITIAL PLAN:
    [r1] Load and explore data
    [r2] Build baseline model
    [r3] Extract features
    [r4] Tune and validate
    
    AFTER STEP 1 (Observation: Data has severe class imbalance):
    [REVISION 1] Diagnosis: Can't use default validation. 
                 New approach: Use StratifiedKFold + SMOTE
                 Revised plan: [r1] → [r2'] SMOTE + rebalance → [r3] → [r4]
    
    AFTER STEP 3 (Observation: Features have high correlation):
    [REVISION 2] Diagnosis: Multicollinearity.
                 New approach: Feature selection or dimensionality reduction.
                 Revised plan: [r1] → [r2'] → [r3'] PCA → [r4]
    """)


def example_7_expert_agentic():
    """Example 7: Use expert-agentic for hard research problems."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Expert-Level Agentic (expert-agentic variant)")
    print("=" * 80)
    
    print("\nScenario: Hard Kaggle competition requiring state-of-the-art approach")
    print("Command: dsgym eval --dataset kaggle --model claude-opus --prompt-variant expert-agentic\n")
    
    metadata = get_variant_metadata("expert-agentic")
    print(f"Variant characteristics:")
    print(f"  - Complexity: {metadata['complexity_level']}/10 (maximum)")
    print(f"  - Implements complete agentic workflow")
    print(f"  - DAG Strategy: {metadata['dag_strategy']}")
    print(f"  - Max Revisions: {metadata['revision_limit']}")
    print(f"  - Based on 2026 ML best practices\n")
    
    print("Workflow structure:")
    print("""
    PHASE 0: PROBLEM ANCHORING
    - Identify primary metric (AUC, RMSE, accuracy, etc.)
    - List constraints (time, memory, data distribution)
    - Document assumptions (no distribution shift, no leakage, etc.)
    - Define success criteria (target metric value)
    
    PHASE 1: REASONING DAG (6-20 nodes)
    Types:
      - ASSESSMENT nodes: data quality, distributions, patterns
      - DECISION nodes: model class, preprocessing, validation
      - GATE nodes: explicit failure points
      - VALIDATION nodes: CV strategy, metrics, baselines
    
    PHASE 2: IMPLEMENTATION DAG (1:1 with reasoning)
    - Concrete executable steps
    - Primary library names (sklearn, xgboost, etc.)
    - Data flow specifications
    
    PHASE 3: EXECUTION WITH DIAGNOSTICS
    - Before: What, why, expected outcome
    - Execute and evaluate (report metrics)
    - Diagnose immediately if unexpected
    
    PHASE 4: SYSTEMATIC REVISION (up to 5)
    Triggers:
      - Exception (code fails)
      - Overfitting (train >> val)
      - Near-random (at chance level)
      - Data anomaly (unexpected distribution)
      - Format failure (submission validation)
    
    PHASE 5: FINAL ANSWER
    - Only after complete execution
    - Explain approach, revisions, trade-offs
    """)


def example_8_comparison():
    """Example 8: Compare multiple variants."""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Compare Variants")
    print("=" * 80)
    
    variants_to_compare = ["lightweight", "structured", "detailed", "expert-agentic"]
    print(f"\nComparing: {', '.join(variants_to_compare)}\n")
    
    print(f"{'Variant':<30} {'Complexity':<12} {'Length':<10} {'DAG Strategy':<30}")
    print("-" * 82)
    
    for name in variants_to_compare:
        metadata = get_variant_metadata(name)
        complexity = f"{metadata['complexity_level']}/10"
        length = metadata['length_class']
        dag = metadata['dag_strategy']
        print(f"{name:<30} {complexity:<12} {length:<10} {dag:<30}")
    
    print("\n\nRecommendations:")
    print("  - lightweight:        Use for quick baselines and rapid iteration")
    print("  - structured:         Use for standard benchmark tasks")
    print("  - detailed:           Use for complex problems requiring rich guidance")
    print("  - expert-agentic:     Use for hard research problems and optimization")


def example_9_cli_commands():
    """Example 9: Show useful CLI commands."""
    print("\n" + "=" * 80)
    print("EXAMPLE 9: CLI Commands")
    print("=" * 80)
    
    commands = [
        ("List all variants", "dsgym prompts list"),
        ("Show variant details", "dsgym prompts show expert-agentic"),
        ("Compare variants", "dsgym prompts compare lightweight structured expert-agentic"),
        ("Export variant", "dsgym prompts export expert-agentic --format markdown"),
        ("Eval with lightweight", "dsgym eval --dataset qrdata --model gpt-4 --prompt-variant lightweight"),
        ("Eval with expert-agentic", "dsgym eval --dataset kaggle --model gpt-4 --prompt-variant expert-agentic"),
        ("Eval with reasoning-only", "dsgym eval --dataset dsbio --model gpt-4 --prompt-variant reasoning-only"),
    ]
    
    for description, command in commands:
        print(f"\n{description}:")
        print(f"  $ {command}")


def main():
    """Run all examples."""
    examples = [
        example_1_list_variants,
        example_2_variant_metadata,
        example_3_quick_iteration,
        example_4_analytical_planning,
        example_5_dual_dag,
        example_6_adaptive_iteration,
        example_7_expert_agentic,
        example_8_comparison,
        example_9_cli_commands,
    ]
    
    print("\n" + "=" * 80)
    print("DSGym Prompt Variants: Usage Examples")
    print("=" * 80)
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠️  Example {i} error: {e}")
    
    print("\n" + "=" * 80)
    print("END OF EXAMPLES")
    print("=" * 80)
    print("\nFor more information, see docs/PROMPT_VARIANTS.md")


if __name__ == "__main__":
    main()
