"""
CLI command for managing and exploring prompt variants.

Provides:
  - List available prompt variants
  - Show detailed information about a variant
  - Export a prompt variant for inspection
"""

import argparse
import json
from typing import Dict, Any

from dsgym.datasets.prompts import (
    list_variants,
    get_variant_metadata,
    get_variant,
)


def add_prompts_parser(subparsers):
    """Add the 'prompts' subcommand to the main parser."""
    parser = subparsers.add_parser(
        "prompts",
        help="Explore and manage prompt variants",
        description="View available prompt variants, their characteristics, and usage."
    )
    
    subcommand = parser.add_subparsers(
        dest="prompts_action",
        help="Prompt variant action",
        metavar="ACTION"
    )
    
    # List action
    list_parser = subcommand.add_parser(
        "list",
        help="List all available prompt variants"
    )
    list_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    # Show action
    show_parser = subcommand.add_parser(
        "show",
        help="Show detailed information about a prompt variant"
    )
    show_parser.add_argument(
        "variant",
        help="Variant name (e.g., 'lightweight', 'expert-agentic')"
    )
    show_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    # Export action
    export_parser = subcommand.add_parser(
        "export",
        help="Export a prompt variant to a file"
    )
    export_parser.add_argument(
        "variant",
        help="Variant name to export"
    )
    export_parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)"
    )
    export_parser.add_argument(
        "--format",
        choices=["json", "text", "markdown"],
        default="json",
        help="Export format (default: json)"
    )
    
    # Compare action
    compare_parser = subcommand.add_parser(
        "compare",
        help="Compare multiple prompt variants"
    )
    compare_parser.add_argument(
        "variants",
        nargs="+",
        help="Variant names to compare (e.g., lightweight structured expert-agentic)"
    )
    compare_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.set_defaults(func=run_prompts)
    return parser


def run_prompts(args) -> int:
    """
    Execute the prompts command.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    if not hasattr(args, 'prompts_action') or not args.prompts_action:
        print("❌ Please specify an action: list, show, export, or compare")
        return 1
    
    if args.prompts_action == "list":
        return cmd_list(args)
    elif args.prompts_action == "show":
        return cmd_show(args)
    elif args.prompts_action == "export":
        return cmd_export(args)
    elif args.prompts_action == "compare":
        return cmd_compare(args)
    else:
        print(f"❌ Unknown action: {args.prompts_action}")
        return 1


def cmd_list(args) -> int:
    """List all available prompt variants."""
    variants = list_variants()
    
    if args.format == "json":
        output = {
            "variants": variants,
            "count": len(variants)
        }
        print(json.dumps(output, indent=2))
    else:
        print("📋 Available Prompt Variants")
        print("=" * 80)
        for name, description in variants.items():
            print(f"\n✓ {name}")
            print(f"  {description}")
        print(f"\n(Total: {len(variants)} variants)")
        print("\nUsage:")
        print("  dsgym prompts show <variant-name>    # Show details")
        print("  dsgym prompts export <variant-name>  # Export to file")
        print("  dsgym prompts compare <v1> <v2> ...  # Compare variants")
    
    return 0


def cmd_show(args) -> int:
    """Show detailed information about a prompt variant."""
    metadata = get_variant_metadata(args.variant)
    
    if metadata is None:
        print(f"❌ Unknown variant: {args.variant}")
        print("Available variants:", ", ".join(list_variants().keys()))
        return 1
    
    if args.format == "json":
        print(json.dumps(metadata, indent=2))
    else:
        _print_variant_details(metadata)
    
    return 0


def _print_variant_details(metadata: Dict[str, Any]) -> None:
    """Pretty-print variant metadata."""
    print(f"📌 Prompt Variant: {metadata['name']}")
    print("=" * 80)
    print(f"\nDescription:")
    print(f"  {metadata['description']}")
    print(f"\nCharacteristics:")
    print(f"  Version:                {metadata['version']}")
    print(f"  DAG Strategy:           {metadata['dag_strategy']}")
    print(f"  Length:                 {metadata['length_class']}")
    print(f"  Complexity Level:       {metadata['complexity_level']}/10")
    print(f"  Generality:             {metadata['generality']}")
    print(f"  Strict DAG Enforcement: {metadata['strict_dag_enforcement']}")
    if metadata['revision_strategy']:
        print(f"  Revision Strategy:      {metadata['revision_strategy']}")
        print(f"  Max Revisions:          {metadata['revision_limit']}")
    print()


def cmd_export(args) -> int:
    """Export a prompt variant to a file."""
    variant = get_variant(args.variant)
    
    if variant is None:
        print(f"❌ Unknown variant: {args.variant}")
        return 1
    
    if args.format == "json":
        output = _variant_to_dict(variant)
        content = json.dumps(output, indent=2)
    elif args.format == "markdown":
        content = _variant_to_markdown(variant)
    else:  # text
        content = _variant_to_text(variant)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(content)
        print(f"✓ Exported '{args.variant}' to {args.output}")
    else:
        print(content)
    
    return 0


def _variant_to_dict(variant) -> Dict[str, Any]:
    """Convert variant to dictionary."""
    return {
        "name": variant.name,
        "description": variant.description,
        "version": variant.version,
        "dag_strategy": variant.dag_strategy,
        "length_class": variant.length_class,
        "complexity_level": variant.complexity_level,
        "generality": variant.generality,
        "revision_strategy": variant.revision_strategy,
        "revision_limit": variant.revision_limit,
        "strict_dag_enforcement": variant.strict_dag_enforcement,
        "system_prompt_template": variant.system_prompt_template,
        "user_prompt_preamble": variant.user_prompt_preamble,
        "format_instructions": variant.format_instructions,
    }


def _variant_to_markdown(variant) -> str:
    """Convert variant to markdown."""
    lines = [
        f"# Prompt Variant: {variant.name}",
        "",
        f"**Version**: {variant.version}",
        f"**Description**: {variant.description}",
        "",
        "## Characteristics",
        f"- **DAG Strategy**: {variant.dag_strategy}",
        f"- **Length**: {variant.length_class}",
        f"- **Complexity**: {variant.complexity_level}/10",
        f"- **Generality**: {variant.generality}",
        f"- **Strict DAG Enforcement**: {variant.strict_dag_enforcement}",
        "",
    ]
    
    if variant.revision_strategy:
        lines.extend([
            "## Revision Strategy",
            f"- **Strategy**: {variant.revision_strategy}",
            f"- **Max Revisions**: {variant.revision_limit}",
            "",
        ])
    
    lines.extend([
        "## System Prompt",
        "```",
        variant.system_prompt_template,
        "```",
        "",
        "## Format Instructions",
        "```",
        variant.format_instructions,
        "```",
        "",
    ])
    
    return "\n".join(lines)


def _variant_to_text(variant) -> str:
    """Convert variant to plain text."""
    lines = [
        f"PROMPT VARIANT: {variant.name}",
        f"{'=' * 80}",
        f"\nDescription: {variant.description}",
        f"Version: {variant.version}",
        f"\nCharacteristics:",
        f"  DAG Strategy:           {variant.dag_strategy}",
        f"  Length:                 {variant.length_class}",
        f"  Complexity Level:       {variant.complexity_level}/10",
        f"  Generality:             {variant.generality}",
        f"  Strict DAG Enforcement: {variant.strict_dag_enforcement}",
    ]
    
    if variant.revision_strategy:
        lines.extend([
            f"  Revision Strategy:      {variant.revision_strategy}",
            f"  Max Revisions:          {variant.revision_limit}",
        ])
    
    lines.extend([
        f"\nSYSTEM PROMPT:",
        f"{'-' * 80}",
        variant.system_prompt_template,
        f"\n\nFORMAT INSTRUCTIONS:",
        f"{'-' * 80}",
        variant.format_instructions,
        "",
    ])
    
    return "\n".join(lines)


def cmd_compare(args) -> int:
    """Compare multiple prompt variants."""
    variants_data = []
    
    for variant_name in args.variants:
        metadata = get_variant_metadata(variant_name)
        if metadata is None:
            print(f"❌ Unknown variant: {variant_name}")
            return 1
        variants_data.append(metadata)
    
    if args.format == "json":
        output = {
            "comparison": variants_data,
            "count": len(variants_data)
        }
        print(json.dumps(output, indent=2))
    else:
        _print_variants_comparison(variants_data)
    
    return 0


def _print_variants_comparison(variants_data) -> None:
    """Pretty-print comparison of multiple variants."""
    print("📊 Prompt Variants Comparison")
    print("=" * 100)
    
    # Extract dimensions for comparison
    dimensions = {
        "DAG Strategy": [v["dag_strategy"] for v in variants_data],
        "Length": [v["length_class"] for v in variants_data],
        "Complexity": [v["complexity_level"] for v in variants_data],
        "Generality": [v["generality"] for v in variants_data],
        "Strict DAG": [v["strict_dag_enforcement"] for v in variants_data],
        "Max Revisions": [v["revision_limit"] for v in variants_data],
    }
    
    variant_names = [v["name"] for v in variants_data]
    max_name_len = max(len(name) for name in variant_names)
    
    # Print header
    header = "Dimension".ljust(20) + "  |  "
    header += "  |  ".join(name.ljust(max_name_len) for name in variant_names)
    print(header)
    print("-" * 100)
    
    # Print comparison rows
    for dim, values in dimensions.items():
        row = dim.ljust(20) + "  |  "
        row += "  |  ".join(str(v).ljust(max_name_len) for v in values)
        print(row)
    
    print("\nDetailed Descriptions:")
    print("-" * 100)
    for variant in variants_data:
        print(f"\n{variant['name']}:")
        print(f"  {variant['description']}")


__all__ = [
    "add_prompts_parser",
    "run_prompts",
]
