#!/usr/bin/env python3
"""
Main execution script for the LLM Prompt Optimizer.
"""

import argparse
from optimizer import PromptOptimizer


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Optimize prompts using Claude API")
    parser.add_argument("--test-data", required=True, help="Path to test data JSON file")
    parser.add_argument("--initial-prompt-name", default="initial_prompt", help="Name of initial prompt in YAML file")
    parser.add_argument("--threshold", type=float, default=0.85, help="Target accuracy (0.0-1.0)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--prompts-file", default="prompts.yaml", help="YAML file to store prompts and iterations")

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = PromptOptimizer(api_key=args.api_key, prompts_file=args.prompts_file)

    # Run optimization
    final_prompt_name = optimizer.optimize(
        test_data_path=args.test_data,
        initial_prompt_name=args.initial_prompt_name,
        threshold=args.threshold,
        max_iterations=args.max_iterations
    )

    print(f"\n🎯 Final optimized prompt name: {final_prompt_name}")
    print("-" * 50)
    print(optimizer.prompts[final_prompt_name])
    print("-" * 50)
    print(f"All prompts saved to: {args.prompts_file}")


if __name__ == "__main__":
    main()
