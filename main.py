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
    parser.add_argument("--initial-prompt", required=True, help="Initial prompt to optimize")
    parser.add_argument("--threshold", type=float, default=0.85, help="Target accuracy (0.0-1.0)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--prompts-file", default="prompts.yaml", help="YAML file to store prompts and iterations")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = PromptOptimizer(api_key=args.api_key, prompts_file=args.prompts_file)
    
    # Run optimization
    optimized_prompt = optimizer.optimize(
        test_data_path=args.test_data,
        initial_prompt=args.initial_prompt,
        threshold=args.threshold,
        max_iterations=args.max_iterations
    )
    
    print(f"\n🎯 Final optimized prompt:")
    print("-" * 50)
    print(optimized_prompt)
    print("-" * 50)
    log_file = args.prompts_file.replace('.yaml', '_optimization_log.yaml')
    print(f"System prompts: {args.prompts_file}")
    print(f"Optimization log: {log_file}")


if __name__ == "__main__":
    main()
