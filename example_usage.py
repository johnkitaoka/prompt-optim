#!/usr/bin/env python3
"""
Example usage of the prompt optimizer.
This demonstrates how to use the optimizer programmatically.
"""

import os
from optimizer import PromptOptimizer


def example_sentiment_analysis():
    """Example: Optimizing a sentiment analysis prompt."""
    print("🎯 Example: Sentiment Analysis Prompt Optimization")
    print("=" * 60)
    
    # Check if API key is available
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set. This is a demonstration of the API.")
        print("To run this example, set your API key:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        return
    
    # Initialize optimizer
    optimizer = PromptOptimizer()
    
    # Define initial prompt (intentionally simple/suboptimal)
    initial_prompt = "What is the sentiment?"
    
    print(f"Initial prompt: '{initial_prompt}'")
    print(f"Test data: test_data.json")
    print(f"Target accuracy: 85%")
    print(f"Max iterations: 5")
    print()
    
    try:
        # Run optimization
        optimized_prompt = optimizer.optimize(
            test_data_path="test_data.json",
            initial_prompt=initial_prompt,
            threshold=0.85,
            max_iterations=5
        )
        
        print("\n" + "=" * 60)
        print("🎉 Optimization Complete!")
        print("=" * 60)
        print(f"Optimized prompt:\n{optimized_prompt}")
        print("\nResults saved to: prompts.yaml")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")


def example_custom_task():
    """Example: Creating a custom task for optimization."""
    print("\n🎯 Example: Custom Task Setup")
    print("=" * 60)
    
    # Example of how to structure test data for different tasks
    custom_tasks = {
        "text_classification": {
            "task_description": "Classify text into categories",
            "test_cases": [
                {"input": "Breaking news: Stock market hits record high", "expected_output": "news"},
                {"input": "How to bake chocolate chip cookies", "expected_output": "recipe"},
                {"input": "Python tutorial for beginners", "expected_output": "tutorial"}
            ]
        },
        
        "question_answering": {
            "task_description": "Answer questions based on context",
            "test_cases": [
                {
                    "input": "Context: The capital of France is Paris. Question: What is the capital of France?",
                    "expected_output": "Paris"
                },
                {
                    "input": "Context: Python was created by Guido van Rossum. Question: Who created Python?",
                    "expected_output": "Guido van Rossum"
                }
            ]
        },
        
        "data_extraction": {
            "task_description": "Extract specific information from text",
            "test_cases": [
                {
                    "input": "Contact John Smith at john.smith@email.com or call (555) 123-4567",
                    "expected_output": "john.smith@email.com"
                },
                {
                    "input": "Meeting scheduled for March 15, 2024 at 2:00 PM",
                    "expected_output": "March 15, 2024"
                }
            ]
        }
    }
    
    print("Available task templates:")
    for task_name, task_data in custom_tasks.items():
        print(f"\n📋 {task_name.replace('_', ' ').title()}:")
        print(f"   Description: {task_data['task_description']}")
        print(f"   Test cases: {len(task_data['test_cases'])}")
        print(f"   Example input: {task_data['test_cases'][0]['input'][:50]}...")
    
    print(f"\n💡 To use these templates:")
    print(f"1. Save any template as a JSON file")
    print(f"2. Run: python main.py --test-data your_task.json --initial-prompt 'Your prompt'")


def show_optimization_tips():
    """Show tips for effective prompt optimization."""
    print("\n🎯 Prompt Optimization Tips")
    print("=" * 60)
    
    tips = [
        "Start Simple: Begin with a basic prompt and let the optimizer improve it",
        "Quality Test Data: Ensure your test cases cover edge cases and variations",
        "Reasonable Threshold: Set achievable accuracy targets (80-90% is often good)",
        "Iteration Limits: Start with 5-10 iterations, increase if needed",
        "Monitor Progress: Check prompts.yaml to see how prompts evolve",
        "Task Clarity: Make sure your task description is clear and specific",
        "Diverse Examples: Include varied test cases to avoid overfitting",
        "Output Format: Be consistent with expected output formats"
    ]
    
    for i, tip in enumerate(tips, 1):
        title, description = tip.split(": ", 1)
        print(f"{i:2d}. {title}: {description}")
    
    print(f"\n📊 Understanding Results:")
    print(f"   • Accuracy: Percentage of test cases passed")
    print(f"   • Analysis: Claude's assessment of prompt weaknesses")
    print(f"   • Iterations: Each refinement attempt")
    print(f"   • YAML File: Complete optimization history")


def main():
    """Run examples and show usage information."""
    print("🚀 Prompt Optimizer Examples & Usage Guide")
    print("=" * 60)
    
    # Show the main example
    example_sentiment_analysis()
    
    # Show custom task examples
    example_custom_task()
    
    # Show optimization tips
    show_optimization_tips()
    
    print("\n" + "=" * 60)
    print("📚 Additional Resources:")
    print("   • README.md: Setup and basic usage")
    print("   • test_optimizer.py: Run tests without API key")
    print("   • main.py --help: CLI options")
    print("=" * 60)


if __name__ == "__main__":
    main()
