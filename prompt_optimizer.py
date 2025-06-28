#!/usr/bin/env python3
"""
LLM Prompt Optimizer using Anthropic Claude API
Automatically improves prompts through iterative testing and refinement.
"""

import os
import json
import yaml
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PromptOptimizer:
    """Main class for optimizing prompts using Claude API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the optimizer with Anthropic client."""
        self.client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-3-5-sonnet-latest"
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
    def load_test_data(self, test_data_path: str) -> Dict[str, Any]:
        """Load test data from JSON file."""
        with open(test_data_path, 'r') as f:
            return json.load(f)
    
    def _api_call_with_retry(self, call_func, *args, **kwargs) -> Any:
        """Execute API call with retry logic for rate limits and transient errors."""
        for attempt in range(self.max_retries):
            try:
                return call_func(*args, **kwargs)
            except anthropic.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                else:
                    print(f"Rate limit error after {self.max_retries} attempts: {e}")
                    raise
            except anthropic.APIConnectionError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Connection error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Connection error after {self.max_retries} attempts: {e}")
                    raise
            except anthropic.APIStatusError as e:
                if e.status_code >= 500 and attempt < self.max_retries - 1:
                    # Retry on server errors
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Server error {e.status_code}, retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    print(f"API error: {e.status_code} - {e}")
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise

    def execute_prompt(self, prompt: str, test_input: str) -> str:
        """Execute prompt on a single test input using Claude API."""
        try:
            def make_call():
                return self.client.messages.create(
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt}\n\nInput: {test_input}"
                        }
                    ],
                    model=self.model,
                )

            message = self._api_call_with_retry(make_call)
            return message.content[0].text.strip()
        except Exception as e:
            print(f"Error executing prompt: {e}")
            return ""
    
    def evaluate_response(self, response: str, expected: str, task_description: str) -> bool:
        """Use Claude to evaluate if response matches expected output."""
        evaluation_prompt = f"""
        Task: {task_description}

        Expected output: {expected}
        Actual output: {response}

        Does the actual output correctly match the expected output for this task?
        Respond with only "YES" or "NO".
        """

        try:
            def make_call():
                return self.client.messages.create(
                    max_tokens=10,
                    messages=[
                        {
                            "role": "user",
                            "content": evaluation_prompt
                        }
                    ],
                    model=self.model,
                )

            message = self._api_call_with_retry(make_call)
            result = message.content[0].text.strip().upper()
            return result == "YES"
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return False
    
    def analyze_failures(self, failures: List[Dict[str, str]], current_prompt: str) -> str:
        """Use Claude to analyze failures and identify prompt weaknesses."""
        failure_examples = "\n".join([
            f"Input: {f['input']}\nExpected: {f['expected']}\nActual: {f['actual']}\n"
            for f in failures[:5]  # Limit to first 5 failures
        ])

        analysis_prompt = f"""
        Current prompt: {current_prompt}

        Failed test cases:
        {failure_examples}

        Analyze these failures and identify specific weaknesses in the prompt.
        What improvements are needed? Be specific and concise.
        """

        try:
            def make_call():
                return self.client.messages.create(
                    max_tokens=500,
                    messages=[
                        {
                            "role": "user",
                            "content": analysis_prompt
                        }
                    ],
                    model=self.model,
                )

            message = self._api_call_with_retry(make_call)
            return message.content[0].text.strip()
        except Exception as e:
            print(f"Error analyzing failures: {e}")
            return "Unable to analyze failures"
    
    def refine_prompt(self, current_prompt: str, analysis: str, task_description: str) -> str:
        """Use Claude to rewrite the prompt based on analysis."""
        refinement_prompt = f"""
        Task: {task_description}

        Current prompt: {current_prompt}

        Analysis of failures: {analysis}

        Rewrite the prompt to address these issues. Make it more specific, clear, and effective.
        Return only the improved prompt, nothing else.
        """

        try:
            def make_call():
                return self.client.messages.create(
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": refinement_prompt
                        }
                    ],
                    model=self.model,
                )

            message = self._api_call_with_retry(make_call)
            return message.content[0].text.strip()
        except Exception as e:
            print(f"Error refining prompt: {e}")
            return current_prompt
    
    def calculate_accuracy(self, test_data: Dict[str, Any], prompt: str) -> Tuple[float, List[Dict[str, str]]]:
        """Calculate accuracy score and return failures."""
        test_cases = test_data["test_cases"]
        task_description = test_data["task_description"]
        
        correct = 0
        failures = []
        
        for case in test_cases:
            response = self.execute_prompt(prompt, case["input"])
            is_correct = self.evaluate_response(response, case["expected_output"], task_description)
            
            if is_correct:
                correct += 1
            else:
                failures.append({
                    "input": case["input"],
                    "expected": case["expected_output"],
                    "actual": response
                })
        
        accuracy = correct / len(test_cases)
        return accuracy, failures
    
    def save_prompt_iteration(self, prompt: str, score: float, iteration: int, 
                            analysis: str = "", prompts_file: str = "prompts.yaml"):
        """Save prompt iteration to YAML file."""
        timestamp = datetime.now().isoformat()
        
        iteration_data = {
            "iteration": iteration,
            "timestamp": timestamp,
            "prompt": prompt,
            "accuracy_score": score,
            "analysis": analysis
        }
        
        # Load existing data or create new
        try:
            with open(prompts_file, 'r') as f:
                data = yaml.safe_load(f) or {"iterations": []}
        except FileNotFoundError:
            data = {"iterations": []}
        
        data["iterations"].append(iteration_data)
        
        with open(prompts_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def optimize(self, test_data_path: str, initial_prompt: str,
                threshold: float = 0.85, max_iterations: int = 10) -> str:
        """Main optimization loop."""
        print(f"Starting prompt optimization...")
        print(f"Target accuracy: {threshold:.1%}")
        print(f"Max iterations: {max_iterations}")
        print("-" * 50)

        # Load test data
        test_data = self.load_test_data(test_data_path)
        current_prompt = initial_prompt
        accuracy = 0.0  # Initialize accuracy

        for iteration in range(1, max_iterations + 1):
            print(f"\nIteration {iteration}:")

            # Calculate current accuracy
            accuracy, failures = self.calculate_accuracy(test_data, current_prompt)
            print(f"Accuracy: {accuracy:.1%} ({len(failures)} failures)")

            # Save current iteration
            analysis = ""
            if failures:
                analysis = self.analyze_failures(failures, current_prompt)
                print(f"Analysis: {analysis[:100]}...")

            self.save_prompt_iteration(current_prompt, accuracy, iteration, analysis)

            # Check if we've reached the threshold
            if accuracy >= threshold:
                print(f"\n✅ Target accuracy reached! Final score: {accuracy:.1%}")
                return current_prompt

            # If not the last iteration, refine the prompt
            if iteration < max_iterations and failures:
                print("Refining prompt...")
                current_prompt = self.refine_prompt(
                    current_prompt, analysis, test_data["task_description"]
                )

        print(f"\n⚠️  Max iterations reached. Final accuracy: {accuracy:.1%}")
        return current_prompt


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Optimize prompts using Claude API")
    parser.add_argument("--test-data", required=True, help="Path to test data JSON file")
    parser.add_argument("--initial-prompt", required=True, help="Initial prompt to optimize")
    parser.add_argument("--threshold", type=float, default=0.85, help="Target accuracy (0.0-1.0)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = PromptOptimizer(api_key=args.api_key)
    
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
    print(f"Results saved to prompts.yaml")


if __name__ == "__main__":
    main()
