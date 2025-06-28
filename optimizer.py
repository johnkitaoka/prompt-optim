"""
PromptOptimizer class for optimizing prompts using Anthropic Claude API.
"""

import os
import json
import yaml
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import anthropic
from dotenv import load_dotenv
from jinja2 import Template

# Load environment variables
load_dotenv()

class PromptOptimizer:
    """Main class for optimizing prompts using Claude API."""
    
    def __init__(self, api_key: Optional[str] = None, prompts_file: str = "prompts.yaml"):
        """Initialize the optimizer with Anthropic client."""
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-3-5-sonnet-latest"
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.prompts_file = prompts_file


    def _load_prompts_data(self) -> Dict[str, Any]:
        """Load the prompts data from YAML file."""
        with open(self.prompts_file, 'r') as f:
            return yaml.safe_load(f)
    
    def load_test_data(self, test_data_path: str) -> Dict[str, Any]:
        """Load test data from JSON file."""
        with open(test_data_path, 'r') as f:
            return json.load(f)
    
    def _make_api_call(self, content: str, max_tokens: int = 1024) -> str:
        """Make a single API call with retry logic for rate limits and transient errors."""
        for attempt in range(self.max_retries):
            try:
                message = self.client.messages.create(
                    max_tokens=max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    model=self.model,
                )
                return message.content[0].text.strip()
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

        # This should never be reached due to the raise statements above
        return ""

    def execute_prompt(self, prompt: str, test_input: str) -> str:
        """Execute prompt on a single test input using Claude API."""
        try:
            content = f"{prompt}\n\nInput: {test_input}"
            return self._make_api_call(content, max_tokens=1024)
        except Exception as e:
            print(f"Error executing prompt: {e}")
            return ""

    def evaluate_response(self, response: str, expected: str, task_description: str) -> bool:
        """Use Claude to evaluate if response matches expected output."""
        data = self._load_prompts_data()
        template = Template(data["evaluation_prompt"])
        evaluation_prompt = template.render(
            task_description=task_description,
            expected=expected,
            response=response
        )

        try:
            result = self._make_api_call(evaluation_prompt, max_tokens=10)
            return result.upper() == "YES"
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return False

    def analyze_failures(self, failures: List[Dict[str, str]], current_prompt: str) -> str:
        """Use Claude to analyze failures and identify prompt weaknesses."""
        failure_examples = "\n".join([
            f"Input: {f['input']}\nExpected: {f['expected']}\nActual: {f['actual']}\n"
            for f in failures[:5]  # Limit to first 5 failures
        ])

        data = self._load_prompts_data()
        template = Template(data["analysis_prompt"])
        analysis_prompt = template.render(
            current_prompt=current_prompt,
            failure_examples=failure_examples
        )

        try:
            return self._make_api_call(analysis_prompt, max_tokens=500)
        except Exception as e:
            print(f"Error analyzing failures: {e}")
            return "Unable to analyze failures"

    def refine_prompt(self, current_prompt: str, analysis: str, task_description: str) -> str:
        """Use Claude to rewrite the prompt based on analysis."""
        data = self._load_prompts_data()
        template = Template(data["refinement_prompt"])
        refinement_prompt = template.render(
            task_description=task_description,
            current_prompt=current_prompt,
            analysis=analysis
        )

        try:
            return self._make_api_call(refinement_prompt, max_tokens=1000)
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

    def save_prompt_iteration(self, prompt: str, score: float, iteration: int, analysis: str = ""):
        """Save prompt iteration to optimization log file."""
        timestamp = datetime.now().isoformat()

        iteration_data = {
            "iteration": iteration,
            "timestamp": timestamp,
            "prompt": prompt,
            "accuracy_score": score,
            "analysis": analysis
        }

        # Create log filename based on prompts file
        log_file = self.prompts_file.replace('.yaml', '_optimization_log.yaml')

        # Load existing log data or create new
        try:
            with open(log_file, 'r') as f:
                log_data = yaml.safe_load(f) or {"iterations": []}
        except FileNotFoundError:
            log_data = {"iterations": []}

        log_data["iterations"].append(iteration_data)

        # Save updated log data
        with open(log_file, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False, indent=2)

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
                log_file = self.prompts_file.replace('.yaml', '_optimization_log.yaml')
                print(f"Optimization history saved to: {log_file}")
                return current_prompt
            
            # If not the last iteration, refine the prompt
            if iteration < max_iterations and failures:
                print("Refining prompt...")
                current_prompt = self.refine_prompt(
                    current_prompt, analysis, test_data["task_description"]
                )
        
        print(f"\n⚠️  Max iterations reached. Final accuracy: {accuracy:.1%}")

        # Show log file location
        log_file = self.prompts_file.replace('.yaml', '_optimization_log.yaml')
        print(f"Optimization history saved to: {log_file}")

        return current_prompt
