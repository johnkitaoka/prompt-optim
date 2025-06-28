import os, json, yaml, time, anthropic
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv
from jinja2 import Template

load_dotenv()

class PromptOptimizer:
    def __init__(self, api_key: Optional[str] = None, prompts_file: str = "prompts.yaml"):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-3-5-sonnet-latest"
        self.max_retries = 3
        self.retry_delay = 1.0
        self.prompts_file = prompts_file
        self.prompts = self._load_prompts_data()
        
    def _load_prompts_data(self) -> Dict[str, Any]:
        with open(self.prompts_file, 'r') as f:
            return yaml.safe_load(f)

    def load_test_data(self, test_data_path: str) -> Dict[str, Any]:
        with open(test_data_path, 'r') as f:
            return json.load(f)
    
    def _make_api_call(self, content: str, max_tokens: int = 1024) -> str:
        for attempt in range(self.max_retries):
            try:
                message = self.client.messages.create(max_tokens=max_tokens, messages=[{"role": "user", "content": content}], model=self.model)
                return message.content[0].text.strip()
            except anthropic.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                else:
                    raise
            except anthropic.APIConnectionError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Connection error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    raise
            except anthropic.APIStatusError as e:
                if e.status_code >= 500 and attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Server error {e.status_code}, retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    raise
        return ""

    def _save_prompt_to_yaml(self, prompt_name: str, prompt_content: str):
        self.prompts[prompt_name] = prompt_content
        with open(self.prompts_file, 'w') as f:
            yaml.dump(self.prompts, f, default_flow_style=False, indent=2)

    def execute_prompt(self, prompt_name: str, test_input: str) -> str:
        prompt_template = Template(self.prompts[prompt_name])
        prompt_content = prompt_template.render(input=test_input)
        return self._make_api_call(prompt_content)

    def evaluate_response(self, response: str, expected: str, task_description: str) -> bool:
        evaluation_prompt = Template(self.prompts["evaluation_prompt"]).render(task_description=task_description, expected=expected, response=response)
        return self._make_api_call(evaluation_prompt, 10).upper() == "YES"

    def analyze_failures(self, failures: List[Dict[str, str]], current_prompt: str) -> str:
        failure_examples = "\n".join([f"Input: {f['input']}\nExpected: {f['expected']}\nActual: {f['actual']}\n" for f in failures[:5]])
        analysis_prompt = Template(self.prompts["analysis_prompt"]).render(current_prompt=current_prompt, failure_examples=failure_examples)
        return self._make_api_call(analysis_prompt, 500)

    def refine_prompt(self, current_prompt: str, analysis: str, task_description: str) -> str:
        refinement_prompt = Template(self.prompts["refinement_prompt"]).render(task_description=task_description, current_prompt=current_prompt, analysis=analysis)
        return self._make_api_call(refinement_prompt, 1000)

    def calculate_accuracy(self, test_data: Dict[str, Any], prompt_name: str) -> Tuple[float, List[Dict[str, str]]]:
        correct, failures = 0, []
        for case in test_data["test_cases"]:
            response = self.execute_prompt(prompt_name, case["input"])
            if self.evaluate_response(response, case["expected_output"], test_data["task_description"]):
                correct += 1
            else:
                failures.append({"input": case["input"], "expected": case["expected_output"], "actual": response})
        return correct / len(test_data["test_cases"]), failures

    def optimize(self, test_data_path: str, initial_prompt_name: str = "initial_prompt", threshold: float = 0.85, max_iterations: int = 10) -> str:
        print(f"Starting prompt optimization...\nTarget accuracy: {threshold:.1%}\nMax iterations: {max_iterations}\n{'-' * 50}")
        test_data = self.load_test_data(test_data_path)
        current_prompt_name = initial_prompt_name
        accuracy = 0.0

        for iteration in range(1, max_iterations + 1):
            print(f"\nIteration {iteration}:")
            accuracy, failures = self.calculate_accuracy(test_data, current_prompt_name)
            print(f"Accuracy: {accuracy:.1%} ({len(failures)} failures)")

            if failures:
                analysis = self.analyze_failures(failures, self.prompts[current_prompt_name])
                if analysis:
                    print(f"Analysis: {analysis[:100]}...")

                if iteration < max_iterations:
                    print("Refining prompt...")
                    refined_prompt = self.refine_prompt(self.prompts[current_prompt_name], analysis, test_data["task_description"])
                    new_prompt_name = f"prompt_{iteration}"
                    self._save_prompt_to_yaml(new_prompt_name, refined_prompt)
                    current_prompt_name = new_prompt_name

            if accuracy >= threshold:
                print(f"\n✅ Target accuracy reached! Final score: {accuracy:.1%}")
                return current_prompt_name

        print(f"\n⚠️  Max iterations reached. Final accuracy: {accuracy:.1%}")
        return current_prompt_name