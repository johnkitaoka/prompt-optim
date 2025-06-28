# LLM Prompt Optimizer

Automatically improves prompts through iterative testing and refinement, in under 100 lines of code.

## 🚀 Features

- **Automated Optimization**: Iteratively improves prompts using Claude's analysis
- **YAML-Based**: All prompts stored in YAML with Jinja2 templating
- **No File Logging**: Clean operation without operational logs
- **Robust Error Handling**: Built-in retry logic for rate limits and API failures
- **CLI Interface**: Simple command-line interface

## 📋 Requirements

- Python 3.7+ • Anthropic API key • Dependencies: `anthropic`, `PyYAML`, `python-dotenv`, `jinja2`

## 🛠️ Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-api-key-here"
```

## 🎯 Usage

```bash
python main.py \
  --test-data test_data.json \
  --initial-prompt-name "initial_prompt" \
  --prompts-file example_prompt.yaml
```

## 📊 How It Works

Execute → Evaluate → Analyze → Refine → Repeat until target accuracy reached

## 📁 Files

- `main.py` - CLI interface
- `optimizer.py` - Core optimization logic
- `prompts.yaml` - System prompts for evaluation/analysis/refinement
- `example_prompt.yaml` - Your prompts with iterative improvements
- `test_data.json` - Input/output test cases

## 📝 Test Data Format

```json
{
  "task_description": "Sentiment analysis",
  "test_cases": [
    {"input": "I love this!", "expected_output": "positive"},
    {"input": "This is terrible", "expected_output": "negative"}
  ]
}
```

## 📈 Example Output

```
Iteration 1: Accuracy: 60.0% (4 failures)
Iteration 2: Accuracy: 80.0% (2 failures)
Iteration 3: Accuracy: 90.0% (1 failures)
✅ Target accuracy reached!
```

## 🎯 Key Features

- **Jinja2 Templating**: All prompts use `{{variable}}` syntax
- **YAML Storage**: Iterative prompts saved as `prompt_1`, `prompt_2`, etc.
- **No Logging**: Clean operation without operational log files

That's it! The optimizer automatically saves refined prompts as `prompt_1`, `prompt_2`, etc. in your YAML file.
