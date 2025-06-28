# LLM Prompt Optimizer

A Python-based tool that automatically improves prompts through iterative testing and refinement using the Anthropic Claude API.

## 🚀 Features

- **Automated Optimization**: Iteratively improves prompts using Claude's analysis
- **Robust Error Handling**: Built-in retry logic for rate limits and API failures
- **Progress Tracking**: Saves all iterations with scores and analysis to YAML
- **Flexible Testing**: Support for any task with input-output test cases
- **CLI Interface**: Easy-to-use command-line interface
- **Comprehensive Logging**: Detailed progress reporting and error messages

## 📋 Requirements

- Python 3.7+
- Anthropic API key
- Dependencies: `anthropic`, `PyYAML`, `python-dotenv`

## 🛠️ Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up your Anthropic API key:**
```bash
cp .env.example .env
# Edit .env and add your actual API key
```

Or set the environment variable directly:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## 🎯 Quick Start

1. **Basic usage:**
```bash
python main.py \
  --test-data test_data.json \
  --initial-prompt "Classify the sentiment of this text" \
  --threshold 0.85 \
  --max-iterations 10
```

2. **Run tests (no API key required):**
```bash
python test_optimizer.py
```

3. **See examples and tips:**
```bash
python example_usage.py
```

## 📊 How It Works

1. **Execution**: Apply current prompt to all test inputs
2. **Evaluation**: Claude scores each response as correct/incorrect
3. **Analysis**: Claude identifies specific prompt weaknesses
4. **Refinement**: Claude rewrites the prompt based on analysis
5. **Iteration**: Repeat until target accuracy or max iterations reached

## 📁 File Structure

```
prompt-optim/
├── main.py              # Main execution script
├── optimizer.py         # PromptOptimizer class
├── test_data.json       # Sample test cases (sentiment analysis)
├── test_optimizer.py    # Test suite (no API key needed)
├── example_usage.py     # Examples and usage guide
├── prompts.yaml         # Generated: all prompts and optimization history
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variable template
└── README.md          # This file
```

## 🔧 CLI Options

```bash
python main.py [OPTIONS]

Required:
  --test-data PATH          JSON file with test cases
  --initial-prompt TEXT     Starting prompt to optimize

Optional:
  --threshold FLOAT         Target accuracy (0.0-1.0, default: 0.85)
  --max-iterations INT      Maximum iterations (default: 10)
  --api-key TEXT           Anthropic API key (or use env var)
  --help                   Show help message
```

## 📝 Test Data Format

Create a JSON file with your test cases:

```json
{
  "task_description": "Brief description of the task",
  "test_cases": [
    {
      "input": "Input text for the prompt",
      "expected_output": "Expected response"
    },
    {
      "input": "Another input example",
      "expected_output": "Another expected response"
    }
  ]
}
```

## 📈 Example Output

```
Starting prompt optimization...
Target accuracy: 85.0%
Max iterations: 10
--------------------------------------------------

Iteration 1:
Accuracy: 60.0% (4 failures)
Analysis: Prompt lacks specificity about output format...
Refining prompt...

Iteration 2:
Accuracy: 80.0% (2 failures)
Analysis: Need better handling of edge cases...
Refining prompt...

Iteration 3:
Accuracy: 90.0% (1 failures)

✅ Target accuracy reached! Final score: 90.0%
```

## 🎯 Optimization Tips

1. **Start Simple**: Begin with basic prompts and let the optimizer improve them
2. **Quality Test Data**: Include diverse examples and edge cases
3. **Reasonable Targets**: Set achievable accuracy goals (80-90% is often good)
4. **Monitor Progress**: Check `prompts.yaml` to see how prompts evolve
5. **Task Clarity**: Ensure your task description is clear and specific

## 🔍 Understanding Results

- **prompts.yaml**: Contains complete optimization history
- **Accuracy Score**: Percentage of test cases that passed
- **Analysis**: Claude's assessment of prompt weaknesses
- **Iterations**: Each refinement attempt with timestamp

## 🚨 Error Handling

The optimizer includes robust error handling for:
- **Rate Limits**: Automatic retry with exponential backoff
- **API Failures**: Connection errors and server issues
- **Invalid Responses**: Malformed API responses
- **File Errors**: Missing or corrupted test data files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `python test_optimizer.py`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Troubleshooting

**API Key Issues:**
```bash
# Check if API key is set
echo $ANTHROPIC_API_KEY

# Set API key for current session
export ANTHROPIC_API_KEY="your-key-here"
```

**Rate Limit Errors:**
- The optimizer automatically handles rate limits with exponential backoff
- Consider reducing the number of test cases for faster iteration

**Low Accuracy:**
- Ensure test data quality and consistency
- Try starting with a more specific initial prompt
- Increase max iterations if needed

**Import Errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```
