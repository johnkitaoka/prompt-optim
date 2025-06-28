#!/usr/bin/env python3
"""
Test script for the prompt optimizer.
This script tests the core functionality without requiring API calls.
"""

import json
import yaml
import os
import tempfile
from unittest.mock import Mock, patch
from optimizer import PromptOptimizer


def test_load_test_data():
    """Test loading test data from JSON file."""
    print("Testing test data loading...")
    
    # Create temporary test data
    test_data = {
        "task_description": "Test task",
        "test_cases": [
            {"input": "test input", "expected_output": "test output"}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        optimizer = PromptOptimizer()
        loaded_data = optimizer.load_test_data(temp_file)
        assert loaded_data == test_data
        print("✅ Test data loading works correctly")
    finally:
        os.unlink(temp_file)


def test_yaml_storage():
    """Test YAML prompt storage functionality."""
    print("Testing YAML storage...")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_file = f.name

    try:
        optimizer = PromptOptimizer(prompts_file=temp_file)

        # Save a prompt iteration
        optimizer.save_prompt_iteration(
            prompt="Test prompt",
            score=0.75,
            iteration=1,
            analysis="Test analysis"
        )

        # Load and verify
        with open(temp_file, 'r') as f:
            data = yaml.safe_load(f)

        assert len(data["iterations"]) == 1
        assert data["iterations"][0]["prompt"] == "Test prompt"
        assert data["iterations"][0]["accuracy_score"] == 0.75
        assert data["iterations"][0]["iteration"] == 1
        assert "system_prompts" in data
        print("✅ YAML storage works correctly")

    finally:
        os.unlink(temp_file)


def test_mock_optimization():
    """Test optimization loop with mocked API calls."""
    print("Testing optimization loop with mocked API calls...")
    
    # Create test data
    test_data = {
        "task_description": "Sentiment analysis",
        "test_cases": [
            {"input": "I love this!", "expected_output": "positive"},
            {"input": "This is terrible", "expected_output": "negative"},
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        test_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        prompts_file = f.name
    
    try:
        optimizer = PromptOptimizer(prompts_file=prompts_file)

        # Mock the API calls
        with patch.object(optimizer, 'execute_prompt') as mock_execute, \
             patch.object(optimizer, 'evaluate_response') as mock_evaluate, \
             patch.object(optimizer, 'analyze_failures') as mock_analyze, \
             patch.object(optimizer, 'refine_prompt') as mock_refine:
            
            # Set up mock responses
            mock_execute.return_value = "positive"
            mock_evaluate.side_effect = [True, False, True, True]  # Improve over iterations
            mock_analyze.return_value = "Need to handle negative sentiment better"
            mock_refine.return_value = "Improved prompt for sentiment analysis"
            
            # Run optimization
            result = optimizer.optimize(
                test_data_path=test_file,
                initial_prompt="Classify the sentiment",
                threshold=0.8,
                max_iterations=2
            )
            
            assert result is not None
            print("✅ Optimization loop works correctly")
            
            # Verify YAML file was created
            with open(prompts_file, 'r') as f:
                data = yaml.safe_load(f)
            
            if data and "iterations" in data:
                print(f"✅ Created {len(data['iterations'])} iterations in YAML file")
            
    finally:
        os.unlink(test_file)
        if os.path.exists(prompts_file):
            os.unlink(prompts_file)


def test_error_handling():
    """Test error handling functionality."""
    print("Testing error handling...")
    
    optimizer = PromptOptimizer()
    
    # Test with invalid file path
    try:
        optimizer.load_test_data("nonexistent_file.json")
        assert False, "Should have raised an exception"
    except FileNotFoundError:
        print("✅ File not found error handled correctly")
    
    # Test execute_prompt with mock exception
    with patch.object(optimizer.client.messages, 'create') as mock_create:
        mock_create.side_effect = Exception("Test error")
        result = optimizer.execute_prompt("test prompt", "test input")
        assert result == ""
        print("✅ API error in execute_prompt handled correctly")


def main():
    """Run all tests."""
    print("🧪 Running prompt optimizer tests...")
    print("=" * 50)
    
    try:
        test_load_test_data()
        test_yaml_storage()
        test_mock_optimization()
        test_error_handling()
        
        print("=" * 50)
        print("🎉 All tests passed!")
        print("\nTo run the actual optimizer with Claude API:")
        print("1. Set your ANTHROPIC_API_KEY environment variable")
        print("2. Run: python main.py --test-data test_data.json --initial-prompt 'Your prompt here'")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
