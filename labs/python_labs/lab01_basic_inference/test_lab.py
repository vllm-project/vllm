"""
Lab 01: Basic vLLM Offline Inference - Tests

pytest tests to verify your implementation.
"""

import pytest
from typing import List
from unittest.mock import Mock, MagicMock, patch

# Import the solution for testing
# In practice, students would import from their starter.py
from solution import (
    create_llm,
    create_sampling_params,
    generate_completions,
    display_outputs
)


class TestLLMCreation:
    """Tests for LLM initialization."""

    @patch('solution.LLM')
    def test_create_llm_with_defaults(self, mock_llm_class):
        """Test that create_llm initializes with default parameters."""
        mock_instance = Mock()
        mock_llm_class.return_value = mock_instance

        result = create_llm()

        mock_llm_class.assert_called_once_with(
            model="facebook/opt-125m",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        assert result == mock_instance

    @patch('solution.LLM')
    def test_create_llm_with_custom_params(self, mock_llm_class):
        """Test that create_llm accepts custom parameters."""
        mock_instance = Mock()
        mock_llm_class.return_value = mock_instance

        result = create_llm(
            model_name="gpt2",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8
        )

        mock_llm_class.assert_called_once_with(
            model="gpt2",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
        )


class TestSamplingParams:
    """Tests for sampling parameter creation."""

    @patch('solution.SamplingParams')
    def test_create_sampling_params_defaults(self, mock_sampling_class):
        """Test sampling params creation with defaults."""
        mock_instance = Mock()
        mock_sampling_class.return_value = mock_instance

        result = create_sampling_params()

        mock_sampling_class.assert_called_once_with(
            temperature=0.8,
            top_p=0.95,
            max_tokens=100,
        )
        assert result == mock_instance

    @patch('solution.SamplingParams')
    def test_create_sampling_params_custom(self, mock_sampling_class):
        """Test sampling params with custom values."""
        mock_instance = Mock()
        mock_sampling_class.return_value = mock_instance

        result = create_sampling_params(
            temperature=0.5,
            top_p=0.9,
            max_tokens=50
        )

        mock_sampling_class.assert_called_once_with(
            temperature=0.5,
            top_p=0.9,
            max_tokens=50,
        )


class TestGenerateCompletions:
    """Tests for generation functionality."""

    def test_generate_completions(self):
        """Test that generate_completions calls LLM.generate correctly."""
        mock_llm = Mock()
        mock_outputs = [Mock(), Mock()]
        mock_llm.generate.return_value = mock_outputs

        prompts = ["Test prompt 1", "Test prompt 2"]
        mock_sampling_params = Mock()

        result = generate_completions(mock_llm, prompts, mock_sampling_params)

        mock_llm.generate.assert_called_once_with(prompts, mock_sampling_params)
        assert result == mock_outputs


class TestDisplayOutputs:
    """Tests for output display functionality."""

    def test_display_outputs(self, capsys):
        """Test that display_outputs prints correctly."""
        # Create mock outputs
        mock_output1 = Mock()
        mock_output1.prompt = "Test prompt"
        mock_output1.outputs = [Mock(text="Generated text")]

        outputs = [mock_output1]

        display_outputs(outputs)

        captured = capsys.readouterr()
        assert "Generated Completions" in captured.out
        assert "Test prompt" in captured.out
        assert "Generated text" in captured.out


class TestIntegration:
    """Integration tests."""

    @patch('solution.LLM')
    @patch('solution.SamplingParams')
    def test_full_pipeline(self, mock_sampling_class, mock_llm_class):
        """Test the full inference pipeline."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance

        mock_sampling_instance = Mock()
        mock_sampling_class.return_value = mock_sampling_instance

        mock_output = Mock()
        mock_output.prompt = "Test"
        mock_output.outputs = [Mock(text="Output")]
        mock_llm_instance.generate.return_value = [mock_output]

        # Run pipeline
        llm = create_llm()
        sampling_params = create_sampling_params()
        prompts = ["Test"]
        outputs = generate_completions(llm, prompts, sampling_params)

        # Verify
        assert len(outputs) == 1
        assert outputs[0].prompt == "Test"
        assert outputs[0].outputs[0].text == "Output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
