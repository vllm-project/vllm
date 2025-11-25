# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""

from unittest.mock import Mock

import pytest

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.reasoning import ReasoningParser
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


class TestReasoningStructuredOutput:
    """Test reasoning-aware structured output functionality."""

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock ModelConfig."""
        config = Mock(spec=ModelConfig)
        config.skip_tokenizer_init = True  # Skip tokenizer init to avoid network calls
        config.get_vocab_size = Mock(return_value=50000)
        # Add missing runner_type attribute that tokenizer initialization expects
        config.runner_type = "generate"
        # Add other attributes that tokenizer initialization might need
        config.tokenizer = "test-tokenizer"
        config.tokenizer_mode = "auto"
        config.trust_remote_code = False
        config.tokenizer_revision = None
        return config

    @pytest.fixture
    def mock_scheduler_config(self):
        """Create a mock SchedulerConfig."""
        config = Mock(spec=SchedulerConfig)
        config.max_num_seqs = 128
        return config

    @pytest.fixture
    def mock_vllm_config(self, mock_model_config, mock_scheduler_config):
        """Create a mock VllmConfig."""
        config = Mock(spec=VllmConfig)
        config.model_config = mock_model_config
        config.scheduler_config = mock_scheduler_config
        config.structured_outputs_config = Mock()
        config.structured_outputs_config.reasoning_parser = None
        config.structured_outputs_config.enable_in_reasoning = False
        config.speculative_config = None
        return config

    @pytest.fixture
    def mock_reasoning_parser(self):
        """Create a mock ReasoningParser."""
        parser = Mock(spec=ReasoningParser)
        parser.is_reasoning_end = Mock(return_value=False)
        return parser

    @pytest.fixture
    def mock_request_with_structured_output(self):
        """Create a mock request with structured output."""
        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = None
        request.structured_output_request.grammar = Mock()
        request.structured_output_request.grammar.is_terminated = Mock(
            return_value=False
        )
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        request.all_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        return request

    def test_should_fill_bitmask_with_enable_in_reasoning(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """Test should_fill_bitmask when enable_in_reasoning is True."""
        # Enable enable_in_reasoning
        mock_vllm_config.structured_outputs_config.enable_in_reasoning = True

        manager = StructuredOutputManager(mock_vllm_config)

        # Should always return True when enable_in_reasoning is enabled
        result = manager.should_fill_bitmask(mock_request_with_structured_output)
        assert result is True

    def test_should_fill_bitmask_without_enable_in_reasoning(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_fill_bitmask when enable_in_reasoning is False."""
        # Keep enable_in_reasoning as False (default)
        config = mock_vllm_config.structured_outputs_config
        assert config.enable_in_reasoning is False

        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Mock reasoning not ended
        mock_reasoning_parser.is_reasoning_end.return_value = False

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        # Should set reasoning_ended and return its value
        assert (
            mock_request_with_structured_output.structured_output_request.reasoning_ended
            is False
        )
        assert result is False

    def test_should_fill_bitmask_no_reasoner(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """Test should_fill_bitmask when no reasoner is configured."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = None

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        # Should default to True when no reasoner
        assert result is True

    def test_should_advance_with_enable_in_reasoning(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when enable_in_reasoning is True."""
        # Enable enable_in_reasoning
        mock_vllm_config.structured_outputs_config.enable_in_reasoning = True

        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Should always return True when enable_in_reasoning is enabled
        result = manager.should_advance(mock_request_with_structured_output)
        assert result is True

    def test_should_advance_reasoning_not_ended(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when reasoning has not ended."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Set reasoning as not ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False
        mock_reasoning_parser.is_reasoning_end.return_value = False

        result = manager.should_advance(mock_request_with_structured_output)

        # Should return False since reasoning hasn't ended
        assert result is False

    def test_should_advance_reasoning_just_ended(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when reasoning ends in current step."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Set reasoning as not ended initially, but ends in this step
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False
        mock_reasoning_parser.is_reasoning_end.return_value = True

        result = manager.should_advance(mock_request_with_structured_output)

        # Should set reasoning_ended to True but return False for this step
        assert (
            mock_request_with_structured_output.structured_output_request.reasoning_ended
            is True
        )
        assert result is False

    def test_should_advance_reasoning_already_ended(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when reasoning has already ended."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Set reasoning as already ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = True

        result = manager.should_advance(mock_request_with_structured_output)

        # Should return True since reasoning has ended
        assert result is True
