# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""

from unittest.mock import Mock

import pytest

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.reasoning import ReasoningParser
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

THINK_END_TOKEN = 99
REASONING_TOKEN_A = 10
REASONING_TOKEN_B = 11
JSON_TOKEN_A = 20
JSON_TOKEN_B = 21


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

        def mock_streaming(prefix, delta):
            return THINK_END_TOKEN in delta

        parser.is_reasoning_end_streaming = mock_streaming
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
        request.num_computed_tokens = 5
        request.num_output_placeholders = 0
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

    def test_update_reasoning_ended_with_new_token_ids_mid_batch(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        struct_req = mock_request_with_structured_output.structured_output_request
        struct_req.reasoning_ended = False

        new_token_ids = [REASONING_TOKEN_A, THINK_END_TOKEN, JSON_TOKEN_A]

        manager.update_reasoning_ended(
            mock_request_with_structured_output,
            new_token_ids=new_token_ids,
        )

        assert struct_req.reasoning_ended is True

    def test_update_reasoning_ended_no_end_found(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        struct_req = mock_request_with_structured_output.structured_output_request
        struct_req.reasoning_ended = False

        manager.update_reasoning_ended(
            mock_request_with_structured_output,
            new_token_ids=[REASONING_TOKEN_A, REASONING_TOKEN_B],
        )

        assert struct_req.reasoning_ended is False

    def test_identify_constrained_draft_tokens_reasoning_ends_mid_draft(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        struct_req = mock_request_with_structured_output.structured_output_request
        struct_req.reasoning_ended = False

        draft_tokens = [
            REASONING_TOKEN_A,
            THINK_END_TOKEN,
            JSON_TOKEN_A,
            JSON_TOKEN_B,
        ]
        unconstrained, constrained = manager.identify_constrained_draft_tokens(
            mock_request_with_structured_output, draft_tokens
        )

        # Tokens up to and including THINK_END_TOKEN are unconstrained
        assert unconstrained == [REASONING_TOKEN_A, THINK_END_TOKEN]
        # Tokens after the marker are constrained
        assert constrained == [JSON_TOKEN_A, JSON_TOKEN_B]

    def test_identify_constrained_draft_tokens_reasoning_already_ended(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        struct_req = mock_request_with_structured_output.structured_output_request
        struct_req.reasoning_ended = True

        # Include a spurious </think> — it should NOT cause a split
        draft_tokens = [
            JSON_TOKEN_A,
            THINK_END_TOKEN,
            JSON_TOKEN_B,
        ]
        unconstrained, constrained = manager.identify_constrained_draft_tokens(
            mock_request_with_structured_output, draft_tokens
        )

        # When reasoning_ended=True, ALL draft tokens should be constrained
        assert unconstrained == []
        assert constrained == [JSON_TOKEN_A, THINK_END_TOKEN, JSON_TOKEN_B]

    def test_validate_tokens_reasoning_aware_reasoning_ended(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        struct_req = mock_request_with_structured_output.structured_output_request
        struct_req.reasoning_ended = True
        # Mock validate_tokens to return a subset (simulating rejection)
        struct_req.grammar.validate_tokens = Mock(return_value=[JSON_TOKEN_A])

        draft_tokens = [JSON_TOKEN_A, JSON_TOKEN_B]
        result = manager.validate_tokens_reasoning_aware(
            mock_request_with_structured_output, draft_tokens
        )

        # validate_tokens was called with all draft tokens (since reasoning_ended=True)
        struct_req.grammar.validate_tokens.assert_called_once_with(
            [JSON_TOKEN_A, JSON_TOKEN_B]
        )
        # result must not contain draft token JSON_TOKEN_B, which didn't match grammar
        assert result == [JSON_TOKEN_A]

    def test_validate_tokens_reasoning_aware_reasoning_ends_mid_draft(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        struct_req = mock_request_with_structured_output.structured_output_request
        struct_req.reasoning_ended = False

        # Mock validate_tokens to return only the first valid token
        struct_req.grammar.validate_tokens = Mock(return_value=[JSON_TOKEN_A])

        draft_tokens = [
            REASONING_TOKEN_A,
            THINK_END_TOKEN,
            JSON_TOKEN_A,
            JSON_TOKEN_B,
        ]
        result = manager.validate_tokens_reasoning_aware(
            mock_request_with_structured_output, draft_tokens
        )

        # Only post-reasoning tokens should be validated (since reasoning_ended=False)
        struct_req.grammar.validate_tokens.assert_called_once_with(
            [JSON_TOKEN_A, JSON_TOKEN_B]
        )
        # Result: unconstrained prefix + validated suffix
        assert result == [REASONING_TOKEN_A, THINK_END_TOKEN, JSON_TOKEN_A]

    def _make_manager_for_bitmask_test(
        self, mock_vllm_config, mock_reasoning_parser, num_spec_tokens=5
    ):
        """Helper: create a StructuredOutputManager wired up for bitmask
        tests with a mock backend and pre-allocated bitmask tensor."""
        import torch

        mock_vllm_config.speculative_config = Mock()
        mock_vllm_config.speculative_config.num_speculative_tokens = num_spec_tokens
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        max_entries = mock_vllm_config.scheduler_config.max_num_seqs * (
            1 + num_spec_tokens
        )
        manager._grammar_bitmask = torch.zeros(max_entries, 1, dtype=torch.int32)
        manager.backend = Mock()
        return manager

    def _make_grammar_mock(self):
        """Helper: create a mock grammar that tracks calls."""
        grammar = Mock()
        grammar.is_terminated.return_value = False
        grammar.accept_tokens.return_value = True
        grammar.fill_bitmask = Mock()
        grammar.rollback = Mock()
        return grammar

    def test_grammar_bitmask_reasoning_ends_mid_speculation(
        self,
        mock_vllm_config,
        mock_reasoning_parser,
    ):
        """When reasoning_end appears mid-speculation, only post-reasoning
        tokens should be grammar-constrained and accepted (then rolled
        back)."""
        manager = self._make_manager_for_bitmask_test(
            mock_vllm_config, mock_reasoning_parser, num_spec_tokens=5
        )
        grammar = self._make_grammar_mock()

        mock_reasoning_parser.is_reasoning_end.return_value = False

        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = False
        request.structured_output_request.grammar = grammar
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3]

        req_id = "req-mid-spec"
        spec_tokens = [
            REASONING_TOKEN_A,
            THINK_END_TOKEN,
            JSON_TOKEN_A,
            JSON_TOKEN_B,
        ]

        result = manager.grammar_bitmask(
            requests={req_id: request},
            structured_output_request_ids=[req_id],
            scheduled_spec_decode_tokens={req_id: spec_tokens},
        )

        assert result is not None
        assert request.structured_output_request.reasoning_ended is False

        # Only post-reasoning tokens should have been accepted
        accepted_tokens = [
            call[0][1][0] for call in grammar.accept_tokens.call_args_list
        ]
        assert accepted_tokens == [JSON_TOKEN_A, JSON_TOKEN_B]

        # Grammar advancements should be rolled back
        if grammar.accept_tokens.call_count > 0:
            grammar.rollback.assert_called_once_with(grammar.accept_tokens.call_count)

    def test_grammar_bitmask_all_constrained_when_reasoning_ended(
        self,
        mock_vllm_config,
        mock_reasoning_parser,
    ):
        """After reasoning ended, ALL bitmask positions must be grammar-constrained"""
        manager = self._make_manager_for_bitmask_test(
            mock_vllm_config, mock_reasoning_parser, num_spec_tokens=5
        )
        grammar = self._make_grammar_mock()

        mock_reasoning_parser.is_reasoning_end.return_value = True

        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = True
        request.structured_output_request.grammar = grammar
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3]

        req_id = "req-root-cause-bitmask"
        # Spurious </think> at index 3 in the draft tokens
        spec_tokens = [
            JSON_TOKEN_A,
            JSON_TOKEN_B,
            REASONING_TOKEN_A,
            THINK_END_TOKEN,
            JSON_TOKEN_A,
        ]

        result = manager.grammar_bitmask(
            requests={req_id: request},
            structured_output_request_ids=[req_id],
            scheduled_spec_decode_tokens={req_id: spec_tokens},
        )

        assert result is not None

        # ALL spec_tokens should have been accepted, since reasoning_ended=True
        accepted_tokens = [
            call[0][1][0] for call in grammar.accept_tokens.call_args_list
        ]
        assert accepted_tokens == spec_tokens

        # All state advancements should be rolled back
        if grammar.accept_tokens.call_count > 0:
            grammar.rollback.assert_called_once_with(grammar.accept_tokens.call_count)
