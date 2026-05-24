# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""

from unittest.mock import Mock

import pytest

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputOptions


class MockReasoner:
    def __init__(self, tokenizer):
        self.is_reasoning_end = Mock(return_value=False)
        self.is_reasoning_end_streaming = Mock(return_value=False)


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
    def mock_request_with_structured_output(self):
        """Create a mock request with structured output."""
        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = None
        request.structured_output_request.grammar = Mock()
        request.structured_output_request.reasoning_parser_kwargs = None
        request.structured_output_request.reasoner = None
        request.structured_output_request.grammar.is_terminated = Mock(
            return_value=False
        )
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        request.all_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        request.num_computed_tokens = 5
        request.num_output_placeholders = 0
        return request

    @pytest.fixture
    def manager_with_reasoner(self, mock_vllm_config):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner_cls = MockReasoner
        manager.tokenizer = Mock()
        return manager

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
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_fill_bitmask when enable_in_reasoning is False."""
        # Keep enable_in_reasoning as False (default)
        config = manager_with_reasoner.vllm_config.structured_outputs_config
        assert config.enable_in_reasoning is False

        result = manager_with_reasoner.should_fill_bitmask(
            mock_request_with_structured_output
        )

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

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        # Should default to True when no reasoner
        assert result is True

    def test_should_fill_bitmask_uses_request_reasoning_parser_kwargs(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """Test request-level parser kwargs override the default reasoner."""

        class KwargReasoner:
            def __init__(self, tokenizer, chat_template_kwargs=None):
                self.chat_template_kwargs = chat_template_kwargs or {}

            def is_reasoning_end(self, input_ids):
                return not self.chat_template_kwargs.get("enable_thinking", False)

        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner_cls = KwargReasoner
        manager.tokenizer = Mock()

        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_parser_kwargs = {
            "chat_template_kwargs": {"enable_thinking": True}
        }

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        assert result is False
        assert (
            mock_request_with_structured_output.structured_output_request.reasoner
            is not None
        )

    def test_should_advance_with_enable_in_reasoning(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when enable_in_reasoning is True."""
        # Enable enable_in_reasoning
        manager_with_reasoner.enable_in_reasoning = True

        # Should always return True when enable_in_reasoning is enabled
        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )
        assert result is True

    def test_should_advance_reasoning_not_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when reasoning has not ended."""
        # Set reasoning as not ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        # Should return False since reasoning hasn't ended
        assert result is False

    def test_should_advance_reasoning_just_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when reasoning ends in current step."""
        # Set reasoning as not ended initially, but ends in this step
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = True
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoner = reasoner

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        # Should set reasoning_ended to True but return False for this step
        assert (
            mock_request_with_structured_output.structured_output_request.reasoning_ended
            is True
        )
        assert result is False

    def test_should_advance_reasoning_just_ended_with_spec_decode_structural_tag(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """When reasoning ends this step, advance immediately for structural
        tags with speculative decoding."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False
        structured_req.structured_output_key = (
            StructuredOutputOptions.STRUCTURAL_TAG,
            "{}",
        )
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = True
        structured_req.reasoner = reasoner

        manager_with_reasoner.vllm_config.speculative_config = Mock()

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        assert structured_req.reasoning_ended is True
        assert result is True

    def test_should_advance_reasoning_already_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when reasoning has already ended."""
        # Set reasoning as already ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = True

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        # Should return True since reasoning has ended
        assert result is True

    def test_should_advance_with_new_token_ids_detects_reasoning_end(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """When new_token_ids is passed containing the end token,
        reasoning_ended should be set regardless of placeholder arithmetic."""
        END_TOKEN = 999
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming = Mock(
            side_effect=lambda all_ids, delta: END_TOKEN in delta
        )
        structured_req.reasoner = reasoner

        # Simulate async + spec decode where placeholder math would produce
        # an empty delta window: num_computed_tokens == len(all_token_ids)
        mock_request_with_structured_output.all_token_ids = [
            1,
            2,
            3,
            END_TOKEN,
            10,
        ]
        mock_request_with_structured_output.num_computed_tokens = 5
        mock_request_with_structured_output.num_output_placeholders = 0

        new_token_ids = [9, 198, END_TOKEN, 271]

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output, new_token_ids=new_token_ids
        )

        assert structured_req.reasoning_ended is True
        # JSON type defers FSM advance to next step
        assert result is False
        # Verify we used new_token_ids, not the placeholder-derived delta
        reasoner.is_reasoning_end_streaming.assert_called_once()

    def test_should_advance_async_spec_decode_empty_delta_misses_end_token(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Reproduce the bug: without new_token_ids, async + spec decode
        placeholder arithmetic produces start == len(all_token_ids), yielding
        an empty delta that misses the reasoning end token.

        This test documents the known limitation of the fallback path."""
        END_TOKEN = 999
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False

        actual_deltas_seen = []

        def capture_delta(all_ids, delta):
            delta_list = list(delta)
            actual_deltas_seen.append(delta_list)
            return END_TOKEN in delta_list

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming = Mock(side_effect=capture_delta)
        structured_req.reasoner = reasoner

        # After async scheduling + spec decode token append:
        # 4 new tokens appended, num_computed_tokens adjusted to match
        mock_request_with_structured_output.all_token_ids = [
            1,
            2,
            3,
            4,
            5,
            9,
            198,
            END_TOKEN,
            271,
        ]
        mock_request_with_structured_output.num_computed_tokens = 9
        mock_request_with_structured_output.num_output_placeholders = 0

        # Fallback path (no new_token_ids) computes start = 9 - 0 = 9,
        # but len(all_token_ids) = 9, so islice yields nothing.
        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        assert result is False
        # The delta was empty, so the end token was missed
        assert actual_deltas_seen == [[]]
        assert structured_req.reasoning_ended is False

        # Now try with new_token_ids -- this should find the end token
        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=[9, 198, END_TOKEN, 271],
        )

        assert structured_req.reasoning_ended is True
        assert result is False  # JSON defers

    def test_should_advance_new_token_ids_structural_tag_spec_decode(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Structural tags with spec decode should return True on the same
        step, even when detected via new_token_ids."""
        END_TOKEN = 999
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False
        structured_req.structured_output_key = (
            StructuredOutputOptions.STRUCTURAL_TAG,
            "{}",
        )

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming = Mock(
            side_effect=lambda all_ids, delta: END_TOKEN in delta
        )
        structured_req.reasoner = reasoner

        manager_with_reasoner.vllm_config.speculative_config = Mock()

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=[END_TOKEN, 42],
        )

        assert structured_req.reasoning_ended is True
        assert result is True

    def test_should_advance_new_token_ids_no_end_token(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """When new_token_ids does not contain the end token,
        reasoning_ended should stay False."""
        END_TOKEN = 999
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming = Mock(
            side_effect=lambda all_ids, delta: END_TOKEN in delta
        )
        structured_req.reasoner = reasoner

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=[10, 20, 30],
        )

        assert structured_req.reasoning_ended is False
        assert result is False
