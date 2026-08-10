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
        request.request_id = "mock_req"
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

    # ------------------------------------------------------------------
    # advance_grammar — replaces the former should_advance tests.
    # The scheduler now calls this single method instead of
    # should_advance + trim_reasoning_for_advance + grammar.accept_tokens.
    # ------------------------------------------------------------------

    def test_advance_grammar_with_enable_in_reasoning(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """advance_grammar passes tokens directly when enable_in_reasoning."""
        manager_with_reasoner.enable_in_reasoning = True
        grammar = mock_request_with_structured_output.structured_output_request.grammar
        grammar.accept_tokens = Mock(return_value=True)

        new_tokens = [10, 20]
        result = manager_with_reasoner.advance_grammar(
            mock_request_with_structured_output, new_tokens
        )
        assert result is True
        grammar.accept_tokens.assert_called_once_with("mock_req", new_tokens)

    def test_advance_grammar_reasoning_not_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """advance_grammar skips FSM advancement while reasoning is active."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False
        grammar = structured_req.grammar
        grammar.accept_tokens = Mock(return_value=True)

        result = manager_with_reasoner.advance_grammar(
            mock_request_with_structured_output, [6, 7, 8]
        )

        # Grammar should NOT have been advanced.
        grammar.accept_tokens.assert_not_called()
        assert result is True  # no rejection

    def test_advance_grammar_reasoning_just_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """advance_grammar detects reasoning-end mid-step and trims prefix."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = True
        structured_req.reasoner = reasoner
        grammar = structured_req.grammar
        grammar.accept_tokens = Mock(return_value=True)

        result = manager_with_reasoner.advance_grammar(
            mock_request_with_structured_output, [6, 7, 8]
        )

        assert structured_req.reasoning_ended is True
        assert result is True

    def test_advance_grammar_reasoning_already_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """advance_grammar advances directly when reasoning already ended."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = True
        structured_req.reasoning_end_token_index = None  # prior step
        grammar = structured_req.grammar
        grammar.accept_tokens = Mock(return_value=True)

        new_tokens = [10, 20]
        result = manager_with_reasoner.advance_grammar(
            mock_request_with_structured_output, new_tokens
        )

        assert result is True
        grammar.accept_tokens.assert_called_once_with("mock_req", new_tokens)

    def test_advance_grammar_uses_new_token_ids_for_delta(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Regression for #43388: advance_grammar uses the exact multi-token
        delta rather than a placeholder-derived window.
        """
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False

        end_token_id = 248069
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming = Mock(
            side_effect=lambda input_ids, delta_ids: end_token_id in list(delta_ids)
        )
        structured_req.reasoner = reasoner
        structured_req.grammar.accept_tokens = Mock(return_value=True)

        new_token_ids = [9, 198, end_token_id, 271]
        mock_request_with_structured_output.all_token_ids = [
            1, 2, 3, 4, 5,
        ] + new_token_ids
        mock_request_with_structured_output.num_computed_tokens = 9
        mock_request_with_structured_output.num_output_placeholders = 1

        result = manager_with_reasoner.advance_grammar(
            mock_request_with_structured_output,
            new_token_ids=new_token_ids,
        )

        first_call = reasoner.is_reasoning_end_streaming.call_args_list[0]
        _, called_delta = first_call.args
        assert list(called_delta) == new_token_ids

        assert structured_req.reasoning_ended is True
        assert result is True

    def test_advance_grammar_trims_reasoning_prefix_for_json(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """advance_grammar trims the reasoning prefix before advancing the
        FSM at the boundary, so the grammar only sees post-reasoning tokens.
        """
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False
        structured_req.structured_output_key = (
            StructuredOutputOptions.JSON_OBJECT,
            "{}",
        )

        marker = 248069

        class MarkerReasoner:
            def __init__(self, *_, **__):
                pass

            def is_reasoning_end_streaming(self, input_ids, delta_ids):
                return marker in list(delta_ids)

        structured_req.reasoner = MarkerReasoner()

        new_token_ids = [9, 198, marker, 271, 5005]
        mock_request_with_structured_output.all_token_ids = [1, 2, 3] + new_token_ids

        result = manager_with_reasoner.advance_grammar(
            mock_request_with_structured_output,
            new_token_ids=new_token_ids,
        )

        # Grammar should have received only the trimmed suffix.
        structured_req.grammar.accept_tokens.assert_called_once_with(
            "mock_req", [271, 5005]
        )
        assert structured_req.reasoning_ended is True
        assert result is True
        assert structured_req.reasoning_end_token_index == 5

    # ------------------------------------------------------------------
    # filter_draft_tokens — replaces should_advance + validate_tokens
    # at the draft-token call sites.
    # ------------------------------------------------------------------

    def test_filter_draft_tokens_passthrough_while_reasoning(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """filter_draft_tokens returns drafts unchanged while reasoning
        is still in progress.
        """
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False
        grammar = structured_req.grammar
        grammar.validate_tokens = Mock(return_value=[1, 2, 3])

        spec_tokens = [10, 20, 30]
        result = manager_with_reasoner.filter_draft_tokens(
            mock_request_with_structured_output, spec_tokens
        )

        assert result == spec_tokens
        grammar.validate_tokens.assert_not_called()

    def test_filter_draft_tokens_validates_after_reasoning(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """filter_draft_tokens delegates to grammar.validate_tokens once
        reasoning has ended.
        """
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = True
        grammar = structured_req.grammar
        validated = [10, 20]
        grammar.validate_tokens = Mock(return_value=validated)

        result = manager_with_reasoner.filter_draft_tokens(
            mock_request_with_structured_output, [10, 20, 30]
        )

        assert result == validated
        grammar.validate_tokens.assert_called_once_with([10, 20, 30])

    def test_filter_draft_tokens_no_reasoner(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """filter_draft_tokens validates when no reasoner is configured."""
        manager = StructuredOutputManager(mock_vllm_config)
        grammar = mock_request_with_structured_output.structured_output_request.grammar
        grammar.validate_tokens = Mock(return_value=[10])

        result = manager.filter_draft_tokens(
            mock_request_with_structured_output, [10, 20]
        )

        assert result == [10]
        grammar.validate_tokens.assert_called_once_with([10, 20])
