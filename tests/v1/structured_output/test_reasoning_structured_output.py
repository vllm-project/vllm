# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""

from unittest.mock import Mock, patch

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

    def test_should_advance_uses_new_token_ids_when_provided(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Regression for #43388: when caller passes new_token_ids, the
        reasoner sees the exact multi-token delta rather than the
        placeholder-derived window.
        """
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False

        end_token_id = 248069

        reasoner = MockReasoner(tokenizer=Mock())
        # Detection mirrors the real Qwen3 parser: end token in the delta.
        reasoner.is_reasoning_end_streaming = Mock(
            side_effect=lambda input_ids, delta_ids: end_token_id
            in list(delta_ids)
        )
        structured_req.reasoner = reasoner

        # Scenario from #43388: async + spec decode K=4, 4 tokens accepted
        # but only 1 placeholder remains (some drafts were rejected).
        # The placeholder math would yield delta=[271] and miss </think>.
        # Passing new_token_ids must override that.
        new_token_ids = [9, 198, end_token_id, 271]
        mock_request_with_structured_output.all_token_ids = (
            [1, 2, 3, 4, 5] + new_token_ids
        )
        mock_request_with_structured_output.num_computed_tokens = 9
        mock_request_with_structured_output.num_output_placeholders = 1

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=new_token_ids,
        )

        # First call to is_reasoning_end_streaming was with the full
        # new_token_ids (not the truncated placeholder window).
        first_call = reasoner.is_reasoning_end_streaming.call_args_list[0]
        _, called_delta = first_call.args
        assert list(called_delta) == new_token_ids

        assert structured_req.reasoning_ended is True
        assert result is False

    def test_should_advance_without_new_token_ids_falls_back(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Backward compat: callers that don't pass new_token_ids keep
        the original placeholder-derived delta window.
        """
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = False
        structured_req.reasoner = reasoner

        mock_request_with_structured_output.all_token_ids = [1, 2, 3, 4, 5]
        mock_request_with_structured_output.num_computed_tokens = 5
        mock_request_with_structured_output.num_output_placeholders = 2

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        # placeholder window: start = 5 - 2 = 3, delta = [4, 5]
        _, called_delta = reasoner.is_reasoning_end_streaming.call_args[0]
        assert list(called_delta) == [4, 5]
        assert result is False

    def test_should_advance_drains_post_marker_into_grammar(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """On the step that ends reasoning, post-marker content tokens are
        fed to the grammar so the next step's bitmask reflects the post-
        marker FSM state. Without this, the model can emit a duplicate
        opening token (e.g. "{{" for json_object).
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
        mock_request_with_structured_output.all_token_ids = (
            [1, 2, 3] + new_token_ids
        )

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=new_token_ids,
        )

        # grammar.accept_tokens was called exactly once with the post-marker
        # portion of new_token_ids, excluding the reasoning prefix and the
        # marker itself.
        accept_calls = structured_req.grammar.accept_tokens.call_args_list
        assert len(accept_calls) == 1
        _, fed_tokens = accept_calls[0].args
        assert fed_tokens == [271, 5005]

        assert structured_req.reasoning_ended is True
        # Deferred backend: still return False so the scheduler does not
        # also feed full new_token_ids (which would include reasoning).
        assert result is False

    def test_should_advance_no_postmarker_skips_grammar_accept(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """When the marker is the last token in new_token_ids, there is no
        post-marker tail to drain, so grammar.accept_tokens is not called.
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

        new_token_ids = [9, 198, marker]
        mock_request_with_structured_output.all_token_ids = (
            [1, 2, 3] + new_token_ids
        )

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=new_token_ids,
        )

        structured_req.grammar.accept_tokens.assert_not_called()
        assert structured_req.reasoning_ended is True
        assert result is False

    def test_should_advance_structural_tag_with_new_token_ids(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """The structural-tag path uses the exact current-step window."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False
        structured_req.structured_output_key = (
            StructuredOutputOptions.STRUCTURAL_TAG,
            "{}",
        )

        marker = 248069
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming = Mock(
            side_effect=lambda input_ids, delta_ids: marker in list(delta_ids)
        )
        structured_req.reasoner = reasoner
        manager_with_reasoner.vllm_config.speculative_config = Mock()

        new_token_ids = [9, marker, 271]
        mock_request_with_structured_output.all_token_ids = (
            [1, 2, 3] + new_token_ids
        )

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=new_token_ids,
        )

        assert result is True
        assert structured_req.reasoning_end_token_index == 4
        assert manager_with_reasoner.trim_reasoning_for_advance(
            mock_request_with_structured_output, new_token_ids
        ) == [271]

    def test_should_advance_tolerates_rejected_post_marker_tokens(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """A rejected speculative tail must not fail the request."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False
        structured_req.structured_output_key = (
            StructuredOutputOptions.JSON_OBJECT,
            "{}",
        )
        structured_req.grammar.accept_tokens.return_value = False

        marker = 248069
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming = Mock(
            side_effect=lambda input_ids, delta_ids: marker in list(delta_ids)
        )
        structured_req.reasoner = reasoner

        new_token_ids = [9, marker, 271]
        mock_request_with_structured_output.all_token_ids = (
            [1, 2, 3] + new_token_ids
        )

        with patch("vllm.v1.structured_output.logger.warning") as warning:
            result = manager_with_reasoner.should_advance(
                mock_request_with_structured_output,
                new_token_ids=new_token_ids,
            )

        assert result is False
        assert structured_req.reasoning_ended is True
        structured_req.grammar.accept_tokens.assert_called_once_with(
            "mock_req", [271]
        )
        warning.assert_called_once()
