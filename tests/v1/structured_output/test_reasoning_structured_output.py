# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""

from unittest.mock import Mock

import pytest
import torch

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputOptions


class MockReasoner:
    def __init__(self, tokenizer):
        self.is_reasoning_end = Mock(return_value=False)
        self.is_reasoning_end_from_prompt = Mock(return_value=False)
        self.is_reasoning_end_streaming = Mock(return_value=False)
        self.extract_content_ids = Mock(return_value=[])


class AdaptiveMarkerReasoner(MockReasoner):
    reasoning_marker_token_ids = ((5, 6), (7, 8))


def _bitmask_allows(bitmask, token_id: int) -> bool:
    word_index, bit_index = divmod(token_id, 32)
    return bool((int(bitmask[word_index]) >> bit_index) & 1)


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
        request.structured_output_request.reasoning_end_token_index = None
        request.structured_output_request.deferred_grammar_start_index = None
        request.structured_output_request.reasoning_prompt_state_initialized = False
        request.structured_output_request.grammar = Mock()
        request.structured_output_request.reasoning_parser_kwargs = None
        request.structured_output_request.reasoner = None
        request.structured_output_request.grammar.is_terminated = Mock(
            return_value=False
        )
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        request.num_prompt_tokens = 5
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

        reasoner = (
            mock_request_with_structured_output.structured_output_request.reasoner
        )
        reasoner.is_reasoning_end_from_prompt.assert_called_once_with([1, 2, 3, 4, 5])
        reasoner.is_reasoning_end.assert_not_called()

    def test_should_fill_bitmask_preserves_deferred_prompt_state(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test adaptive reasoning remains deferred after prompt inspection."""
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end.return_value = True
        reasoner.is_reasoning_end_from_prompt.return_value = None
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoner = reasoner

        result = manager_with_reasoner.should_fill_bitmask(
            mock_request_with_structured_output
        )

        assert result is False
        assert structured_req.reasoning_ended is None
        reasoner.is_reasoning_end_from_prompt.assert_called_once_with([1, 2, 3, 4, 5])
        reasoner.is_reasoning_end.assert_not_called()

    def test_adaptive_initial_bitmask_unions_grammar_and_reasoning_markers(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Illegal direct tokens are masked before adaptive mode is resolved."""
        structured_req = mock_request_with_structured_output.structured_output_request
        reasoner = AdaptiveMarkerReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_from_prompt.return_value = None
        structured_req.reasoner = reasoner
        structured_req.reasoning_ended = None

        prompt_token_ids = [1, 2, 3]
        mock_request_with_structured_output.prompt_token_ids = prompt_token_ids
        mock_request_with_structured_output.num_prompt_tokens = len(prompt_token_ids)
        mock_request_with_structured_output.all_token_ids = prompt_token_ids

        grammar_token = 10
        structured_req.grammar.fill_bitmask.side_effect = lambda bitmask, index: (
            bitmask[index, 0].bitwise_or_(1 << grammar_token)
        )
        manager_with_reasoner.vllm_config.num_speculative_tokens = 0
        manager_with_reasoner.vllm_config.model_config.is_diffusion = False
        manager_with_reasoner.backend = Mock()
        manager_with_reasoner.backend.allocate_token_bitmask.return_value = torch.zeros(
            (1, 2), dtype=torch.int32
        )

        request_id = mock_request_with_structured_output.request_id
        (bitmask,) = manager_with_reasoner.grammar_bitmask(
            requests={request_id: mock_request_with_structured_output},
            structured_output_request_ids=[request_id],
            scheduled_spec_decode_tokens={},
        )

        assert _bitmask_allows(bitmask, grammar_token)
        assert _bitmask_allows(bitmask, 5)
        assert _bitmask_allows(bitmask, 7)
        assert not _bitmask_allows(bitmask, 11)

    def test_adaptive_invalid_grammar_prefix_only_allows_marker_continuation(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """A grammar-invalid marker prefix cannot branch into direct content."""
        structured_req = mock_request_with_structured_output.structured_output_request
        reasoner = AdaptiveMarkerReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_from_prompt.return_value = None
        structured_req.reasoner = reasoner
        structured_req.reasoning_ended = None
        structured_req.grammar.validate_tokens.return_value = []

        prompt_token_ids = [1, 2, 3]
        mock_request_with_structured_output.prompt_token_ids = prompt_token_ids
        mock_request_with_structured_output.num_prompt_tokens = len(prompt_token_ids)
        mock_request_with_structured_output.all_token_ids = [*prompt_token_ids, 5]

        manager_with_reasoner.vllm_config.num_speculative_tokens = 0
        manager_with_reasoner.vllm_config.model_config.is_diffusion = False
        manager_with_reasoner.backend = Mock()
        manager_with_reasoner.backend.allocate_token_bitmask.return_value = torch.zeros(
            (1, 2), dtype=torch.int32
        )

        request_id = mock_request_with_structured_output.request_id
        (bitmask,) = manager_with_reasoner.grammar_bitmask(
            requests={request_id: mock_request_with_structured_output},
            structured_output_request_ids=[request_id],
            scheduled_spec_decode_tokens={},
        )

        assert _bitmask_allows(bitmask, 6)
        assert not _bitmask_allows(bitmask, 10)
        structured_req.grammar.fill_bitmask.assert_not_called()

    def test_should_fill_bitmask_initializes_forwarded_open_state(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Engine parsers inspect open continuations despite a forwarded state."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False

        result = manager_with_reasoner.should_fill_bitmask(
            mock_request_with_structured_output
        )

        assert result is False
        assert structured_req.reasoning_ended is False
        assert structured_req.reasoning_prompt_state_initialized is True
        structured_req.reasoner.is_reasoning_end_from_prompt.assert_called_once_with(
            [1, 2, 3, 4, 5]
        )

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

            def is_reasoning_end_from_prompt(self, prompt_token_ids):
                return self.is_reasoning_end(prompt_token_ids)

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

    def test_should_advance_ignores_prompt_reasoning_markers(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Adaptive reasoning detection only inspects generated tokens."""
        start_marker = 101
        end_marker = 102
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = None

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_from_prompt.return_value = None
        reasoner.is_reasoning_end_streaming = Mock(
            side_effect=lambda input_ids, delta_ids: end_marker in list(input_ids)
        )
        structured_req.reasoner = reasoner

        mock_request_with_structured_output.prompt_token_ids = [
            start_marker,
            end_marker,
        ]
        mock_request_with_structured_output.num_prompt_tokens = 2
        mock_request_with_structured_output.all_token_ids = [
            start_marker,
            end_marker,
            start_marker,
        ]

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=[start_marker],
        )

        assert result is False
        assert structured_req.reasoning_ended is None
        input_ids, delta_ids = reasoner.is_reasoning_end_streaming.call_args.args
        assert list(input_ids) == [start_marker]
        assert list(delta_ids) == [start_marker]

    def test_should_advance_replays_adaptive_direct_content(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """The token that resolves adaptive mode as content reaches the grammar."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = None

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_from_prompt.return_value = None
        reasoner.is_reasoning_end_streaming.return_value = True
        reasoner.extract_content_ids.side_effect = lambda token_ids: list(token_ids)
        structured_req.reasoner = reasoner

        prompt_token_ids = [1, 2, 3]
        direct_content_token = 400
        mock_request_with_structured_output.prompt_token_ids = prompt_token_ids
        mock_request_with_structured_output.num_prompt_tokens = len(prompt_token_ids)
        mock_request_with_structured_output.all_token_ids = [
            *prompt_token_ids,
            direct_content_token,
        ]

        assert manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=[direct_content_token],
        )
        assert structured_req.reasoning_ended is True
        assert structured_req.reasoning_end_token_index is None
        assert structured_req.deferred_grammar_start_index == len(prompt_token_ids)
        assert manager_with_reasoner.trim_reasoning_for_advance(
            mock_request_with_structured_output, [direct_content_token]
        ) == [direct_content_token]
        assert structured_req.deferred_grammar_start_index is None

    def test_should_advance_replays_ambiguous_adaptive_prefix(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Earlier unconstrained prefix tokens are replayed with direct content."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = None

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_from_prompt.return_value = None
        reasoner.is_reasoning_end_streaming.return_value = True
        reasoner.extract_content_ids.side_effect = lambda token_ids: list(token_ids)
        structured_req.reasoner = reasoner

        prompt_token_ids = [1, 2, 3]
        output_token_ids = [401, 402]
        mock_request_with_structured_output.prompt_token_ids = prompt_token_ids
        mock_request_with_structured_output.num_prompt_tokens = len(prompt_token_ids)
        mock_request_with_structured_output.all_token_ids = [
            *prompt_token_ids,
            *output_token_ids,
        ]

        assert manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=[output_token_ids[-1]],
        )
        assert (
            manager_with_reasoner.trim_reasoning_for_advance(
                mock_request_with_structured_output, [output_token_ids[-1]]
            )
            == output_token_ids
        )

    def test_speculative_reasoning_detection_ignores_prompt_markers(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Speculative draft simulation starts after the prompt boundary."""
        start_marker = 101
        end_marker = 102
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = None

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_from_prompt.return_value = None
        reasoner.is_reasoning_end_streaming = Mock(
            side_effect=lambda input_ids, delta_ids: end_marker in list(input_ids)
        )
        structured_req.reasoner = reasoner

        mock_request_with_structured_output.prompt_token_ids = [
            start_marker,
            end_marker,
        ]
        mock_request_with_structured_output.num_prompt_tokens = 2
        mock_request_with_structured_output.all_token_ids = [
            start_marker,
            end_marker,
        ]

        manager_with_reasoner.vllm_config.num_speculative_tokens = 1
        manager_with_reasoner.vllm_config.model_config.is_diffusion = False
        manager_with_reasoner.backend = Mock()
        manager_with_reasoner.backend.allocate_token_bitmask.return_value = torch.zeros(
            (2, 1), dtype=torch.int32
        )

        request_id = mock_request_with_structured_output.request_id
        manager_with_reasoner.grammar_bitmask(
            requests={request_id: mock_request_with_structured_output},
            structured_output_request_ids=[request_id],
            scheduled_spec_decode_tokens={request_id: [start_marker]},
        )

        input_ids, delta_ids = reasoner.is_reasoning_end_streaming.call_args.args
        assert list(input_ids) == [start_marker]
        assert list(delta_ids) == [start_marker]
        structured_req.grammar.fill_bitmask.assert_not_called()

    def test_speculative_adaptive_direct_content_advances_grammar(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Later speculative rows include direct adaptive content state."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = None

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_from_prompt.return_value = None
        reasoner.is_reasoning_end_streaming.return_value = True
        reasoner.extract_content_ids.side_effect = lambda token_ids: list(token_ids)
        structured_req.reasoner = reasoner
        structured_req.grammar.validate_tokens.side_effect = lambda token_ids: list(
            token_ids
        )
        structured_req.grammar.accept_tokens.return_value = True

        prompt_token_ids = [1, 2, 3]
        mock_request_with_structured_output.prompt_token_ids = prompt_token_ids
        mock_request_with_structured_output.num_prompt_tokens = len(prompt_token_ids)
        mock_request_with_structured_output.all_token_ids = prompt_token_ids

        manager_with_reasoner.vllm_config.num_speculative_tokens = 2
        manager_with_reasoner.vllm_config.model_config.is_diffusion = False
        manager_with_reasoner.backend = Mock()
        manager_with_reasoner.backend.allocate_token_bitmask.return_value = torch.zeros(
            (3, 1), dtype=torch.int32
        )

        request_id = mock_request_with_structured_output.request_id
        direct_tokens = [501, 502]
        manager_with_reasoner.grammar_bitmask(
            requests={request_id: mock_request_with_structured_output},
            structured_output_request_ids=[request_id],
            scheduled_spec_decode_tokens={request_id: direct_tokens},
        )

        accepted_token_batches = [
            mock_call.args[1]
            for mock_call in structured_req.grammar.accept_tokens.call_args_list
        ]
        assert accepted_token_batches == [[direct_tokens[0]], [direct_tokens[1]]]
        structured_req.grammar.rollback.assert_called_once_with(len(direct_tokens))

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

        # The scheduler trims the reasoning prefix before advancing the grammar.
        assert (
            mock_request_with_structured_output.structured_output_request.reasoning_ended
            is True
        )
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
        seen_delta_ids: list[list[int]] = []

        def detects_reasoning_end(input_ids, delta_ids):
            delta_ids = list(delta_ids)
            seen_delta_ids.append(delta_ids)
            return end_token_id in delta_ids

        reasoner = MockReasoner(tokenizer=Mock())
        # Detection mirrors the real Qwen3 parser: end token in the delta.
        reasoner.is_reasoning_end_streaming = Mock(side_effect=detects_reasoning_end)
        structured_req.reasoner = reasoner

        # Scenario from #43388: async + spec decode K=4, 4 tokens accepted
        # but only 1 placeholder remains (some drafts were rejected).
        # The placeholder math would yield delta=[271] and miss </think>.
        # Passing new_token_ids must override that.
        new_token_ids = [9, 198, end_token_id, 271]
        mock_request_with_structured_output.all_token_ids = [
            1,
            2,
            3,
            4,
            5,
        ] + new_token_ids
        mock_request_with_structured_output.num_computed_tokens = 9
        mock_request_with_structured_output.num_output_placeholders = 1

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=new_token_ids,
        )

        # First call to is_reasoning_end_streaming was with the full
        # new_token_ids (not the truncated placeholder window).
        assert seen_delta_ids[0] == new_token_ids

        assert structured_req.reasoning_ended is True
        assert result is True

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

        mock_request_with_structured_output.all_token_ids = [1, 2, 3, 4, 5, 6, 7]
        mock_request_with_structured_output.num_computed_tokens = 7
        mock_request_with_structured_output.num_output_placeholders = 2

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        # placeholder window: start = 7 - 2 = 5, delta = [6, 7]
        _, called_delta = reasoner.is_reasoning_end_streaming.call_args[0]
        assert list(called_delta) == [6, 7]
        assert result is False

    def test_should_advance_trims_reasoning_prefix_for_json(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """JSON uses the common trim-then-advance path at the boundary."""
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

            def is_reasoning_end_from_prompt(self, prompt_token_ids):
                return False

            def is_reasoning_end_streaming(self, input_ids, delta_ids):
                return marker in list(delta_ids)

        structured_req.reasoner = MarkerReasoner()

        new_token_ids = [9, 198, marker, 271, 5005]
        mock_request_with_structured_output.prompt_token_ids = [1, 2, 3]
        mock_request_with_structured_output.num_prompt_tokens = 3
        mock_request_with_structured_output.all_token_ids = [1, 2, 3] + new_token_ids

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output,
            new_token_ids=new_token_ids,
        )

        structured_req.grammar.accept_tokens.assert_not_called()
        assert structured_req.reasoning_ended is True
        assert result is True
        assert structured_req.reasoning_end_token_index == 5
        assert manager_with_reasoner.trim_reasoning_for_advance(
            mock_request_with_structured_output, new_token_ids
        ) == [271, 5005]
