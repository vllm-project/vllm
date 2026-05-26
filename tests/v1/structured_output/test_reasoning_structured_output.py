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
        request.request_id = "test-request-123"
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = None
        request.structured_output_request.grammar = Mock()
        request.structured_output_request.reasoning_parser_kwargs = None
        request.structured_output_request.reasoner = None
        request.structured_output_request.bonus_requires_grammar = False
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

    def _make_detecting_reasoner(self, end_token_id: int = 99):
        """Return a reasoner whose is_reasoning_end_streaming detects
        ``end_token_id`` and records every call in ``detected_tokens``."""
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.detected_tokens = []

        def side_effect(input_ids, delta_ids):
            delta_list = (
                list(delta_ids) if not isinstance(delta_ids, list) else delta_ids
            )
            reasoner.detected_tokens.append(delta_list)
            return end_token_id in delta_list

        reasoner.is_reasoning_end_streaming = Mock(side_effect=side_effect)
        return reasoner

    def _make_mock_grammar(self, accept_result: bool = True):
        grammar = Mock()
        grammar.is_terminated = Mock(return_value=False)
        grammar.fill_bitmask = Mock()
        grammar.accept_tokens = Mock(return_value=accept_result)
        grammar.rollback = Mock()
        return grammar

    def _setup_manager_backend(self, manager):
        import torch

        manager.backend = Mock()
        manager.backend.allocate_token_bitmask = Mock(
            return_value=torch.zeros((10, 50000), dtype=torch.int32)
        )
        manager._full_mask = torch.tensor(-1, dtype=torch.int32)

    # ------------------------------------------------------------------ #
    # Test: reasoning ends at the LAST draft token                       #
    #   Draft:   [10, 20, 30, 99]   ← 99 = reasoning-end                #
    #   Expect:  bonus slot (idx 4) gets constrained bitmask             #
    # ------------------------------------------------------------------ #

    def test_grammar_bitmask_reasoning_ends_mid_batch(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Grammar_bitmask constrains bonus token when reasoning-end is
        the last draft token."""

        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False

        reasoner = self._make_detecting_reasoner(end_token_id=99)
        structured_req.reasoner = reasoner

        grammar = self._make_mock_grammar()
        structured_req.grammar = grammar

        self._setup_manager_backend(manager_with_reasoner)

        requests = {
            mock_request_with_structured_output.request_id: mock_request_with_structured_output
        }
        scheduled_spec_decode_tokens = {
            mock_request_with_structured_output.request_id: [10, 20, 30, 99]
        }

        manager_with_reasoner.grammar_bitmask(
            requests,
            [mock_request_with_structured_output.request_id],
            scheduled_spec_decode_tokens,
        )

        # --- assertions ---

        # is_reasoning_end_streaming was called per draft token
        assert reasoner.is_reasoning_end_streaming.called

        # reasoning_ended flag is NOT set by grammar_bitmask — that is
        # managed by should_advance() post-batch to avoid prematurely
        # claiming reasoning is done while the current step's output
        # still contains the reasoning-end token.
        assert structured_req.reasoning_ended is False

        # bonus_requires_grammar flag IS set — tells update_from_output()
        # to advance the grammar with the bonus token even though
        # should_advance() will return False this step.
        assert structured_req.bonus_requires_grammar is True

        # fill_bitmask should have been called exactly once — for the
        # bonus token position (index 4).  Positions 0-3 were filled
        # with the full mask (no constraint).
        fill_calls = grammar.fill_bitmask.call_args_list
        assert len(fill_calls) == 1, (
            f"Expected 1 fill_bitmask (bonus pos), got {len(fill_calls)}"
        )
        # The second argument is the index into the bitmask tensor
        fill_index = fill_calls[0][0][1]
        assert fill_index == 4, f"Expected bonus position index 4, got {fill_index}"

        # accept_tokens should NOT have been called (only the bonus
        # position had the bitmask enabled, and its token is the
        # sentinel -1 which is handled before the accept call).
        grammar.accept_tokens.assert_not_called()

    # ------------------------------------------------------------------ #
    # Test: reasoning ends MID-DRAFT (not at last position)              #
    #   Draft:   [10, 99, 30, 40]   ← 99 = reasoning-end at pos 1       #
    #   Expect:  positions 2, 3, and bonus (idx 4) constrained           #
    #   NOTE: positions 2-3 are draft tokens generated WITHOUT the       #
    #         constraint; accept_tokens is skipped to avoid xgrammar     #
    #         state corruption risk.                                     #
    # ------------------------------------------------------------------ #

    def test_grammar_bitmask_reasoning_ends_mid_draft(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Grammar_bitmask constrains remaining draft + bonus when
        reasoning-end appears mid-draft (not at the last position)."""

        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False

        reasoner = self._make_detecting_reasoner(end_token_id=99)
        structured_req.reasoner = reasoner

        grammar = self._make_mock_grammar(accept_result=False)
        structured_req.grammar = grammar

        self._setup_manager_backend(manager_with_reasoner)

        requests = {
            mock_request_with_structured_output.request_id: mock_request_with_structured_output
        }
        # reasoning-end (99) at position 1, two more draft tokens follow
        scheduled_spec_decode_tokens = {
            mock_request_with_structured_output.request_id: [10, 99, 30, 40]
        }

        manager_with_reasoner.grammar_bitmask(
            requests,
            [mock_request_with_structured_output.request_id],
            scheduled_spec_decode_tokens,
        )

        # --- assertions ---

        assert reasoner.is_reasoning_end_streaming.called
        # reasoning_ended is NOT set by grammar_bitmask — done by
        # should_advance() post-batch to avoid feeding reasoning-end
        # tokens into the grammar FSM in the same step.
        assert structured_req.reasoning_ended is False

        # bonus_requires_grammar IS set
        assert structured_req.bonus_requires_grammar is True

        # fill_bitmask for constrained positions: indices 2, 3, 4
        # (draft after end, last draft after end, bonus)
        fill_calls = grammar.fill_bitmask.call_args_list
        fill_indices = sorted(c[0][1] for c in fill_calls)
        assert fill_indices == [2, 3, 4], (
            f"Expected constrained indices [2, 3, 4], got {fill_indices}"
        )

        # accept_tokens is NOT called for any position — draft tokens
        # after reasoning-end were generated without constraint and
        # are skipped to avoid xgrammar state corruption risk.
        grammar.accept_tokens.assert_not_called()

        # rollback is not called (state_advancements stayed at 0)
        grammar.rollback.assert_not_called()

    # ------------------------------------------------------------------ #
    # Test: NO reasoning-end token in the batch                          #
    #   Draft:   [10, 20, 30, 40]                                        #
    #   Expect:  everything unconstrained (existing behaviour)           #
    # ------------------------------------------------------------------ #

    def test_grammar_bitmask_no_reasoning_end(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Grammar_bitmask leaves everything unconstrained when no
        reasoning-end token appears in the draft tokens."""

        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False

        reasoner = self._make_detecting_reasoner(end_token_id=99)
        structured_req.reasoner = reasoner

        grammar = self._make_mock_grammar()
        structured_req.grammar = grammar

        self._setup_manager_backend(manager_with_reasoner)

        requests = {
            mock_request_with_structured_output.request_id: mock_request_with_structured_output
        }
        # No reasoning-end token (99) in draft
        scheduled_spec_decode_tokens = {
            mock_request_with_structured_output.request_id: [10, 20, 30, 40]
        }

        manager_with_reasoner.grammar_bitmask(
            requests,
            [mock_request_with_structured_output.request_id],
            scheduled_spec_decode_tokens,
        )

        # --- assertions ---

        # is_reasoning_end_streaming was called but never returned True
        assert reasoner.is_reasoning_end_streaming.called

        # reasoning_ended flag is still False
        assert structured_req.reasoning_ended is False

        # fill_bitmask was never called — all positions stayed
        # unconstrained (fill with full mask instead)
        grammar.fill_bitmask.assert_not_called()

        # accept_tokens was never called
        grammar.accept_tokens.assert_not_called()

    # ------------------------------------------------------------------ #
    # Test: reasoning already ended BEFORE this batch                     #
    #   Draft: [10, 20, 30, 40]  with reasoning_ended=True               #
    #   Expect: all positions constrained normally                        #
    # ------------------------------------------------------------------ #

    def test_grammar_bitmask_reasoning_already_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Grammar_bitmask constrains everything when reasoning already
        ended before this batch."""

        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = True  # already ended

        grammar = self._make_mock_grammar()
        structured_req.grammar = grammar

        self._setup_manager_backend(manager_with_reasoner)

        requests = {
            mock_request_with_structured_output.request_id: mock_request_with_structured_output
        }
        scheduled_spec_decode_tokens = {
            mock_request_with_structured_output.request_id: [10, 20, 30, 40]
        }

        manager_with_reasoner.grammar_bitmask(
            requests,
            [mock_request_with_structured_output.request_id],
            scheduled_spec_decode_tokens,
        )

        # --- assertions ---

        # fill_bitmask was called for every position (4 draft + 1 bonus)
        fill_calls = grammar.fill_bitmask.call_args_list
        assert len(fill_calls) == 5, (
            f"Expected 5 fill_bitmask calls, got {len(fill_calls)}"
        )

        # accept_tokens was called for each draft token only (4).  The
        # bonus position uses sentinel -1 which sets apply_bitmask=False
        # before the accept check, so no accept call for the bonus.
        assert grammar.accept_tokens.call_count == 4, (
            f"Expected 4 accept_tokens calls, got {grammar.accept_tokens.call_count}"
        )

        # rollback was called (state_advancements == 4 > 0)
        grammar.rollback.assert_called_once()
