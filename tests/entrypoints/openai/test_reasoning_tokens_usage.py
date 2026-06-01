# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for reasoning_tokens in completion_tokens_details (issue #14335).

Verifies that the CompletionTokensDetails model is correctly defined,
integrated into UsageInfo, and populated by the reasoning parser's
count_reasoning_tokens method.
"""

from vllm.entrypoints.openai.engine.protocol import (
    CompletionTokensDetails,
    UsageInfo,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class TestCompletionTokensDetails:
    """Tests for the CompletionTokensDetails protocol model."""

    def test_default_reasoning_tokens_is_zero(self):
        details = CompletionTokensDetails()
        assert details.reasoning_tokens == 0

    def test_reasoning_tokens_set(self):
        details = CompletionTokensDetails(reasoning_tokens=42)
        assert details.reasoning_tokens == 42

    def test_serialization_round_trip(self):
        details = CompletionTokensDetails(reasoning_tokens=10)
        data = details.model_dump()
        assert data == {"reasoning_tokens": 10}
        restored = CompletionTokensDetails.model_validate(data)
        assert restored.reasoning_tokens == 10


class TestUsageInfoWithCompletionDetails:
    """Tests for UsageInfo with completion_tokens_details field."""

    def test_usage_info_no_details_by_default(self):
        usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert usage.completion_tokens_details is None

    def test_usage_info_with_completion_details(self):
        usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=5,
            ),
        )
        assert usage.completion_tokens_details is not None
        assert usage.completion_tokens_details.reasoning_tokens == 5

    def test_usage_info_serialization_excludes_none_details(self):
        usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        data = usage.model_dump(exclude_none=True)
        assert "completion_tokens_details" not in data

    def test_usage_info_serialization_includes_details(self):
        usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=7,
            ),
        )
        data = usage.model_dump()
        assert data["completion_tokens_details"]["reasoning_tokens"] == 7


class TestCountReasoningTokens:
    """Tests for BaseThinkingReasoningParser.count_reasoning_tokens."""

    def _make_parser(self, start_token_id: int, end_token_id: int):
        """Create a minimal parser with the needed attributes."""
        _start_id = start_token_id
        _end_id = end_token_id

        class _TestParser(BaseThinkingReasoningParser):
            @property
            def start_token(self) -> str:
                return "<think>"

            @property
            def end_token(self) -> str:
                return "</think>"

            def __init__(self):
                # Skip parent __init__ which requires a tokenizer.
                # Only set attributes needed by count_reasoning_tokens.
                self.start_token_id = _start_id
                self.end_token_id = _end_id

        return _TestParser()

    def test_no_reasoning_tokens(self):
        parser = self._make_parser(start_token_id=100, end_token_id=101)
        token_ids = [1, 2, 3, 4, 5]
        assert parser.count_reasoning_tokens(token_ids) == 0

    def test_simple_reasoning_span(self):
        parser = self._make_parser(start_token_id=100, end_token_id=101)
        # <think> a b c </think> d e
        token_ids = [100, 10, 11, 12, 101, 20, 21]
        assert parser.count_reasoning_tokens(token_ids) == 3

    def test_reasoning_at_end_no_close(self):
        parser = self._make_parser(start_token_id=100, end_token_id=101)
        # <think> a b c  (no close tag - generation stopped mid-think)
        token_ids = [100, 10, 11, 12]
        assert parser.count_reasoning_tokens(token_ids) == 3

    def test_empty_reasoning_span(self):
        parser = self._make_parser(start_token_id=100, end_token_id=101)
        # <think></think> a b
        token_ids = [100, 101, 20, 21]
        assert parser.count_reasoning_tokens(token_ids) == 0

    def test_only_reasoning(self):
        parser = self._make_parser(start_token_id=100, end_token_id=101)
        # <think> a b c </think>
        token_ids = [100, 10, 11, 12, 101]
        assert parser.count_reasoning_tokens(token_ids) == 3

    def test_empty_sequence(self):
        parser = self._make_parser(start_token_id=100, end_token_id=101)
        assert parser.count_reasoning_tokens([]) == 0

    def test_stray_end_token_ignored(self):
        parser = self._make_parser(start_token_id=100, end_token_id=101)
        # </think> a b (stray end token at start)
        token_ids = [101, 10, 11]
        assert parser.count_reasoning_tokens(token_ids) == 0


class TestReasoningParserBaseDefault:
    """Test that the base ReasoningParser.count_reasoning_tokens returns 0."""

    def test_base_parser_returns_zero(self):
        from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

        # count_reasoning_tokens is a concrete method on the base class
        # that returns 0 by default.
        assert ReasoningParser.count_reasoning_tokens(None, [1, 2, 3]) == 0
