# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for should_include_usage base method and subclass overrides.

Covers default behavior only — no server-level flags (enable_force_include_usage
or usage_policy). All request-level params (include_usage / continuous_usage)
are tested per endpoint.
"""

from tests.entrypoints.conftest import (
    make_anthropic,
    make_base,
    make_chat,
    make_completion,
    make_disagg,
)


class TestBaseClassShouldIncludeUsage:
    """Base class (OpenAIServing) — used by Speech-to-Text endpoints."""

    def test_non_streaming(self):
        """Usage in final response, never in chunks."""
        s = make_base()
        assert s.should_include_usage(is_streaming=False) == (True, False)

    def test_streaming_default(self):
        """Usage in last chunk only, not every chunk."""
        s = make_base()
        assert s.should_include_usage(is_streaming=True) == (True, False)

    def test_request_include_usage_true(self):
        """include_usage=True → usage in last chunk only."""
        s = make_base()
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            False,
        )

    def test_request_include_usage_false(self):
        """include_usage=False → no usage."""
        s = make_base()
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            False,
            False,
        )

    def test_request_both_flags_true(self):
        """include_usage=True + continuous_usage=True → usage in every chunk."""
        s = make_base()
        assert s.should_include_usage(
            is_streaming=True, include_usage=True, continuous_usage=True
        ) == (True, True)


class TestChatShouldIncludeUsage:
    """Chat Completions endpoint (OpenAIServingChat)."""

    def test_non_streaming(self):
        """Usage in final response, never in chunks."""
        s = make_chat()
        assert s.should_include_usage(is_streaming=False) == (True, False)

    def test_streaming_default(self):
        """No usage in streaming by default."""
        s = make_chat()
        assert s.should_include_usage(is_streaming=True) == (False, False)

    def test_request_include_usage_true(self):
        """include_usage=True → usage in last chunk only."""
        s = make_chat()
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            False,
        )

    def test_request_include_usage_false(self):
        """include_usage=False → no usage."""
        s = make_chat()
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            False,
            False,
        )

    def test_request_both_flags_true(self):
        """include_usage=True + continuous_usage=True → usage in every chunk."""
        s = make_chat()
        assert s.should_include_usage(
            is_streaming=True, include_usage=True, continuous_usage=True
        ) == (True, True)


class TestCompletionShouldIncludeUsage:
    """Completions endpoint (OpenAIServingCompletion)."""

    def test_streaming_default(self):
        """No usage in streaming by default."""
        s = make_completion()
        assert s.should_include_usage(is_streaming=True) == (False, False)

    def test_request_include_usage_true(self):
        """include_usage=True → usage in last chunk only."""
        s = make_completion()
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            False,
        )


class TestDisaggShouldIncludeUsage:
    """Disaggregated serving endpoint (ServingTokens)."""

    def test_streaming_default(self):
        """No usage in streaming by default."""
        s = make_disagg()
        assert s.should_include_usage(is_streaming=True) == (False, False)

    def test_request_include_usage_true(self):
        """include_usage=True → usage in last chunk only."""
        s = make_disagg()
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            False,
        )


class TestAnthropicShouldIncludeUsage:
    """Anthropic Messages endpoint (AnthropicServingMessages)."""

    def test_non_streaming(self):
        """Always (True, True) — Anthropic API spec."""
        s = make_anthropic()
        assert s.should_include_usage(is_streaming=False) == (True, True)

    def test_streaming_default(self):
        """Always (True, True) regardless of parameters."""
        s = make_anthropic()
        assert s.should_include_usage(is_streaming=True) == (True, True)

    def test_request_include_usage_false(self):
        """Request include_usage=False is ignored."""
        s = make_anthropic()
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )
