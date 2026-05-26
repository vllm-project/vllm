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


class TestNonStreaming:
    """Non-streaming: usage always in final response."""

    def test_openai_endpoints(self):
        """Usage in final response, never in chunks."""
        for make in (make_base, make_chat, make_completion, make_disagg):
            entryoint = make()
            assert entryoint.should_include_usage(is_streaming=False) == (True, False)

    def test_anthropic(self):
        """Anthropic always (True, True) regardless."""
        entryoint = make_anthropic()
        assert entryoint.should_include_usage(is_streaming=False) == (True, True)
        assert entryoint.should_include_usage(
            is_streaming=True, include_usage=False
        ) == (True, True)


class TestStreamingDefault:
    """Default streaming behavior (no request params)."""

    def test_base(self):
        """Base: usage in last chunk only."""
        entryoint = make_base()
        assert entryoint.should_include_usage(is_streaming=True) == (True, False)

    def test_openai_endpoints(self):
        """Chat/Completion/Disagg: no usage by default."""
        for make in (make_chat, make_completion, make_disagg):
            entryoint = make()
            assert entryoint.should_include_usage(is_streaming=True) == (False, False)
            assert entryoint.should_include_usage(
                is_streaming=True, include_usage=False
            ) == (False, False)
            assert entryoint.should_include_usage(
                is_streaming=True, include_usage=True
            ) == (True, False)
            assert entryoint.should_include_usage(
                is_streaming=True, include_usage=True, continuous_usage=True
            ) == (True, True)

    def test_anthropic(self):
        """Anthropic always (True, True)."""
        entryoint = make_anthropic()
        assert entryoint.should_include_usage(is_streaming=True) == (True, True)
