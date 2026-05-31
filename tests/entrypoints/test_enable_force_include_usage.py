# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests: --enable-force-include-usage behavior.
"""

from tests.entrypoints.conftest import (
    make_anthropic,
    make_base,
    make_chat,
    make_completion,
    make_disagg,
)


class TestNonStreaming:
    """Non-streaming with enable_force_include_usage=True."""

    def test_openai_endpoints(self):
        """Usage in final response, never in chunks."""
        for make in (make_base, make_chat, make_completion, make_disagg):
            assert make(enable_force_include_usage=True).should_include_usage(
                is_streaming=False
            ) == (True, False)

    def test_anthropic(self):
        """Always (True, True) regardless."""
        assert make_anthropic(enable_force_include_usage=True).should_include_usage(
            is_streaming=False
        ) == (True, True)


class TestStreaming:
    """Streaming with enable_force_include_usage=True."""

    def test_base(self):
        """Force: usage in last chunk only (include_usage=always)."""
        s = make_base(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True) == (True, False)
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            False,
        )
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            False,
        )

    def test_chat_completion(self):
        """Force: (True, True) regardless of request params."""
        for make in (make_chat, make_completion):
            s = make(enable_force_include_usage=True)
            assert s.should_include_usage(is_streaming=True) == (True, True)
            assert s.should_include_usage(is_streaming=True, include_usage=False) == (
                True,
                True,
            )
            assert s.should_include_usage(is_streaming=True, include_usage=True) == (
                True,
                True,
            )
            assert s.should_include_usage(
                is_streaming=True, include_usage=True, continuous_usage=True
            ) == (True, True)

    def test_disagg(self):
        """Disagg ignores the flag, behaves as default."""
        s = make_disagg(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True) == (False, False)
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            False,
            False,
        )
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            False,
        )
        assert s.should_include_usage(
            is_streaming=True, include_usage=True, continuous_usage=True
        ) == (True, True)

    def test_anthropic(self):
        """Always (True, True) regardless."""
        s = make_anthropic(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True) == (True, True)
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )
