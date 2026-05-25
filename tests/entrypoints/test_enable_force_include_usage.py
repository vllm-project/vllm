# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests: --enable-force-include-usage behavior across all entrypoints.

Each test class covers one scenario with --enable-force-include-usage=True.
"""

from tests.entrypoints.conftest import (
    make_anthropic,
    make_base,
    make_chat,
    make_completion,
    make_disagg,
)


class TestForceNonStreaming:
    """Non-streaming with enable_force_include_usage=True."""

    def test_base(self):
        """Usage in final response, never in chunks."""
        s = make_base(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=False) == (True, False)

    def test_chat(self):
        """Usage in final response, never in chunks."""
        s = make_chat(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=False) == (True, False)

    def test_completion(self):
        """Usage in final response, never in chunks."""
        s = make_completion(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=False) == (True, False)

    def test_disagg(self):
        """Usage in final response, never in chunks."""
        s = make_disagg(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=False) == (True, False)

    def test_anthropic(self):
        """Always (True, True) regardless."""
        s = make_anthropic(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=False) == (True, True)


class TestForceStreamingDefault:
    """Streaming with enable_force_include_usage=True, no request-level params."""

    def test_base(self):
        """Usage in last chunk only, not every chunk."""
        s = make_base(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True) == (True, False)

    def test_chat(self):
        """Usage in last chunk, and every chunk."""
        s = make_chat(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True) == (True, True)

    def test_completion(self):
        """Usage in last chunk, and every chunk."""
        s = make_completion(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True) == (True, True)

    def test_disagg(self):
        """Usage in last chunk, and every chunk."""
        s = make_disagg(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True) == (True, True)

    def test_anthropic(self):
        """Always (True, True) regardless."""
        s = make_anthropic(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True) == (True, True)


class TestForceOverridesRequestIncludeUsageFalse:
    """
    Streaming with enable_force_include_usage=True
    and request-level include_usage=False.
    """

    def test_base(self):
        """Force overrides to usage in last chunk only."""
        s = make_base(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            False,
        )

    def test_chat(self):
        """Force overrides to usage in last chunk and every chunk."""
        s = make_chat(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )

    def test_completion(self):
        """Force overrides to usage in last chunk and every chunk."""
        s = make_completion(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )

    def test_disagg(self):
        """Force overrides to usage in last chunk and every chunk."""
        s = make_disagg(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )

    def test_anthropic(self):
        """Always (True, True) regardless."""
        s = make_anthropic(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )


class TestForceWithRequestIncludeUsageTrue:
    """
    Streaming with enable_force_include_usage=True
    and request-level include_usage=True.
    """

    def test_base(self):
        """Usage in last chunk only."""
        s = make_base(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            False,
        )

    def test_chat(self):
        """Usage in last chunk and every chunk."""
        s = make_chat(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            True,
        )

    def test_completion(self):
        """Usage in last chunk and every chunk."""
        s = make_completion(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            True,
        )

    def test_disagg(self):
        """Usage in last chunk and every chunk."""
        s = make_disagg(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            True,
        )

    def test_anthropic(self):
        """Always (True, True) regardless."""
        s = make_anthropic(enable_force_include_usage=True)
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            True,
        )


class TestForceWithRequestIncludeUsageAndContinuousTrue:
    """
    Streaming with enable_force_include_usage=True
    and request-level include_usage=True + continuous_usage=True.
    """

    def test_base(self):
        """Force short-circuits before reaching request-level continuous."""
        s = make_base(enable_force_include_usage=True)
        assert s.should_include_usage(
            is_streaming=True, include_usage=True, continuous_usage=True
        ) == (True, False)

    def test_chat(self):
        """Usage in last chunk and every chunk."""
        s = make_chat(enable_force_include_usage=True)
        assert s.should_include_usage(
            is_streaming=True, include_usage=True, continuous_usage=True
        ) == (True, True)

    def test_completion(self):
        """Usage in last chunk and every chunk."""
        s = make_completion(enable_force_include_usage=True)
        assert s.should_include_usage(
            is_streaming=True, include_usage=True, continuous_usage=True
        ) == (True, True)

    def test_disagg(self):
        """Usage in last chunk and every chunk."""
        s = make_disagg(enable_force_include_usage=True)
        assert s.should_include_usage(
            is_streaming=True, include_usage=True, continuous_usage=True
        ) == (True, True)

    def test_anthropic(self):
        """Always (True, True) regardless."""
        s = make_anthropic(enable_force_include_usage=True)
        assert s.should_include_usage(
            is_streaming=True, include_usage=True, continuous_usage=True
        ) == (True, True)
