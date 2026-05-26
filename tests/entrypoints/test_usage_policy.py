# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests: --include-usage-policy / --continuous-usage-policy
combinations across all entrypoints.

Each test class covers one UsagePolicy configuration.
"""

from tests.entrypoints.conftest import (
    make_anthropic,
    make_base,
    make_chat,
    make_completion,
    make_disagg,
)
from vllm.entrypoints.chat_utils import UsagePolicy


class TestPolicyIncludeAlways:
    """UsagePolicy(include_usage="always"), force usage in final chunk."""

    def test_endpoints(self):
        """Overrides request include_usage=False; continuous falls back to request."""
        for make in (make_base, make_chat, make_completion, make_disagg):
            s = make(UsagePolicy(include_usage="always"))

            assert s.should_include_usage(
                is_streaming=True, include_usage=False, continuous_usage=False
            ) == (True, False)
            assert s.should_include_usage(
                is_streaming=True, include_usage=True, continuous_usage=True
            ) == (True, True)
            assert s.should_include_usage(
                is_streaming=True, include_usage=False, continuous_usage=True
            ) == (True, True)

    def test_anthropic(self):
        """Anthropic ignores policy, still (True, True)."""
        s = make_anthropic(UsagePolicy(include_usage="always"))

        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )
        assert s.should_include_usage(
            is_streaming=True, include_usage=True, continuous_usage=True
        ) == (True, True)

    def test_non_streaming(self):
        """all non-streaming return (True, False)."""
        p = UsagePolicy(include_usage="always")
        for make in (make_base, make_chat, make_completion, make_disagg):
            s = make(p)
            assert s.should_include_usage(is_streaming=False) == (True, False)


class TestPolicyBothAlways:
    """UsagePolicy(include_usage="always", continuous_usage="always")."""

    def test_non_streaming(self):
        """all non-streaming return (True, False)."""
        for make in (make_base, make_chat, make_completion, make_disagg):
            s = make(UsagePolicy(include_usage="always", continuous_usage="always"))
            assert s.should_include_usage(is_streaming=False) == (True, False)

    def test_endpoints(self):
        """Usage in every chunk by default."""
        for make in (make_base, make_chat, make_completion, make_disagg):
            s = make(UsagePolicy(include_usage="always", continuous_usage="always"))

            assert s.should_include_usage(is_streaming=True) == (True, True)
            assert s.should_include_usage(is_streaming=True, include_usage=False) == (
                True,
                True,
            )
            assert s.should_include_usage(
                is_streaming=True, include_usage=False, continuous_usage=True
            ) == (True, True)
            assert s.should_include_usage(
                is_streaming=True, include_usage=True, continuous_usage=True
            ) == (True, True)

    def test_anthropic(self):
        """Anthropic ignores policy, still (True, True)."""
        s = make_anthropic(
            UsagePolicy(include_usage="always", continuous_usage="always")
        )

        assert s.should_include_usage(is_streaming=True) == (True, True)
