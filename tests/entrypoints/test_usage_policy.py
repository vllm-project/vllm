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


class TestPolicyNone:
    """UsagePolicy() with all None fields — default behavior per endpoint."""

    def test_base(self):
        """Base defaults to usage in last chunk only."""
        s = make_base(UsagePolicy())
        assert s.should_include_usage(is_streaming=True) == (True, False)

    def test_chat(self):
        """Chat defaults to no usage in streaming."""
        s = make_chat(UsagePolicy())
        assert s.should_include_usage(is_streaming=True) == (False, False)

    def test_completion(self):
        """Completion defaults to no usage in streaming."""
        s = make_completion(UsagePolicy())
        assert s.should_include_usage(is_streaming=True) == (False, False)

    def test_disagg(self):
        """Disagg defaults to no usage in streaming."""
        s = make_disagg(UsagePolicy())
        assert s.should_include_usage(is_streaming=True) == (False, False)

    def test_anthropic(self):
        """Anthropic always (True, True) regardless of policy."""
        s = make_anthropic(UsagePolicy())
        assert s.should_include_usage(is_streaming=True) == (True, True)

    def test_base_request_include_usage_true(self):
        """Base + request include_usage=True → usage in last chunk only."""
        s = make_base(UsagePolicy())
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            False,
        )

    def test_chat_request_include_usage_true(self):
        """Chat + request include_usage=True → usage in last chunk only."""
        s = make_chat(UsagePolicy())
        assert s.should_include_usage(is_streaming=True, include_usage=True) == (
            True,
            False,
        )

    def test_chat_request_both_flags_true(self):
        """Chat + both request flags → usage in every chunk."""
        s = make_chat(UsagePolicy())
        assert s.should_include_usage(
            is_streaming=True, include_usage=True, continuous_usage=True
        ) == (True, True)

    def test_chat_request_include_usage_false(self):
        """Chat + request include_usage=False → no usage."""
        s = make_chat(UsagePolicy())
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            False,
            False,
        )


class TestPolicyIncludeAlways:
    """UsagePolicy(include_usage="always") — force usage in final chunk."""

    def test_base(self):
        """Overrides request include_usage=False."""
        s = make_base(UsagePolicy(include_usage="always"))
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            False,
        )

    def test_chat(self):
        """Overrides request include_usage=False."""
        s = make_chat(UsagePolicy(include_usage="always"))
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            False,
        )

    def test_completion(self):
        """Overrides request include_usage=False."""
        s = make_completion(UsagePolicy(include_usage="always"))
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            False,
        )

    def test_disagg(self):
        """Overrides request include_usage=False."""
        s = make_disagg(UsagePolicy(include_usage="always"))
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            False,
        )

    def test_anthropic(self):
        """Anthropic ignores policy, still (True, True)."""
        s = make_anthropic(UsagePolicy(include_usage="always"))
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )

    def test_chat_continuous_true(self):
        """Chat + request continuous_usage=True is ignored by policy."""
        s = make_chat(UsagePolicy(include_usage="always"))
        assert s.should_include_usage(
            is_streaming=True, include_usage=True, continuous_usage=True
        ) == (True, False)


class TestPolicyBothAlways:
    """UsagePolicy(include_usage="always", continuous_usage="always")."""

    def test_base(self):
        """Usage in every chunk."""
        s = make_base(UsagePolicy(include_usage="always", continuous_usage="always"))
        assert s.should_include_usage(is_streaming=True) == (True, True)

    def test_chat(self):
        """Usage in every chunk."""
        s = make_chat(UsagePolicy(include_usage="always", continuous_usage="always"))
        assert s.should_include_usage(is_streaming=True) == (True, True)

    def test_completion(self):
        """Usage in every chunk."""
        s = make_completion(
            UsagePolicy(include_usage="always", continuous_usage="always")
        )
        assert s.should_include_usage(is_streaming=True) == (True, True)

    def test_disagg(self):
        """Usage in every chunk."""
        s = make_disagg(UsagePolicy(include_usage="always", continuous_usage="always"))
        assert s.should_include_usage(is_streaming=True) == (True, True)

    def test_anthropic(self):
        """Anthropic ignores policy, still (True, True)."""
        s = make_anthropic(
            UsagePolicy(include_usage="always", continuous_usage="always")
        )
        assert s.should_include_usage(is_streaming=True) == (True, True)

    def test_base_overrides_request(self):
        """Overrides request include_usage=False."""
        s = make_base(UsagePolicy(include_usage="always", continuous_usage="always"))
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )

    def test_chat_overrides_request(self):
        """Overrides request include_usage=False."""
        s = make_chat(UsagePolicy(include_usage="always", continuous_usage="always"))
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )

    def test_completion_overrides_request(self):
        """Overrides request include_usage=False."""
        s = make_completion(
            UsagePolicy(include_usage="always", continuous_usage="always")
        )
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )

    def test_disagg_overrides_request(self):
        """Overrides request include_usage=False."""
        s = make_disagg(UsagePolicy(include_usage="always", continuous_usage="always"))
        assert s.should_include_usage(is_streaming=True, include_usage=False) == (
            True,
            True,
        )


class TestPolicyNonStreaming:
    """Non-streaming with various UsagePolicy configurations."""

    def test_all_endpoints_none_policy(self):
        """All non-streaming endpoints return (True, False)."""
        for make in [make_base, make_chat, make_completion, make_disagg]:
            s = make(UsagePolicy())
            assert s.should_include_usage(is_streaming=False) == (True, False)

    def test_anthropic(self):
        """Anthropic non-streaming → always (True, True)."""
        s = make_anthropic(UsagePolicy())
        assert s.should_include_usage(is_streaming=False) == (True, True)

    def test_all_endpoints_always_policy(self):
        """Policy include_usage=always: all non-streaming return (True, False)."""
        p = UsagePolicy(include_usage="always")
        for make in [make_base, make_chat, make_completion, make_disagg]:
            s = make(p)
            assert s.should_include_usage(is_streaming=False) == (True, False)

    def test_all_endpoints_both_always_policy(self):
        """Policy both=always: all non-streaming return (True, False)."""
        p = UsagePolicy(include_usage="always", continuous_usage="always")
        for make in [make_base, make_chat, make_completion, make_disagg]:
            s = make(p)
            assert s.should_include_usage(is_streaming=False) == (True, False)
