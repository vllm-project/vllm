# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for --supported-reasoning-efforts normalization.

Models whose chat templates accept only a subset of the reasoning effort
ladder (e.g. low/medium/xhigh) raise at template render time when a client
sends an unsupported value. `normalize_reasoning_effort` remaps such values
to the nearest supported level in the configured rounding direction so the
request succeeds instead.
"""

from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.reasoning_effort import (
    normalize_reasoning_effort,
    validate_supported_reasoning_efforts,
)

SUPPORTED = ["low", "medium", "xhigh"]


class TestNormalizeReasoningEffort:
    @pytest.mark.parametrize(
        ("effort", "rounding", "expected"),
        [
            # In-range: down picks the highest supported level at or below.
            ("high", "down", "medium"),
            # In-range: up picks the lowest supported level at or above.
            ("high", "up", "xhigh"),
            # Below the supported range: both directions land on the floor.
            ("minimal", "down", "low"),
            ("minimal", "up", "low"),
            # Above the supported range: both directions land on the ceiling.
            ("max", "down", "xhigh"),
            ("max", "up", "xhigh"),
        ],
    )
    def test_unsupported_efforts_are_remapped(self, effort, rounding, expected):
        assert normalize_reasoning_effort(effort, SUPPORTED, rounding) == expected

    @pytest.mark.parametrize("effort", SUPPORTED)
    def test_supported_efforts_pass_through(self, effort):
        assert normalize_reasoning_effort(effort, SUPPORTED, "down") == effort
        assert normalize_reasoning_effort(effort, SUPPORTED, "up") == effort

    @pytest.mark.parametrize("effort", [None, "none"])
    def test_none_always_passes_through(self, effort):
        assert normalize_reasoning_effort(effort, SUPPORTED, "down") == effort
        assert normalize_reasoning_effort(effort, SUPPORTED, "up") == effort

    def test_no_supported_set_is_a_no_op(self):
        assert normalize_reasoning_effort("high", None, "down") == "high"

    def test_unordered_supported_set(self):
        assert normalize_reasoning_effort("high", ["xhigh", "low"], "down") == "low"
        assert normalize_reasoning_effort("high", ["xhigh", "low"], "up") == "xhigh"


class TestValidateSupportedReasoningEfforts:
    def test_none_and_valid_values_accepted(self):
        validate_supported_reasoning_efforts(None)
        validate_supported_reasoning_efforts(["low", "medium", "xhigh"])

    @pytest.mark.parametrize("supported", [[], ["none"], ["medium", "extreme"]])
    def test_invalid_values_rejected(self, supported):
        with pytest.raises(ValueError):
            validate_supported_reasoning_efforts(supported)


class TestServingChatNormalization:
    """The chat serving path applies the policy before rendering."""

    def _effective_kwargs(self, request, supported, rounding):
        serving = SimpleNamespace(
            chat_template=None,
            chat_template_content_format="auto",
            default_chat_template_kwargs={},
            supported_reasoning_efforts=supported,
            reasoning_effort_rounding=rounding,
        )
        return OpenAIServingChat._effective_chat_template_kwargs(serving, request)

    def test_request_and_template_kwargs_see_effective_effort(self):
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="high",
        )
        kwargs = self._effective_kwargs(request, SUPPORTED, "down")
        assert request.reasoning_effort == "medium"
        assert kwargs["reasoning_effort"] == "medium"

    def test_unset_policy_leaves_request_untouched(self):
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="high",
        )
        kwargs = self._effective_kwargs(request, None, "down")
        assert request.reasoning_effort == "high"
        assert kwargs["reasoning_effort"] == "high"
