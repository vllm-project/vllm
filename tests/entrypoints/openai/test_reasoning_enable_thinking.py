# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for reasoning_effort -> enable_thinking mapping.

Models like Gemma4 require enable_thinking=True in chat_template_kwargs to
activate thinking mode. This mapping ensures that when a user requests
reasoning (via reasoning_effort or reasoning.effort), the template kwarg
is injected automatically.
"""

import pytest
from openai.types.shared import Reasoning
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _build_chat_request(**kwargs) -> ChatCompletionRequest:
    defaults = dict(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _build_responses_request(**kwargs) -> ResponsesRequest:
    defaults = dict(
        model="test-model",
        input=[{"role": "user", "content": "Hello"}],
    )
    defaults.update(kwargs)
    return ResponsesRequest(**defaults)


class TestChatCompletionReasoningEffort:
    """Chat Completions: reasoning_effort -> enable_thinking."""

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_non_none_effort_injects_enable_thinking_true(self, effort):
        request = _build_chat_request(reasoning_effort=effort)
        params = request.build_chat_params(None, "auto")
        assert params.chat_template_kwargs["enable_thinking"] is True

    def test_none_effort_injects_enable_thinking_false(self):
        request = _build_chat_request(reasoning_effort="none")
        params = request.build_chat_params(None, "auto")
        assert params.chat_template_kwargs["enable_thinking"] is False

    def test_no_effort_does_not_inject(self):
        request = _build_chat_request()
        params = request.build_chat_params(None, "auto")
        assert "enable_thinking" not in params.chat_template_kwargs

    def test_explicit_user_kwarg_not_overridden(self):
        request = _build_chat_request(
            reasoning_effort="high",
            chat_template_kwargs={"enable_thinking": False},
        )
        params = request.build_chat_params(None, "auto")
        assert params.chat_template_kwargs["enable_thinking"] is False

    def test_reasoning_effort_still_in_kwargs(self):
        request = _build_chat_request(reasoning_effort="high")
        params = request.build_chat_params(None, "auto")
        assert params.chat_template_kwargs["reasoning_effort"] == "high"

    def test_effort_map_rewrites_listed_efforts_only(self):
        """Unsupported efforts are rewritten; the rest reach the template as-is."""
        effort_map = {"medium": "low"}

        mapped = _build_chat_request(reasoning_effort="medium")
        params = mapped.build_chat_params(None, "auto", effort_map)
        assert params.chat_template_kwargs["reasoning_effort"] == "low"

        unmapped = _build_chat_request(reasoning_effort="high")
        params = unmapped.build_chat_params(None, "auto", effort_map)
        assert params.chat_template_kwargs["reasoning_effort"] == "high"

    @pytest.mark.parametrize(
        ("effort", "effort_map", "enable_thinking"),
        [("minimal", {"minimal": "none"}, False), ("none", {"none": "low"}, True)],
    )
    def test_enable_thinking_follows_mapped_effort(
        self, effort, effort_map, enable_thinking
    ):
        request = _build_chat_request(reasoning_effort=effort)
        params = request.build_chat_params(None, "auto", effort_map)
        assert params.chat_template_kwargs["enable_thinking"] is enable_thinking

    def test_effort_map_leaves_chat_template_kwargs_alone(self):
        """An effort passed straight to the template bypasses the map."""
        request = _build_chat_request(
            chat_template_kwargs={"reasoning_effort": "medium"},
        )
        params = request.build_chat_params(None, "auto", {"medium": "low"})
        assert params.chat_template_kwargs["reasoning_effort"] == "medium"


class TestChatCompletionThinkingObject:
    """Chat Completions: top-level vendor-style `thinking` object."""

    @pytest.mark.parametrize(
        ("mode", "enabled"), [("enabled", True), ("disabled", False)]
    )
    def test_type_overrides_template_toggles(self, mode, enabled):
        request = _build_chat_request(
            thinking={"type": mode},
            reasoning_effort="none" if enabled else "high",
            chat_template_kwargs={
                "thinking": not enabled,
                "enable_thinking": not enabled,
            },
        )
        kwargs = request.build_chat_params(None, "auto").chat_template_kwargs
        assert kwargs["thinking"] is enabled
        assert kwargs["enable_thinking"] is enabled

    def test_effort_overrides_reasoning_effort(self):
        """Renderers that read `request.reasoning_effort` see the override too."""
        request = _build_chat_request(
            thinking={"effort": "low"},
            reasoning_effort="max",
        )
        kwargs = request.build_chat_params(None, "auto").chat_template_kwargs
        assert request.reasoning_effort == "low"
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["enable_thinking"] is True
        assert "thinking" not in kwargs

    def test_empty_object_preserves_existing_behavior(self):
        request = _build_chat_request(thinking={}, reasoning_effort="high")
        kwargs = request.build_chat_params(None, "auto").chat_template_kwargs
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["enable_thinking"] is True
        assert "thinking" not in kwargs

    @pytest.mark.parametrize("thinking", [True, {"type": "adaptive"}, {"keep": "none"}])
    def test_unsupported_controls_are_rejected(self, thinking):
        """Controls vLLM cannot honor fail validation instead of being dropped."""
        with pytest.raises(ValidationError):
            _build_chat_request(thinking=thinking)


class TestResponsesReasoningEffort:
    """Responses API: reasoning.effort -> enable_thinking."""

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_non_none_effort_injects_enable_thinking_true(self, effort):
        request = _build_responses_request(
            reasoning=Reasoning(effort=effort),
        )
        params = request.build_chat_params(None, "auto")
        assert params.chat_template_kwargs["enable_thinking"] is True

    def test_none_effort_injects_enable_thinking_false(self):
        request = _build_responses_request(
            reasoning=Reasoning(effort="none"),
        )
        params = request.build_chat_params(None, "auto")
        assert params.chat_template_kwargs["enable_thinking"] is False

    def test_no_reasoning_does_not_inject(self):
        request = _build_responses_request()
        params = request.build_chat_params(None, "auto")
        assert "enable_thinking" not in params.chat_template_kwargs

    def test_explicit_user_kwarg_not_overridden(self):
        request = _build_responses_request(
            reasoning=Reasoning(effort="high"),
            chat_template_kwargs={"enable_thinking": False},
        )
        params = request.build_chat_params(None, "auto")
        assert params.chat_template_kwargs["enable_thinking"] is False

    def test_reasoning_effort_still_in_kwargs(self):
        request = _build_responses_request(
            reasoning=Reasoning(effort="high"),
        )
        params = request.build_chat_params(None, "auto")
        assert params.chat_template_kwargs["reasoning_effort"] == "high"

    def test_effort_map_rewrites_effort(self):
        request = _build_responses_request(
            reasoning=Reasoning(effort="medium"),
        )
        params = request.build_chat_params(None, "auto", {"medium": "low"})
        assert params.chat_template_kwargs["reasoning_effort"] == "low"
