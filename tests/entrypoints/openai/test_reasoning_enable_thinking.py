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
