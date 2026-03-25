# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for reasoning_effort -> enable_thinking mapping."""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)


def _build_request(**kwargs) -> ChatCompletionRequest:
    """Helper to create a minimal ChatCompletionRequest."""
    defaults = dict(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


class TestReasoningEffortEnableThinking:
    """Test that reasoning_effort='none' injects enable_thinking=False
    into chat_template_kwargs at request level (model validator) so that
    both the Jinja template and the reasoning parser see it."""

    def test_none_injects_enable_thinking_in_request(self):
        """enable_thinking=False should be in request.chat_template_kwargs."""
        request = _build_request(reasoning_effort="none")
        assert request.chat_template_kwargs is not None
        assert request.chat_template_kwargs["enable_thinking"] is False

    def test_none_propagates_to_build_chat_params(self):
        """enable_thinking=False should also appear in ChatParams kwargs."""
        request = _build_request(reasoning_effort="none")
        params = request.build_chat_params(None, "auto")
        assert params.chat_template_kwargs["enable_thinking"] is False

    def test_none_preserves_reasoning_effort_in_kwargs(self):
        request = _build_request(reasoning_effort="none")
        params = request.build_chat_params(None, "auto")
        assert params.chat_template_kwargs["reasoning_effort"] == "none"

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_non_none_does_not_inject_enable_thinking(self, effort):
        request = _build_request(reasoning_effort=effort)
        if request.chat_template_kwargs:
            assert "enable_thinking" not in request.chat_template_kwargs
        params = request.build_chat_params(None, "auto")
        assert "enable_thinking" not in params.chat_template_kwargs

    def test_no_reasoning_effort_does_not_inject(self):
        request = _build_request()
        params = request.build_chat_params(None, "auto")
        assert "enable_thinking" not in params.chat_template_kwargs

    def test_explicit_enable_thinking_not_overridden(self):
        """User's explicit enable_thinking=True via chat_template_kwargs
        should not be overridden by reasoning_effort='none'."""
        request = _build_request(
            reasoning_effort="none",
            chat_template_kwargs={"enable_thinking": True},
        )
        assert request.chat_template_kwargs["enable_thinking"] is True
        params = request.build_chat_params(None, "auto")
        assert params.chat_template_kwargs["enable_thinking"] is True

    def test_none_also_sets_include_reasoning_false(self):
        """Verify the existing validator still works alongside our change."""
        request = _build_request(reasoning_effort="none")
        assert request.include_reasoning is False

    def test_existing_chat_template_kwargs_preserved(self):
        """Other user-provided chat_template_kwargs should be preserved."""
        request = _build_request(
            reasoning_effort="none",
            chat_template_kwargs={"custom_key": "value"},
        )
        assert request.chat_template_kwargs["custom_key"] == "value"
        assert request.chat_template_kwargs["enable_thinking"] is False
