# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for separate_reasoning parameter."""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import (
    _merge_reasoning_into_content,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)


class TestSeparateReasoningParameter:
    """Tests for the separate_reasoning request parameter."""

    def test_default_separate_reasoning_is_true(self):
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert request.separate_reasoning is True

    def test_separate_reasoning_false(self):
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            separate_reasoning=False,
        )
        assert request.separate_reasoning is False

    def test_separate_reasoning_true_explicit(self):
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            separate_reasoning=True,
        )
        assert request.separate_reasoning is True


class TestMergeReasoningIntoContent:
    """Tests for the _merge_reasoning_into_content helper."""

    def test_merge_reasoning_into_content(self):
        reasoning, content = _merge_reasoning_into_content(
            "Let me think...", "The answer is 42."
        )
        assert reasoning is None
        assert content == "Let me think...The answer is 42."

    def test_merge_reasoning_only(self):
        reasoning, content = _merge_reasoning_into_content("Thinking...", None)
        assert reasoning is None
        assert content == "Thinking..."

    def test_no_merge_when_no_reasoning(self):
        reasoning, content = _merge_reasoning_into_content(None, "Hello!")
        assert reasoning is None
        assert content == "Hello!"

    def test_both_none(self):
        reasoning, content = _merge_reasoning_into_content(None, None)
        assert reasoning is None
        assert content is None

    @pytest.mark.parametrize(
        "reasoning,content,expected_reasoning,expected_content",
        [
            ("thinking", "answer", None, "thinkinganswer"),
            ("thinking", None, None, "thinking"),
            (None, "answer", None, "answer"),
            (None, None, None, None),
        ],
    )
    def test_merge_matrix(
        self, reasoning, content, expected_reasoning, expected_content
    ):
        result_reasoning, result_content = _merge_reasoning_into_content(
            reasoning, content
        )
        assert result_reasoning == expected_reasoning
        assert result_content == expected_content


class TestDeltaMessageReasoningMerge:
    """Tests for merging reasoning into content in streaming DeltaMessage."""

    def test_streaming_merge_with_reasoning(self):
        """Simulate the streaming merge path from serving.py."""
        delta = DeltaMessage(reasoning="Let me think...", content="The answer is 42.")
        merged_reasoning, merged_content = _merge_reasoning_into_content(
            delta.reasoning, delta.content
        )
        delta = DeltaMessage(
            content=merged_content,
            reasoning=merged_reasoning,
            tool_calls=delta.tool_calls,
        )
        assert delta.reasoning is None
        assert delta.content == "Let me think...The answer is 42."

    def test_streaming_merge_preserves_tool_calls(self):
        delta = DeltaMessage(
            reasoning="Thinking...",
            content="",
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    function=DeltaFunctionCall(name="test", arguments="{}"),
                )
            ],
        )
        merged_reasoning, merged_content = _merge_reasoning_into_content(
            delta.reasoning, delta.content
        )
        delta = DeltaMessage(
            content=merged_content,
            reasoning=merged_reasoning,
            tool_calls=delta.tool_calls,
        )
        assert delta.reasoning is None
        assert len(delta.tool_calls) == 1
        assert delta.tool_calls[0].function.name == "test"
