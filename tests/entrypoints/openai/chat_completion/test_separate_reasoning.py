# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for separate_reasoning parameter."""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage


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


class TestDeltaMessageReasoningMerge:
    """Tests for merging reasoning into content in DeltaMessage."""

    def test_merge_reasoning_into_content(self):
        """When separate_reasoning=False, reasoning should merge into content."""
        delta = DeltaMessage(reasoning="Let me think...", content="The answer is 42.")
        merged = DeltaMessage(
            content=(delta.reasoning or "") + (delta.content or ""),
            reasoning=None,
            tool_calls=delta.tool_calls,
        )
        assert merged.reasoning is None
        assert merged.content == "Let me think...The answer is 42."

    def test_merge_reasoning_only(self):
        """When there is reasoning but no content, reasoning becomes content."""
        delta = DeltaMessage(reasoning="Thinking...", content=None)
        merged = DeltaMessage(
            content=(delta.reasoning or "") + (delta.content or ""),
            reasoning=None,
            tool_calls=delta.tool_calls,
        )
        assert merged.reasoning is None
        assert merged.content == "Thinking..."

    def test_no_merge_when_no_reasoning(self):
        """When there is no reasoning, content stays unchanged."""
        delta = DeltaMessage(reasoning=None, content="Hello!")
        merged = DeltaMessage(
            content=(delta.reasoning or "") + (delta.content or ""),
            reasoning=None,
            tool_calls=delta.tool_calls,
        )
        assert merged.reasoning is None
        assert merged.content == "Hello!"

    def test_merge_preserves_tool_calls(self):
        """Tool calls should be preserved during merge."""
        from vllm.entrypoints.openai.engine.protocol import (
            DeltaFunctionCall,
            DeltaToolCall,
        )

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
        merged = DeltaMessage(
            content=(delta.reasoning or "") + (delta.content or ""),
            reasoning=None,
            tool_calls=delta.tool_calls,
        )
        assert merged.reasoning is None
        assert len(merged.tool_calls) == 1
        assert merged.tool_calls[0].function.name == "test"

    @pytest.mark.parametrize(
        "separate_reasoning,has_reasoning,expected_content,expected_reasoning",
        [
            (True, True, "answer", "thinking"),
            (False, True, "thinkinganswer", None),
            (True, False, "answer", None),
            (False, False, "answer", None),
        ],
    )
    def test_merge_matrix(
        self, separate_reasoning, has_reasoning, expected_content, expected_reasoning
    ):
        """Test all combinations of separate_reasoning and reasoning presence."""
        reasoning = "thinking" if has_reasoning else None
        content = "answer"

        if not separate_reasoning and reasoning:
            content = reasoning + content
            reasoning = None

        assert content == expected_content
        assert reasoning == expected_reasoning
