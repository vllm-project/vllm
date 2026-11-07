# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.generate.base.protocol import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser

pytestmark = pytest.mark.skip_global_cleanup


class _MockToolParser:
    """Mock tool parser implementation for testing."""

    supports_required_and_named = False


class _MockReasoningToolParser(DelegatingParser):
    """Mock delegating parser combining reasoning and tool call extraction."""

    def __init__(self):
        """Initialize mock parser with a mock tool parser."""
        super().__init__(tokenizer=None)
        self._tool_parser = _MockToolParser()

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """Mock check if reasoning has completed."""
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Mock extraction of content token IDs."""
        return input_ids

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int],
        current_token_ids: list[int],
        delta_token_ids: list[int],
    ):
        """Mock streaming extraction of reasoning."""
        return None

    def extract_reasoning(self, model_output: str, request):
        """Extract reasoning span delimited by <think>...</think> tags."""
        if "</think>" in model_output:
            parts = model_output.split("</think>")
            reasoning = parts[0].replace("<think>", "").strip()
            content = parts[1].strip() or None
            return reasoning, content
        return model_output, None

    def extract_tool_calls(self, content: str, request):
        """Extract tool calls containing <tool_call> tags."""
        if "<tool_call>" in content:
            tool_call = ToolCall(
                function=FunctionCall(
                    name="get_weather",
                    arguments='{"city": "Paris"}',
                )
            )
            clean_content = content.split("<tool_call>")[0].strip() or None
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[tool_call],
                content=clean_content,
            )
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=content,
        )


def test_tool_call_inside_think_region_is_extracted():
    """Verify that tool calls emitted inside the reasoning region (<think>...</think>)
    are properly extracted and not lost (regression test for Issue 39056).
    """
    parser = _MockReasoningToolParser()
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )

    # Case 1: Standard output where tool call is after </think>
    standard_output = (
        "<think>Let me think</think><tool_call><function=get_weather></tool_call>"
    )
    reasoning, content, tool_calls = parser.parse(
        standard_output, request, enable_auto_tools=True
    )
    assert reasoning == "Let me think"
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"

    # Case 2: Bug in Issue 39056 where tool call is inside <think>...</think>
    think_with_tool_call = (
        "<think>Let me think\n<tool_call><function=get_weather></tool_call></think>"
    )
    reasoning, content, tool_calls = parser.parse(
        think_with_tool_call, request, enable_auto_tools=True
    )
    # The tool call must NOT be lost
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    # The reasoning must not retain the raw tool_call markup and must preserve
    # non-tool reasoning
    assert "<tool_call>" not in (reasoning or "")
    assert reasoning == "Let me think"

    # Case 3: Named tool_choice with empty content and reasoning should not
    # convert reasoning into arguments
    named_request = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
        }
    )
    parser._tool_parser.supports_required_and_named = True
    reasoning, content, tool_calls = parser.parse(
        "<think>Just thinking, no tool call</think>",
        named_request,
        enable_auto_tools=True,
    )
    assert reasoning == "Just thinking, no tool call"
    assert tool_calls == []
