# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for consecutive function_call merging in
construct_chat_messages_with_tool_call.

Regression test: when Responses API input contains consecutive
ResponseFunctionToolCall items (parallel tool calls), they should be merged
into a single assistant message with multiple tool_calls entries, matching
the Chat Completions format that models were trained on.
"""

import pytest
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall,
)
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText

from vllm.entrypoints.openai.responses.utils import (
    construct_chat_messages_with_tool_call,
)

pytestmark = pytest.mark.cpu_test


def _make_function_call(name: str, args: str, call_id: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{call_id}",
        call_id=f"call_{call_id}",
        type="function_call",
        status="completed",
        name=name,
        arguments=args,
    )


def _make_tool_output(call_id: str, output: str) -> ResponseFunctionToolCallOutputItem:
    return ResponseFunctionToolCallOutputItem(
        id=f"fco_{call_id}",
        type="function_call_output",
        call_id=f"call_{call_id}",
        output=output,
        status="completed",
    )


def _make_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1",
        content=[
            ResponseOutputText(
                annotations=[],
                text=text,
                type="output_text",
                logprobs=None,
            )
        ],
        role="assistant",
        status="completed",
        type="message",
    )


class TestConsecutiveFunctionCallMerge:

    def test_two_consecutive_calls_merged(self):
        """Two consecutive function_call items → 1 assistant message with 2 tool_calls."""
        items = [
            _make_function_call("get_weather", '{"city": "NYC"}', "1"),
            _make_function_call("get_weather", '{"city": "LA"}', "2"),
        ]
        messages = construct_chat_messages_with_tool_call(items)

        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"city": "NYC"}'
        assert msg["tool_calls"][0]["id"] == "call_1"
        assert msg["tool_calls"][1]["function"]["name"] == "get_weather"
        assert msg["tool_calls"][1]["function"]["arguments"] == '{"city": "LA"}'
        assert msg["tool_calls"][1]["id"] == "call_2"

    def test_three_consecutive_calls_merged(self):
        """Three consecutive function_call items → 1 assistant message with 3 tool_calls."""
        items = [
            _make_function_call("fn_a", '{"x": 1}', "a"),
            _make_function_call("fn_b", '{"y": 2}', "b"),
            _make_function_call("fn_c", '{"z": 3}', "c"),
        ]
        messages = construct_chat_messages_with_tool_call(items)

        assert len(messages) == 1
        assert len(messages[0]["tool_calls"]) == 3

    def test_message_then_call_not_merged(self):
        """ResponseOutputMessage followed by function_call → 2 separate messages."""
        items = [
            _make_message("Hello"),
            _make_function_call("fn", '{}', "1"),
        ]
        messages = construct_chat_messages_with_tool_call(items)

        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 1

    def test_tool_output_between_calls_breaks_merge(self):
        """function_call, function_call_output, function_call → 3 messages (not merged)."""
        items = [
            _make_function_call("fn_a", '{}', "a"),
            _make_tool_output("a", "result_a"),
            _make_function_call("fn_b", '{}', "b"),
        ]
        messages = construct_chat_messages_with_tool_call(items)

        assert len(messages) == 3
        # First: assistant with tool_calls
        assert messages[0]["role"] == "assistant"
        assert len(messages[0]["tool_calls"]) == 1
        # Second: tool output
        assert messages[1]["role"] == "tool"
        assert messages[1]["content"] == "result_a"
        # Third: new assistant with tool_calls
        assert messages[2]["role"] == "assistant"
        assert len(messages[2]["tool_calls"]) == 1

    def test_single_call_unchanged(self):
        """A single function_call still produces 1 assistant message with 1 tool_call."""
        items = [_make_function_call("fn", '{"k": "v"}', "1")]
        messages = construct_chat_messages_with_tool_call(items)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert len(messages[0]["tool_calls"]) == 1

    def test_parallel_calls_then_outputs(self):
        """Two parallel calls followed by two outputs → assistant(2 tools) + 2 tool msgs."""
        items = [
            _make_function_call("fn_a", '{}', "a"),
            _make_function_call("fn_b", '{}', "b"),
            _make_tool_output("a", "res_a"),
            _make_tool_output("b", "res_b"),
        ]
        messages = construct_chat_messages_with_tool_call(items)

        assert len(messages) == 3
        # Merged assistant message
        assert messages[0]["role"] == "assistant"
        assert len(messages[0]["tool_calls"]) == 2
        # Tool outputs
        assert messages[1]["role"] == "tool"
        assert messages[2]["role"] == "tool"
