# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import (
    Content,
    ResponseReasoningItem,
    Summary,
)

from vllm.entrypoints.constants import MCP_PREFIX
from vllm.entrypoints.responses_utils import (
    _construct_single_message_from_response_item,
    _maybe_combine_reasoning_and_tool_call,
    construct_chat_messages_with_tool_call,
    convert_tool_responses_to_completions_format,
)


class TestResponsesUtils:
    """Tests for convert_tool_responses_to_completions_format function."""

    def test_convert_tool_responses_to_completions_format(self):
        """Test basic conversion of a flat tool schema to nested format."""
        input_tool = {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        }

        result = convert_tool_responses_to_completions_format(input_tool)

        assert result == {"type": "function", "function": input_tool}

    def test_construct_chat_messages_with_tool_call(self):
        """Test construction of chat messages with tool calls."""
        reasoning_item = ResponseReasoningItem(
            id="lol",
            summary=[],
            type="reasoning",
            content=[
                Content(
                    text="Leroy Jenkins",
                    type="reasoning_text",
                )
            ],
            encrypted_content=None,
            status=None,
        )
        mcp_tool_item = ResponseFunctionToolCall(
            id="mcp_123",
            call_id="call_123",
            type="function_call",
            status="completed",
            name="python",
            arguments='{"code": "123+456"}',
        )
        input_items = [reasoning_item, mcp_tool_item]
        messages = construct_chat_messages_with_tool_call(input_items)

        assert len(messages) == 1
        message = messages[0]
        assert message["role"] == "assistant"
        assert message["reasoning"] == "Leroy Jenkins"
        assert message["tool_calls"][0]["id"] == "call_123"
        assert message["tool_calls"][0]["function"]["name"] == "python"
        assert (
            message["tool_calls"][0]["function"]["arguments"] == '{"code": "123+456"}'
        )

    def test_construct_single_message_from_response_item(self):
        item = ResponseReasoningItem(
            id="lol",
            summary=[],
            type="reasoning",
            content=[
                Content(
                    text="Leroy Jenkins",
                    type="reasoning_text",
                )
            ],
            encrypted_content=None,
            status=None,
        )
        formatted_item = _construct_single_message_from_response_item(item)
        assert formatted_item["role"] == "assistant"
        assert formatted_item["reasoning"] == "Leroy Jenkins"

        item = ResponseReasoningItem(
            id="lol",
            summary=[
                Summary(
                    text='Hmm, the user has just started with a simple "Hello,"',
                    type="summary_text",
                )
            ],
            type="reasoning",
            content=None,
            encrypted_content=None,
            status=None,
        )

        formatted_item = _construct_single_message_from_response_item(item)
        assert formatted_item["role"] == "assistant"
        assert (
            formatted_item["reasoning"]
            == 'Hmm, the user has just started with a simple "Hello,"'
        )

        tool_call_output = ResponseFunctionToolCallOutputItem(
            id="temp_id",
            type="function_call_output",
            call_id="temp",
            output="1234",
            status="completed",
        )
        formatted_item = _construct_single_message_from_response_item(tool_call_output)
        assert formatted_item["role"] == "tool"
        assert formatted_item["content"] == "1234"
        assert formatted_item["tool_call_id"] == "temp"

        item = ResponseReasoningItem(
            id="lol",
            summary=[],
            type="reasoning",
            content=None,
            encrypted_content="TOP_SECRET_MESSAGE",
            status=None,
        )
        with pytest.raises(ValueError):
            _construct_single_message_from_response_item(item)

        output_item = ResponseOutputMessage(
            id="msg_bf585bbbe3d500e0",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text="dongyi",
                    type="output_text",
                    logprobs=None,
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )

        formatted_item = _construct_single_message_from_response_item(output_item)
        assert formatted_item["role"] == "assistant"
        assert formatted_item["content"] == "dongyi"


class TestMaybeCombineReasoningAndToolCall:
    """Tests for _maybe_combine_reasoning_and_tool_call function."""

    def test_returns_none_when_item_id_is_none(self):
        """
        Test fix from PR #31999: when item.id is None, should return None
        instead of raising TypeError on startswith().
        """
        item = ResponseFunctionToolCall(
            type="function_call",
            id=None,  # This was causing TypeError before the fix
            call_id="call_123",
            name="test_function",
            arguments="{}",
        )
        messages: list[ChatCompletionMessageParam] = []

        result = _maybe_combine_reasoning_and_tool_call(item, messages)

        assert result is None

    def test_returns_none_when_id_does_not_start_with_mcp_prefix(self):
        """Test that non-MCP tool calls are not combined."""
        item = ResponseFunctionToolCall(
            type="function_call",
            id="regular_id",  # Does not start with MCP_PREFIX
            call_id="call_123",
            name="test_function",
            arguments="{}",
        )
        messages = [{"role": "assistant", "reasoning": "some reasoning"}]

        result = _maybe_combine_reasoning_and_tool_call(item, messages)

        assert result is None

    def test_returns_none_when_last_message_is_not_assistant(self):
        """Test that non-assistant last message returns None."""
        item = ResponseFunctionToolCall(
            type="function_call",
            id=f"{MCP_PREFIX}tool_id",
            call_id="call_123",
            name="test_function",
            arguments="{}",
        )
        messages = [{"role": "user", "content": "hello"}]

        result = _maybe_combine_reasoning_and_tool_call(item, messages)

        assert result is None

    def test_returns_none_when_last_message_has_no_reasoning(self):
        """Test that assistant message without reasoning returns None."""
        item = ResponseFunctionToolCall(
            type="function_call",
            id=f"{MCP_PREFIX}tool_id",
            call_id="call_123",
            name="test_function",
            arguments="{}",
        )
        messages = [{"role": "assistant", "content": "some content"}]

        result = _maybe_combine_reasoning_and_tool_call(item, messages)

        assert result is None

    def test_combines_reasoning_and_mcp_tool_call(self):
        """Test successful combination of reasoning message and MCP tool call."""
        item = ResponseFunctionToolCall(
            type="function_call",
            id=f"{MCP_PREFIX}tool_id",
            call_id="call_123",
            name="test_function",
            arguments='{"arg": "value"}',
        )
        messages = [{"role": "assistant", "reasoning": "I need to call this tool"}]

        result = _maybe_combine_reasoning_and_tool_call(item, messages)

        assert result is not None
        assert result["role"] == "assistant"
        assert result["reasoning"] == "I need to call this tool"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["function"]["name"] == "test_function"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"arg": "value"}'
        assert result["tool_calls"][0]["type"] == "function"

    def test_returns_none_for_non_function_tool_call_type(self):
        """Test that non-ResponseFunctionToolCall items return None."""
        # Pass a dict instead of ResponseFunctionToolCall
        item = {"type": "message", "content": "hello"}
        messages = [{"role": "assistant", "reasoning": "some reasoning"}]

        result = _maybe_combine_reasoning_and_tool_call(item, messages)

        assert result is None

    def test_returns_none_when_id_is_empty_string(self):
        """Test that empty string id returns None (falsy check)."""
        item = ResponseFunctionToolCall(
            type="function_call",
            id="",  # Empty string is falsy
            call_id="call_123",
            name="test_function",
            arguments="{}",
        )
        messages = [{"role": "assistant", "reasoning": "some reasoning"}]

        result = _maybe_combine_reasoning_and_tool_call(item, messages)

        assert result is None
