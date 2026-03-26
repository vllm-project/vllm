# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
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

from vllm.entrypoints.openai.responses.utils import (
    _construct_single_message_from_response_item,
    _maybe_combine_prevmsg_and_tool_call,
    construct_chat_messages_with_tool_call,
    convert_tool_responses_to_completions_format,
    should_continue_final_message,
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


class TestReasoningItemContentPriority:
    """Tests that content is prioritized over summary for reasoning items."""

    def test_content_preferred_over_summary(self):
        """When both content and summary are present, content should win."""
        item = ResponseReasoningItem(
            id="reasoning_1",
            summary=[
                Summary(
                    text="This is a summary",
                    type="summary_text",
                )
            ],
            type="reasoning",
            content=[
                Content(
                    text="This is the actual content",
                    type="reasoning_text",
                )
            ],
            encrypted_content=None,
            status=None,
        )
        formatted = _construct_single_message_from_response_item(item)
        assert formatted["reasoning"] == "This is the actual content"

    def test_content_only(self):
        """When only content is present (no summary), content is used."""
        item = ResponseReasoningItem(
            id="reasoning_2",
            summary=[],
            type="reasoning",
            content=[
                Content(
                    text="Content without summary",
                    type="reasoning_text",
                )
            ],
            encrypted_content=None,
            status=None,
        )
        formatted = _construct_single_message_from_response_item(item)
        assert formatted["reasoning"] == "Content without summary"

    @patch("vllm.entrypoints.openai.responses.utils.logger")
    def test_summary_fallback_when_no_content(self, mock_logger):
        """When content is absent, summary is used as fallback with warning."""
        item = ResponseReasoningItem(
            id="reasoning_3",
            summary=[
                Summary(
                    text="Fallback summary text",
                    type="summary_text",
                )
            ],
            type="reasoning",
            content=None,
            encrypted_content=None,
            status=None,
        )
        formatted = _construct_single_message_from_response_item(item)
        assert formatted["reasoning"] == "Fallback summary text"
        mock_logger.warning.assert_called_once()
        assert (
            "summary text as reasoning content" in mock_logger.warning.call_args[0][0]
        )

    @patch("vllm.entrypoints.openai.responses.utils.logger")
    def test_summary_fallback_when_content_empty(self, mock_logger):
        """When content is an empty list, summary is used as fallback."""
        item = ResponseReasoningItem(
            id="reasoning_4",
            summary=[
                Summary(
                    text="Summary when content empty",
                    type="summary_text",
                )
            ],
            type="reasoning",
            content=[],
            encrypted_content=None,
            status=None,
        )
        formatted = _construct_single_message_from_response_item(item)
        assert formatted["reasoning"] == "Summary when content empty"
        mock_logger.warning.assert_called_once()
        assert (
            "summary text as reasoning content" in mock_logger.warning.call_args[0][0]
        )

    def test_neither_content_nor_summary(self):
        """When neither content nor summary is present, reasoning is empty."""
        item = ResponseReasoningItem(
            id="reasoning_5",
            summary=[],
            type="reasoning",
            content=None,
            encrypted_content=None,
            status=None,
        )
        formatted = _construct_single_message_from_response_item(item)
        assert formatted["reasoning"] == ""

    def test_encrypted_content_raises(self):
        """Encrypted content should still raise ValueError."""
        item = ResponseReasoningItem(
            id="reasoning_6",
            summary=[
                Summary(
                    text="Some summary",
                    type="summary_text",
                )
            ],
            type="reasoning",
            content=[
                Content(
                    text="Some content",
                    type="reasoning_text",
                )
            ],
            encrypted_content="ENCRYPTED",
            status=None,
        )
        with pytest.raises(ValueError):
            _construct_single_message_from_response_item(item)

    @patch("vllm.entrypoints.openai.responses.utils.logger")
    def test_summary_with_multiple_entries_uses_first(self, mock_logger):
        """When multiple summary entries exist, the first one is used."""
        item = ResponseReasoningItem(
            id="reasoning_7",
            summary=[
                Summary(
                    text="First summary",
                    type="summary_text",
                ),
                Summary(
                    text="Second summary",
                    type="summary_text",
                ),
            ],
            type="reasoning",
            content=None,
            encrypted_content=None,
            status=None,
        )
        formatted = _construct_single_message_from_response_item(item)
        assert formatted["reasoning"] == "First summary"
        mock_logger.warning.assert_called_once()
        assert (
            "summary text as reasoning content" in mock_logger.warning.call_args[0][0]
        )

    @patch("vllm.entrypoints.openai.responses.utils.logger")
    def test_no_warning_when_content_used(self, mock_logger):
        """No warning should be emitted when content is available."""
        item = ResponseReasoningItem(
            id="reasoning_8",
            summary=[
                Summary(
                    text="Summary text",
                    type="summary_text",
                )
            ],
            type="reasoning",
            content=[
                Content(
                    text="Content text",
                    type="reasoning_text",
                )
            ],
            encrypted_content=None,
            status=None,
        )
        _construct_single_message_from_response_item(item)
        mock_logger.warning.assert_not_called()


class TestShouldContinueFinalMessage:
    """Tests for should_continue_final_message function.

    This function enables Anthropic-style partial message completion, where
    users can provide an incomplete assistant message and have the model
    continue from where it left off.
    """

    def test_string_input_returns_false(self):
        """String input is always a user message, so should not continue."""
        assert should_continue_final_message("Hello, world!") is False

    def test_empty_list_returns_false(self):
        """Empty list should not continue."""
        assert should_continue_final_message([]) is False

    def test_completed_message_returns_false(self):
        """Completed message should not be continued."""
        output_item = ResponseOutputMessage(
            id="msg_123",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text="The answer is 42.",
                    type="output_text",
                    logprobs=None,
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        assert should_continue_final_message([output_item]) is False

    def test_in_progress_message_returns_true(self):
        """In-progress message should be continued.

        This is the key use case for partial message completion.
        Example: The user provides "The best answer is (" and wants
        the model to continue from there.
        """
        output_item = ResponseOutputMessage(
            id="msg_123",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text="The best answer is (",
                    type="output_text",
                    logprobs=None,
                )
            ],
            role="assistant",
            status="in_progress",
            type="message",
        )
        assert should_continue_final_message([output_item]) is True

    def test_incomplete_message_returns_true(self):
        """Incomplete message should be continued."""
        output_item = ResponseOutputMessage(
            id="msg_123",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text="The answer",
                    type="output_text",
                    logprobs=None,
                )
            ],
            role="assistant",
            status="incomplete",
            type="message",
        )
        assert should_continue_final_message([output_item]) is True

    def test_in_progress_reasoning_returns_true(self):
        """In-progress reasoning should be continued."""
        reasoning_item = ResponseReasoningItem(
            id="reasoning_123",
            summary=[],
            type="reasoning",
            content=[
                Content(
                    text="Let me think about this...",
                    type="reasoning_text",
                )
            ],
            encrypted_content=None,
            status="in_progress",
        )
        assert should_continue_final_message([reasoning_item]) is True

    def test_incomplete_reasoning_returns_true(self):
        """Incomplete reasoning should be continued."""
        reasoning_item = ResponseReasoningItem(
            id="reasoning_123",
            summary=[],
            type="reasoning",
            content=[
                Content(
                    text="Let me think",
                    type="reasoning_text",
                )
            ],
            encrypted_content=None,
            status="incomplete",
        )
        assert should_continue_final_message([reasoning_item]) is True

        reasoning_item = {
            "id": "reasoning_123",
            "summary": [],
            "type": "reasoning",
            "content": [],
            "status": "incomplete",
        }
        assert should_continue_final_message([reasoning_item]) is True

    def test_completed_reasoning_returns_false(self):
        """Completed reasoning should not be continued."""
        reasoning_item = ResponseReasoningItem(
            id="reasoning_123",
            summary=[],
            type="reasoning",
            content=[
                Content(
                    text="I have thought about this.",
                    type="reasoning_text",
                )
            ],
            encrypted_content=None,
            status="completed",
        )
        assert should_continue_final_message([reasoning_item]) is False

    def test_reasoning_with_none_status_returns_false(self):
        """Reasoning with None status should not be continued."""
        reasoning_item = ResponseReasoningItem(
            id="reasoning_123",
            summary=[],
            type="reasoning",
            content=[
                Content(
                    text="Some reasoning",
                    type="reasoning_text",
                )
            ],
            encrypted_content=None,
            status=None,
        )
        assert should_continue_final_message([reasoning_item]) is False

    def test_only_last_item_matters(self):
        """Only the last item in the list determines continuation."""
        completed_item = ResponseOutputMessage(
            id="msg_1",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text="Complete message.",
                    type="output_text",
                    logprobs=None,
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        in_progress_item = ResponseOutputMessage(
            id="msg_2",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text="Partial message...",
                    type="output_text",
                    logprobs=None,
                )
            ],
            role="assistant",
            status="in_progress",
            type="message",
        )

        # In-progress as last item -> should continue
        assert should_continue_final_message([completed_item, in_progress_item]) is True

        # Completed as last item -> should not continue
        assert (
            should_continue_final_message([in_progress_item, completed_item]) is False
        )

    def test_tool_call_returns_false(self):
        """Tool calls should not trigger continuation."""
        tool_call = ResponseFunctionToolCall(
            id="fc_123",
            call_id="call_123",
            type="function_call",
            status="in_progress",
            name="get_weather",
            arguments='{"location": "NYC"}',
        )
        assert should_continue_final_message([tool_call]) is False

        tool_call = {
            "id": "msg_123",
            "call_id": "call_123",
            "type": "function_call",
            "status": "in_progress",
            "name": "get_weather",
            "arguments": '{"location": "NYC"}',
        }
        assert should_continue_final_message([tool_call]) is False

    # Tests for dict inputs (e.g., from curl requests)
    def test_dict_in_progress_message_returns_true(self):
        """Dict with in_progress status should be continued (curl input)."""
        dict_item = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [{"type": "output_text", "text": "The answer is ("}],
        }
        assert should_continue_final_message([dict_item]) is True

    def test_dict_incomplete_message_returns_true(self):
        """Dict with incomplete status should be continued (curl input)."""
        dict_item = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "status": "incomplete",
            "content": [{"type": "output_text", "text": "Partial answer"}],
        }
        assert should_continue_final_message([dict_item]) is True

    def test_dict_completed_message_returns_false(self):
        """Dict with completed status should not be continued (curl input)."""
        dict_item = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Complete answer."}],
        }
        assert should_continue_final_message([dict_item]) is False

    def test_dict_reasoning_in_progress_returns_true(self):
        """Dict reasoning item with in_progress status should be continued."""
        dict_item = {
            "id": "reasoning_123",
            "type": "reasoning",
            "status": "in_progress",
            "content": [{"type": "reasoning_text", "text": "Let me think..."}],
        }
        assert should_continue_final_message([dict_item]) is True

    def test_dict_without_status_returns_false(self):
        """Dict without status field should not be continued."""
        dict_item = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Some text"}],
        }
        assert should_continue_final_message([dict_item]) is False

    def test_dict_with_none_status_returns_false(self):
        """Dict with None status should not be continued."""
        dict_item = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "status": None,
            "content": [{"type": "output_text", "text": "Some text"}],
        }
        assert should_continue_final_message([dict_item]) is False


class TestMaybeCombinePrevmsgAndToolCall:
    """Tests for _maybe_combine_prevmsg_and_tool_call function."""

    def test_combines_reasoning_and_tool_call(self):
        """Test successful combination of reasoning message and tool call."""
        item = ResponseFunctionToolCall(
            type="function_call",
            id="tool_id",
            call_id="call_123",
            name="test_function",
            arguments='{"arg": "value"}',
        )
        messages = [{"role": "assistant", "reasoning": "I need to call this tool"}]

        result = _maybe_combine_prevmsg_and_tool_call(item, messages)

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

        result = _maybe_combine_prevmsg_and_tool_call(item, messages)

        assert result is None

    def test_combines_content_and_tool_call(self):
        """Test combining content message (not reasoning) with tool call."""
        item = ResponseFunctionToolCall(
            type="function_call",
            id="tool_id",
            call_id="call_456",
            name="get_weather",
            arguments='{"city": "Tokyo"}',
        )
        messages = [{"role": "assistant", "content": "Let me check the weather"}]

        result = _maybe_combine_prevmsg_and_tool_call(item, messages)

        assert result is not None
        assert result["role"] == "assistant"
        assert result["content"] == "Let me check the weather"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_456"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_appends_multiple_tool_calls(self):
        """Test that multiple tool calls are appended to the same message."""
        # First tool call
        item1 = ResponseFunctionToolCall(
            type="function_call",
            id="tool_id_1",
            call_id="call_001",
            name="function_a",
            arguments='{"x": 1}',
        )
        messages = [{"role": "assistant", "content": "calling tools"}]
        result1 = _maybe_combine_prevmsg_and_tool_call(item1, messages)

        assert result1 is not None
        assert len(result1["tool_calls"]) == 1
        assert result1["tool_calls"][0]["function"]["name"] == "function_a"

        # Second tool call - should append, not replace
        item2 = ResponseFunctionToolCall(
            type="function_call",
            id="tool_id_2",
            call_id="call_002",
            name="function_b",
            arguments='{"y": 2}',
        )
        result2 = _maybe_combine_prevmsg_and_tool_call(item2, messages)

        assert result2 is not None
        assert len(result2["tool_calls"]) == 2
        assert result2["tool_calls"][0]["function"]["name"] == "function_a"
        assert result2["tool_calls"][1]["function"]["name"] == "function_b"

    def test_combines_three_tool_calls(self):
        """Test that three tool calls are combined into a single message."""
        # First tool call
        item1 = ResponseFunctionToolCall(
            type="function_call",
            id="tool_id_1",
            call_id="call_001",
            name="function_a",
            arguments='{"x": 1}',
        )
        messages = [{"role": "assistant", "content": "calling tools"}]
        result1 = _maybe_combine_prevmsg_and_tool_call(item1, messages)

        assert result1 is not None
        assert len(result1["tool_calls"]) == 1

        # Second tool call
        item2 = ResponseFunctionToolCall(
            type="function_call",
            id="tool_id_2",
            call_id="call_002",
            name="function_b",
            arguments='{"y": 2}',
        )
        result2 = _maybe_combine_prevmsg_and_tool_call(item2, messages)

        assert result2 is not None
        assert len(result2["tool_calls"]) == 2

        # Third tool call
        item3 = ResponseFunctionToolCall(
            type="function_call",
            id="tool_id_3",
            call_id="call_003",
            name="function_c",
            arguments='{"z": 3}',
        )
        result3 = _maybe_combine_prevmsg_and_tool_call(item3, messages)

        assert result3 is not None
        assert len(result3["tool_calls"]) == 3
        assert result3["tool_calls"][0]["function"]["name"] == "function_a"
        assert result3["tool_calls"][1]["function"]["name"] == "function_b"
        assert result3["tool_calls"][2]["function"]["name"] == "function_c"
