# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

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
    _construct_message_from_response_item,
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

    def test_construct_message_from_response_item(self):
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
        formatted_item = _construct_message_from_response_item(item)
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

        formatted_item = _construct_message_from_response_item(item)
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
        formatted_item = _construct_message_from_response_item(tool_call_output)
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
            _construct_message_from_response_item(item)

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

        formatted_item = _construct_message_from_response_item(output_item)
        assert formatted_item["role"] == "assistant"
        assert formatted_item["content"] == "dongyi"


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


def _reasoning_item(text: str, item_id: str = "rs_1") -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id=item_id,
        summary=[],
        type="reasoning",
        content=[Content(text=text, type="reasoning_text")],
        encrypted_content=None,
        status=None,
    )


def _output_message(text: str, item_id: str = "msg_1") -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=item_id,
        content=[
            ResponseOutputText(
                annotations=[], text=text, type="output_text", logprobs=None
            )
        ],
        role="assistant",
        status="completed",
        type="message",
    )


def _function_call(
    name: str, arguments: str, call_id: str, item_id: str = "fc_1"
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        id=item_id,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


class TestConstructMessageFromResponseItemMerging:
    """Consecutive assistant output items (reasoning / message / function_call)
    are merged into a single assistant message, keyed purely by position (no
    shared turn id). A non-assistant item (tool result / user) ends the run.
    """

    def test_function_call_merges_into_prev_assistant(self):
        prev: dict[str, Any] = {"role": "assistant", "reasoning": "let me run it"}
        item = _function_call("shell", '{"cmd": "ls"}', "call_1")

        result = _construct_message_from_response_item(item, prev_msg=prev)

        # Merged in place -> no new message returned.
        assert result is None
        assert prev["reasoning"] == "let me run it"
        assert len(prev["tool_calls"]) == 1
        assert prev["tool_calls"][0]["id"] == "call_1"
        assert prev["tool_calls"][0]["function"]["name"] == "shell"

    def test_parallel_function_calls_appended(self):
        prev: dict[str, Any] = {"role": "assistant", "reasoning": "two calls"}
        first = _construct_message_from_response_item(
            _function_call("a", "{}", "call_a"), prev_msg=prev
        )
        second = _construct_message_from_response_item(
            _function_call("b", "{}", "call_b"), prev_msg=prev
        )

        assert first is None and second is None
        names = [tc["function"]["name"] for tc in prev["tool_calls"]]
        assert names == ["a", "b"]

    def test_function_call_new_message_when_prev_not_assistant(self):
        prev = {"role": "user", "content": "hello"}
        item = _function_call("shell", "{}", "call_1")

        result = _construct_message_from_response_item(item, prev_msg=prev)

        assert result is not None
        assert result["role"] == "assistant"
        assert result["tool_calls"][0]["function"]["name"] == "shell"

    def test_reasoning_fills_empty_slot_on_prev_assistant(self):
        # e.g. prev already has content but no reasoning.
        prev = {"role": "assistant", "content": "answer"}
        result = _construct_message_from_response_item(
            _reasoning_item("thinking"), prev_msg=prev
        )
        assert result is None
        assert prev["reasoning"] == "thinking"

    def test_output_message_fills_empty_content_slot(self):
        prev = {"role": "assistant", "reasoning": "thinking"}
        result = _construct_message_from_response_item(
            _output_message("the answer"), prev_msg=prev
        )
        assert result is None
        assert prev["content"] == "the answer"

    def test_namespaced_function_call_uses_flattened_name(self):
        prev: dict[str, Any] = {"role": "assistant", "reasoning": "call it"}
        item = ResponseFunctionToolCall(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            name="spawn_agent",
            namespace="multi_agent_v1",
            arguments="{}",
        )
        result = _construct_message_from_response_item(item, prev_msg=prev)
        assert result is None
        assert (
            prev["tool_calls"][0]["function"]["name"] == "multi_agent_v1__spawn_agent"
        )


class TestConstructChatMessagesEndToEnd:
    """End-to-end grouping via construct_chat_messages_with_tool_call."""

    def test_single_turn_reasoning_message_and_tool_call_merged(self):
        items = [
            _reasoning_item("think"),
            _output_message("intro text"),
            _function_call("shell", '{"cmd": "ls"}', "call_1"),
        ]
        messages = construct_chat_messages_with_tool_call(items)

        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert msg["reasoning"] == "think"
        assert msg["content"] == "intro text"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "shell"

    def test_tool_result_starts_new_turn(self):
        """A function_call_output ends the assistant run; the following
        reasoning starts a fresh assistant message."""
        items = [
            _reasoning_item("first turn think", item_id="rs_1"),
            _function_call("shell", "{}", "call_1"),
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result-data",
            },
            _reasoning_item("second turn think", item_id="rs_2"),
            _function_call("shell", "{}", "call_2"),
        ]
        messages = construct_chat_messages_with_tool_call(items)

        assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]
        assert messages[0]["reasoning"] == "first turn think"
        assert messages[0]["tool_calls"][0]["id"] == "call_1"
        assert messages[1]["content"] == "result-data"
        assert messages[2]["reasoning"] == "second turn think"
        assert messages[2]["tool_calls"][0]["id"] == "call_2"

    def test_parallel_tool_calls_same_turn(self):
        items = [
            _reasoning_item("do two things"),
            _function_call("a", "{}", "call_a"),
            _function_call("b", "{}", "call_b"),
        ]
        messages = construct_chat_messages_with_tool_call(items)

        assert len(messages) == 1
        names = [tc["function"]["name"] for tc in messages[0]["tool_calls"]]
        assert names == ["a", "b"]
