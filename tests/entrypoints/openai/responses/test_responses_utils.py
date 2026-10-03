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
    _construct_message_from_response_item,
    apply_streamed_item_ids,
    construct_chat_messages_with_tool_call,
    construct_input_messages,
    should_continue_final_message,
)
from vllm.exceptions import VLLMValidationError


def _single_chat_message(item):
    message = _construct_message_from_response_item(item)
    assert message is not None
    return message


def make_output_message(
    text: str,
    *,
    id: str = "msg_1",
    status: str = "completed",
) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=id,
        content=[
            ResponseOutputText(
                annotations=[],
                text=text,
                type="output_text",
                logprobs=None,
            )
        ],
        role="assistant",
        status=status,
        type="message",
    )


def make_reasoning_item(
    *,
    content_text: str | None = None,
    summary_text: str | None = None,
    content: list[Content] | None = None,
    summary: list[Summary] | None = None,
    encrypted_content: str | None = None,
    id: str = "reasoning_1",
    status: str | None = None,
) -> ResponseReasoningItem:
    if content is None and content_text is not None:
        content = [Content(text=content_text, type="reasoning_text")]
    if summary is None and summary_text is not None:
        summary = [Summary(text=summary_text, type="summary_text")]

    return ResponseReasoningItem(
        id=id,
        summary=[] if summary is None else summary,
        type="reasoning",
        content=content,
        encrypted_content=encrypted_content,
        status=status,
    )


def make_function_call(
    *,
    call_id: str,
    name: str = "test_function",
    arguments: str = "{}",
    id: str = "tool_id",
    status: str | None = None,
) -> ResponseFunctionToolCall:
    kwargs = {
        "type": "function_call",
        "id": id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }
    if status is not None:
        kwargs["status"] = status

    return ResponseFunctionToolCall(**kwargs)


def make_function_call_output(
    *,
    call_id: str,
    output: str = "42",
    id: str = "output_1",
    status: str = "completed",
) -> ResponseFunctionToolCallOutputItem:
    return ResponseFunctionToolCallOutputItem(
        id=id,
        type="function_call_output",
        call_id=call_id,
        output=output,
        status=status,
    )


class TestResponsesUtils:
    """Tests for Responses API utils."""

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

    def test_construct_chat_messages_preserves_single_item_conversions(self):
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
        formatted_item = _single_chat_message(item)
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

        formatted_item = _single_chat_message(item)
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
        formatted_item = _single_chat_message(tool_call_output)
        assert formatted_item["role"] == "tool"
        assert formatted_item["content"] == "1234"
        assert formatted_item["tool_call_id"] == "temp"

        formatted_item = _single_chat_message(
            {
                "type": "function_call_output",
                "call_id": "temp_dict",
                "output": "5678",
            }
        )
        assert formatted_item["role"] == "tool"
        assert formatted_item["content"] == "5678"
        assert formatted_item["tool_call_id"] == "temp_dict"

        item = ResponseReasoningItem(
            id="lol",
            summary=[],
            type="reasoning",
            content=None,
            encrypted_content="TOP_SECRET_MESSAGE",
            status=None,
        )
        with pytest.raises(VLLMValidationError) as exc_info:
            construct_chat_messages_with_tool_call([item])
        assert exc_info.value.parameter == "input"

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

        formatted_item = _single_chat_message(output_item)
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
        formatted = _single_chat_message(item)
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
        formatted = _single_chat_message(item)
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
        formatted = _single_chat_message(item)
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
        formatted = _single_chat_message(item)
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
        formatted = _single_chat_message(item)
        assert formatted["reasoning"] == ""

    def test_encrypted_content_raises(self):
        """Encrypted content should raise VLLMValidationError."""
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
        with pytest.raises(VLLMValidationError) as exc_info:
            construct_chat_messages_with_tool_call([item])
        assert exc_info.value.parameter == "input"

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
        formatted = _single_chat_message(item)
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
        construct_chat_messages_with_tool_call([item])
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


class TestConstructChatMessagesCombinePolicy:
    """Tests for contiguous assistant-side merging."""

    @pytest.mark.parametrize(
        ("items", "expected_content", "expected_reasoning", "expected_tool_call_ids"),
        [
            pytest.param(
                [
                    make_reasoning_item(content_text="Let me think"),
                    make_output_message("Hello"),
                ],
                "Hello",
                "Let me think",
                None,
                id="reasoning-output-messages",
            ),
            pytest.param(
                [
                    make_function_call(call_id="call_123"),
                    make_function_call(call_id="call_456"),
                ],
                None,
                None,
                ["call_123", "call_456"],
                id="consecutive-tool-calls",
            ),
            pytest.param(
                [
                    make_reasoning_item(content_text="Let me think"),
                    make_function_call(call_id="call_123"),
                ],
                None,
                "Let me think",
                ["call_123"],
                id="reasoning-tool-call",
            ),
            pytest.param(
                [
                    make_output_message("Hello"),
                    make_function_call(call_id="call_123"),
                ],
                "Hello",
                None,
                ["call_123"],
                id="output-tool-call",
            ),
            pytest.param(
                [
                    make_reasoning_item(content_text="Thinking"),
                    make_output_message("Hello"),
                    make_function_call(call_id="call_123"),
                    make_function_call(call_id="call_456"),
                ],
                "Hello",
                "Thinking",
                ["call_123", "call_456"],
                id="reasoning-output-tool-call",
            ),
            pytest.param(
                [
                    make_reasoning_item(content_text="Let me think"),
                    {"type": "message", "role": "assistant", "content": "Hello"},
                ],
                "Hello",
                "Let me think",
                None,
                id="reasoning-easyinput-assistant",
            ),
        ],
    )
    def test_assistant_side_items_merge_until_tool_output(
        self,
        items,
        expected_content,
        expected_reasoning,
        expected_tool_call_ids,
    ):
        messages = construct_chat_messages_with_tool_call(items)

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        if expected_content is None:
            assert "content" not in messages[0]
        else:
            assert messages[0]["content"] == expected_content
        if expected_reasoning is None:
            assert "reasoning" not in messages[0]
        else:
            assert messages[0]["reasoning"] == expected_reasoning
        if expected_tool_call_ids is None:
            assert "tool_calls" not in messages[0]
        else:
            assert [tool_call["id"] for tool_call in messages[0]["tool_calls"]] == (
                expected_tool_call_ids
            )

    @pytest.mark.parametrize(
        ("items", "num_expected_messages"),
        [
            pytest.param(
                [
                    make_output_message("Hello"),
                    make_output_message("World"),
                ],
                2,
                id="consecutive-output-messages",
            ),
            pytest.param(
                [
                    make_reasoning_item(content_text="Let me think"),
                    make_reasoning_item(content_text="Let me think more"),
                ],
                2,
                id="consecutive-reasoning-messages",
            ),
            pytest.param(
                [
                    make_function_call(call_id="call_123"),
                    make_function_call_output(call_id="call_123", output="42"),
                    make_function_call(call_id="call_456"),
                ],
                3,
                id="interrupted-by-non-assistant-item",
            ),
        ],
    )
    def test_merge_chain_breaks(self, items, num_expected_messages):
        messages = construct_chat_messages_with_tool_call(items)
        assert len(messages) == num_expected_messages


class TestConstructInputMessagesInstructionsLeak:
    """Regression tests for #37697: instructions from a prior response
    should NOT leak through previous_response_id."""

    def test_old_instructions_stripped_from_prev_msg(self):
        """System message in prev_msg must be dropped so the new request's
        instructions are the only system message in the conversation."""
        prev = [
            {"role": "system", "content": "old instructions"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        msgs = construct_input_messages(
            request_instructions="new instructions",
            request_input="What is 3+3?",
            prev_msg=prev,
        )
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "new instructions"

    def test_no_instructions_in_new_request(self):
        """If the new request has no instructions, old ones should still
        be stripped -- they must not carry over."""
        prev = [
            {"role": "system", "content": "old instructions"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        msgs = construct_input_messages(
            request_instructions=None,
            request_input="What is 3+3?",
            prev_msg=prev,
        )
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        assert len(system_msgs) == 0

    def test_non_system_messages_preserved(self):
        """User/assistant messages from prev_msg must remain intact."""
        prev = [
            {"role": "system", "content": "old instructions"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        msgs = construct_input_messages(
            request_instructions="new instructions",
            request_input="Follow up",
            prev_msg=prev,
        )
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]
        assert msgs[0]["content"] == "new instructions"
        assert msgs[1]["content"] == "Hi"
        assert msgs[2]["content"] == "Hello"
        assert msgs[3]["content"] == "Follow up"

    def test_no_prev_msg(self):
        """Baseline: when there's no prev_msg, instructions work normally."""
        msgs = construct_input_messages(
            request_instructions="be helpful",
            request_input="hello",
            prev_msg=None,
        )
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "be helpful"}
        assert msgs[1] == {"role": "user", "content": "hello"}


class TestApplyStreamedItemIds:
    def test_overlays_id_and_call_id(self):
        output = [
            make_output_message("rebuilt text", id="msg_rebuilt"),
            make_function_call(
                name="get_weather", id="fc_rebuilt", call_id="call_rebuilt"
            ),
        ]
        streamed = [
            make_output_message("streamed text", id="msg_streamed"),
            make_function_call(
                name="get_weather", id="fc_streamed", call_id="call_streamed"
            ),
        ]

        result = apply_streamed_item_ids(output, streamed)

        assert [item.id for item in result] == ["msg_streamed", "fc_streamed"]
        assert result[1].call_id == "call_streamed"
        # Only ids are overlaid; the rebuilt content wins.
        assert result[0].content[0].text == "rebuilt text"

    def test_same_name_calls_pair_in_order(self):
        output = [
            make_function_call(name="get_weather", id="fc_r1", call_id="call_r1"),
            make_function_call(name="get_weather", id="fc_r2", call_id="call_r2"),
        ]
        streamed = [
            make_function_call(name="get_weather", id="fc_s1", call_id="call_s1"),
            make_function_call(name="get_weather", id="fc_s2", call_id="call_s2"),
        ]

        result = apply_streamed_item_ids(output, streamed)

        assert [item.id for item in result] == ["fc_s1", "fc_s2"]
        assert [item.call_id for item in result] == ["call_s1", "call_s2"]

    def test_function_calls_matched_by_name(self):
        output = [
            make_function_call(name="get_weather", id="fc_rw", call_id="call_rw"),
            make_function_call(name="get_news", id="fc_rn", call_id="call_rn"),
        ]
        streamed = [
            make_function_call(name="get_weather", id="fc_sw", call_id="call_sw"),
            make_function_call(name="get_news", id="fc_sn", call_id="call_sn"),
        ]

        result = apply_streamed_item_ids(output, streamed)

        assert [item.id for item in result] == ["fc_sw", "fc_sn"]

    def test_item_without_streamed_counterpart_keeps_fresh_id(self):
        # Zero-delta items never emitted a done event, so they have no
        # streamed counterpart and must keep their freshly minted id.
        output = [
            make_reasoning_item(id="rs_rebuilt"),
            make_output_message("hi", id="msg_rebuilt"),
        ]
        streamed = [make_output_message("hi", id="msg_streamed")]

        result = apply_streamed_item_ids(output, streamed)

        assert result[0].id == "rs_rebuilt"
        assert result[1].id == "msg_streamed"

    def test_streamed_gap_does_not_shift_later_matches(self):
        output = [
            make_output_message("one", id="msg_r1"),
            make_function_call(name="get_weather", id="fc_r", call_id="call_r"),
            make_output_message("two", id="msg_r2"),
        ]
        streamed = [
            make_output_message("one", id="msg_s1"),
            make_output_message("two", id="msg_s2"),
        ]

        result = apply_streamed_item_ids(output, streamed)

        assert [item.id for item in result] == ["msg_s1", "fc_r", "msg_s2"]

    def test_no_streamed_items_is_noop(self):
        output = [make_output_message("hi", id="msg_rebuilt")]

        assert apply_streamed_item_ids(output, [])[0].id == "msg_rebuilt"
        assert apply_streamed_item_ids(output, None)[0].id == "msg_rebuilt"
