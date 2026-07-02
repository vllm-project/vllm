# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.entrypoints.openai.responses.harmony."""

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from openai.types.responses.response_output_item import McpCall
from openai_harmony import Author, Message, Role, TextContent

from vllm.entrypoints.openai.responses.harmony import (
    harmony_to_response_output,
    response_previous_input_to_harmony,
)


class TestResponsePreviousInputToHarmony:
    """
    Tests for scenarios that are specific to the Responses API
    response_previous_input_to_harmony function.
    """

    def test_message_with_empty_content(self):
        """Test parsing message with empty string content."""
        chat_msg = {
            "role": "user",
            "content": "",
        }

        messages = response_previous_input_to_harmony(chat_msg)

        assert len(messages) == 1
        assert messages[0].content[0].text == ""

    def test_tool_message_with_string_content(self):
        """Test parsing tool message with string content."""
        chat_msg = {
            "role": "tool",
            "name": "get_weather",
            "content": "The weather in San Francisco is sunny, 72°F",
        }

        messages = response_previous_input_to_harmony(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.TOOL
        assert messages[0].author.name == "functions.get_weather"
        assert (
            messages[0].content[0].text == "The weather in San Francisco is sunny, 72°F"
        )
        assert messages[0].channel == "commentary"

    def test_tool_message_with_array_content(self):
        """Test parsing tool message with array content."""
        chat_msg = {
            "role": "tool",
            "name": "search_results",
            "content": [
                {"type": "text", "text": "Result 1: "},
                {"type": "text", "text": "Result 2: "},
                {
                    "type": "image",
                    "url": "http://example.com/img.png",
                },  # Should be ignored
                {"type": "text", "text": "Result 3"},
            ],
        }

        messages = response_previous_input_to_harmony(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.TOOL
        assert messages[0].author.name == "functions.search_results"
        assert messages[0].content[0].text == "Result 1: Result 2: Result 3"

    def test_tool_message_with_empty_content(self):
        """Test parsing tool message with None content."""
        chat_msg = {
            "role": "tool",
            "name": "empty_tool",
            "content": None,
        }

        messages = response_previous_input_to_harmony(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.TOOL
        assert messages[0].author.name == "functions.empty_tool"
        assert messages[0].content[0].text == ""


class TestHarmonyToResponseOutput:
    """Tests for harmony_to_response_output function."""

    @pytest.mark.parametrize("incomplete", [False, True])
    def test_commentary_with_no_recipient_creates_message(self, incomplete):
        """Test that commentary with recipient=None (preambles) creates message items.

        Per Harmony format, preambles are intended to be shown to end-users,
        unlike analysis channel content which is hidden reasoning.
        See: https://cookbook.openai.com/articles/openai-harmony
        """
        message = Message.from_role_and_content(
            Role.ASSISTANT, "I will now search for the weather information."
        )
        message = message.with_channel("commentary")
        # recipient is None by default, representing a preamble

        output_items = harmony_to_response_output(
            message, frozenset(), incomplete=incomplete
        )

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseOutputMessage)
        assert output_items[0].type == "message"
        assert output_items[0].role == "assistant"
        assert output_items[0].status == ("incomplete" if incomplete else "completed")
        assert len(output_items[0].content) == 1
        assert output_items[0].content[0].type == "output_text"
        assert (
            output_items[0].content[0].text
            == "I will now search for the weather information."
        )

    @pytest.mark.parametrize("channel", ["commentary", "comment", "analysis", "final"])
    @pytest.mark.parametrize(
        ("recipient", "fn_names", "expected_name"),
        [
            ("functions.get_weather", frozenset(), "get_weather"),
            ("get_weather", frozenset({"get_weather"}), "get_weather"),
            ("math.sum", frozenset({"math.sum"}), "math.sum"),
        ],
    )
    @pytest.mark.parametrize("incomplete", [False, True])
    def test_function_recipient_creates_function_call(
        self, channel, recipient, fn_names, expected_name, incomplete
    ):
        """Function recipients create function calls across channels."""
        content = '{"location": "San Francisco"}'
        if recipient == "math.sum":
            content = '{"a": 1, "b": 2}'

        message = Message.from_role_and_content(Role.ASSISTANT, content)
        message = message.with_channel(channel)
        message = message.with_recipient(recipient)

        output_items = harmony_to_response_output(
            message, fn_names, incomplete=incomplete
        )

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionToolCall)
        assert output_items[0].type == "function_call"
        assert output_items[0].name == expected_name
        assert output_items[0].arguments == content
        assert output_items[0].call_id.startswith("call_")
        assert output_items[0].id.startswith("fc_")
        assert output_items[0].status == ("incomplete" if incomplete else "completed")

    @pytest.mark.parametrize("channel", ["commentary", "comment", "analysis", "final"])
    @pytest.mark.parametrize(
        ("recipient", "content"),
        [
            ("python", "import numpy as np\nprint(np.array([1, 2, 3]))"),
            ("browser", "Navigating to the specified URL"),
            ("container", "Running command in container"),
        ],
    )
    @pytest.mark.parametrize("incomplete", [False, True])
    def test_builtin_recipient_creates_reasoning(
        self, channel, recipient, content, incomplete
    ):
        """Built-in recipients create reasoning items."""
        message = Message.from_role_and_content(Role.ASSISTANT, content)
        message = message.with_channel(channel)
        message = message.with_recipient(recipient)

        output_items = harmony_to_response_output(
            message, frozenset(), incomplete=incomplete
        )

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].type == "reasoning"
        assert output_items[0].content[0].text == content
        assert output_items[0].status is None

    @pytest.mark.parametrize("channel", ["commentary", "comment", "analysis", "final"])
    @pytest.mark.parametrize(
        ("recipient", "fn_names", "content", "expected_name", "expected_server_label"),
        [
            (
                "get_weather",
                frozenset(),
                '{"arg": "value"}',
                "get_weather",
                "get_weather",
            ),
            (
                "not_get_weather",
                frozenset({"get_weather"}),
                '{"arg": "value"}',
                "not_get_weather",
                "not_get_weather",
            ),
            ("repo_browser.list", frozenset(), '{"cmd": "ls"}', "list", "repo_browser"),
        ],
    )
    @pytest.mark.parametrize("incomplete", [False, True])
    def test_non_function_non_builtin_recipient_creates_mcp_call(
        self,
        channel,
        recipient,
        fn_names,
        content,
        expected_name,
        expected_server_label,
        incomplete,
    ):
        """Non-function, non-built-in recipients create MCP calls."""
        message = Message.from_role_and_content(Role.ASSISTANT, content)
        message = message.with_channel(channel)
        message = message.with_recipient(recipient)

        output_items = harmony_to_response_output(
            message, fn_names, incomplete=incomplete
        )

        assert len(output_items) == 1
        assert isinstance(output_items[0], McpCall)
        assert output_items[0].type == "mcp_call"
        assert output_items[0].name == expected_name
        assert output_items[0].server_label == expected_server_label
        assert output_items[0].arguments == content
        assert output_items[0].status == ("incomplete" if incomplete else "completed")

    @pytest.mark.parametrize("incomplete", [False, True])
    def test_browser_search_recipient_respects_incomplete(self, incomplete):
        """browser.search emits a web search call unless the item is incomplete."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, '{"query": "weather in San Francisco"}'
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("browser.search")

        output_items = harmony_to_response_output(
            message, frozenset(), incomplete=incomplete
        )

        if incomplete:
            assert output_items == []
            return

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionWebSearch)
        assert output_items[0].type == "web_search_call"
        assert output_items[0].status == "completed"
        assert output_items[0].action.type == "search"
        assert output_items[0].action.query == "cursor:weather in San Francisco"

    def test_commentary_with_empty_content_and_no_recipient(self):
        """Test edge case: empty commentary with recipient=None."""
        message = Message.from_role_and_content(Role.ASSISTANT, "")
        message = message.with_channel("commentary")

        output_items = harmony_to_response_output(message, frozenset())

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseOutputMessage)
        assert output_items[0].content[0].text == ""

    def test_commentary_with_multiple_contents_and_no_recipient(self):
        """Test multiple content items in commentary with no recipient."""
        contents = [
            TextContent(text="Step 1: Analyze the request"),
            TextContent(text="Step 2: Prepare to call functions"),
        ]
        message = Message.from_role_and_contents(Role.ASSISTANT, contents)
        message = message.with_channel("commentary")

        output_items = harmony_to_response_output(message, frozenset())

        # _parse_final_message returns single ResponseOutputMessage with
        # multiple contents
        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseOutputMessage)
        assert len(output_items[0].content) == 2
        assert output_items[0].content[0].text == "Step 1: Analyze the request"
        assert output_items[0].content[1].text == "Step 2: Prepare to call functions"

    def test_commentary_with_multiple_function_calls(self):
        """Test multiple function calls in commentary channel."""
        contents = [
            TextContent(text='{"location": "San Francisco"}'),
            TextContent(text='{"location": "New York"}'),
        ]
        message = Message.from_role_and_contents(Role.ASSISTANT, contents)
        message = message.with_channel("commentary")
        message = message.with_recipient("functions.get_weather")

        output_items = harmony_to_response_output(message, frozenset())

        assert len(output_items) == 2
        assert all(isinstance(item, ResponseFunctionToolCall) for item in output_items)
        assert output_items[0].name == "get_weather"
        assert output_items[1].name == "get_weather"
        assert output_items[0].arguments == '{"location": "San Francisco"}'
        assert output_items[1].arguments == '{"location": "New York"}'

    def test_analysis_channel_creates_reasoning(self):
        """Test that analysis channel creates reasoning items."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, "Analyzing the problem step by step..."
        )
        message = message.with_channel("analysis")

        output_items = harmony_to_response_output(message, frozenset())

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].type == "reasoning"
        assert (
            output_items[0].content[0].text == "Analyzing the problem step by step..."
        )

    def test_non_assistant_message_returns_empty(self):
        """Test that non-assistant messages return empty list.

        Per the implementation, tool messages to assistant (e.g., search results)
        are not included in final output to align with OpenAI behavior.
        """
        message = Message.from_author_and_content(
            Author.new(Role.TOOL, "functions.get_weather"),
            "The weather is sunny, 72°F",
        )

        output_items = harmony_to_response_output(message, frozenset())

        assert len(output_items) == 0
