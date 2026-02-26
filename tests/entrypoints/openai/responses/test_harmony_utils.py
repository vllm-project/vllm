# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.entrypoints.openai.responses.harmony."""

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from openai.types.responses.response_output_item import McpCall
from openai_harmony import Author, Message, Role, TextContent

from vllm.entrypoints.openai.responses.harmony import (
    harmony_to_response_output,
    parser_state_to_response_output,
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

    def test_commentary_with_no_recipient_creates_message(self):
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

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseOutputMessage)
        assert output_items[0].type == "message"
        assert output_items[0].role == "assistant"
        assert output_items[0].status == "completed"
        assert len(output_items[0].content) == 1
        assert output_items[0].content[0].type == "output_text"
        assert (
            output_items[0].content[0].text
            == "I will now search for the weather information."
        )

    def test_commentary_with_function_recipient_creates_function_call(self):
        """Test commentary with recipient='functions.X' creates function calls."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, '{"location": "San Francisco", "units": "celsius"}'
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("functions.get_weather")

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionToolCall)
        assert output_items[0].type == "function_call"
        assert output_items[0].name == "get_weather"
        assert (
            output_items[0].arguments
            == '{"location": "San Francisco", "units": "celsius"}'
        )
        assert output_items[0].call_id.startswith("call_")
        assert output_items[0].id.startswith("fc_")

    def test_commentary_with_python_recipient_creates_reasoning(self):
        """Test that commentary with recipient='python' creates reasoning items."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, "import numpy as np\nprint(np.array([1, 2, 3]))"
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("python")

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].type == "reasoning"
        assert (
            output_items[0].content[0].text
            == "import numpy as np\nprint(np.array([1, 2, 3]))"
        )

    def test_commentary_with_browser_recipient_creates_reasoning(self):
        """Test that commentary with recipient='browser' creates reasoning items."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, "Navigating to the specified URL"
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("browser")

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].type == "reasoning"
        assert output_items[0].content[0].text == "Navigating to the specified URL"

    def test_commentary_with_container_recipient_creates_reasoning(self):
        """Test that commentary with recipient='container' creates reasoning items."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, "Running command in container"
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("container")

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].type == "reasoning"
        assert output_items[0].content[0].text == "Running command in container"

    def test_commentary_with_empty_content_and_no_recipient(self):
        """Test edge case: empty commentary with recipient=None."""
        message = Message.from_role_and_content(Role.ASSISTANT, "")
        message = message.with_channel("commentary")

        output_items = harmony_to_response_output(message)

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

        output_items = harmony_to_response_output(message)

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

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 2
        assert all(isinstance(item, ResponseFunctionToolCall) for item in output_items)
        assert output_items[0].name == "get_weather"
        assert output_items[1].name == "get_weather"
        assert output_items[0].arguments == '{"location": "San Francisco"}'
        assert output_items[1].arguments == '{"location": "New York"}'

    def test_commentary_with_unknown_recipient_creates_mcp_call(self):
        """Test that commentary with unknown recipient creates MCP call."""
        message = Message.from_role_and_content(Role.ASSISTANT, '{"arg": "value"}')
        message = message.with_channel("commentary")
        message = message.with_recipient("custom_tool")

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], McpCall)
        assert output_items[0].type == "mcp_call"
        assert output_items[0].name == "custom_tool"
        assert output_items[0].server_label == "custom_tool"

    def test_analysis_channel_creates_reasoning(self):
        """Test that analysis channel creates reasoning items."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, "Analyzing the problem step by step..."
        )
        message = message.with_channel("analysis")

        output_items = harmony_to_response_output(message)

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

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 0


def test_parse_mcp_call_basic() -> None:
    """Test that MCP calls are parsed with correct type and server_label."""
    message = Message.from_role_and_content(Role.ASSISTANT, '{"path": "/tmp"}')
    message = message.with_recipient("filesystem")
    message = message.with_channel("commentary")

    output_items = harmony_to_response_output(message)

    assert len(output_items) == 1
    assert isinstance(output_items[0], McpCall)
    assert output_items[0].type == "mcp_call"
    assert output_items[0].name == "filesystem"
    assert output_items[0].server_label == "filesystem"
    assert output_items[0].arguments == '{"path": "/tmp"}'
    assert output_items[0].status == "completed"


def test_parse_mcp_call_dotted_recipient() -> None:
    """Test that dotted recipients extract the tool name correctly."""
    message = Message.from_role_and_content(Role.ASSISTANT, '{"cmd": "ls"}')
    message = message.with_recipient("repo_browser.list")
    message = message.with_channel("commentary")

    output_items = harmony_to_response_output(message)

    assert len(output_items) == 1
    assert isinstance(output_items[0], McpCall)
    assert output_items[0].name == "list"
    assert output_items[0].server_label == "repo_browser"


def test_mcp_vs_function_call() -> None:
    """Test that function calls are not parsed as MCP calls."""
    func_message = Message.from_role_and_content(Role.ASSISTANT, '{"arg": "value"}')
    func_message = func_message.with_recipient("functions.my_tool")
    func_message = func_message.with_channel("commentary")

    func_items = harmony_to_response_output(func_message)

    assert len(func_items) == 1
    assert not isinstance(func_items[0], McpCall)
    assert func_items[0].type == "function_call"


def test_mcp_vs_builtin_tools() -> None:
    """Test that built-in tools (python, container) are not parsed as MCP calls."""
    # Test python (built-in tool) - should be reasoning, not MCP
    python_message = Message.from_role_and_content(Role.ASSISTANT, "print('hello')")
    python_message = python_message.with_recipient("python")
    python_message = python_message.with_channel("commentary")

    python_items = harmony_to_response_output(python_message)

    assert len(python_items) == 1
    assert not isinstance(python_items[0], McpCall)
    assert python_items[0].type == "reasoning"


def test_parser_state_to_response_output_commentary_channel() -> None:
    """Test parser_state_to_response_output with commentary
    channel and various recipients."""
    from unittest.mock import Mock

    # Test 1: functions.* recipient -> should return function tool call
    parser_func = Mock()
    parser_func.current_content = '{"arg": "value"}'
    parser_func.current_role = Role.ASSISTANT
    parser_func.current_channel = "commentary"
    parser_func.current_recipient = "functions.my_tool"

    func_items = parser_state_to_response_output(parser_func)

    assert len(func_items) == 1
    assert not isinstance(func_items[0], McpCall)
    assert func_items[0].type == "function_call"
    assert func_items[0].name == "my_tool"
    assert func_items[0].status == "in_progress"

    # Test 2: MCP tool (not builtin) -> should return MCP call
    parser_mcp = Mock()
    parser_mcp.current_content = '{"path": "/tmp"}'
    parser_mcp.current_role = Role.ASSISTANT
    parser_mcp.current_channel = "commentary"
    parser_mcp.current_recipient = "filesystem"

    mcp_items = parser_state_to_response_output(parser_mcp)

    assert len(mcp_items) == 1
    assert isinstance(mcp_items[0], McpCall)
    assert mcp_items[0].type == "mcp_call"
    assert mcp_items[0].name == "filesystem"
    assert mcp_items[0].server_label == "filesystem"
    assert mcp_items[0].status == "in_progress"

    # Test 3: Built-in tool (python)
    # should NOT return MCP call, returns reasoning (internal tool interaction)
    parser_builtin = Mock()
    parser_builtin.current_content = "print('hello')"
    parser_builtin.current_role = Role.ASSISTANT
    parser_builtin.current_channel = "commentary"
    parser_builtin.current_recipient = "python"

    builtin_items = parser_state_to_response_output(parser_builtin)

    # Built-in tools explicitly return reasoning
    assert len(builtin_items) == 1
    assert not isinstance(builtin_items[0], McpCall)
    assert builtin_items[0].type == "reasoning"

    # Test 4: No recipient (preamble) → should return message, not reasoning
    parser_preamble = Mock()
    parser_preamble.current_content = "I'll search for that information now."
    parser_preamble.current_role = Role.ASSISTANT
    parser_preamble.current_channel = "commentary"
    parser_preamble.current_recipient = None

    preamble_items = parser_state_to_response_output(parser_preamble)

    assert len(preamble_items) == 1
    assert isinstance(preamble_items[0], ResponseOutputMessage)
    assert preamble_items[0].type == "message"
    assert preamble_items[0].content[0].text == "I'll search for that information now."
    assert preamble_items[0].status == "incomplete"  # streaming


def test_parser_state_to_response_output_analysis_channel() -> None:
    """Test parser_state_to_response_output with analysis
    channel and various recipients."""
    from unittest.mock import Mock

    # Test 1: functions.* recipient -> should return function tool call
    parser_func = Mock()
    parser_func.current_content = '{"arg": "value"}'
    parser_func.current_role = Role.ASSISTANT
    parser_func.current_channel = "analysis"
    parser_func.current_recipient = "functions.my_tool"

    func_items = parser_state_to_response_output(parser_func)

    assert len(func_items) == 1
    assert not isinstance(func_items[0], McpCall)
    assert func_items[0].type == "function_call"
    assert func_items[0].name == "my_tool"
    assert func_items[0].status == "in_progress"

    # Test 2: MCP tool (not builtin) -> should return MCP call
    parser_mcp = Mock()
    parser_mcp.current_content = '{"query": "test"}'
    parser_mcp.current_role = Role.ASSISTANT
    parser_mcp.current_channel = "analysis"
    parser_mcp.current_recipient = "database"

    mcp_items = parser_state_to_response_output(parser_mcp)

    assert len(mcp_items) == 1
    assert isinstance(mcp_items[0], McpCall)
    assert mcp_items[0].type == "mcp_call"
    assert mcp_items[0].name == "database"
    assert mcp_items[0].server_label == "database"
    assert mcp_items[0].status == "in_progress"

    # Test 3: Built-in tool (container)
    # should NOT return MCP call, falls through to reasoning
    parser_builtin = Mock()
    parser_builtin.current_content = "docker run"
    parser_builtin.current_role = Role.ASSISTANT
    parser_builtin.current_channel = "analysis"
    parser_builtin.current_recipient = "container"

    builtin_items = parser_state_to_response_output(parser_builtin)

    # Should fall through to reasoning logic
    assert len(builtin_items) == 1
    assert not isinstance(builtin_items[0], McpCall)
    assert builtin_items[0].type == "reasoning"
