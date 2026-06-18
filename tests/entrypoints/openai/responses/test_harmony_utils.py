# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.entrypoints.openai.responses.harmony."""

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from openai.types.responses.response_output_item import McpCall
from openai_harmony import Message, Role, TextContent

from vllm.entrypoints.openai.responses.harmony import (
    harmony_to_response_output,
    response_previous_input_to_harmony,
)


def _item_value(item, field: str):
    """Read *field* from an object that may be a dict or an attrs/pydantic model."""
    if isinstance(item, dict):
        return item[field]
    return getattr(item, field)


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

    def test_commentary_with_python_recipient_creates_code_interpreter_call(self):
        """Test that commentary with recipient='python' creates code calls."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, "import numpy as np\nprint(np.array([1, 2, 3]))"
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("python")

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 1
        assert _item_value(output_items[0], "type") == "code_interpreter_call"
        assert _item_value(output_items[0], "status") == "completed"
        assert _item_value(output_items[0], "code") == (
            "import numpy as np\nprint(np.array([1, 2, 3]))"
        )

    def test_commentary_with_browser_recipient_creates_web_search_call(self):
        """Test that bare browser recipients default to search actions."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, '{"query": "weather in seoul"}'
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("browser")

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionWebSearch)
        assert output_items[0].type == "web_search_call"
        assert output_items[0].status == "completed"
        assert output_items[0].action.type == "search"
        assert output_items[0].action.query == "cursor:weather in seoul"

    def test_commentary_with_browser_search_recipient_creates_web_search_call(self):
        """Test that dotted browser recipients share the same browser branch."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, '{"url": "https://example.com"}'
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("browser.open")

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionWebSearch)
        assert output_items[0].type == "web_search_call"
        assert output_items[0].action.type == "open_page"
        assert output_items[0].action.url == "cursor:https://example.com"

    def test_commentary_with_container_recipient_creates_mcp_call(self):
        """Test that bare container recipients default to exec MCP calls."""
        message = Message.from_role_and_content(Role.ASSISTANT, '{"cmd": ["ls"]}')
        message = message.with_channel("commentary")
        message = message.with_recipient("container")

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], McpCall)
        assert output_items[0].type == "mcp_call"
        assert output_items[0].name == "exec"
        assert output_items[0].server_label == "container"
        assert output_items[0].arguments == '{"cmd": ["ls"]}'
        assert output_items[0].status == "completed"
        assert output_items[0].error is None

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

    def test_commentary_with_unknown_recipient_creates_mcp_call(self):
        """Test that commentary with unknown recipient creates MCP call."""
        message = Message.from_role_and_content(Role.ASSISTANT, '{"arg": "value"}')
        message = message.with_channel("commentary")
        message = message.with_recipient("custom_tool")

        fn_names = frozenset({"other_tool"})
        output_items = harmony_to_response_output(message, fn_names)

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


class TestHarmonyToResponseOutputWithFunctionToolNames:
    """Tests for bare function name handling with function_tool_names."""

    def test_bare_name_creates_function_call_when_in_tool_names(self):
        """Bare function name matching a known tool creates function call."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, '{"location": "San Francisco"}'
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("get_weather")

        fn_names = frozenset({"get_weather"})
        output_items = harmony_to_response_output(message, fn_names)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionToolCall)
        assert output_items[0].type == "function_call"
        assert output_items[0].name == "get_weather"
        assert output_items[0].arguments == '{"location": "San Francisco"}'

    def test_bare_name_creates_mcp_call_when_not_in_tool_names(self):
        """Bare name not matching any known tool creates MCP call."""
        message = Message.from_role_and_content(Role.ASSISTANT, '{"arg": "value"}')
        message = message.with_channel("commentary")
        message = message.with_recipient("custom_tool")

        fn_names = frozenset({"get_weather"})
        output_items = harmony_to_response_output(message, fn_names)

        assert len(output_items) == 1
        assert isinstance(output_items[0], McpCall)
        assert output_items[0].type == "mcp_call"

    def test_dotted_function_name_creates_function_call(self):
        """Dotted function name in tool names creates function call."""
        message = Message.from_role_and_content(Role.ASSISTANT, '{"a": 1, "b": 2}')
        message = message.with_channel("commentary")
        message = message.with_recipient("math.sum")

        fn_names = frozenset({"math.sum"})
        output_items = harmony_to_response_output(message, fn_names)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionToolCall)
        assert output_items[0].name == "math.sum"

    def test_empty_tool_names_defaults_to_mcp(self):
        """With empty function_tool_names, bare names become MCP calls."""
        message = Message.from_role_and_content(Role.ASSISTANT, '{"arg": "value"}')
        message = message.with_channel("commentary")
        message = message.with_recipient("get_weather")

        output_items = harmony_to_response_output(message, frozenset())

        assert len(output_items) == 1
        assert isinstance(output_items[0], McpCall)

    def test_prefixed_name_always_function_call(self):
        """functions. prefix always creates function call even with empty tool names."""
        message = Message.from_role_and_content(Role.ASSISTANT, '{"arg": "value"}')
        message = message.with_channel("commentary")
        message = message.with_recipient("functions.get_weather")

        output_items = harmony_to_response_output(message, frozenset())

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionToolCall)
        assert output_items[0].name == "get_weather"


class TestToolCallsOnNonStandardChannels:
    """Tests verifying tool calls are detected regardless of channel."""

    def test_function_call_on_comment_channel(self):
        message = Message.from_role_and_content(Role.ASSISTANT, '{"query": "weather"}')
        message = message.with_channel("comment")
        message = message.with_recipient("functions.get_weather")

        output_items = harmony_to_response_output(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionToolCall)
        assert output_items[0].type == "function_call"
        assert output_items[0].name == "get_weather"

    def test_bare_function_on_comment_channel(self):
        message = Message.from_role_and_content(Role.ASSISTANT, '{"query": "weather"}')
        message = message.with_channel("comment")
        message = message.with_recipient("get_weather")

        fn_names = frozenset({"get_weather"})
        output_items = harmony_to_response_output(message, fn_names)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionToolCall)
        assert output_items[0].name == "get_weather"


def test_parse_mcp_call_basic() -> None:
    """Test that MCP calls are parsed with correct type and server_label."""
    message = Message.from_role_and_content(Role.ASSISTANT, '{"path": "/tmp"}')
    message = message.with_recipient("filesystem")
    message = message.with_channel("commentary")

    fn_names: frozenset[str] = frozenset()
    output_items = harmony_to_response_output(message, fn_names)

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

    fn_names: frozenset[str] = frozenset()
    output_items = harmony_to_response_output(message, fn_names)

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
    # Test python (built-in tool) - should be code interpreter, not MCP
    python_message = Message.from_role_and_content(Role.ASSISTANT, "print('hello')")
    python_message = python_message.with_recipient("python")
    python_message = python_message.with_channel("commentary")

    python_items = harmony_to_response_output(python_message)

    assert len(python_items) == 1
    assert not isinstance(python_items[0], McpCall)
    assert _item_value(python_items[0], "type") == "code_interpreter_call"
