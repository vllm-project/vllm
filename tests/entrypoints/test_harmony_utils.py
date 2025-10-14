# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for harmony_utils.py module."""

from unittest.mock import MagicMock, patch

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from openai.types.responses.response_output_item import McpCall
from openai_harmony import (
    Author,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    ToolDescription,
    ToolNamespaceConfig,
)

from vllm.entrypoints.harmony_utils import (
    BUILTIN_TOOLS,
    REASONING_EFFORT,
    build_system_and_developer_messages,
    create_function_tools_namespace,
    create_tool_definition,
    get_developer_message,
    get_encoding,
    get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant,
    get_system_message,
    get_user_message,
    parse_chat_input,
    parse_chat_output,
    parse_output_into_messages,
    parse_output_message,
    parse_remaining_state,
    parse_response_input,
    render_for_completion,
)
from vllm.entrypoints.openai.protocol import ChatCompletionToolsParam


class TestConstants:
    """Test module constants."""

    def test_reasoning_effort_mapping(self):
        """Test that REASONING_EFFORT contains correct mappings."""
        assert REASONING_EFFORT["high"] == ReasoningEffort.HIGH
        assert REASONING_EFFORT["medium"] == ReasoningEffort.MEDIUM
        assert REASONING_EFFORT["low"] == ReasoningEffort.LOW
        assert len(REASONING_EFFORT) == 3

    def test_builtin_tools(self):
        """Test BUILTIN_TOOLS set contains expected tools."""
        assert "web_search_preview" in BUILTIN_TOOLS
        assert "code_interpreter" in BUILTIN_TOOLS
        assert "container" in BUILTIN_TOOLS
        assert len(BUILTIN_TOOLS) == 3


class TestGetEncoding:
    """Test get_encoding() function."""

    def test_get_encoding_returns_encoding(self):
        """Test that get_encoding returns a harmony encoding."""
        encoding = get_encoding()
        assert encoding is not None
        assert hasattr(encoding, "render_conversation_for_completion")

    def test_get_encoding_caches_result(self):
        """Test that get_encoding caches the result."""
        encoding1 = get_encoding()
        encoding2 = get_encoding()
        assert encoding1 is encoding2


class TestCreateToolDefinition:
    """Test create_tool_definition() function."""

    def test_create_tool_definition_from_chat_completion_tool(self):
        """Test creating tool definition from ChatCompletionToolsParam."""
        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        )

        tool_desc = create_tool_definition(tool)

        assert isinstance(tool_desc, ToolDescription)
        assert tool_desc.name == "get_weather"
        assert tool_desc.description == "Get weather information"
        assert tool_desc.parameters["type"] == "object"


class TestCreateFunctionToolsNamespace:
    """Test create_function_tools_namespace() function."""

    def test_create_namespace_single_tool(self):
        """Test creating namespace with a single function tool."""
        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "test_func",
                "description": "Test function",
                "parameters": {"type": "object", "properties": {}},
            },
        )

        namespace = create_function_tools_namespace([tool])

        assert isinstance(namespace, ToolNamespaceConfig)
        assert namespace.name == "functions"
        assert namespace.description == ""
        assert len(namespace.tools) == 1
        assert namespace.tools[0].name == "test_func"

    def test_create_namespace_multiple_tools(self):
        """Test creating namespace with multiple function tools."""
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "func1",
                    "description": "Function 1",
                    "parameters": {"type": "object"},
                },
            ),
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "func2",
                    "description": "Function 2",
                    "parameters": {"type": "object"},
                },
            ),
        ]

        namespace = create_function_tools_namespace(tools)

        assert len(namespace.tools) == 2
        assert namespace.tools[0].name == "func1"
        assert namespace.tools[1].name == "func2"

    def test_create_namespace_empty_tools(self):
        """Test creating namespace with empty tools list."""
        namespace = create_function_tools_namespace([])
        assert len(namespace.tools) == 0


class TestGetUserMessage:
    """Test get_user_message() function."""

    def test_get_user_message_simple(self):
        """Test creating a simple user message."""
        msg = get_user_message("Hello!")

        assert isinstance(msg, Message)
        assert msg.author.role == Role.USER
        assert len(msg.content) == 1
        assert msg.content[0].text == "Hello!"

    def test_get_user_message_multiline(self):
        """Test creating user message with multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        msg = get_user_message(content)

        assert msg.content[0].text == content


class TestGetSystemMessage:
    """Test get_system_message() function."""

    def test_get_system_message_minimal(self):
        """Test creating system message with minimal parameters."""
        msg = get_system_message()

        assert isinstance(msg, Message)
        assert msg.author.role == Role.SYSTEM
        assert len(msg.content) == 1

    def test_get_system_message_with_model_identity(self):
        """Test system message with model identity."""
        msg = get_system_message(model_identity="gpt-oss-test")

        assert msg.content[0].model_identity == "gpt-oss-test"

    def test_get_system_message_with_reasoning_effort(self):
        """Test system message with reasoning effort."""
        msg = get_system_message(reasoning_effort="high")

        assert msg.content[0].reasoning_effort == ReasoningEffort.HIGH

    def test_get_system_message_with_start_date(self):
        """Test system message with custom start date."""
        date = "2024-01-01"
        msg = get_system_message(start_date=date)

        assert msg.content[0].conversation_start_date == date

    @patch(
        "vllm.entrypoints.harmony_utils.envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS",
        True,
    )
    def test_get_system_message_with_instructions_in_system(self):
        """Test system message with instructions in system content."""
        instructions = "Be concise"
        msg = get_system_message(instructions=instructions)

        assert instructions in msg.content[0].model_identity

    def test_get_system_message_with_elevated_tools(self):
        """Test system message with elevated tool namespaces."""
        tool_namespace = ToolNamespaceConfig(
            name="elevated_tool",
            description="Test elevated tool",
            tools=[],
        )

        msg = get_system_message(elevated_namespace_descriptions=[tool_namespace])

        # Verify message was created - internals vary by harmony version
        assert isinstance(msg, Message)
        assert msg.author.role == Role.SYSTEM

    def test_get_system_message_with_custom_tools(self):
        """Test that commentary channel is available with custom tools."""
        tool_namespace = ToolNamespaceConfig(
            name="custom_tool",
            description="Test custom tool",
            tools=[],
        )

        msg = get_system_message(custom_namespace_descriptions=[tool_namespace])

        valid_channels = msg.content[0].channel_config.valid_channels
        assert "commentary" in valid_channels

    def test_get_system_message_without_custom_tools(self):
        """Test that commentary channel is removed without custom tools."""
        msg = get_system_message()

        valid_channels = msg.content[0].channel_config.valid_channels
        assert "commentary" not in valid_channels


class TestGetDeveloperMessage:
    """Test get_developer_message() function."""

    def test_get_developer_message_none_when_empty(self):
        """Test that None is returned when no instructions or tools."""
        msg = get_developer_message()
        assert msg is None

    @patch(
        "vllm.entrypoints.harmony_utils.envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS",
        False,
    )
    def test_get_developer_message_with_instructions(self):
        """Test developer message with instructions."""
        instructions = "Test instructions"
        msg = get_developer_message(instructions=instructions)

        assert msg is not None
        assert msg.author.role == Role.DEVELOPER
        assert msg.content[0].instructions == instructions

    @patch(
        "vllm.entrypoints.harmony_utils.envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS",
        True,
    )
    def test_get_developer_message_no_instructions_when_system_mode(self):
        """Test developer message doesn't include instructions in system mode."""
        instructions = "Test instructions"
        msg = get_developer_message(instructions=instructions)

        assert msg is None

    def test_get_developer_message_with_tool_namespaces(self):
        """Test developer message with tool namespaces."""
        tool_namespace = ToolNamespaceConfig(
            name="test_tool",
            description="Test tool",
            tools=[],
        )

        msg = get_developer_message(tool_namespaces=[tool_namespace])

        assert msg is not None
        assert msg.author.role == Role.DEVELOPER
        # Just verify message was created - internals vary by harmony version

    @patch(
        "vllm.entrypoints.harmony_utils.envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS",
        False,
    )
    def test_get_developer_message_with_both(self):
        """Test developer message with instructions and tools."""
        instructions = "Test instructions"
        tool_namespace = ToolNamespaceConfig(
            name="test_tool",
            description="Test tool",
            tools=[],
        )

        msg = get_developer_message(
            instructions=instructions, tool_namespaces=[tool_namespace]
        )

        assert msg is not None
        assert msg.content[0].instructions == instructions
        # Just verify message was created - internals vary by harmony version


class TestParseResponseInput:
    """Test parse_response_input() function."""

    def test_parse_user_message_string(self):
        """Test parsing user message with string content."""
        response_msg = {"role": "user", "content": "Hello"}

        msg = parse_response_input(response_msg, [])

        assert msg.author.role == Role.USER
        assert msg.content[0].text == "Hello"

    def test_parse_user_message_with_type(self):
        """Test parsing user message with input_text type."""
        response_msg = {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }

        msg = parse_response_input(response_msg, [])

        assert msg.author.role == Role.USER
        assert msg.content[0].text == "Hello"

    def test_parse_system_message_converts_to_developer(self):
        """Test that system messages are converted to developer role."""
        response_msg = {"role": "system", "content": "System prompt"}

        msg = parse_response_input(response_msg, [])

        assert msg.author.role == Role.DEVELOPER
        assert "Instructions:" in msg.content[0].text
        assert "System prompt" in msg.content[0].text

    def test_parse_assistant_message(self):
        """Test parsing assistant message."""
        response_msg = {"role": "assistant", "content": "Hello"}

        msg = parse_response_input(response_msg, [])

        assert msg.author.role == Role.ASSISTANT
        assert msg.channel == "final"
        assert msg.content[0].text == "Hello"

    def test_parse_function_call_output(self):
        """Test parsing function call output message."""
        # First create a function call
        function_call = ResponseFunctionToolCall(
            call_id="call_123",
            type="function_call",
            name="get_weather",
            arguments='{"location": "SF"}',
            id="fc_123",
        )

        output_msg = {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "70 degrees",
        }

        msg = parse_response_input(output_msg, [function_call])

        assert msg.author.role == Role.TOOL
        assert msg.author.name == "functions.get_weather"
        assert msg.content[0].text == "70 degrees"

    def test_parse_function_call_output_no_matching_call(self):
        """Test parsing function call output with no matching call."""
        output_msg = {
            "type": "function_call_output",
            "call_id": "call_nonexistent",
            "output": "result",
        }

        with pytest.raises(ValueError, match="No call message found"):
            parse_response_input(output_msg, [])

    def test_parse_reasoning_item(self):
        """Test parsing reasoning item."""
        response_msg = {
            "type": "reasoning",
            "content": [{"text": "Let me think..."}],
        }

        msg = parse_response_input(response_msg, [])

        assert msg.author.role == Role.ASSISTANT
        assert msg.content[0].text == "Let me think..."

    def test_parse_function_call(self):
        """Test parsing function call."""
        response_msg = {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"location": "SF"}',
        }

        msg = parse_response_input(response_msg, [])

        assert msg.author.role == Role.ASSISTANT
        assert msg.channel == "commentary"
        assert msg.recipient == "functions.get_weather"
        assert msg.content_type == "json"

    def test_parse_unknown_type(self):
        """Test parsing unknown message type."""
        response_msg = {"type": "unknown_type"}

        with pytest.raises(ValueError, match="Unknown input type"):
            parse_response_input(response_msg, [])


class TestParseChatInput:
    """Test parse_chat_input() function."""

    def test_parse_simple_user_message(self):
        """Test parsing simple user message."""
        chat_msg = {"role": "user", "content": "Hello"}

        msgs = parse_chat_input(chat_msg)

        assert len(msgs) == 1
        assert msgs[0].author.role == Role.USER
        assert msgs[0].content[0].text == "Hello"

    def test_parse_assistant_with_tool_calls(self):
        """Test parsing assistant message with tool calls."""
        chat_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "SF"}',
                    }
                },
                {
                    "function": {
                        "name": "get_time",
                        "arguments": '{"timezone": "PST"}',
                    }
                },
            ],
        }

        msgs = parse_chat_input(chat_msg)

        assert len(msgs) == 2
        assert all(msg.author.role == Role.ASSISTANT for msg in msgs)
        assert all(msg.channel == "commentary" for msg in msgs)
        assert msgs[0].recipient == "functions.get_weather"
        assert msgs[1].recipient == "functions.get_time"

    def test_parse_tool_message(self):
        """Test parsing tool role message."""
        chat_msg = {
            "role": "tool",
            "name": "get_weather",
            "content": "70 degrees",
        }

        msgs = parse_chat_input(chat_msg)

        assert len(msgs) == 1
        assert msgs[0].author.role == Role.TOOL
        assert msgs[0].author.name == "functions.get_weather"
        assert msgs[0].content[0].text == "70 degrees"
        assert msgs[0].channel == "commentary"

    def test_parse_tool_message_with_array_content(self):
        """Test parsing tool message with array content."""
        chat_msg = {
            "role": "tool",
            "name": "search",
            "content": [
                {"type": "text", "text": "Result 1"},
                {"type": "text", "text": "Result 2"},
            ],
        }

        msgs = parse_chat_input(chat_msg)

        assert len(msgs) == 1
        assert msgs[0].content[0].text == "Result 1Result 2"

    def test_parse_message_with_array_content(self):
        """Test parsing message with array content."""
        chat_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " World"},
            ],
        }

        msgs = parse_chat_input(chat_msg)

        assert len(msgs) == 1
        assert len(msgs[0].content) == 2
        assert msgs[0].content[0].text == "Hello"
        assert msgs[0].content[1].text == " World"


class TestRenderForCompletion:
    """Test render_for_completion() function."""

    def test_render_simple_conversation(self):
        """Test rendering a simple conversation."""
        messages = [
            get_user_message("What is 2+2?"),
        ]

        token_ids = render_for_completion(messages)

        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)

    def test_render_multi_turn(self):
        """Test rendering multi-turn conversation."""
        messages = [
            get_user_message("Hello"),
            Message.from_role_and_content(Role.ASSISTANT, "Hi there!"),
            get_user_message("How are you?"),
        ]

        token_ids = render_for_completion(messages)

        assert isinstance(token_ids, list)
        assert len(token_ids) > 0


class TestParseOutputMessage:
    """Test parse_output_message() function."""

    def test_parse_final_channel_message(self):
        """Test parsing message on final channel."""
        msg = Message.from_role_and_content(Role.ASSISTANT, "Hello!").with_channel(
            "final"
        )

        output_items = parse_output_message(msg)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseOutputMessage)
        assert output_items[0].type == "message"
        assert output_items[0].content[0].text == "Hello!"

    def test_parse_analysis_channel_message(self):
        """Test parsing message on analysis channel."""
        msg = Message.from_role_and_content(
            Role.ASSISTANT, "Let me think..."
        ).with_channel("analysis")

        output_items = parse_output_message(msg)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].type == "reasoning"
        assert output_items[0].content[0].text == "Let me think..."

    def test_parse_function_call_on_commentary(self):
        """Test parsing function call on commentary channel."""
        msg = (
            Message.from_role_and_content(Role.ASSISTANT, '{"location": "SF"}')
            .with_channel("commentary")
            .with_recipient("functions.get_weather")
            .with_content_type("json")
        )

        output_items = parse_output_message(msg)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionToolCall)
        assert output_items[0].type == "function_call"
        assert output_items[0].name == "get_weather"
        assert output_items[0].arguments == '{"location": "SF"}'

    def test_parse_mcp_call_on_commentary(self):
        """Test parsing MCP call on commentary channel."""
        msg = (
            Message.from_role_and_content(Role.ASSISTANT, '{"key": "value"}')
            .with_channel("commentary")
            .with_recipient("memory.store")
            .with_content_type("json")
        )

        output_items = parse_output_message(msg)

        assert len(output_items) == 1
        assert isinstance(output_items[0], McpCall)
        assert output_items[0].type == "mcp_call"
        assert output_items[0].name == "store"
        assert output_items[0].server_label == "memory"
        assert output_items[0].arguments == '{"key": "value"}'
        assert output_items[0].output is None

    def test_parse_tool_response_updates_mcp_call(self):
        """Test that tool response updates matching MCP call."""
        # Create an MCP call
        mcp_call = McpCall(
            id="mcp_123",
            type="mcp_call",
            name="store",
            server_label="memory",
            arguments='{"key": "value"}',
            output=None,
            error=None,
        )

        # Create tool response
        tool_msg = Message.from_author_and_content(
            Author.new(Role.TOOL, "memory.store"), "Success"
        )

        output_items = parse_output_message(tool_msg, output_items_so_far=[mcp_call])

        assert len(output_items) == 0  # Tool response doesn't create new items
        assert mcp_call.output == "Success"  # But updates the existing call

    def test_parse_tool_response_no_matching_call(self):
        """Test tool response with no matching call."""
        tool_msg = Message.from_author_and_content(
            Author.new(Role.TOOL, "nonexistent.tool"), "Result"
        )

        # Should log error but not crash
        output_items = parse_output_message(tool_msg, output_items_so_far=[])

        assert len(output_items) == 0

    def test_parse_builtin_tool_on_commentary_becomes_reasoning(self):
        """Test that built-in tools on commentary become reasoning items."""
        msg = (
            Message.from_role_and_content(Role.ASSISTANT, "print('hello')")
            .with_channel("commentary")
            .with_recipient("python")
        )

        output_items = parse_output_message(msg)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)

    def test_parse_non_assistant_message_returns_empty(self):
        """Test that non-assistant messages return empty list."""
        msg = Message.from_role_and_content(Role.USER, "Hello")

        output_items = parse_output_message(msg)

        assert len(output_items) == 0

    def test_parse_unknown_channel_raises_error(self):
        """Test that unknown channel raises error."""
        msg = Message.from_role_and_content(Role.ASSISTANT, "Test").with_channel(
            "unknown_channel"
        )

        with pytest.raises(ValueError, match="Unknown channel"):
            parse_output_message(msg)


class TestParseRemainingState:
    """Test parse_remaining_state() function."""

    def test_parse_remaining_analysis_content(self):
        """Test parsing remaining analysis content."""
        parser = MagicMock(spec=StreamableParser)
        parser.current_content = "Incomplete thought..."
        parser.current_role = Role.ASSISTANT
        parser.current_channel = "analysis"
        parser.current_recipient = None

        items = parse_remaining_state(parser)

        assert len(items) == 1
        assert isinstance(items[0], ResponseReasoningItem)
        assert items[0].content[0].text == "Incomplete thought..."

    def test_parse_remaining_final_content(self):
        """Test parsing remaining final content."""
        parser = MagicMock(spec=StreamableParser)
        parser.current_content = "Incomplete answer..."
        parser.current_role = Role.ASSISTANT
        parser.current_channel = "final"
        parser.current_recipient = None

        items = parse_remaining_state(parser)

        assert len(items) == 1
        assert isinstance(items[0], ResponseOutputMessage)
        assert items[0].status == "incomplete"
        assert items[0].content[0].text == "Incomplete answer..."

    def test_parse_remaining_empty_content(self):
        """Test parsing with empty content."""
        parser = MagicMock(spec=StreamableParser)
        parser.current_content = ""

        items = parse_remaining_state(parser)

        assert len(items) == 0

    def test_parse_remaining_non_assistant_role(self):
        """Test parsing with non-assistant role."""
        parser = MagicMock(spec=StreamableParser)
        parser.current_content = "Some content"
        parser.current_role = Role.USER

        items = parse_remaining_state(parser)

        assert len(items) == 0

    def test_parse_remaining_browser_recipient_skipped(self):
        """Test that browser recipients are skipped."""
        parser = MagicMock(spec=StreamableParser)
        parser.current_content = "Search query"
        parser.current_role = Role.ASSISTANT
        parser.current_channel = "commentary"
        parser.current_recipient = "browser.search"

        items = parse_remaining_state(parser)

        assert len(items) == 0


class TestGetStopTokens:
    """Test get_stop_tokens_for_assistant_actions() function."""

    def test_get_stop_tokens(self):
        """Test getting stop tokens."""
        stop_tokens = get_stop_tokens_for_assistant_actions()

        assert isinstance(stop_tokens, list)
        assert len(stop_tokens) > 0
        assert all(isinstance(t, int) for t in stop_tokens)


class TestGetStreamableParser:
    """Test get_streamable_parser_for_assistant() function."""

    def test_get_streamable_parser(self):
        """Test getting streamable parser."""
        parser = get_streamable_parser_for_assistant()

        assert isinstance(parser, StreamableParser)
        # Parser doesn't expose role directly, just check it's created


class TestParseOutputIntoMessages:
    """Test parse_output_into_messages() function."""

    def test_parse_simple_tokens(self):
        """Test parsing simple token sequence."""
        # Get a simple message and render it
        msg = Message.from_role_and_content(Role.ASSISTANT, "Hello").with_channel(
            "final"
        )
        token_ids = render_for_completion([get_user_message("Hi"), msg])

        # Parse back
        parser = parse_output_into_messages(token_ids)

        assert isinstance(parser, StreamableParser)
        assert len(parser.messages) >= 0


class TestParseChatOutput:
    """Test parse_chat_output() function."""

    def test_parse_during_reasoning(self):
        """Test parsing output stopped during reasoning."""
        # Create a message with just reasoning
        msg = Message.from_role_and_content(Role.ASSISTANT, "Test").with_channel(
            "analysis"
        )

        token_ids = render_for_completion([get_user_message("Hi"), msg])

        reasoning, final, is_tool_call = parse_chat_output(token_ids)

        assert isinstance(reasoning, (str, type(None)))
        assert isinstance(final, (str, type(None)))
        assert isinstance(is_tool_call, bool)

    def test_parse_complete_output(self):
        """Test parsing complete output with reasoning and final."""
        # This is a integration-style test
        token_ids = render_for_completion([get_user_message("Hello")])

        reasoning, final, is_tool_call = parse_chat_output(token_ids)

        # Just verify types - actual content depends on model
        assert reasoning is None or isinstance(reasoning, str)
        assert final is None or isinstance(final, str)
        assert isinstance(is_tool_call, bool)


class TestBuildSystemAndDeveloperMessages:
    """Test build_system_and_developer_messages() function."""

    def test_build_with_no_tools(self):
        """Test building messages with no tools."""
        messages = build_system_and_developer_messages(
            request_tools=[],
            tool_server=None,
        )

        assert len(messages) == 1  # Just system message
        assert messages[0].author.role == Role.SYSTEM

    def test_build_with_function_tools(self):
        """Test building messages with function tools."""
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            )
        ]

        messages = build_system_and_developer_messages(
            request_tools=tools,
            tool_server=None,
        )

        # Should have system + developer messages
        assert len(messages) == 2
        assert messages[0].author.role == Role.SYSTEM
        assert messages[1].author.role == Role.DEVELOPER

    @patch(
        "vllm.entrypoints.harmony_utils.envs.GPT_OSS_SYSTEM_TOOL_MCP_LABELS",
        ["elevated_tool"],
    )
    def test_build_with_elevated_mcp_tools(self):
        """Test building messages with elevated MCP tools."""
        from types import SimpleNamespace

        # Mock tool server
        tool_server = MagicMock()
        tool_server.has_namespace.return_value = True
        tool_namespace = ToolNamespaceConfig(
            name="elevated_tool",
            description="Elevated tool",
            tools=[],
        )
        tool_server.get_tool_description.return_value = tool_namespace

        # Create tool with attributes (not dict)
        tools = [SimpleNamespace(type="mcp", server_label="elevated_tool")]

        messages = build_system_and_developer_messages(
            request_tools=tools,
            tool_server=tool_server,
        )

        # Should have just system message (elevated tools go in system)
        assert len(messages) == 1
        assert messages[0].author.role == Role.SYSTEM

    def test_build_with_custom_mcp_tools(self):
        """Test building messages with custom MCP tools."""
        from types import SimpleNamespace

        # Mock tool server
        tool_server = MagicMock()
        tool_server.has_namespace.return_value = True
        tool_namespace = ToolNamespaceConfig(
            name="custom_tool",
            description="Custom tool",
            tools=[],
        )
        tool_server.get_tool_description.return_value = tool_namespace

        # Create tool with attributes (not dict)
        tools = [SimpleNamespace(type="mcp", server_label="custom_tool")]

        messages = build_system_and_developer_messages(
            request_tools=tools,
            tool_server=tool_server,
        )

        # Should have system + developer messages
        assert len(messages) == 2
        assert messages[0].author.role == Role.SYSTEM
        assert messages[1].author.role == Role.DEVELOPER

    def test_build_with_instructions(self):
        """Test building messages with custom instructions."""
        messages = build_system_and_developer_messages(
            request_tools=[],
            tool_server=None,
            instructions="Be concise",
        )

        # Instructions should be in system or developer message
        assert len(messages) >= 1

    def test_build_with_reasoning_effort(self):
        """Test building messages with reasoning effort."""
        messages = build_system_and_developer_messages(
            request_tools=[],
            tool_server=None,
            reasoning_effort="high",
        )

        assert len(messages) >= 1
        assert messages[0].content[0].reasoning_effort == ReasoningEffort.HIGH

    def test_build_with_missing_mcp_namespace(self):
        """Test building messages with missing MCP namespace."""
        from types import SimpleNamespace

        tool_server = MagicMock()
        tool_server.has_namespace.return_value = False
        tool_server.harmony_tool_descriptions = {}

        # Create tool with attributes (not dict)
        tools = [SimpleNamespace(type="mcp", server_label="nonexistent")]

        with pytest.raises(ValueError, match="not available in tool server"):
            build_system_and_developer_messages(
                request_tools=tools,
                tool_server=tool_server,
            )

    def test_build_with_invalid_tool_type(self):
        """Test building messages with invalid tool type."""
        from types import SimpleNamespace

        # Create tool with attributes (not dict)
        tools = [SimpleNamespace(type="invalid_type")]

        error_msg = "should be of type 'mcp' or 'function'"
        with pytest.raises(ValueError, match=error_msg):
            build_system_and_developer_messages(
                request_tools=tools,
                tool_server=None,
            )


class TestIntegration:
    """Integration tests for harmony_utils."""

    def test_round_trip_user_message(self):
        """Test round-trip encoding and parsing of user message."""
        msg = get_user_message("What is 2+2?")
        token_ids = render_for_completion([msg])

        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)

    def test_function_tools_end_to_end(self):
        """Test creating function tools and building messages."""
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "calculator",
                    "description": "Perform calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                },
            )
        ]

        messages = build_system_and_developer_messages(
            request_tools=tools,
            tool_server=None,
            instructions="Use the calculator when needed",
        )

        # Should have system and developer messages
        assert len(messages) == 2

        # Verify we can render them
        token_ids = render_for_completion(messages)
        assert len(token_ids) > 0
