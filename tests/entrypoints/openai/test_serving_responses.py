# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AsyncExitStack
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from openai.types.responses.tool import (
    CodeInterpreterContainerCodeInterpreterToolAuto,
    LocalShell,
    Mcp,
    Tool,
)

from vllm.entrypoints.context import ConversationContext
from vllm.entrypoints.openai.protocol import ErrorResponse, ResponsesRequest
from vllm.entrypoints.openai.serving_responses import (
    OpenAIServingResponses,
    _extract_allowed_tools_from_mcp_requests,
    extract_tool_types,
)
from vllm.entrypoints.tool_server import ToolServer
from vllm.inputs.data import TokensPrompt


class MockConversationContext(ConversationContext):
    """Mock conversation context for testing"""

    def __init__(self):
        self.init_tool_sessions_called = False
        self.init_tool_sessions_args = None
        self.init_tool_sessions_kwargs = None

    def append_output(self, output) -> None:
        pass

    def append_tool_output(self, output) -> None:
        pass

    async def call_tool(self):
        return []

    def need_builtin_tool_call(self) -> bool:
        return False

    def render_for_completion(self):
        return []

    async def init_tool_sessions(self, tool_server, exit_stack, request_id, mcp_tools):
        self.init_tool_sessions_called = True
        self.init_tool_sessions_args = (tool_server, exit_stack, request_id, mcp_tools)

    async def cleanup_session(self) -> None:
        pass


@pytest.fixture
def mock_serving_responses():
    """Create a mock OpenAIServingResponses instance"""
    serving_responses = MagicMock(spec=OpenAIServingResponses)
    serving_responses.tool_server = MagicMock(spec=ToolServer)
    return serving_responses


@pytest.fixture
def mock_context():
    """Create a mock conversation context"""
    return MockConversationContext()


@pytest.fixture
def mock_exit_stack():
    """Create a mock async exit stack"""
    return MagicMock(spec=AsyncExitStack)


def test_extract_tool_types(monkeypatch: pytest.MonkeyPatch) -> None:
    tools: list[Tool] = []
    assert extract_tool_types(tools) == set()

    tools.append(LocalShell(type="local_shell"))
    assert extract_tool_types(tools) == {"local_shell"}

    tools.append(CodeInterpreterContainerCodeInterpreterToolAuto(type="auto"))
    assert extract_tool_types(tools) == {"local_shell", "auto"}

    tools.extend(
        [
            Mcp(type="mcp", server_label="random", server_url=""),
            Mcp(type="mcp", server_label="container", server_url=""),
            Mcp(type="mcp", server_label="code_interpreter", server_url=""),
            Mcp(type="mcp", server_label="web_search_preview", server_url=""),
        ]
    )
    # When envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS is not set,
    # mcp tool types are all ignored.
    assert extract_tool_types(tools) == {"local_shell", "auto"}

    # container is allowed, it would be extracted
    monkeypatch.setenv("VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "container")
    assert extract_tool_types(tools) == {"local_shell", "auto", "container"}

    # code_interpreter and web_search_preview are allowed,
    # they would be extracted
    monkeypatch.setenv(
        "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "code_interpreter,web_search_preview"
    )
    assert extract_tool_types(tools) == {
        "local_shell",
        "auto",
        "code_interpreter",
        "web_search_preview",
    }


class TestInitializeToolSessions:
    """Test class for _initialize_tool_sessions method"""

    @pytest_asyncio.fixture
    async def serving_responses_instance(self):
        """Create a real OpenAIServingResponses instance for testing"""
        # Create minimal mocks for required dependencies
        engine_client = MagicMock()

        model_config = MagicMock()
        model_config.hf_config.model_type = "test"
        model_config.get_diff_sampling_param.return_value = {}
        engine_client.model_config = model_config

        engine_client.input_processor = MagicMock()
        engine_client.io_processor = MagicMock()

        models = MagicMock()

        tool_server = MagicMock(spec=ToolServer)

        # Create the actual instance
        instance = OpenAIServingResponses(
            engine_client=engine_client,
            models=models,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            tool_server=tool_server,
        )

        return instance

    @pytest.mark.asyncio
    async def test_initialize_tool_sessions(
        self, serving_responses_instance, mock_context, mock_exit_stack
    ):
        """Test that method works correctly with only MCP tools"""

        request = ResponsesRequest(input="test input", tools=[])

        # Call the method
        await serving_responses_instance._initialize_tool_sessions(
            request, mock_context, mock_exit_stack
        )
        assert mock_context.init_tool_sessions_called is False

        # Create only MCP tools
        tools = [
            {"type": "web_search_preview"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
        ]

        request = ResponsesRequest(input="test input", tools=tools)

        # Call the method
        await serving_responses_instance._initialize_tool_sessions(
            request, mock_context, mock_exit_stack
        )

        # Verify that init_tool_sessions was called
        assert mock_context.init_tool_sessions_called

    def test_validate_create_responses_input(
        self, serving_responses_instance, mock_context, mock_exit_stack
    ):
        request = ResponsesRequest(
            input="test input",
            previous_input_messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is my horoscope? I am an Aquarius.",
                        }
                    ],
                }
            ],
            previous_response_id="lol",
        )
        error = serving_responses_instance._validate_create_responses_input(request)
        assert error is not None
        assert error.error.type == "invalid_request_error"


class TestValidateGeneratorInput:
    """Test class for _validate_generator_input method"""

    @pytest_asyncio.fixture
    async def serving_responses_instance(self):
        """Create a real OpenAIServingResponses instance for testing"""
        # Create minimal mocks for required dependencies
        engine_client = MagicMock()

        model_config = MagicMock()
        model_config.hf_config.model_type = "test"
        model_config.get_diff_sampling_param.return_value = {}
        engine_client.model_config = model_config

        engine_client.input_processor = MagicMock()
        engine_client.io_processor = MagicMock()

        models = MagicMock()

        # Create the actual instance
        instance = OpenAIServingResponses(
            engine_client=engine_client,
            models=models,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )

        # Set max_model_len for testing
        instance.max_model_len = 100

        return instance

    def test_validate_generator_input(self, serving_responses_instance):
        """Test _validate_generator_input with valid prompt length"""
        # Create an engine prompt with valid length (less than max_model_len)
        valid_prompt_token_ids = list(range(5))  # 5 tokens < 100 max_model_len
        engine_prompt = TokensPrompt(prompt_token_ids=valid_prompt_token_ids)

        # Call the method
        result = serving_responses_instance._validate_generator_input(engine_prompt)

        # Should return None for valid input
        assert result is None

        # create an invalid engine prompt
        invalid_prompt_token_ids = list(range(200))  # 100 tokens >= 100 max_model_len
        engine_prompt = TokensPrompt(prompt_token_ids=invalid_prompt_token_ids)

        # Call the method
        result = serving_responses_instance._validate_generator_input(engine_prompt)

        # Should return an ErrorResponse
        assert result is not None
        assert isinstance(result, ErrorResponse)


class TestExtractAllowedToolsFromMcpRequests:
    """Test class for _extract_allowed_tools_from_mcp_requests function"""

    def test_extract_allowed_tools_basic_formats(self):
        """Test extraction with list format, object format, and None."""
        from openai.types.responses.tool import McpAllowedToolsMcpToolFilter

        tools = [
            # List format
            Mcp(
                type="mcp",
                server_label="server1",
                allowed_tools=["tool1", "tool2"],
            ),
            # Object format
            Mcp(
                type="mcp",
                server_label="server2",
                allowed_tools=McpAllowedToolsMcpToolFilter(
                    tool_names=["tool3", "tool4"]
                ),
            ),
            # None (no filter)
            Mcp(
                type="mcp",
                server_label="server3",
                allowed_tools=None,
            ),
        ]
        result = _extract_allowed_tools_from_mcp_requests(tools)
        assert result == {
            "server1": ["tool1", "tool2"],
            "server2": ["tool3", "tool4"],
            "server3": None,
        }

    def test_extract_allowed_tools_star_normalization(self):
        """Test that '*' wildcard is normalized to None (select all tools).

        This is the key test requested by reviewers to explicitly demonstrate
        that the "*" select-all scenario is handled correctly.
        """
        from openai.types.responses.tool import McpAllowedToolsMcpToolFilter

        tools = [
            # Star in list format
            Mcp(
                type="mcp",
                server_label="server1",
                allowed_tools=["*"],
            ),
            # Star mixed with other tools in list
            Mcp(
                type="mcp",
                server_label="server2",
                allowed_tools=["tool1", "*"],
            ),
            # Star in object format
            Mcp(
                type="mcp",
                server_label="server3",
                allowed_tools=McpAllowedToolsMcpToolFilter(tool_names=["*"]),
            ),
        ]
        result = _extract_allowed_tools_from_mcp_requests(tools)
        # All should be normalized to None (allows all tools)
        assert result == {
            "server1": None,
            "server2": None,
            "server3": None,
        }

    def test_extract_allowed_tools_filters_non_mcp(self):
        """Test that non-MCP tools are ignored during extraction."""
        tools = [
            Mcp(
                type="mcp",
                server_label="server1",
                allowed_tools=["tool1"],
            ),
            LocalShell(type="local_shell"),  # Non-MCP tool should be ignored
            Mcp(
                type="mcp",
                server_label="server2",
                allowed_tools=["tool2"],
            ),
        ]
        result = _extract_allowed_tools_from_mcp_requests(tools)
        # Non-MCP tools should be ignored
        assert result == {
            "server1": ["tool1"],
            "server2": ["tool2"],
        }


def test_tool_parser_runs_before_reasoning_parser():
    """
    Ensure tool calls are extracted before reasoning parsing, even without <think> tags.
    """
    from vllm.entrypoints.openai.protocol import (
        ExtractedToolCallInformation,
        FunctionCall,
        ResponsesRequest,
        ToolCall,
    )
    from vllm.outputs import CompletionOutput

    engine_client = MagicMock()
    engine_client.model_config.hf_config.model_type = "test"
    engine_client.model_config.get_diff_sampling_param.return_value = {}

    instance = OpenAIServingResponses(
        engine_client=engine_client,
        models=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    # Simulate reasoning parser fallback: when no </think> tag, treats ALL
    # input as reasoning and returns None for content
    def mock_extract_reasoning(text, request=None):
        if "</think>" not in text:
            # Fallback: entire text becomes reasoning, content is None
            return (text, None)
        # Normal case: would extract between <think>...</think>
        return (None, text)

    mock_reasoning = MagicMock()
    mock_reasoning.return_value.extract_reasoning.side_effect = mock_extract_reasoning
    instance.reasoning_parser = mock_reasoning

    # Simulate tool parser: extracts tool calls if markers present
    def mock_extract_tool_calls(text, request=None):
        if "<|tool_calls_section_begin|>" in text:
            # Extract the tool call, return remaining text
            remaining = text.split("<|tool_calls_section_begin|>")[0].strip()
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[
                    ToolCall(
                        id="1",
                        type="function",
                        function=FunctionCall(name="Bash", arguments="{}"),
                    )
                ],
                content=remaining if remaining else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=text
        )

    mock_tool = MagicMock()
    mock_tool.return_value.extract_tool_calls.side_effect = mock_extract_tool_calls
    instance.tool_parser = mock_tool
    instance.enable_auto_tools = True

    # Model output WITHOUT <think> tags - this triggers reasoning parser fallback
    output = CompletionOutput(
        index=0,
        text=(
            " Let me run it. "
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.Bash:0<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        ),
        token_ids=[1],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )

    result = instance._make_response_output_items(
        request=ResponsesRequest(input="test", model="test"),
        final_output=output,
        tokenizer=MagicMock(),
    )

    # Verify tool call was extracted (not lost to reasoning parser)
    tool_items = [item for item in result if item.type == "function_call"]
    assert len(tool_items) == 1, (
        f"Tool call should be extracted even without <think> tags. "
        f"Got output types: {[i.type for i in result]}"
    )

    # Verify we have a completed status
    completed_items = [
        item for item in result if getattr(item, "status", None) == "completed"
    ]
    assert len(completed_items) >= 1, (
        f"Expected at least one item with status='completed'. "
        f"Got: {[(i.type, getattr(i, 'status', None)) for i in result]}"
    )

    # Edge case: tool call with NO preceding text (text_for_reasoning is None)
    # This should NOT fall back to original text containing tool markers
    output_no_text = CompletionOutput(
        index=0,
        text=(
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.Bash:0<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        ),
        token_ids=[1],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )

    result_no_text = instance._make_response_output_items(
        request=ResponsesRequest(input="test", model="test"),
        final_output=output_no_text,
        tokenizer=MagicMock(),
    )

    # Should still extract tool call
    tool_items_no_text = [
        item for item in result_no_text if item.type == "function_call"
    ]
    assert len(tool_items_no_text) == 1, (
        f"Tool call should be extracted even with no preceding text. "
        f"Got: {[i.type for i in result_no_text]}"
    )

    # Reasoning should NOT contain tool markers
    reasoning_items = [item for item in result_no_text if item.type == "reasoning"]
    for item in reasoning_items:
        for content in item.content:
            assert "<|tool_call" not in content.text, (
                f"Tool markers should not appear in reasoning. Got: {content.text}"
            )
