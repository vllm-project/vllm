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
    extract_tool_types,
)
from vllm.entrypoints.tool_server import ToolServer
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt


class MockConversationContext(ConversationContext):
    """Mock conversation context for testing"""

    def __init__(self):
        self.init_tool_sessions_called = False
        self.init_tool_sessions_args = None
        self.init_tool_sessions_kwargs = None

    def append_output(self, output) -> None:
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

        engine_client.processor = MagicMock()
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

        engine_client.processor = MagicMock()
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
        engine_prompt = EngineTokensPrompt(prompt_token_ids=valid_prompt_token_ids)

        # Call the method
        result = serving_responses_instance._validate_generator_input(engine_prompt)

        # Should return None for valid input
        assert result is None

        # create an invalid engine prompt
        invalid_prompt_token_ids = list(range(200))  # 100 tokens >= 100 max_model_len
        engine_prompt = EngineTokensPrompt(prompt_token_ids=invalid_prompt_token_ids)

        # Call the method
        result = serving_responses_instance._validate_generator_input(engine_prompt)

        # Should return an ErrorResponse
        assert result is not None
        assert isinstance(result, ErrorResponse)
