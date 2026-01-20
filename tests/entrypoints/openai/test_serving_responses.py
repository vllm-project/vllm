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

from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.responses.context import ConversationContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import (
    OpenAIServingResponses,
    _extract_allowed_tools_from_mcp_requests,
    extract_tool_types,
)
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


class TestSplitCompletionOutput:
    """Test class for _split_completion_output method"""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()

        # Define token mappings for delimiters
        token_map = {
            "]~b]": [100],  # start delimiter
            "[e~[": [200],  # end delimiter
            "hello": [1],
            "world": [2],
            "test": [3],
        }

        decode_map = {
            100: "]~b]",
            200: "[e~[",
            1: "hello",
            2: "world",
            3: "test",
        }

        def mock_encode(text, add_special_tokens=False):
            return token_map.get(text, [999])

        def mock_decode(token_ids):
            return "".join(decode_map.get(tid, "?") for tid in token_ids)

        tokenizer.encode = mock_encode
        tokenizer.decode = mock_decode
        return tokenizer

    @pytest.fixture
    def mock_reasoning_parser_class(self):
        """Create a mock reasoning parser class with start_message and end_message."""

        class MockReasoningParser:
            start_message = ["]~b]"]
            end_message = ["[e~["]

            def __init__(self, tokenizer):
                pass

        return MockReasoningParser

    @pytest.fixture
    def mock_reasoning_parser_no_messages(self):
        """Create a mock reasoning parser without start_message/end_message."""

        class MockReasoningParserNoMessages:
            def __init__(self, tokenizer):
                pass

        return MockReasoningParserNoMessages

    @pytest_asyncio.fixture
    async def serving_responses_instance(self):
        """Create a real OpenAIServingResponses instance for testing."""
        engine_client = MagicMock()

        model_config = MagicMock()
        model_config.hf_config.model_type = "test"
        model_config.get_diff_sampling_param.return_value = {}
        engine_client.model_config = model_config

        engine_client.input_processor = MagicMock()
        engine_client.io_processor = MagicMock()

        models = MagicMock()

        instance = OpenAIServingResponses(
            engine_client=engine_client,
            models=models,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )

        return instance

    def test_no_reasoning_parser(self, serving_responses_instance, mock_tokenizer):
        """Test that original output is returned when no reasoning parser."""
        from vllm.outputs import CompletionOutput

        output = CompletionOutput(
            index=0,
            text="hello world",
            token_ids=[1, 2],
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
            stop_reason=None,
        )

        serving_responses_instance.reasoning_parser = None

        result = serving_responses_instance._split_completion_output(
            output, mock_tokenizer
        )

        assert len(result) == 1
        assert result[0] is output

    def test_reasoning_parser_without_start_message(
        self,
        serving_responses_instance,
        mock_tokenizer,
        mock_reasoning_parser_no_messages,
    ):
        """Test parser without start_message attribute returns original output."""
        from vllm.outputs import CompletionOutput

        output = CompletionOutput(
            index=0,
            text="hello world",
            token_ids=[1, 2],
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
            stop_reason=None,
        )

        serving_responses_instance.reasoning_parser = mock_reasoning_parser_no_messages

        result = serving_responses_instance._split_completion_output(
            output, mock_tokenizer
        )

        assert len(result) == 1
        assert result[0] is output

    def test_no_delimiter_in_output(
        self,
        serving_responses_instance,
        mock_tokenizer,
        mock_reasoning_parser_class,
    ):
        """Test that original output is returned when no delimiter found."""
        from vllm.outputs import CompletionOutput

        output = CompletionOutput(
            index=0,
            text="hello world",
            token_ids=[1, 2],  # No delimiter tokens
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
            stop_reason=None,
        )

        serving_responses_instance.reasoning_parser = mock_reasoning_parser_class

        result = serving_responses_instance._split_completion_output(
            output, mock_tokenizer
        )

        assert len(result) == 1
        assert result[0] is output

    def test_split_with_delimiter(
        self,
        serving_responses_instance,
        mock_tokenizer,
        mock_reasoning_parser_class,
    ):
        """Test splitting output at delimiter token."""
        from vllm.outputs import CompletionOutput

        # Output: hello, delimiter, world
        output = CompletionOutput(
            index=0,
            text="hello ]~b] world",
            token_ids=[1, 100, 2],  # 100 is delimiter
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
            stop_reason=None,
        )

        serving_responses_instance.reasoning_parser = mock_reasoning_parser_class

        result = serving_responses_instance._split_completion_output(
            output, mock_tokenizer
        )

        assert len(result) == 2
        # First segment: before delimiter
        assert result[0].token_ids == [1]
        assert result[0].finish_reason is None
        # Second segment: delimiter + after
        assert result[1].token_ids == [100, 2]
        assert result[1].finish_reason == "stop"

    def test_split_with_end_delimiter_stripping(
        self,
        serving_responses_instance,
        mock_tokenizer,
        mock_reasoning_parser_class,
    ):
        """Test that end delimiters are stripped from segment ends."""
        from vllm.outputs import CompletionOutput

        # Output: hello, end_delimiter, start_delimiter, world
        output = CompletionOutput(
            index=0,
            text="hello [e~[ ]~b] world",
            token_ids=[1, 200, 100, 2],  # 200 is end delimiter, 100 is start
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
            stop_reason=None,
        )

        serving_responses_instance.reasoning_parser = mock_reasoning_parser_class

        result = serving_responses_instance._split_completion_output(
            output, mock_tokenizer
        )

        # First segment should have end delimiter stripped
        assert len(result) == 2
        assert result[0].token_ids == [1]  # end delimiter stripped
        assert result[1].token_ids == [100, 2]

    def test_split_preserves_logprobs(
        self,
        serving_responses_instance,
        mock_tokenizer,
        mock_reasoning_parser_class,
    ):
        """Test that logprobs are correctly split across segments."""
        from vllm.outputs import CompletionOutput

        logprobs_data = [
            {1: MagicMock()},
            {100: MagicMock()},
            {2: MagicMock()},
        ]

        output = CompletionOutput(
            index=0,
            text="hello ]~b] world",
            token_ids=[1, 100, 2],
            cumulative_logprob=None,
            logprobs=logprobs_data,
            finish_reason="stop",
            stop_reason=None,
        )

        serving_responses_instance.reasoning_parser = mock_reasoning_parser_class

        result = serving_responses_instance._split_completion_output(
            output, mock_tokenizer
        )

        assert len(result) == 2
        assert result[0].logprobs == [logprobs_data[0]]
        assert result[1].logprobs == [logprobs_data[1], logprobs_data[2]]

    def test_delimiter_at_start(
        self,
        serving_responses_instance,
        mock_tokenizer,
        mock_reasoning_parser_class,
    ):
        """Test handling when delimiter is at the start of output."""
        from vllm.outputs import CompletionOutput

        output = CompletionOutput(
            index=0,
            text="]~b] hello world",
            token_ids=[100, 1, 2],  # Delimiter at start
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
            stop_reason=None,
        )

        serving_responses_instance.reasoning_parser = mock_reasoning_parser_class

        result = serving_responses_instance._split_completion_output(
            output, mock_tokenizer
        )

        # Should return single segment starting with delimiter
        assert len(result) == 1
        assert result[0].token_ids == [100, 1, 2]
        assert result[0].finish_reason == "stop"

    def test_multiple_delimiters(
        self,
        serving_responses_instance,
        mock_tokenizer,
        mock_reasoning_parser_class,
    ):
        """Test splitting with multiple delimiters."""
        from vllm.outputs import CompletionOutput

        # Output: hello, delimiter, world, delimiter, test
        output = CompletionOutput(
            index=0,
            text="hello ]~b] world ]~b] test",
            token_ids=[1, 100, 2, 100, 3],
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
            stop_reason=None,
        )

        serving_responses_instance.reasoning_parser = mock_reasoning_parser_class

        result = serving_responses_instance._split_completion_output(
            output, mock_tokenizer
        )

        assert len(result) == 3
        assert result[0].token_ids == [1]
        assert result[0].finish_reason is None
        assert result[1].token_ids == [100, 2]
        assert result[1].finish_reason is None
        assert result[2].token_ids == [100, 3]
        assert result[2].finish_reason == "stop"
