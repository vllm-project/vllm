# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from openai.types.responses.tool import (
    CodeInterpreterContainerCodeInterpreterToolAuto,
    LocalShell,
    Mcp,
    Tool,
)

import vllm.envs as envs
from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.responses.context import ConversationContext, SimpleContext
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesContextCheckpointDeleteRequest,
    ResponsesContextCheckpointRequest,
    ResponsesContextRevertRequest,
    ResponsesRequest,
)
from vllm.entrypoints.openai.responses.serving import (
    OpenAIServingResponses,
    _extract_allowed_tools_from_mcp_requests,
    extract_tool_types,
)
from vllm.inputs.data import TokensPrompt
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_checkpointing import (
    KV_CHECKPOINT_RESTORE_ID_ARG,
    KV_CHECKPOINT_SAVE_ID_ARG,
)


def _make_serving_responses_instance() -> OpenAIServingResponses:
    engine_client = MagicMock()

    model_config = MagicMock()
    model_config.max_model_len = 100
    model_config.hf_config.model_type = "test"
    model_config.get_diff_sampling_param.return_value = {}
    engine_client.model_config = model_config

    engine_client.input_processor = MagicMock()
    engine_client.io_processor = MagicMock()
    engine_client.renderer = MagicMock()
    engine_client.drop_kv_checkpoints = AsyncMock(return_value=0)

    models = MagicMock()

    return OpenAIServingResponses(
        engine_client=engine_client,
        models=models,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


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
        model_config.max_model_len = 100
        model_config.hf_config.model_type = "test"
        model_config.get_diff_sampling_param.return_value = {}
        engine_client.model_config = model_config

        engine_client.input_processor = MagicMock()
        engine_client.io_processor = MagicMock()
        engine_client.renderer = MagicMock()
        engine_client.drop_kv_checkpoints = AsyncMock(return_value=0)

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
        model_config.max_model_len = 100
        model_config.hf_config.model_type = "test"
        model_config.get_diff_sampling_param.return_value = {}
        engine_client.model_config = model_config

        engine_client.input_processor = MagicMock()
        engine_client.io_processor = MagicMock()
        engine_client.renderer = MagicMock()

        models = MagicMock()

        # Create the actual instance
        instance = OpenAIServingResponses(
            engine_client=engine_client,
            models=models,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )

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


@pytest.mark.asyncio
async def test_reasoning_tokens_counted_for_text_reasoning_model(monkeypatch):
    """Ensure reasoning_tokens usage is derived from thinking token spans."""

    class FakeTokenizer:
        def __init__(self):
            self._vocab = {"<think>": 1, "</think>": 2, "reason": 3, "final": 4}

        def get_vocab(self):
            return self._vocab

    # Force non-harmony, SimpleContext path
    monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)

    engine_client = MagicMock()
    model_config = MagicMock()
    model_config.hf_config.model_type = "test"
    model_config.hf_text_config = MagicMock()
    model_config.get_diff_sampling_param.return_value = {}
    engine_client.model_config = model_config
    engine_client.input_processor = MagicMock()
    engine_client.io_processor = MagicMock()
    engine_client.renderer = MagicMock()

    tokenizer = FakeTokenizer()
    engine_client.renderer.get_tokenizer.return_value = tokenizer

    models = MagicMock()

    serving = OpenAIServingResponses(
        engine_client=engine_client,
        models=models,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        reasoning_parser="qwen3",
    )

    # Build a SimpleContext with thinking tokens in the output.
    context = SimpleContext()
    token_ids = [1, 10, 2, 20]  # <think> 10 </think> 20 -> reasoning token count = 1
    completion = CompletionOutput(
        index=0,
        text="<think>reason</think>final",
        token_ids=token_ids,
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    req_output = RequestOutput(
        request_id="req",
        prompt="hi",
        prompt_token_ids=[7, 8],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
        num_cached_tokens=0,
    )
    context.append_output(req_output)

    async def dummy_result_generator():
        yield None

    request = ResponsesRequest(input="hi", tools=[], stream=False)
    sampling_params = SamplingParams(max_tokens=16)
    metadata = RequestResponseMetadata(request_id="req")

    response = await serving.responses_full_generator(
        request=request,
        sampling_params=sampling_params,
        result_generator=dummy_result_generator(),
        context=context,
        model_name="test-model",
        tokenizer=tokenizer,
        request_metadata=metadata,
    )

    assert response.usage.output_tokens_details.reasoning_tokens == 1


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


@pytest.mark.asyncio
async def test_context_session_rewind_injects_summary_for_string_input():
    serving = _make_serving_responses_instance()
    session_id = "session-a"
    response_id = "resp_1"
    checkpoint_label = "before-loop"
    engine_checkpoint_id = f"resp:{response_id}"

    # Register response in store so checkpoint validation can succeed.
    serving.response_store[response_id] = MagicMock()
    serving.context_manager.set_current_state(
        session_id,
        response_id,
        engine_checkpoint_id,
    )

    checkpoint_resp = await serving.drop_context_checkpoint(
        ResponsesContextCheckpointRequest(
            session_id=session_id,
            checkpoint_label=checkpoint_label,
        )
    )
    assert checkpoint_resp.response_id == response_id
    assert checkpoint_resp.engine_checkpoint_id == engine_checkpoint_id

    revert_resp = await serving.revert_context_checkpoint(
        ResponsesContextRevertRequest(
            session_id=session_id,
            checkpoint_label=checkpoint_label,
            summary="Loop produced one deterministic fix and two dead ends.",
        )
    )
    assert revert_resp.response_id == response_id
    assert revert_resp.engine_checkpoint_id == engine_checkpoint_id
    assert revert_resp.queued_summaries == 1

    request = ResponsesRequest(
        input="Continue with the fix implementation.",
        context_session_id=session_id,
        store=True,
    )
    maybe_error = await serving._resolve_context_session(request)
    assert maybe_error is None
    assert request.previous_response_id == response_id
    assert isinstance(request.input, str)
    assert "Loop produced one deterministic fix" in request.input
    assert request.vllm_xargs is not None
    assert request.vllm_xargs[KV_CHECKPOINT_RESTORE_ID_ARG] == engine_checkpoint_id
    assert request.vllm_xargs[KV_CHECKPOINT_SAVE_ID_ARG] == f"resp:{request.request_id}"


@pytest.mark.asyncio
async def test_revert_context_checkpoint_returns_not_found_for_unknown_checkpoint():
    serving = _make_serving_responses_instance()
    response = await serving.revert_context_checkpoint(
        ResponsesContextRevertRequest(
            session_id="session-a",
            checkpoint_label="missing",
            summary="summary",
        )
    )
    assert isinstance(response, ErrorResponse)
    assert response.error.code == 404


@pytest.mark.asyncio
async def test_revert_context_checkpoint_auto_summary_when_missing():
    serving = _make_serving_responses_instance()
    session_id = "session-a"
    response_id = "resp_1"
    checkpoint_label = "before-loop"
    engine_checkpoint_id = f"resp:{response_id}"

    serving.response_store[response_id] = MagicMock(
        output=[
            ResponseOutputMessage(
                id="msg_1",
                role="assistant",
                type="message",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text="Collected traceback and narrowed failure to parser.",
                        annotations=[],
                    )
                ],
            )
        ]
    )
    serving.context_manager.set_current_state(
        session_id,
        response_id,
        engine_checkpoint_id,
    )
    await serving.drop_context_checkpoint(
        ResponsesContextCheckpointRequest(
            session_id=session_id,
            checkpoint_label=checkpoint_label,
        )
    )

    revert_resp = await serving.revert_context_checkpoint(
        ResponsesContextRevertRequest(
            session_id=session_id,
            checkpoint_label=checkpoint_label,
            summary=None,
        )
    )
    assert revert_resp.response_id == response_id
    assert revert_resp.engine_checkpoint_id == engine_checkpoint_id
    assert revert_resp.queued_summaries == 1

    queued = serving.context_manager.consume_summary_queue(session_id)
    assert len(queued) == 1
    assert "Automatic rewind summary generated" in queued[0]
    assert "narrowed failure to parser" in queued[0]


@pytest.mark.asyncio
async def test_delete_context_checkpoint_drops_unreferenced_engine_checkpoint():
    serving = _make_serving_responses_instance()
    session_id = "session-a"
    current_response_id = "resp_current"
    checkpoint_response_id = "resp_saved"
    checkpoint_label = "before-loop"

    serving.response_store[current_response_id] = MagicMock()
    serving.response_store[checkpoint_response_id] = MagicMock()
    serving.context_manager.set_current_state(
        session_id,
        current_response_id,
        f"resp:{current_response_id}",
    )
    await serving.drop_context_checkpoint(
        ResponsesContextCheckpointRequest(
            session_id=session_id,
            checkpoint_label=checkpoint_label,
            response_id=checkpoint_response_id,
        )
    )

    serving.engine_client.drop_kv_checkpoints.return_value = 1
    delete_resp = await serving.delete_context_checkpoint(
        ResponsesContextCheckpointDeleteRequest(
            session_id=session_id,
            checkpoint_label=checkpoint_label,
        )
    )
    assert delete_resp.response_id == checkpoint_response_id
    assert delete_resp.engine_checkpoint_id == f"resp:{checkpoint_response_id}"
    assert delete_resp.dropped_engine_checkpoints == 1
    serving.engine_client.drop_kv_checkpoints.assert_awaited_with(
        [f"resp:{checkpoint_response_id}"]
    )


@pytest.mark.asyncio
async def test_revert_context_checkpoint_drops_stale_current_engine_checkpoint():
    serving = _make_serving_responses_instance()
    session_id = "session-a"
    current_response_id = "resp_current"
    checkpoint_response_id = "resp_saved"
    checkpoint_label = "before-loop"

    serving.response_store[current_response_id] = MagicMock(output=[])
    serving.response_store[checkpoint_response_id] = MagicMock(output=[])
    serving.context_manager.set_current_state(
        session_id,
        current_response_id,
        f"resp:{current_response_id}",
    )
    await serving.drop_context_checkpoint(
        ResponsesContextCheckpointRequest(
            session_id=session_id,
            checkpoint_label=checkpoint_label,
            response_id=checkpoint_response_id,
        )
    )

    serving.engine_client.drop_kv_checkpoints.reset_mock()
    serving.engine_client.drop_kv_checkpoints.return_value = 1
    revert_resp = await serving.revert_context_checkpoint(
        ResponsesContextRevertRequest(
            session_id=session_id,
            checkpoint_label=checkpoint_label,
            summary="summary",
        )
    )
    assert revert_resp.response_id == checkpoint_response_id
    assert revert_resp.engine_checkpoint_id == f"resp:{checkpoint_response_id}"
    serving.engine_client.drop_kv_checkpoints.assert_awaited_once_with(
        [f"resp:{current_response_id}"]
    )


@pytest.mark.asyncio
async def test_drop_context_checkpoint_evicts_lru_when_session_limit_exceeded():
    serving = _make_serving_responses_instance()
    serving.context_manager.max_checkpoints_per_session = 1
    serving.context_manager.checkpoint_ttl_s = None

    session_id = "session-a"
    current_response_id = "resp_current"
    first_checkpoint_response_id = "resp_saved_1"
    second_checkpoint_response_id = "resp_saved_2"

    serving.response_store[current_response_id] = MagicMock()
    serving.response_store[first_checkpoint_response_id] = MagicMock()
    serving.response_store[second_checkpoint_response_id] = MagicMock()
    serving.context_manager.set_current_state(
        session_id,
        current_response_id,
        f"resp:{current_response_id}",
    )

    await serving.drop_context_checkpoint(
        ResponsesContextCheckpointRequest(
            session_id=session_id,
            checkpoint_label="ckpt-1",
            response_id=first_checkpoint_response_id,
        )
    )

    serving.engine_client.drop_kv_checkpoints.reset_mock()
    serving.engine_client.drop_kv_checkpoints.return_value = 1
    await serving.drop_context_checkpoint(
        ResponsesContextCheckpointRequest(
            session_id=session_id,
            checkpoint_label="ckpt-2",
            response_id=second_checkpoint_response_id,
        )
    )

    serving.engine_client.drop_kv_checkpoints.assert_awaited_once_with(
        [f"resp:{first_checkpoint_response_id}"]
    )
