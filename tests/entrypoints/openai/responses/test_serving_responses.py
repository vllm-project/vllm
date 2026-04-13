# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AsyncExitStack
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from openai.types.responses import (
    ResponseOutputItemDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextConfig,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_format_text_json_schema_config import (
    ResponseFormatTextJSONSchemaConfig,
)
from openai.types.responses.tool import (
    CodeInterpreterContainerCodeInterpreterToolAuto,
    LocalShell,
    Mcp,
    Tool,
)

import vllm.envs as envs
from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.responses.context import ConversationContext, SimpleContext
from vllm.entrypoints.openai.responses.protocol import (
    ResponseCreatedEvent,
    ResponseRawMessageAndToken,
    ResponsesRequest,
    ResponsesResponse,
    serialize_message,
)
from vllm.entrypoints.openai.responses.serving import (
    OpenAIServingResponses,
    _extract_allowed_tools_from_mcp_requests,
    extract_tool_types,
)
from vllm.entrypoints.openai.responses.streaming_events import (
    StreamingState,
)
from vllm.inputs import tokens_input
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


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


def test_serialize_message_pydantic_model_returns_dict() -> None:
    msg = ResponseRawMessageAndToken(message="hello", tokens=[1, 2, 3])

    serialized = serialize_message(msg)

    assert isinstance(serialized, dict)
    assert serialized["type"] == "raw_message_tokens"
    assert serialized["message"] == "hello"


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


@pytest.mark.skip_global_cleanup
def test_response_created_event_uses_public_json_schema_alias() -> None:
    schema = {
        "type": "object",
        "properties": {
            "event_name": {"type": "string"},
            "date": {"type": "string"},
            "participants": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["event_name", "date", "participants"],
        "additionalProperties": False,
    }
    text = ResponseTextConfig()
    text.format = ResponseFormatTextJSONSchemaConfig(
        type="json_schema",
        name="calendar_event",
        schema=schema,
        description="A calendar event.",
        strict=True,
    )
    request = ResponsesRequest(
        model="test-model",
        input="Alice and Bob are going to a science fair on Friday.",
        text=text,
    )
    sampling_params = request.to_sampling_params(default_max_tokens=64)
    initial_response = ResponsesResponse.from_request(
        request=request,
        sampling_params=sampling_params,
        model_name="test-model",
        created_time=0,
        output=[],
        status="in_progress",
        usage=None,
    ).model_dump(mode="json", by_alias=True)

    fmt = initial_response["text"]["format"]
    assert fmt["schema"] == schema
    assert "schema_" not in fmt

    event = ResponseCreatedEvent(
        type="response.created",
        sequence_number=0,
        response=initial_response,
    )
    assert event.response.text is not None
    assert event.response.text.format is not None
    assert event.response.text.format.model_dump(by_alias=True)["schema"] == schema


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
        engine_client.renderer = MagicMock()

        models = MagicMock()

        tool_server = MagicMock(spec=ToolServer)

        # Create the actual instance
        instance = OpenAIServingResponses(
            engine_client=engine_client,
            models=models,
            openai_serving_render=MagicMock(),
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
        engine_client.renderer = MagicMock()

        models = MagicMock()

        # Create the actual instance
        instance = OpenAIServingResponses(
            engine_client=engine_client,
            models=models,
            openai_serving_render=MagicMock(),
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )

        return instance

    def test_validate_generator_input(self, serving_responses_instance):
        """Test _validate_generator_input with valid prompt length"""
        # Create an engine prompt with valid length (less than max_model_len)
        valid_prompt_token_ids = list(range(5))  # 5 tokens < 100 max_model_len
        engine_input = tokens_input(valid_prompt_token_ids)

        # Call the method
        result = serving_responses_instance._validate_generator_input(engine_input)

        # Should return None for valid input
        assert result is None

        # create an invalid engine prompt
        invalid_prompt_token_ids = list(range(200))  # 100 tokens >= 100 max_model_len
        engine_input = tokens_input(invalid_prompt_token_ids)

        # Call the method
        result = serving_responses_instance._validate_generator_input(engine_input)

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
    engine_client.renderer = MagicMock()

    tokenizer = FakeTokenizer()
    engine_client.renderer.get_tokenizer.return_value = tokenizer

    models = MagicMock()

    serving = OpenAIServingResponses(
        engine_client=engine_client,
        models=models,
        openai_serving_render=MagicMock(),
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


class TestHarmonyPreambleStreaming:
    """Tests for preamble (commentary with no recipient) streaming events."""

    @staticmethod
    def _make_ctx(*, channel, recipient, delta="hello"):
        """Build a lightweight mock StreamingHarmonyContext."""
        ctx = MagicMock()
        ctx.last_content_delta = delta
        ctx.parser.current_channel = channel
        ctx.parser.current_recipient = recipient
        return ctx

    @staticmethod
    def _make_previous_item(*, channel, recipient, text="preamble text"):
        """Build a lightweight mock previous_item (openai_harmony Message)."""
        content_part = MagicMock()
        content_part.text = text
        item = MagicMock()
        item.channel = channel
        item.recipient = recipient
        item.content = [content_part]
        return item

    def test_preamble_delta_emits_text_events(self) -> None:
        """commentary + recipient=None should emit output_text.delta events."""
        from vllm.entrypoints.openai.responses.streaming_events import (
            emit_content_delta_events,
        )

        ctx = self._make_ctx(channel="commentary", recipient=None)
        state = StreamingState()

        events = emit_content_delta_events(ctx, state)

        type_names = [e.type for e in events]
        assert "response.output_text.delta" in type_names
        assert "response.output_item.added" in type_names

    def test_preamble_delta_second_token_no_added(self) -> None:
        """Second preamble token should emit delta only, not added again."""
        from vllm.entrypoints.openai.responses.streaming_events import (
            emit_content_delta_events,
        )

        ctx = self._make_ctx(channel="commentary", recipient=None, delta="w")
        state = StreamingState()
        state.sent_output_item_added = True
        state.current_item_id = "msg_test"
        state.current_content_index = 0

        events = emit_content_delta_events(ctx, state)

        type_names = [e.type for e in events]
        assert "response.output_text.delta" in type_names
        assert "response.output_item.added" not in type_names

    def test_commentary_with_function_recipient_not_preamble(self) -> None:
        """commentary + recipient='functions.X' must NOT use preamble path."""
        from vllm.entrypoints.openai.responses.streaming_events import (
            emit_content_delta_events,
        )

        ctx = self._make_ctx(
            channel="commentary",
            recipient="functions.get_weather",
        )
        state = StreamingState()

        events = emit_content_delta_events(ctx, state)

        type_names = [e.type for e in events]
        assert "response.output_text.delta" not in type_names

    def test_preamble_done_emits_text_done_events(self) -> None:
        """Completed preamble should emit text done + content_part done +
        output_item done, same shape as final channel."""
        from vllm.entrypoints.openai.responses.streaming_events import (
            emit_previous_item_done_events,
        )

        previous = self._make_previous_item(channel="commentary", recipient=None)
        state = StreamingState()
        state.current_item_id = "msg_test"
        state.current_output_index = 0
        state.current_content_index = 0

        events = emit_previous_item_done_events(previous, state)

        type_names = [e.type for e in events]
        assert "response.output_text.done" in type_names
        assert "response.content_part.done" in type_names
        assert "response.output_item.done" in type_names

    def test_commentary_with_recipient_no_preamble_done(self) -> None:
        """commentary + recipient='functions.X' should route to function call
        done, not preamble done."""
        from vllm.entrypoints.openai.responses.streaming_events import (
            emit_previous_item_done_events,
        )

        previous = self._make_previous_item(
            channel="commentary", recipient="functions.get_weather"
        )
        state = StreamingState()
        state.current_item_id = "fc_test"

        events = emit_previous_item_done_events(previous, state)

        type_names = [e.type for e in events]
        assert "response.output_text.done" not in type_names


def _make_simple_context_with_output(text, token_ids):
    """Create a SimpleContext with a RequestOutput containing the given text."""
    ctx = SimpleContext()
    completion = CompletionOutput(
        index=0,
        text=text,
        token_ids=token_ids,
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason=None,
        stop_reason=None,
    )
    req_output = RequestOutput(
        request_id="req",
        prompt="hi",
        prompt_token_ids=[7, 8],
        prompt_logprobs=None,
        outputs=[completion],
        finished=False,
        num_cached_tokens=0,
    )
    ctx.append_output(req_output)
    return ctx


def _make_serving_instance_with_reasoning():
    """Create an OpenAIServingResponses with a mocked reasoning parser."""
    engine_client = MagicMock()
    model_config = MagicMock()
    model_config.max_model_len = 100
    model_config.hf_config.model_type = "test"
    model_config.hf_text_config = MagicMock()
    model_config.get_diff_sampling_param.return_value = {}
    engine_client.model_config = model_config
    engine_client.input_processor = MagicMock()
    engine_client.renderer = MagicMock()

    models = MagicMock()

    serving = OpenAIServingResponses(
        engine_client=engine_client,
        models=models,
        openai_serving_render=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        reasoning_parser="qwen3",
    )
    return serving


def _identity_increment(event):
    """Simple identity callable for _increment_sequence_number_and_return."""
    seq = getattr(_identity_increment, "_counter", 0)
    if hasattr(event, "sequence_number"):
        event.sequence_number = seq
    _identity_increment._counter = seq + 1  # type: ignore
    return event


def _mock_parser_with_reasoning(serving, delta_sequence: list[DeltaMessage]):
    """Set up serving.parser so that it returns a mock parser instance
    with a reasoning parser that returns the given delta_sequence.

    The mock has reasoning_parser set (truthy) but tool_parser as None,
    so the parser's parse_delta enters the reasoning-only branch.
    """
    call_count = 0

    def mock_parse_delta(**kwargs):
        nonlocal call_count
        if call_count >= len(delta_sequence):
            return None
        result = delta_sequence[call_count]
        call_count += 1
        return result

    mock_parser_instance = MagicMock()
    mock_parser_instance.reasoning_parser = MagicMock()  # truthy
    mock_parser_instance.tool_parser = None
    mock_parser_instance.parse_delta = mock_parse_delta
    mock_parser_instance.is_reasoning_end = MagicMock(return_value=False)
    serving.parser = MagicMock(return_value=mock_parser_instance)


class TestStreamingReasoningToContentTransition:
    """Tests for _process_simple_streaming_events reasoning-to-content
    transition, specifically the fix for mixed deltas that carry both
    reasoning and content simultaneously."""

    @pytest.mark.asyncio
    async def test_mixed_delta_reasoning_and_content_emits_reasoning_delta(
        self, monkeypatch
    ):
        """When the reasoning parser produces a delta with both reasoning
        and content set (e.g. reasoning end and content start in the same
        chunk), the trailing reasoning text must be emitted as a
        ResponseReasoningTextDeltaEvent and included in the
        ResponseReasoningTextDoneEvent text."""

        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        serving = _make_serving_instance_with_reasoning()

        # Sequence of DeltaMessages the mock orchestrator will return
        delta_sequence = [
            DeltaMessage(reasoning="thinking..."),
            DeltaMessage(reasoning=" end", content="hello"),  # mixed delta
            DeltaMessage(content=" world"),
        ]
        _mock_parser_with_reasoning(serving, delta_sequence)
        # Create contexts for each streaming chunk
        contexts = [
            _make_simple_context_with_output("chunk1", [10]),
            _make_simple_context_with_output("chunk2", [20]),
            _make_simple_context_with_output("chunk3", [30]),
        ]

        async def result_generator():
            for ctx in contexts:
                yield ctx

        request = ResponsesRequest(input="hi", tools=[], stream=True)
        sampling_params = SamplingParams(max_tokens=64)
        metadata = RequestResponseMetadata(request_id="req")
        _identity_increment._counter = 0  # type: ignore

        events = []
        async for event in serving._process_simple_streaming_events(
            request=request,
            sampling_params=sampling_params,
            result_generator=result_generator(),
            context=SimpleContext(),
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
            created_time=0,
            _increment_sequence_number_and_return=_identity_increment,
        ):
            events.append(event)

        # The first reasoning delta should be emitted
        reasoning_deltas = [
            e for e in events if isinstance(e, ResponseReasoningTextDeltaEvent)
        ]
        assert len(reasoning_deltas) == 2
        assert reasoning_deltas[0].delta == "thinking..."
        # The trailing reasoning from the mixed delta must also be emitted
        assert reasoning_deltas[1].delta == " end"

        # The done event must include both reasoning parts
        reasoning_done = [
            e for e in events if isinstance(e, ResponseReasoningTextDoneEvent)
        ]
        assert len(reasoning_done) == 1
        assert reasoning_done[0].text == "thinking... end"

        # Content deltas should be emitted for both the mixed delta's
        # content and the pure content delta
        text_deltas = [e for e in events if isinstance(e, ResponseTextDeltaEvent)]
        assert len(text_deltas) == 2
        assert text_deltas[0].delta == "hello"
        assert text_deltas[1].delta == " world"

    @pytest.mark.asyncio
    async def test_transition_without_mixed_delta_no_extra_reasoning_event(
        self, monkeypatch
    ):
        """When the transition from reasoning to content is clean (no mixed
        delta), no extra reasoning delta event should be emitted."""

        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        serving = _make_serving_instance_with_reasoning()

        delta_sequence = [
            DeltaMessage(reasoning="thinking"),
            DeltaMessage(content="answer"),
        ]
        _mock_parser_with_reasoning(serving, delta_sequence)

        contexts = [
            _make_simple_context_with_output("chunk1", [10]),
            _make_simple_context_with_output("chunk2", [20]),
        ]

        async def result_generator():
            for ctx in contexts:
                yield ctx

        request = ResponsesRequest(input="hi", tools=[], stream=True)
        sampling_params = SamplingParams(max_tokens=64)
        metadata = RequestResponseMetadata(request_id="req")
        _identity_increment._counter = 0  # type: ignore

        events = []
        async for event in serving._process_simple_streaming_events(
            request=request,
            sampling_params=sampling_params,
            result_generator=result_generator(),
            context=SimpleContext(),
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
            created_time=0,
            _increment_sequence_number_and_return=_identity_increment,
        ):
            events.append(event)

        # Exactly one reasoning delta
        reasoning_deltas = [
            e for e in events if isinstance(e, ResponseReasoningTextDeltaEvent)
        ]
        assert len(reasoning_deltas) == 1
        assert reasoning_deltas[0].delta == "thinking"

        # Done event has just "thinking"
        reasoning_done = [
            e for e in events if isinstance(e, ResponseReasoningTextDoneEvent)
        ]
        assert len(reasoning_done) == 1
        assert reasoning_done[0].text == "thinking"

        # One content delta
        text_deltas = [e for e in events if isinstance(e, ResponseTextDeltaEvent)]
        assert len(text_deltas) == 1
        assert text_deltas[0].delta == "answer"

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_no_content(self, monkeypatch):
        """When the stream has only reasoning deltas and no content, the
        reasoning done event should be emitted at finalization with the
        full accumulated text, and no text delta events should appear."""

        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        serving = _make_serving_instance_with_reasoning()

        delta_sequence = [
            DeltaMessage(reasoning="step 1"),
            DeltaMessage(reasoning=" step 2"),
        ]
        _mock_parser_with_reasoning(serving, delta_sequence)

        contexts = [
            _make_simple_context_with_output("chunk1", [10]),
            _make_simple_context_with_output("chunk2", [20]),
        ]

        async def result_generator():
            for ctx in contexts:
                yield ctx

        request = ResponsesRequest(input="hi", tools=[], stream=True)
        sampling_params = SamplingParams(max_tokens=64)
        metadata = RequestResponseMetadata(request_id="req")
        _identity_increment._counter = 0  # type: ignore

        events = []
        async for event in serving._process_simple_streaming_events(
            request=request,
            sampling_params=sampling_params,
            result_generator=result_generator(),
            context=SimpleContext(),
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
            created_time=0,
            _increment_sequence_number_and_return=_identity_increment,
        ):
            events.append(event)

        # Two reasoning deltas
        reasoning_deltas = [
            e for e in events if isinstance(e, ResponseReasoningTextDeltaEvent)
        ]
        assert len(reasoning_deltas) == 2
        assert reasoning_deltas[0].delta == "step 1"
        assert reasoning_deltas[1].delta == " step 2"

        # Done event at finalization with accumulated text
        reasoning_done = [
            e for e in events if isinstance(e, ResponseReasoningTextDoneEvent)
        ]
        assert len(reasoning_done) == 1
        assert reasoning_done[0].text == "step 1 step 2"

        # No content text deltas
        text_deltas = [e for e in events if isinstance(e, ResponseTextDeltaEvent)]
        assert len(text_deltas) == 0

        # Final item should be a reasoning item
        item_done_events = [
            e for e in events if isinstance(e, ResponseOutputItemDoneEvent)
        ]
        assert len(item_done_events) == 1
        assert isinstance(item_done_events[0].item, ResponseReasoningItem)


class TestAutoToolStreaming:
    @staticmethod
    async def _collect_events(delta_sequence: list[DeltaMessage]):
        serving = _make_serving_instance_with_reasoning()
        _mock_parser_with_reasoning(serving, delta_sequence)

        contexts = [
            _make_simple_context_with_output("chunk", [i])
            for i in range(len(delta_sequence))
        ]

        async def result_generator():
            for ctx in contexts:
                yield ctx

        request = ResponsesRequest(
            input="hi",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                }
            ],
            tool_choice="auto",
            stream=True,
        )
        sampling_params = SamplingParams(max_tokens=64)
        metadata = RequestResponseMetadata(request_id="req")
        _identity_increment._counter = 0  # type: ignore

        events = []
        async for event in serving._process_simple_streaming_events(
            request=request,
            sampling_params=sampling_params,
            result_generator=result_generator(),
            context=SimpleContext(),
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
            created_time=0,
            _increment_sequence_number_and_return=_identity_increment,
        ):
            events.append(event)
        return events

    @pytest.mark.skip_global_cleanup
    @pytest.mark.asyncio
    async def test_auto_multi_tool_streaming_opens_one_item_per_tool(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)

        delta_sequence = [
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        id="call_vienna",
                        type="function",
                        index=0,
                        function=DeltaFunctionCall(
                            name="get_weather",
                            arguments="",
                        ),
                    )
                ]
            ),
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        function=DeltaFunctionCall(
                            arguments='{"location":"Vienna"}',
                        ),
                    )
                ]
            ),
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        id="call_berlin",
                        type="function",
                        index=1,
                        function=DeltaFunctionCall(
                            name="get_weather",
                            arguments='{"location":"Berlin"}',
                        ),
                    )
                ]
            ),
        ]
        events = await self._collect_events(delta_sequence)

        function_items = [
            event
            for event in events
            if event.type == "response.output_item.added"
            and getattr(event.item, "type", None) == "function_call"
        ]
        assert len(function_items) == 2
        assert [event.item.name for event in function_items] == [
            "get_weather",
            "get_weather",
        ]
        assert [event.output_index for event in function_items] == [0, 1]

        argument_deltas = [
            event.delta
            for event in events
            if event.type == "response.function_call_arguments.delta"
        ]
        assert argument_deltas == [
            '{"location":"Vienna"}',
            '{"location":"Berlin"}',
        ]

        argument_done = [
            event
            for event in events
            if event.type == "response.function_call_arguments.done"
        ]
        assert [event.arguments for event in argument_done] == [
            '{"location":"Vienna"}',
            '{"location":"Berlin"}',
        ]
        assert [event.output_index for event in argument_done] == [0, 1]

        function_done = [
            event
            for event in events
            if event.type == "response.output_item.done"
            and getattr(event.item, "type", None) == "function_call"
        ]
        assert [event.item.arguments for event in function_done] == [
            '{"location":"Vienna"}',
            '{"location":"Berlin"}',
        ]
        assert [event.output_index for event in function_done] == [0, 1]

    @pytest.mark.skip_global_cleanup
    @pytest.mark.asyncio
    async def test_auto_tool_choice_first_delta_tool_call_does_not_duplicate_item(
        self, monkeypatch
    ):
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)

        delta_sequence = [
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        id="call_test",
                        type="function",
                        index=0,
                        function=DeltaFunctionCall(
                            name="get_weather",
                            arguments="",
                        ),
                    )
                ]
            ),
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        function=DeltaFunctionCall(
                            arguments='{"location":"Berlin"}',
                        ),
                    )
                ]
            ),
        ]
        events = await self._collect_events(delta_sequence)

        function_items = [
            event
            for event in events
            if event.type == "response.output_item.added"
            and getattr(event.item, "type", None) == "function_call"
        ]
        assert len(function_items) == 1
        assert function_items[0].item.name == "get_weather"

        argument_deltas = [
            event.delta
            for event in events
            if event.type == "response.function_call_arguments.delta"
        ]
        assert "".join(argument_deltas) == '{"location":"Berlin"}'
# ── helpers for parallel tool call tests ─────────────────────────────────────

def _make_serving_instance_for_tool_calls():
    """Create a minimal OpenAIServingResponses instance for tool call tests."""
    engine_client = MagicMock()
    model_config = MagicMock()
    model_config.max_model_len = 1000
    model_config.hf_config.model_type = "test"
    model_config.get_diff_sampling_param.return_value = {}
    engine_client.model_config = model_config
    engine_client.input_processor = MagicMock()
    engine_client.renderer = MagicMock()
    return OpenAIServingResponses(
        engine_client=engine_client,
        models=MagicMock(),
        openai_serving_render=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


def _mock_parser_with_tool_calls(serving, delta_sequence: list[DeltaMessage]):
    """Attach a mock parser to *serving* that returns *delta_sequence* in order."""
    call_count = 0

    def _parse_delta(**kwargs):
        nonlocal call_count
        if call_count >= len(delta_sequence):
            return None
        result = delta_sequence[call_count]
        call_count += 1
        return result

    mock_parser_instance = MagicMock()
    mock_parser_instance.tool_parser = MagicMock()  # truthy so TCs are expected
    mock_parser_instance.reasoning_parser = None
    mock_parser_instance.parse_delta = _parse_delta
    serving.parser = MagicMock(return_value=mock_parser_instance)


def _tc(index: int, name: str | None = None, args: str | None = None) -> DeltaToolCall:
    """Build a DeltaToolCall with an optional function name and/or argument chunk."""
    return DeltaToolCall(
        index=index,
        function=DeltaFunctionCall(name=name, arguments=args),
    )


async def _run_simple_streaming(
    serving,
    delta_sequence: list[DeltaMessage],
    *,
    parallel_tool_calls: bool | None = None,
) -> list:
    """Drive _process_simple_streaming_events and return all emitted events."""
    _mock_parser_with_tool_calls(serving, delta_sequence)

    contexts = [
        _make_simple_context_with_output(f"chunk{i}", [i + 1])
        for i in range(len(delta_sequence))
    ]

    async def _result_gen():
        for ctx in contexts:
            yield ctx

    request = ResponsesRequest(
        input="test",
        tools=[],
        stream=True,
        parallel_tool_calls=parallel_tool_calls,
    )
    sampling_params = SamplingParams(max_tokens=100)
    metadata = RequestResponseMetadata(request_id="test-req")
    _identity_increment._counter = 0  # type: ignore

    events: list = []
    async for event in serving._process_simple_streaming_events(
        request=request,
        sampling_params=sampling_params,
        result_generator=_result_gen(),
        context=SimpleContext(),
        model_name="test-model",
        tokenizer=MagicMock(),
        request_metadata=metadata,
        created_time=0,
        _increment_sequence_number_and_return=_identity_increment,
    ):
        events.append(event)
    return events


class TestParallelToolCallStreaming:
    """Unit tests for parallel tool call streaming (GitHub issue #39584).

    Verifies that _process_simple_streaming_events correctly handles multiple
    tool calls in a single SSE delta using per-call ToolCallStreamState tracking.
    Prior to the fix, these cases crashed with AssertionError or silently dropped
    all tool calls beyond the first.
    """

    @pytest.mark.asyncio
    async def test_single_tool_call_no_regression(self, monkeypatch):
        """Single TC streaming still works correctly after the parallel-TC fix."""
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        serving = _make_serving_instance_for_tool_calls()

        delta_sequence = [
            DeltaMessage(tool_calls=[_tc(0, "get_weather")]),
            DeltaMessage(tool_calls=[_tc(0, args='{"city": "B')]),
            DeltaMessage(tool_calls=[_tc(0, args='erlin"}')]),
        ]
        events = await _run_simple_streaming(serving, delta_sequence)

        types = [e.type for e in events]
        assert types.count("response.output_item.added") == 1
        assert types.count("response.function_call_arguments.done") == 1
        assert types.count("response.output_item.done") == 1

        added = next(e for e in events if e.type == "response.output_item.added")
        assert added.output_index == 0
        assert added.item.name == "get_weather"

        args_done = next(
            e for e in events if e.type == "response.function_call_arguments.done"
        )
        assert args_done.arguments == '{"city": "Berlin"}'

    @pytest.mark.asyncio
    async def test_two_tool_calls_in_first_delta(self, monkeypatch):
        """Two TCs in the first delta produce two separate output items at
        consecutive output_indexes (0 and 1)."""
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        serving = _make_serving_instance_for_tool_calls()

        delta_sequence = [
            DeltaMessage(tool_calls=[
                _tc(0, "get_weather"),
                _tc(1, "get_forecast"),
            ]),
            DeltaMessage(tool_calls=[
                _tc(0, args='{"city": "Berlin"}'),
                _tc(1, args='{"city": "London"}'),
            ]),
        ]
        events = await _run_simple_streaming(serving, delta_sequence)

        types = [e.type for e in events]
        # One output_item.added per tool call
        assert types.count("response.output_item.added") == 2
        # One args delta per TC per args-carrying delta
        assert types.count("response.function_call_arguments.delta") == 2
        # One args done + item done per TC
        assert types.count("response.function_call_arguments.done") == 2
        assert types.count("response.output_item.done") == 2

        added_events = [e for e in events if e.type == "response.output_item.added"]
        output_indexes = sorted(e.output_index for e in added_events)
        assert output_indexes == [0, 1], (
            "Output indexes must be distinct and sequential"
        )

        names = {e.output_index: e.item.name for e in added_events}
        assert names[0] == "get_weather"
        assert names[1] == "get_forecast"

    @pytest.mark.asyncio
    async def test_parallel_args_attributed_correctly_by_index(self, monkeypatch):
        """Argument fragments across multiple deltas are routed to the correct
        TC by DeltaToolCall.index, not by arrival order."""
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        serving = _make_serving_instance_for_tool_calls()

        delta_sequence = [
            # Both TCs announced together in first delta
            DeltaMessage(tool_calls=[_tc(0, "tc_a"), _tc(1, "tc_b")]),
            # First fragment — index 0 gets '{"x":', index 1 gets '{"y":'
            DeltaMessage(tool_calls=[
                _tc(0, args='{"x":'),
                _tc(1, args='{"y":'),
            ]),
            # Second fragment
            DeltaMessage(tool_calls=[
                _tc(0, args="1}"),
                _tc(1, args="2}"),
            ]),
        ]
        events = await _run_simple_streaming(serving, delta_sequence)

        done_events = [
            e for e in events
            if e.type == "response.function_call_arguments.done"
        ]
        assert len(done_events) == 2

        args_by_name: dict[str, str] = {e.name: e.arguments for e in done_events}
        assert args_by_name["tc_a"] == '{"x":1}'
        assert args_by_name["tc_b"] == '{"y":2}'

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_false_keeps_only_first(self, monkeypatch):
        """parallel_tool_calls=False must filter all TCs with index != 0,
        keeping only the first tool call in the output."""
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        serving = _make_serving_instance_for_tool_calls()

        delta_sequence = [
            DeltaMessage(tool_calls=[
                _tc(0, "tc_a"),
                _tc(1, "tc_b"),  # should be filtered out
            ]),
            DeltaMessage(tool_calls=[
                _tc(0, args='{"x": 1}'),
                _tc(1, args='{"y": 2}'),  # filtered
            ]),
        ]
        events = await _run_simple_streaming(
            serving, delta_sequence, parallel_tool_calls=False
        )

        types = [e.type for e in events]
        assert types.count("response.output_item.added") == 1
        assert types.count("response.function_call_arguments.done") == 1

        added = next(e for e in events if e.type == "response.output_item.added")
        assert added.item.name == "tc_a"

        done = next(
            e for e in events if e.type == "response.function_call_arguments.done"
        )
        assert done.name == "tc_a"
        assert done.arguments == '{"x": 1}'

    @pytest.mark.asyncio
    async def test_first_delta_args_preserved(self, monkeypatch):
        """When name + arguments arrive in the same registration delta,
        arguments must appear in the finalized arguments.done event."""
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        serving = _make_serving_instance_for_tool_calls()

        delta_sequence = [
            # Name AND arguments bundled in one delta (atomic parser)
            DeltaMessage(tool_calls=[
                _tc(0, name="get_weather", args='{"city": "Berlin"}'),
            ]),
        ]
        events = await _run_simple_streaming(serving, delta_sequence)

        done = next(
            e for e in events
            if e.type == "response.function_call_arguments.done"
        )
        assert done.arguments == '{"city": "Berlin"}', (
            "Arguments from the registration delta must not be dropped"
        )

    @pytest.mark.asyncio
    async def test_reasoning_then_tool_call_transition(self, monkeypatch):
        """When reasoning deltas precede tool calls, the reasoning item must
        be properly closed before tool call items are opened."""
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        serving = _make_serving_instance_with_reasoning()
        # Need both reasoning_parser AND tool_parser truthy
        _mock_parser_with_tool_calls(serving, [
            DeltaMessage(reasoning="Let me think..."),
            DeltaMessage(reasoning=" about this."),
            DeltaMessage(tool_calls=[_tc(0, "get_weather")]),
            DeltaMessage(tool_calls=[_tc(0, args='{"city": "NYC"}')]),
        ])
        # Override: set reasoning_parser truthy so reasoning deltas are
        # recognized as reasoning (not text)
        serving.parser.return_value.reasoning_parser = MagicMock()

        events = await _run_simple_streaming(serving, [
            DeltaMessage(reasoning="Let me think..."),
            DeltaMessage(reasoning=" about this."),
            DeltaMessage(tool_calls=[_tc(0, "get_weather")]),
            DeltaMessage(tool_calls=[_tc(0, args='{"city": "NYC"}')]),
        ])

        types = [e.type for e in events]

        # Reasoning must be opened and closed before tool calls
        assert "response.reasoning_text.delta" in types
        assert "response.reasoning_text.done" in types

        # Tool call must be opened and closed
        assert "response.output_item.added" in types
        assert "response.function_call_arguments.done" in types

        # Reasoning close must come BEFORE tool call open
        reasoning_done_idx = types.index("response.reasoning_text.done")
        tc_added_idx = types.index("response.output_item.added")
        assert reasoning_done_idx < tc_added_idx, (
            "Reasoning must be closed before tool call is opened"
        )
