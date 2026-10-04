# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Additional unit tests for OpenAIServingResponses covering:
- Validation failures not tested elsewhere
  (harmony logprobs, background/store, mutual exclusion, cancel constraints)
- Store disabled vs enabled lifecycle (create / retrieve / continue)
- Response status flows (incomplete, cancelled)
- cancel_responses and retrieve_responses route behavior
- Usage per-turn accounting
- Background request failure path
- Tool-session initialization is skipped when no tools are given and invoked
  when tools are given
- Streaming event sequence numbers are monotonically increasing
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

import vllm.envs as envs
from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsage,
)
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.inputs import tokens_input
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_serving(
    *,
    model_type: str = "test",
    max_model_len: int = 100,
    enable_store: bool = False,
    tool_server: ToolServer | None = None,
    reasoning_parser: str = "",
) -> OpenAIServingResponses:
    """Return an OpenAIServingResponses instance backed by MagicMocks."""
    engine_client = MagicMock()
    model_config = MagicMock()
    model_config.max_model_len = max_model_len
    model_config.hf_config.model_type = model_type
    model_config.get_diff_sampling_param.return_value = {}
    engine_client.model_config = model_config
    engine_client.input_processor = MagicMock()
    engine_client.renderer = MagicMock()

    models = MagicMock()

    instance = OpenAIServingResponses(
        engine_client=engine_client,
        models=models,
        online_renderer=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        reasoning_parser=reasoning_parser,
        tool_server=tool_server,
    )
    # Override the store flag without re-reading the environment variable.
    instance.enable_store = enable_store
    return instance


@pytest_asyncio.fixture
async def serving() -> OpenAIServingResponses:
    return _make_serving()


@pytest_asyncio.fixture
async def serving_with_store() -> OpenAIServingResponses:
    return _make_serving(enable_store=True)


@pytest_asyncio.fixture
async def harmony_serving() -> OpenAIServingResponses:
    return _make_serving(model_type="gpt_oss")


# ---------------------------------------------------------------------------
# _validate_create_responses_input
# ---------------------------------------------------------------------------


class TestValidateCreateResponsesInput:
    """Covers every branch in _validate_create_responses_input."""

    def test_harmony_logprobs_rejected(self, harmony_serving):
        """gpt-oss models must reject logprob requests."""
        request = ResponsesRequest(
            input="hi",
            include=["message.output_text.logprobs"],
        )
        error = harmony_serving._validate_create_responses_input(request)
        assert error is not None
        assert error.error.type == "invalid_request_error"
        assert "logprobs" in error.error.message

    def test_background_requires_store_enabled(self, serving):
        """background=True should fail if the server store is disabled."""
        # Pydantic requires store=True for background=True at validation time;
        # the serving layer validates that the store is actually enabled.
        request = ResponsesRequest(input="hi", store=True, background=True)
        # Manually force store to True so the Pydantic model passes:
        error = serving._validate_create_responses_input(request)
        assert error is not None
        assert "background" in error.error.message or "store" in error.error.message

    def test_background_allowed_when_store_enabled(self, serving_with_store):
        """background=True should be accepted when the store is enabled."""
        request = ResponsesRequest(input="hi", store=True, background=True)
        error = serving_with_store._validate_create_responses_input(request)
        assert error is None

    def test_mutual_exclusion_previous_input_and_response_id(self, serving):
        """previous_input_messages and previous_response_id are mutually exclusive."""
        request = ResponsesRequest(
            input="hi",
            previous_input_messages=[
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
            previous_response_id="resp_abc",
        )
        error = serving._validate_create_responses_input(request)
        assert error is not None
        assert error.error.type == "invalid_request_error"

    def test_no_errors_for_plain_request(self, serving):
        """A basic request should pass all validations."""
        request = ResponsesRequest(input="hello")
        error = serving._validate_create_responses_input(request)
        assert error is None

    def test_no_errors_for_plain_harmony_request(self, harmony_serving):
        """A basic request without logprobs should pass harmony validation."""
        request = ResponsesRequest(input="hello")
        error = harmony_serving._validate_create_responses_input(request)
        assert error is None


# ---------------------------------------------------------------------------
# _validate_generator_input
# ---------------------------------------------------------------------------


class TestValidateGeneratorInput:
    """Prompt-length validation."""

    def test_valid_prompt_passes(self, serving):
        engine_input = tokens_input(list(range(5)))  # 5 < 100
        assert serving._validate_generator_input(engine_input) is None

    def test_prompt_at_limit_fails(self, serving):
        engine_input = tokens_input(list(range(100)))  # 100 == max_model_len
        error = serving._validate_generator_input(engine_input)
        assert isinstance(error, ErrorResponse)
        assert "exceeds" in error.error.message

    def test_prompt_over_limit_fails(self, serving):
        engine_input = tokens_input(list(range(200)))  # 200 > max_model_len
        error = serving._validate_generator_input(engine_input)
        assert isinstance(error, ErrorResponse)


# ---------------------------------------------------------------------------
# retrieve_responses
# ---------------------------------------------------------------------------


class TestRetrieveResponses:
    """Covers the retrieve_responses public method."""

    @pytest.mark.asyncio
    async def test_not_found_returns_error(self, serving_with_store):
        error = await serving_with_store.retrieve_responses(
            "resp_nonexistent", starting_after=None, stream=False
        )
        assert isinstance(error, ErrorResponse)
        assert error.error.code == 404

    @pytest.mark.asyncio
    async def test_found_returns_response(self, serving_with_store):
        # Pre-populate the store.
        stored = _make_minimal_response("resp_stored")
        serving_with_store.response_store["resp_stored"] = stored

        result = await serving_with_store.retrieve_responses(
            "resp_stored", starting_after=None, stream=False
        )
        assert isinstance(result, ResponsesResponse)
        assert result.id == "resp_stored"


# ---------------------------------------------------------------------------
# cancel_responses
# ---------------------------------------------------------------------------


class TestCancelResponses:
    """Covers the cancel_responses public method."""

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, serving_with_store):
        error = await serving_with_store.cancel_responses("resp_nope")
        assert isinstance(error, ErrorResponse)
        assert error.error.code == 404

    @pytest.mark.asyncio
    async def test_cancel_completed_fails(self, serving_with_store):
        stored = _make_minimal_response("resp_done", status="completed")
        serving_with_store.response_store["resp_done"] = stored

        error = await serving_with_store.cancel_responses("resp_done")
        assert isinstance(error, ErrorResponse)
        assert "synchronous" in error.error.message.lower() or (
            error.error.type == "invalid_request_error"
        )

    @pytest.mark.asyncio
    async def test_cancel_in_progress_succeeds(self, serving_with_store):
        stored = _make_minimal_response("resp_ip", status="in_progress")
        serving_with_store.response_store["resp_ip"] = stored
        # No background task registered → cancel() just updates status.

        result = await serving_with_store.cancel_responses("resp_ip")
        assert isinstance(result, ResponsesResponse)
        assert result.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_queued_succeeds(self, serving_with_store):
        stored = _make_minimal_response("resp_q", status="queued")
        serving_with_store.response_store["resp_q"] = stored

        result = await serving_with_store.cancel_responses("resp_q")
        assert isinstance(result, ResponsesResponse)
        assert result.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_cancels_background_task(self, serving_with_store):
        """When a background task is registered, cancelling it is attempted."""
        stored = _make_minimal_response("resp_bg", status="in_progress")
        serving_with_store.response_store["resp_bg"] = stored

        # Register a fake (already-done) task so the cancel path is exercised.
        async def _noop():
            pass

        task = asyncio.create_task(_noop())
        await task  # let it finish so cancel() won't block
        serving_with_store.background_tasks["resp_bg"] = task

        result = await serving_with_store.cancel_responses("resp_bg")
        assert isinstance(result, ResponsesResponse)
        assert result.status == "cancelled"


# ---------------------------------------------------------------------------
# responses_full_generator – status flows
# ---------------------------------------------------------------------------


def _make_minimal_response(
    resp_id: str = "resp_test",
    status: str = "completed",
) -> ResponsesResponse:
    """Return a minimal valid ResponsesResponse."""
    request = ResponsesRequest(model="test-model", input="hi")
    request.request_id = resp_id
    sampling_params = request.to_sampling_params(default_max_tokens=16)
    return ResponsesResponse.from_request(
        request=request,
        sampling_params=sampling_params,
        model_name="test-model",
        created_time=0,
        output=[],
        status=status,  # type: ignore[arg-type]
        usage=None,
    )


def _make_simple_context_with_finish(text: str, finish_reason: str) -> SimpleContext:
    """Return a SimpleContext whose final output has the given finish_reason."""
    ctx = SimpleContext(response_parser=None)
    completion = CompletionOutput(
        index=0,
        text=text,
        token_ids=[1, 2, 3],
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason=finish_reason,
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
    ctx.append_output(req_output)
    return ctx


class TestFullGeneratorStatusFlows:
    """Test that responses_full_generator produces the correct status."""

    @pytest.mark.asyncio
    async def test_completed_status(self, serving):
        ctx = _make_simple_context_with_finish("Hello", "stop")

        async def gen():
            yield None

        request = ResponsesRequest(input="hi", stream=False)
        sampling_params = SamplingParams(max_tokens=16)
        metadata = RequestResponseMetadata(request_id="req")

        response = await serving.responses_full_generator(
            request=request,
            sampling_params=sampling_params,
            result_generator=gen(),
            context=ctx,
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
        )
        assert isinstance(response, ResponsesResponse)
        assert response.status == "completed"

    @pytest.mark.asyncio
    async def test_incomplete_status_on_length(self, serving):
        ctx = _make_simple_context_with_finish("Partial answer", "length")

        async def gen():
            yield None

        request = ResponsesRequest(input="hi", stream=False)
        sampling_params = SamplingParams(max_tokens=16)
        metadata = RequestResponseMetadata(request_id="req")

        response = await serving.responses_full_generator(
            request=request,
            sampling_params=sampling_params,
            result_generator=gen(),
            context=ctx,
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
        )
        assert isinstance(response, ResponsesResponse)
        assert response.status == "incomplete"
        assert response.incomplete_details is not None
        assert response.incomplete_details.reason == "max_output_tokens"

    @pytest.mark.asyncio
    async def test_store_saves_response(self, serving_with_store):
        """When store=True and enable_store=True, the response is saved."""
        ctx = _make_simple_context_with_finish("Hello", "stop")
        request = ResponsesRequest(input="hi", stream=False, store=True)

        async def gen():
            yield None

        sampling_params = SamplingParams(max_tokens=16)
        metadata = RequestResponseMetadata(request_id=request.request_id)

        response = await serving_with_store.responses_full_generator(
            request=request,
            sampling_params=sampling_params,
            result_generator=gen(),
            context=ctx,
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
        )
        assert isinstance(response, ResponsesResponse)
        assert response.id in serving_with_store.response_store

    @pytest.mark.asyncio
    async def test_store_disabled_does_not_save(self, serving):
        """When enable_store=False, store is silently ignored."""
        ctx = _make_simple_context_with_finish("Hello", "stop")
        request = ResponsesRequest(input="hi", stream=False, store=True)

        async def gen():
            yield None

        sampling_params = SamplingParams(max_tokens=16)
        metadata = RequestResponseMetadata(request_id=request.request_id)

        response = await serving.responses_full_generator(
            request=request,
            sampling_params=sampling_params,
            result_generator=gen(),
            context=ctx,
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
        )
        assert isinstance(response, ResponsesResponse)
        # Store should be empty because enable_store=False
        assert response.id not in serving.response_store


# ---------------------------------------------------------------------------
# Usage per-turn accounting
# ---------------------------------------------------------------------------


class TestUsageAccounting:
    """Verify that per-turn usage arrays are populated correctly."""

    @pytest.mark.asyncio
    async def test_single_turn_usage_arrays(self, serving):
        ctx = _make_simple_context_with_finish("Hello", "stop")
        request = ResponsesRequest(input="hi", stream=False)
        sampling_params = SamplingParams(max_tokens=16)
        metadata = RequestResponseMetadata(request_id="req")

        response = await serving.responses_full_generator(
            request=request,
            sampling_params=sampling_params,
            result_generator=_trivial_gen(),
            context=ctx,
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
        )
        assert isinstance(response, ResponsesResponse)
        usage = response.usage
        assert usage is not None
        # There should be exactly one entry per turn in the per-turn arrays.
        assert len(usage.input_tokens_details.input_tokens_per_turn) == 1
        assert len(usage.input_tokens_details.cached_tokens_per_turn) == 1
        assert len(usage.output_tokens_details.output_tokens_per_turn) == 1
        assert len(usage.output_tokens_details.tool_output_tokens_per_turn) == 1

    @pytest.mark.asyncio
    async def test_usage_totals_match_per_turn_sums(self, serving):
        ctx = _make_simple_context_with_finish("Hello", "stop")
        request = ResponsesRequest(input="hi", stream=False)
        sampling_params = SamplingParams(max_tokens=16)
        metadata = RequestResponseMetadata(request_id="req")

        response = await serving.responses_full_generator(
            request=request,
            sampling_params=sampling_params,
            result_generator=_trivial_gen(),
            context=ctx,
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
        )
        assert isinstance(response, ResponsesResponse)
        usage = response.usage
        assert usage is not None
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens
        assert usage.input_tokens == sum(
            usage.input_tokens_details.input_tokens_per_turn
        )
        assert usage.output_tokens == sum(
            usage.output_tokens_details.output_tokens_per_turn
        )


async def _trivial_gen():
    yield None


# ---------------------------------------------------------------------------
# ResponsesResponse.from_request – field propagation
# ---------------------------------------------------------------------------


class TestResponsesResponseFromRequest:
    """Verify key fields are propagated from the request to the response."""

    def test_instructions_propagated(self):
        request = ResponsesRequest(
            input="hi",
            instructions="Be concise.",
            model="test-model",
        )
        sp = request.to_sampling_params(default_max_tokens=16)
        resp = ResponsesResponse.from_request(
            request=request,
            sampling_params=sp,
            model_name="test-model",
            created_time=0,
            output=[],
            status="completed",
        )
        assert resp.instructions == "Be concise."

    def test_incomplete_details_set_on_incomplete(self):
        request = ResponsesRequest(input="hi", model="test-model")
        sp = request.to_sampling_params(default_max_tokens=16)
        resp = ResponsesResponse.from_request(
            request=request,
            sampling_params=sp,
            model_name="test-model",
            created_time=0,
            output=[],
            status="incomplete",
        )
        assert resp.incomplete_details is not None
        assert resp.incomplete_details.reason == "max_output_tokens"

    def test_incomplete_details_none_on_completed(self):
        request = ResponsesRequest(input="hi", model="test-model")
        sp = request.to_sampling_params(default_max_tokens=16)
        resp = ResponsesResponse.from_request(
            request=request,
            sampling_params=sp,
            model_name="test-model",
            created_time=0,
            output=[],
            status="completed",
        )
        assert resp.incomplete_details is None

    def test_previous_response_id_propagated(self):
        # Manually set previous_response_id after construction (skip pydantic
        # mutual exclusion validation since we aren't setting prev_messages).
        request = ResponsesRequest(input="hi", model="test-model")
        request.previous_response_id = "resp_prev"
        sp = request.to_sampling_params(default_max_tokens=16)
        resp = ResponsesResponse.from_request(
            request=request,
            sampling_params=sp,
            model_name="test-model",
            created_time=0,
            output=[],
            status="completed",
        )
        assert resp.previous_response_id == "resp_prev"

    @pytest.mark.parametrize(
        "status",
        ["completed", "incomplete", "cancelled", "queued", "in_progress", "failed"],
    )
    def test_valid_statuses_accepted(self, status):
        request = ResponsesRequest(input="hi", model="test-model")
        sp = request.to_sampling_params(default_max_tokens=16)
        resp = ResponsesResponse.from_request(
            request=request,
            sampling_params=sp,
            model_name="test-model",
            created_time=0,
            output=[],
            status=status,  # type: ignore[arg-type]
        )
        assert resp.status == status


# ---------------------------------------------------------------------------
# Streaming: sequence numbers are monotonically increasing
# ---------------------------------------------------------------------------


def _identity_counter(event):
    """Assign monotonically increasing sequence numbers for testing."""
    seq = getattr(_identity_counter, "_counter", 0)
    if hasattr(event, "sequence_number"):
        event.sequence_number = seq
    _identity_counter._counter = seq + 1  # type: ignore[attr-defined]
    return event


class TestStreamingSequenceNumbers:
    """Streaming responses must emit events with strictly increasing seq nums."""

    @pytest.mark.asyncio
    async def test_sequence_numbers_monotone(self, serving, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        ctx = _make_simple_context_with_finish("Hello world", "stop")

        contexts = [ctx]

        async def result_generator():
            for c in contexts:
                yield c

        request = ResponsesRequest(input="hi", tools=[], stream=True)
        sampling_params = SamplingParams(max_tokens=64)
        metadata = RequestResponseMetadata(request_id="req")
        _identity_counter._counter = 0  # type: ignore[attr-defined]

        events = []
        async for event in serving._process_simple_streaming_events(
            request=request,
            sampling_params=sampling_params,
            result_generator=result_generator(),
            context=SimpleContext(response_parser=None),
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
            created_time=0,
            _increment_sequence_number_and_return=_identity_counter,
        ):
            events.append(event)

        # There must be at least created, in_progress, completed events.
        assert len(events) >= 3

        sequence_numbers = [
            e.sequence_number
            for e in events
            if hasattr(e, "sequence_number")
        ]
        # All sequence numbers are present.
        assert len(sequence_numbers) == len(events)
        # They are strictly increasing.
        for i in range(1, len(sequence_numbers)):
            assert sequence_numbers[i] > sequence_numbers[i - 1], (
                f"sequence numbers not increasing: {sequence_numbers}"
            )

    @pytest.mark.asyncio
    async def test_created_event_is_first(self, serving, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        ctx = _make_simple_context_with_finish("Hi", "stop")

        async def result_generator():
            yield ctx

        request = ResponsesRequest(input="hi", tools=[], stream=True)
        sampling_params = SamplingParams(max_tokens=64)
        metadata = RequestResponseMetadata(request_id="req")
        _identity_counter._counter = 0  # type: ignore[attr-defined]

        events = []
        async for event in serving._process_simple_streaming_events(
            request=request,
            sampling_params=sampling_params,
            result_generator=result_generator(),
            context=SimpleContext(response_parser=None),
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
            created_time=0,
            _increment_sequence_number_and_return=_identity_counter,
        ):
            events.append(event)

        assert isinstance(events[0], ResponseCreatedEvent)

    @pytest.mark.asyncio
    async def test_completed_event_is_last(self, serving, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", False)
        ctx = _make_simple_context_with_finish("Hi", "stop")

        async def result_generator():
            yield ctx

        request = ResponsesRequest(input="hi", tools=[], stream=True)
        sampling_params = SamplingParams(max_tokens=64)
        metadata = RequestResponseMetadata(request_id="req")
        _identity_counter._counter = 0  # type: ignore[attr-defined]

        events = []
        async for event in serving._process_simple_streaming_events(
            request=request,
            sampling_params=sampling_params,
            result_generator=result_generator(),
            context=SimpleContext(response_parser=None),
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=metadata,
            created_time=0,
            _increment_sequence_number_and_return=_identity_counter,
        ):
            events.append(event)

        assert isinstance(events[-1], ResponseCompletedEvent)


# ---------------------------------------------------------------------------
# _run_background_request: failure path updates store status to "failed"
# ---------------------------------------------------------------------------


class TestBackgroundRequestFailurePath:
    @pytest.mark.asyncio
    async def test_failed_response_updates_status(self, serving_with_store):
        """When _run_background_request receives an ErrorResponse, it must
        mark the stored response as 'failed'."""
        # Pre-populate the store with an in-progress response.
        request = ResponsesRequest(
            input="hi", model="test-model", store=True, background=True
        )
        stored = _make_minimal_response(request.request_id, status="in_progress")
        serving_with_store.response_store[request.request_id] = stored

        # Make responses_full_generator return an ErrorResponse.
        serving_with_store.responses_full_generator = AsyncMock(
            return_value=ErrorResponse(
                error=ErrorInfo(
                    message="boom",
                    type="server_error",
                    code=500,
                )
            )
        )

        await serving_with_store._run_background_request(
            request,
            MagicMock(),  # sampling_params
            MagicMock(),  # result_generator
            MagicMock(),  # context
            "test-model",
            MagicMock(),  # tokenizer
            MagicMock(),  # request_metadata
        )

        stored_after = serving_with_store.response_store.get(request.request_id)
        assert stored_after is not None
        assert stored_after.status == "failed"

    @pytest.mark.asyncio
    async def test_cancelled_response_not_overwritten_by_failed(
        self, serving_with_store
    ):
        """If the response is already 'cancelled', _run_background_request
        must not overwrite it with 'failed'."""
        request = ResponsesRequest(
            input="hi", model="test-model", store=True, background=True
        )
        stored = _make_minimal_response(request.request_id, status="cancelled")
        serving_with_store.response_store[request.request_id] = stored

        serving_with_store.responses_full_generator = AsyncMock(
            return_value=ErrorResponse(
                error=ErrorInfo(
                    message="too slow",
                    type="server_error",
                    code=500,
                )
            )
        )

        await serving_with_store._run_background_request(
            request,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            "test-model",
            MagicMock(),
            MagicMock(),
        )

        stored_after = serving_with_store.response_store.get(request.request_id)
        assert stored_after is not None
        assert stored_after.status == "cancelled"


# ---------------------------------------------------------------------------
# Tool-session initialization
# ---------------------------------------------------------------------------


class TestToolSessionInit:
    """init_tool_sessions should be called iff tools are present in request."""

    class _MockContext:
        """Minimal mock that satisfies _initialize_tool_sessions."""

        response_parser = None

        def __init__(self):
            self.init_called = False
            self.init_args = None

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

        async def init_tool_sessions(self, tool_server, exit_stack, request_id, mcp):
            self.init_called = True
            self.init_args = (tool_server, exit_stack, request_id, mcp)

        async def cleanup_session(self) -> None:
            pass

    @pytest.mark.asyncio
    async def test_no_tools_no_init(self, serving):
        request = ResponsesRequest(input="hi", tools=[])
        ctx = self._MockContext()
        from contextlib import AsyncExitStack

        await serving._initialize_tool_sessions(
            request, ctx, AsyncExitStack()  # type: ignore[arg-type]
        )
        assert not ctx.init_called

    @pytest.mark.asyncio
    async def test_builtin_tool_triggers_init(self, serving):
        request = ResponsesRequest(
            input="hi",
            tools=[{"type": "web_search_preview"}],
        )
        ctx = self._MockContext()
        from contextlib import AsyncExitStack

        await serving._initialize_tool_sessions(
            request, ctx, AsyncExitStack()  # type: ignore[arg-type]
        )
        # init is called when at least one MCP-type tool is present (mcp tools
        # get collected in _initialize_tool_sessions; web_search_preview is a
        # builtin, not mcp, so mcp_tools will be empty, but init_tool_sessions
        # is still called because tools is non-empty).
        # The actual semantics: init_tool_sessions is called when len(tools) > 0.
        assert ctx.init_called


# ---------------------------------------------------------------------------
# _make_not_found_error
# ---------------------------------------------------------------------------


def test_make_not_found_error():
    serving = _make_serving()
    error = serving._make_not_found_error("resp_xyz")
    assert isinstance(error, ErrorResponse)
    assert error.error.code == 404
    assert "resp_xyz" in error.error.message


# ---------------------------------------------------------------------------
# ResponseUsage construction
# ---------------------------------------------------------------------------


def test_response_usage_fields():
    usage = ResponseUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=InputTokensDetails(
            cached_tokens=2,
            input_tokens_per_turn=[6, 4],
            cached_tokens_per_turn=[2, 0],
        ),
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=1,
            tool_output_tokens=0,
            output_tokens_per_turn=[3, 2],
            tool_output_tokens_per_turn=[0, 0],
        ),
    )
    assert usage.total_tokens == 15
    assert usage.input_tokens_details.cached_tokens == 2
    assert sum(usage.input_tokens_details.input_tokens_per_turn) == 10
    assert sum(usage.output_tokens_details.output_tokens_per_turn) == 5
    assert usage.output_tokens_details.reasoning_tokens == 1
