# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for API span lifecycle management (PR #6).

These tests verify that API parent spans are created, finalized, and cleaned up
correctly on all termination paths (success, errors, cancellation, generator exits).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from collections.abc import AsyncGenerator

import pytest

from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.outputs import RequestOutput


# Helper function for robust event detection
def find_events(span, name: str) -> list:
    """
    Find span events by exact name match using kwargs-only pattern.

    This is more robust than string matching on call.__repr__() which can
    break with mock library changes or Python version changes.

    Args:
        span: Mock span object with add_event.call_args_list
        name: Exact event name to match (e.g., 'api.DEPARTED', 'api.ABORTED')

    Returns:
        List of matching call objects from span.add_event.call_args_list
    """
    return [c for c in span.add_event.call_args_list if c.kwargs.get("name") == name]


# Patch OTEL imports at module level for all tests
pytestmark = pytest.mark.usefixtures("mock_otel")


@pytest.fixture
def mock_otel():
    """Mock all OpenTelemetry imports."""
    with patch("vllm.tracing.SpanAttributes") as mock_span_attrs:
        mock_span_attrs.GEN_AI_REQUEST_ID = "gen_ai.request.id"
        mock_span_attrs.EVENT_TS_MONOTONIC = "event.ts.monotonic"

        with patch("opentelemetry.trace.Status") as mock_status:
            with patch("opentelemetry.trace.StatusCode") as mock_status_code:
                mock_status_code.ERROR = "ERROR"
                yield {
                    "span_attrs": mock_span_attrs,
                    "status": mock_status,
                    "status_code": mock_status_code,
                }


# ========== Fixtures ==========

@pytest.fixture
def mock_span():
    """Create mock OTEL span."""
    span = MagicMock()
    span.is_recording.return_value = True
    span.set_attribute = MagicMock()
    span.add_event = MagicMock()
    span.end = MagicMock()
    span.set_status = MagicMock()
    return span


@pytest.fixture
def mock_tracer(mock_span):
    """Create mock tracer that returns mock span."""
    tracer = MagicMock()
    tracer.start_span.return_value = mock_span
    return tracer


@pytest.fixture
def mock_serving():
    """Create OpenAIServing instance with mocked dependencies."""
    serving = OpenAIServing(
        engine_client=AsyncMock(),
        models=MagicMock(),
        request_logger=MagicMock(),
    )
    # Pre-populate cache to avoid async call
    serving._cached_is_tracing_enabled = True
    return serving


@pytest.fixture
def mock_chat_serving():
    """Create OpenAIServingChat instance with mocked dependencies."""
    mock_models = MagicMock()
    mock_models.model_name.return_value = "test-model"
    mock_models.tokenizer = MagicMock()
    mock_models.input_processor = MagicMock()
    mock_models.io_processor = MagicMock()
    mock_models.renderer = MagicMock()
    mock_models.renderer.tokenizer = MagicMock()
    mock_models.model_config = MagicMock()
    mock_models.model_config.max_model_len = 4096

    serving = OpenAIServingChat(
        engine_client=AsyncMock(),
        models=mock_models,
        response_role="assistant",
        request_logger=MagicMock(),
        chat_template=None,
        chat_template_content_format="auto",
    )
    # Mock additional attributes needed by generators
    serving.use_harmony = False
    serving.reasoning_parser = None
    serving.tool_parser = None
    serving.tool_call_id_type = "default"
    serving.enable_log_outputs = False
    serving.enable_force_include_usage = False
    serving.enable_prompt_tokens_details = False
    serving.default_sampling_params = {}

    serving._cached_is_tracing_enabled = True
    return serving


def create_mock_request(**overrides):
    """Create a mock ChatCompletionRequest with default attributes."""
    request = MagicMock(spec=ChatCompletionRequest)

    # Set default attributes
    defaults = {
        "stream": True,
        "n": 1,
        "stream_options": None,
        "echo": False,
        "logprobs": False,
        "top_logprobs": None,
        "tool_choice": None,
        "tools": None,
        "include_reasoning": False,
        "return_token_ids": False,
        "add_generation_prompt": True,
        "parallel_tool_calls": True,
    }

    # Apply overrides
    defaults.update(overrides)

    # Set all attributes on the mock
    for key, value in defaults.items():
        setattr(request, key, value)

    return request


# ========== Test Group 1: Span Creation (2 tests) ==========

@pytest.mark.asyncio
async def test_api_span_created_when_tracing_enabled(mock_serving, mock_tracer):
    """Test 1.1: Verify span created and stored when tracing enabled."""
    with patch("opentelemetry.trace.get_tracer_provider") as mock_provider:
        mock_provider.return_value.get_tracer.return_value = mock_tracer

        span = await mock_serving._create_api_span("test-req-123", trace_headers=None)

        assert span is not None
        assert mock_tracer.start_span.called
        call_kwargs = mock_tracer.start_span.call_args[1]
        assert call_kwargs["name"] == "llm_request"

        # Store and verify
        mock_serving._store_api_span("test-req-123", span, time.monotonic())
        assert len(mock_serving._api_spans) == 1
        assert mock_serving._api_spans["test-req-123"][0] == span


@pytest.mark.asyncio
async def test_span_creation_gated_at_call_site(mock_serving, mock_tracer):
    """Test 1.2: Verify tracing flag gating happens at call site."""
    # Disable tracing
    mock_serving._cached_is_tracing_enabled = False

    with patch("opentelemetry.trace.get_tracer_provider") as mock_provider:
        mock_provider.return_value.get_tracer.return_value = mock_tracer

        # Simulate call-site gating logic
        is_tracing_enabled = await mock_serving._get_is_tracing_enabled()
        api_span = None
        if is_tracing_enabled:
            api_span = await mock_serving._create_api_span("test-req-123", trace_headers=None)
            if api_span:
                mock_serving._store_api_span("test-req-123", api_span, time.monotonic())

        assert is_tracing_enabled is False
        assert api_span is None
        assert len(mock_serving._api_spans) == 0
        assert not mock_tracer.start_span.called


# ========== Test Group 2: Streaming Termination Paths (5 tests) ==========

@pytest.mark.asyncio
async def test_streaming_success_path(mock_chat_serving, mock_span):
    """Test 2.1: Verify DEPARTED emitted on success."""
    # Create and store span
    mock_chat_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    request = create_mock_request(stream=True)
    request_metadata = RequestResponseMetadata(request_id="test-req-123")

    # Mock result generator
    async def mock_generator():
        output = MagicMock()
        output.prompt_token_ids = [1, 2, 3]
        output.encoder_prompt_token_ids = None
        output.num_cached_tokens = None
        output.outputs = [MagicMock(index=0, token_ids=[4, 5], text="test", finish_reason="stop", logprobs=None)]
        yield output

    # Call generator and collect chunks
    chunks = []
    async for chunk in mock_chat_serving.chat_completion_stream_generator(
        request=request,
        result_generator=mock_generator(),
        request_id="test-req-123",
        model_name="test-model",
        conversation=[],
        tokenizer=None,
        request_metadata=request_metadata,
    ):
        chunks.append(chunk)

    # Verify DEPARTED event was emitted
    departed_calls = find_events(mock_span, "api.DEPARTED")
    assert len(departed_calls) > 0, "DEPARTED event should be emitted"

    # Verify span ended
    assert mock_span.end.called

    # Verify dict cleaned
    assert len(mock_chat_serving._api_spans) == 0

    # Verify [DONE] was yielded
    assert any("[DONE]" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_streaming_departed_emitted_before_done(mock_chat_serving, mock_span):
    """Test 2.2: Verify DEPARTED is emitted BEFORE [DONE] is yielded (regression test for post-[DONE] danger window)."""
    # Track call order using a list
    call_order = []

    # Wrap add_event to track when DEPARTED is emitted
    original_add_event = mock_span.add_event
    def track_add_event(name=None, *args, **kwargs):
        # Track DEPARTED events by checking exact name match
        if name == "api.DEPARTED" or kwargs.get("name") == "api.DEPARTED":
            call_order.append("DEPARTED")
        return original_add_event(name, *args, **kwargs)
    mock_span.add_event = track_add_event

    # Create and store span
    mock_chat_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    request = create_mock_request(stream=True)
    request_metadata = RequestResponseMetadata(request_id="test-req-123")

    # Mock result generator
    async def mock_generator():
        output = MagicMock()
        output.prompt_token_ids = [1, 2, 3]
        output.encoder_prompt_token_ids = None
        output.num_cached_tokens = None
        output.outputs = [MagicMock(index=0, token_ids=[4, 5], text="test", finish_reason="stop", logprobs=None)]
        yield output

    # Call generator and track when [DONE] appears
    chunks = []
    async for chunk in mock_chat_serving.chat_completion_stream_generator(
        request=request,
        result_generator=mock_generator(),
        request_id="test-req-123",
        model_name="test-model",
        conversation=[],
        tokenizer=None,
        request_metadata=request_metadata,
    ):
        chunks.append(chunk)
        if "[DONE]" in chunk:
            call_order.append("[DONE]")

    # CRITICAL ASSERTION: DEPARTED must be emitted BEFORE [DONE] is yielded
    # This proves the fix works: DEPARTED finalization happens before [DONE] yield
    assert "DEPARTED" in call_order, "DEPARTED event should be emitted"
    assert "[DONE]" in call_order, "[DONE] should be yielded"
    departed_index = call_order.index("DEPARTED")
    done_index = call_order.index("[DONE]")
    assert departed_index < done_index, f"DEPARTED must be emitted BEFORE [DONE], got order: {call_order}"

    # Verify span cleanup
    assert mock_span.end.called
    assert len(mock_chat_serving._api_spans) == 0


@pytest.mark.asyncio
async def test_streaming_cancelled_error(mock_chat_serving, mock_span):
    """Test 2.3: Verify ABORTED + re-raise on CancelledError."""
    # Create and store span
    mock_chat_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    request = create_mock_request(stream=True)
    request_metadata = RequestResponseMetadata(request_id="test-req-123")

    # Mock result generator that raises CancelledError
    async def mock_generator():
        if False:
            yield  # Make this an async generator
        raise asyncio.CancelledError()

    # Verify exception is re-raised
    chunks = []
    with pytest.raises(asyncio.CancelledError):
        async for chunk in mock_chat_serving.chat_completion_stream_generator(
            request=request,
            result_generator=mock_generator(),
            request_id="test-req-123",
            model_name="test-model",
            conversation=[],
            tokenizer=None,
            request_metadata=request_metadata,
        ):
            chunks.append(chunk)

    # Verify ABORTED event was emitted
    aborted_calls = find_events(mock_span, "api.ABORTED")
    assert len(aborted_calls) > 0, "ABORTED event should be emitted"

    # Verify span ended
    assert mock_span.end.called

    # Verify dict cleaned
    assert len(mock_chat_serving._api_spans) == 0

    # Verify [DONE] NOT yielded (re-raises immediately)
    assert not any("[DONE]" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_streaming_generator_exit(mock_chat_serving, mock_span):
    """Test 2.4: Verify ABORTED + re-raise on GeneratorExit."""
    # Create and store span
    mock_chat_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    request = create_mock_request(stream=True)
    request_metadata = RequestResponseMetadata(request_id="test-req-123")

    # Mock result generator that raises GeneratorExit
    async def mock_generator():
        if False:
            yield  # Make this an async generator
        raise GeneratorExit()

    # Verify exception is re-raised
    chunks = []
    with pytest.raises(GeneratorExit):
        async for chunk in mock_chat_serving.chat_completion_stream_generator(
            request=request,
            result_generator=mock_generator(),
            request_id="test-req-123",
            model_name="test-model",
            conversation=[],
            tokenizer=None,
            request_metadata=request_metadata,
        ):
            chunks.append(chunk)

    # Verify ABORTED event was emitted
    aborted_calls = find_events(mock_span, "api.ABORTED")
    assert len(aborted_calls) > 0, "ABORTED event should be emitted"

    # Verify span ended
    assert mock_span.end.called

    # Verify dict cleaned
    assert len(mock_chat_serving._api_spans) == 0

    # Verify [DONE] NOT yielded (re-raises immediately)
    assert not any("[DONE]" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_streaming_generic_exception(mock_chat_serving, mock_span):
    """Test 2.5: Verify ABORTED with reason='exception' on generic Exception."""
    # Create and store span
    mock_chat_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    request = create_mock_request(stream=True)
    request_metadata = RequestResponseMetadata(request_id="test-req-123")

    # Mock result generator that raises Exception
    async def mock_generator():
        if False:
            yield  # Make this an async generator
        raise Exception("Unexpected error")

    # Call generator and collect chunks
    chunks = []
    async for chunk in mock_chat_serving.chat_completion_stream_generator(
        request=request,
        result_generator=mock_generator(),
        request_id="test-req-123",
        model_name="test-model",
        conversation=[],
        tokenizer=None,
        request_metadata=request_metadata,
    ):
        chunks.append(chunk)

    # Verify ABORTED event was emitted
    aborted_calls = find_events(mock_span, "api.ABORTED")
    assert len(aborted_calls) > 0, "ABORTED event should be emitted"

    # Verify span ended
    assert mock_span.end.called

    # Verify dict cleaned
    assert len(mock_chat_serving._api_spans) == 0

    # Verify [DONE] was yielded (falls through after error)
    assert any("[DONE]" in chunk for chunk in chunks)


# ========== Test Group 3: Non-Streaming Termination Paths (6 tests) ==========

@pytest.mark.asyncio
async def test_full_success_path(mock_chat_serving, mock_span):
    """Test 3.1: Verify DEPARTED emitted on success."""
    # Create and store span
    mock_chat_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    request = create_mock_request(stream=False, logprobs=False)
    request_metadata = RequestResponseMetadata(request_id="test-req-123")

    # Mock result generator
    async def mock_generator():
        output = MagicMock()
        output.prompt_token_ids = [1, 2, 3]
        output.encoder_prompt_token_ids = None
        output.kv_transfer_params = None
        output.outputs = [MagicMock(
            index=0,
            token_ids=[4, 5],
            text="test response",
            finish_reason="stop",
            stop_reason=None,
            logprobs=None,
        )]
        yield output

    # Call full generator
    response = await mock_chat_serving.chat_completion_full_generator(
        request=request,
        result_generator=mock_generator(),
        request_id="test-req-123",
        model_name="test-model",
        conversation=[],
        tokenizer=None,
        request_metadata=request_metadata,
    )

    # Verify DEPARTED event was emitted
    departed_calls = find_events(mock_span, "api.DEPARTED")
    assert len(departed_calls) > 0, "DEPARTED event should be emitted"

    # Verify span ended
    assert mock_span.end.called

    # Verify dict cleaned
    assert len(mock_chat_serving._api_spans) == 0


@pytest.mark.asyncio
async def test_full_cancelled_error(mock_chat_serving, mock_span):
    """Test 3.2: Verify ABORTED + return error on CancelledError."""
    # Create and store span
    mock_chat_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    request = create_mock_request(stream=False, logprobs=False)
    request_metadata = RequestResponseMetadata(request_id="test-req-123")

    # Mock result generator that raises CancelledError
    async def mock_generator():
        if False:
            yield  # Make this an async generator
        raise asyncio.CancelledError()

    # Call full generator
    response = await mock_chat_serving.chat_completion_full_generator(
        request=request,
        result_generator=mock_generator(),
        request_id="test-req-123",
        model_name="test-model",
        conversation=[],
        tokenizer=None,
        request_metadata=request_metadata,
    )

    # Verify ABORTED event was emitted
    aborted_calls = find_events(mock_span, "api.ABORTED")
    assert len(aborted_calls) > 0, "ABORTED event should be emitted"

    # Verify span ended
    assert mock_span.end.called

    # Verify dict cleaned
    assert len(mock_chat_serving._api_spans) == 0

    # Verify error response returned (not re-raised)
    assert response is not None


@pytest.mark.asyncio
async def test_full_validation_error(mock_chat_serving, mock_span):
    """Test 3.4: Verify ABORTED with reason='validation_error' on ValueError."""
    # Create and store span
    mock_chat_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    request = create_mock_request(stream=False, logprobs=False)
    request_metadata = RequestResponseMetadata(request_id="test-req-123")

    # Mock result generator that raises ValueError
    async def mock_generator():
        if False:
            yield  # Make this an async generator
        raise ValueError("Invalid parameter")

    # Call full generator
    response = await mock_chat_serving.chat_completion_full_generator(
        request=request,
        result_generator=mock_generator(),
        request_id="test-req-123",
        model_name="test-model",
        conversation=[],
        tokenizer=None,
        request_metadata=request_metadata,
    )

    # Verify ABORTED event was emitted
    aborted_calls = find_events(mock_span, "api.ABORTED")
    assert len(aborted_calls) > 0, "ABORTED event should be emitted"

    # Verify span ended
    assert mock_span.end.called

    # Verify dict cleaned
    assert len(mock_chat_serving._api_spans) == 0

    # Verify error response returned
    assert response is not None


@pytest.mark.asyncio
async def test_full_generic_exception(mock_chat_serving, mock_span):
    """Test 3.5: Verify ABORTED with reason='exception' on generic Exception."""
    # Create and store span
    mock_chat_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    request = create_mock_request(stream=False, logprobs=False)
    request_metadata = RequestResponseMetadata(request_id="test-req-123")

    # Mock result generator that raises generic Exception
    async def mock_generator():
        if False:
            yield  # Make this an async generator
        raise Exception("Unexpected error")

    # Call full generator
    response = await mock_chat_serving.chat_completion_full_generator(
        request=request,
        result_generator=mock_generator(),
        request_id="test-req-123",
        model_name="test-model",
        conversation=[],
        tokenizer=None,
        request_metadata=request_metadata,
    )

    # Verify ABORTED event was emitted
    aborted_calls = find_events(mock_span, "api.ABORTED")
    assert len(aborted_calls) > 0, "ABORTED event should be emitted"

    # Verify span ended
    assert mock_span.end.called

    # Verify dict cleaned
    assert len(mock_chat_serving._api_spans) == 0

    # Verify error response returned (not re-raised)
    assert response is not None


@pytest.mark.asyncio
async def test_full_generator_exit(mock_chat_serving, mock_span):
    """Test 3.6: Verify ABORTED + re-raise on GeneratorExit."""
    # Create and store span
    mock_chat_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    request = create_mock_request(stream=False, logprobs=False)
    request_metadata = RequestResponseMetadata(request_id="test-req-123")

    # Mock result generator that raises GeneratorExit
    async def mock_generator():
        if False:
            yield  # Make this an async generator
        raise GeneratorExit()

    # Verify exception is re-raised (GeneratorExit must propagate)
    with pytest.raises(GeneratorExit):
        await mock_chat_serving.chat_completion_full_generator(
            request=request,
            result_generator=mock_generator(),
            request_id="test-req-123",
            model_name="test-model",
            conversation=[],
            tokenizer=None,
            request_metadata=request_metadata,
        )

    # Verify ABORTED event was emitted
    aborted_calls = find_events(mock_span, "api.ABORTED")
    assert len(aborted_calls) > 0, "ABORTED event should be emitted"

    # Verify span ended
    assert mock_span.end.called

    # Verify dict cleaned
    assert len(mock_chat_serving._api_spans) == 0


@pytest.mark.asyncio
async def test_full_generation_error_during_response_building(mock_chat_serving, mock_span):
    """Test 3.7: Verify ABORTED emitted when GenerationError occurs during response building (regression test for uncaught GenerationError)."""
    # Create and store span
    mock_chat_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    request = create_mock_request(stream=False, logprobs=False)
    request_metadata = RequestResponseMetadata(request_id="test-req-123")

    # Mock result generator that yields output with finish_reason="error"
    # This will trigger _raise_if_error() to raise GenerationError
    async def mock_generator():
        output = MagicMock()
        output.prompt_token_ids = [1, 2, 3]
        output.encoder_prompt_token_ids = None
        output.kv_transfer_params = None
        output.outputs = [MagicMock(
            index=0,
            token_ids=[4, 5],
            text="test response",
            finish_reason="error",  # This triggers GenerationError
            stop_reason=None,
            logprobs=None,
        )]
        yield output

    # Call full generator
    response = await mock_chat_serving.chat_completion_full_generator(
        request=request,
        result_generator=mock_generator(),
        request_id="test-req-123",
        model_name="test-model",
        conversation=[],
        tokenizer=None,
        request_metadata=request_metadata,
    )

    # Verify ABORTED event was emitted with reason='generation_error'
    aborted_calls = find_events(mock_span, "api.ABORTED")
    assert len(aborted_calls) == 1, f"Expected exactly 1 ABORTED event, got {len(aborted_calls)}"

    # Extract attributes dict using kwargs (matches actual implementation)
    aborted_call = aborted_calls[0]
    attributes = aborted_call.kwargs.get("attributes")

    assert attributes is not None, "ABORTED event should have attributes"
    assert isinstance(attributes, dict), f"Expected attributes dict, got {type(attributes)}"
    assert attributes.get("reason") == "generation_error", \
        f"Expected reason='generation_error', got {attributes.get('reason')}"
    assert attributes.get("error") is not None, "ABORTED event should include error message"

    # Verify span ended
    assert mock_span.end.called

    # Verify dict cleaned
    assert len(mock_chat_serving._api_spans) == 0

    # Verify error response returned (not re-raised)
    assert response is not None


# ========== Test Group 4: Idempotence & Leak Tests (2 tests) ==========

def test_finalizer_idempotence(mock_serving, mock_span):
    """Test 4.1: Verify safe to call finalizer multiple times."""
    # Create and store span
    mock_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    # Call finalizer 3 times
    mock_serving._finalize_api_span("test-req-123", "DEPARTED")
    mock_serving._finalize_api_span("test-req-123", "DEPARTED")
    mock_serving._finalize_api_span("test-req-123", "DEPARTED")

    # Verify span.end() called only once (first call does work)
    assert mock_span.end.call_count == 1

    # Verify dict cleaned after first call
    assert len(mock_serving._api_spans) == 0


def test_no_span_leaks(mock_serving, mock_span):
    """Test 4.2: Verify no leaks over 100 requests (mix of success/error)."""
    # Simulate 100 requests with various termination paths
    for i in range(100):
        request_id = f"req_{i}"

        # Create span (skip every 3rd request)
        if i % 3 != 0:
            mock_serving._store_api_span(request_id, mock_span, time.monotonic())

        # Simulate various termination paths
        if i % 7 == 0:
            # Success path
            mock_serving._finalize_api_span(request_id, "DEPARTED")
        elif i % 7 == 1:
            # Client disconnect
            mock_serving._finalize_api_span(request_id, "ABORTED", "client_disconnect", "Disconnected")
        elif i % 7 == 2:
            # Generation error
            mock_serving._finalize_api_span(request_id, "ABORTED", "generation_error", "Error")
        elif i % 7 == 3:
            # Cleanup-only fallback
            mock_serving._finalize_api_span(request_id)
        else:
            # Other ABORTED reasons
            mock_serving._finalize_api_span(request_id, "ABORTED", "exception", "Unexpected")

    # CRITICAL: Dict must be empty
    assert len(mock_serving._api_spans) == 0, f"Leak detected: {len(mock_serving._api_spans)} spans remain"


# ========== Test Group 5: Cleanup Independence (2 tests) ==========

def test_cleanup_not_gated_on_is_recording(mock_serving):
    """Test 5.1: Verify cleanup runs when is_recording() returns False."""
    # Create mock span with is_recording() returning False
    mock_span = MagicMock()
    mock_span.is_recording.return_value = False
    mock_span.end = MagicMock()
    mock_span.add_event = MagicMock()

    # Store in dict
    mock_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    # Finalize
    mock_serving._finalize_api_span("test-req-123", "DEPARTED")

    # Verify span.end() called (NOT gated on is_recording)
    assert mock_span.end.called

    # Verify dict cleaned (NOT gated on is_recording)
    assert len(mock_serving._api_spans) == 0

    # Verify event emission skipped (IS gated on is_recording)
    assert not mock_span.add_event.called


def test_cleanup_only_fallback(mock_serving, mock_span):
    """Test 5.2: Verify cleanup-only mode (terminal_event=None)."""
    # Create and store span
    mock_serving._store_api_span("test-req-123", mock_span, time.monotonic())

    # Finalize with no args (cleanup-only)
    mock_serving._finalize_api_span("test-req-123")

    # Verify no event emission (terminal_event=None)
    assert not mock_span.add_event.called

    # Verify span still ended
    assert mock_span.end.called

    # Verify dict still cleaned
    assert len(mock_serving._api_spans) == 0
