# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for API additional events (PR #8).

This test file verifies:
- EVENT_TS_MONOTONIC attribute exists (G1)
- FIRST_RESPONSE_FROM_CORE event emission once per request (G3)
- First response time persisted in _api_spans tuple (G4)
- Request attributes set on API span (G5)
- Early exit when tracing disabled (G6)
- Defensive programming guarantees (G7)

Note: HANDOFF_TO_CORE emission (G2) is verified by code inspection only,
not by automated tests in this file.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.sampling_params import SamplingParams


# ========== Helpers ==========

def find_events(span, name: str) -> list:
    """Find span events by exact name match using kwargs-only pattern.

    This is more robust than string matching on call.__repr__() which can
    break with mock library changes or Python version changes.

    Args:
        span: Mock span object with add_event.call_args_list
        name: Exact event name to match (e.g., 'api.HANDOFF_TO_CORE')

    Returns:
        List of matching call objects from span.add_event.call_args_list
    """
    return [c for c in span.add_event.call_args_list if c.kwargs.get("name") == name]


def create_mock_request(**overrides):
    """Create a mock ChatCompletionRequest with default attributes."""
    request = MagicMock(spec=ChatCompletionRequest)

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
        "return_tokens_as_token_ids": False,
        "add_generation_prompt": True,
        "parallel_tool_calls": True,
    }

    defaults.update(overrides)

    for key, value in defaults.items():
        setattr(request, key, value)

    return request


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
def serving_instance():
    """Create OpenAIServing instance with mocked dependencies."""
    mock_engine = MagicMock()
    mock_engine.is_tracing_enabled = AsyncMock(return_value=False)

    mock_models = MagicMock(spec=OpenAIServingModels)
    mock_models.input_processor = MagicMock()
    mock_models.io_processor = MagicMock()
    mock_models.renderer = MagicMock()
    mock_models.model_config = MagicMock()
    mock_models.model_config.max_model_len = 4096

    serving = OpenAIServing(
        engine_client=mock_engine,
        models=mock_models,
        request_logger=None,
    )
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


# ========== Tests ==========

# P1: test_event_ts_monotonic_attribute_exists
def test_event_ts_monotonic_attribute_exists():
    """Verify EVENT_TS_MONOTONIC attribute defined in SpanAttributes (G1)."""
    from vllm.tracing import SpanAttributes

    assert hasattr(SpanAttributes, 'EVENT_TS_MONOTONIC')
    assert SpanAttributes.EVENT_TS_MONOTONIC == "event.ts.monotonic"


# P3: test_first_response_event_emitted_streaming
@pytest.mark.asyncio
async def test_first_response_event_emitted_streaming(mock_chat_serving, mock_span):
    """Verify FIRST_RESPONSE_FROM_CORE event emitted in streaming path (G3).

    This test drives the actual chat_completion_stream_generator() to verify
    that FIRST_RESPONSE_FROM_CORE event is emitted on first iteration.
    """
    from vllm.tracing import SpanAttributes

    # Store span
    request_id = "test-req-123"
    mock_chat_serving._store_api_span(request_id, mock_span, time.monotonic())

    request = create_mock_request(stream=True)
    request_metadata = RequestResponseMetadata(request_id=request_id)

    # Mock result generator
    async def mock_generator():
        output = MagicMock()
        output.prompt_token_ids = [1, 2, 3]
        output.encoder_prompt_token_ids = None
        output.num_cached_tokens = None
        output.outputs = [MagicMock(
            index=0,
            token_ids=[4, 5],
            text="test",
            finish_reason="stop",
            logprobs=None
        )]
        yield output

    # Call streaming generator
    chunks = []
    async for chunk in mock_chat_serving.chat_completion_stream_generator(
        request=request,
        result_generator=mock_generator(),
        request_id=request_id,
        model_name="test-model",
        conversation=[],
        tokenizer=None,
        request_metadata=request_metadata,
    ):
        chunks.append(chunk)

    # Verify FIRST_RESPONSE_FROM_CORE event was emitted
    first_response_calls = find_events(mock_span, "api.FIRST_RESPONSE_FROM_CORE")
    assert len(first_response_calls) == 1, "FIRST_RESPONSE_FROM_CORE event should be emitted exactly once"

    # Verify event has correct attributes
    first_response_call = first_response_calls[0]
    attributes = first_response_call.kwargs.get("attributes")
    assert attributes is not None
    assert SpanAttributes.EVENT_TS_MONOTONIC in attributes
    assert isinstance(attributes[SpanAttributes.EVENT_TS_MONOTONIC], float)


# P4: test_first_response_event_only_once_streaming
@pytest.mark.asyncio
async def test_first_response_event_only_once_streaming(mock_chat_serving, mock_span):
    """Verify FIRST_RESPONSE event emitted exactly once despite multiple chunks (G3).

    This test verifies idempotence by yielding multiple outputs and confirming
    FIRST_RESPONSE_FROM_CORE is only emitted once.
    """
    # Store span
    request_id = "test-req-456"
    mock_chat_serving._store_api_span(request_id, mock_span, time.monotonic())

    request = create_mock_request(stream=True)
    request_metadata = RequestResponseMetadata(request_id=request_id)

    # Mock result generator that yields multiple outputs
    async def mock_generator():
        for i in range(5):  # Yield 5 outputs
            output = MagicMock()
            output.prompt_token_ids = [1, 2, 3] if i == 0 else None
            output.encoder_prompt_token_ids = None
            output.num_cached_tokens = None
            output.outputs = [MagicMock(
                index=0,
                token_ids=[10 + i],
                text=f"word{i}",
                finish_reason="stop" if i == 4 else None,
                logprobs=None
            )]
            yield output

    # Call streaming generator and consume all chunks
    chunks = []
    async for chunk in mock_chat_serving.chat_completion_stream_generator(
        request=request,
        result_generator=mock_generator(),
        request_id=request_id,
        model_name="test-model",
        conversation=[],
        tokenizer=None,
        request_metadata=request_metadata,
    ):
        chunks.append(chunk)

    # Verify FIRST_RESPONSE_FROM_CORE emitted exactly once
    first_response_calls = find_events(mock_span, "api.FIRST_RESPONSE_FROM_CORE")
    assert len(first_response_calls) == 1, f"Expected exactly 1 FIRST_RESPONSE event, got {len(first_response_calls)}"


# P5: test_first_response_event_emitted_non_streaming
@pytest.mark.asyncio
async def test_first_response_event_emitted_non_streaming(mock_chat_serving, mock_span):
    """Verify FIRST_RESPONSE_FROM_CORE event emitted in non-streaming path (G3).

    This test drives the actual chat_completion_full_generator() to verify
    that FIRST_RESPONSE_FROM_CORE event is emitted.
    """
    from vllm.tracing import SpanAttributes

    # Store span
    request_id = "test-req-789"
    mock_chat_serving._store_api_span(request_id, mock_span, time.monotonic())

    request = create_mock_request(stream=False)
    request_metadata = RequestResponseMetadata(request_id=request_id)

    # Mock result generator
    async def mock_generator():
        output = MagicMock()
        output.prompt_token_ids = [1, 2, 3]
        output.encoder_prompt_token_ids = None
        output.num_cached_tokens = None
        output.kv_transfer_params = None
        output.outputs = [MagicMock(
            index=0,
            token_ids=[4, 5, 6, 7],
            text="complete response",
            finish_reason="stop",
            logprobs=None,
            stop_reason=None,
        )]
        yield output

    # Call non-streaming generator
    response = await mock_chat_serving.chat_completion_full_generator(
        request=request,
        result_generator=mock_generator(),
        request_id=request_id,
        model_name="test-model",
        conversation=[],
        tokenizer=None,
        request_metadata=request_metadata,
    )

    # Verify FIRST_RESPONSE_FROM_CORE event was emitted
    first_response_calls = find_events(mock_span, "api.FIRST_RESPONSE_FROM_CORE")
    assert len(first_response_calls) == 1, "FIRST_RESPONSE_FROM_CORE event should be emitted exactly once"

    # Verify event has correct attributes
    first_response_call = first_response_calls[0]
    attributes = first_response_call.kwargs.get("attributes")
    assert attributes is not None
    assert SpanAttributes.EVENT_TS_MONOTONIC in attributes
    assert isinstance(attributes[SpanAttributes.EVENT_TS_MONOTONIC], float)


# P6: test_first_response_time_persisted_in_tuple
def test_first_response_time_persisted_in_tuple(serving_instance):
    """Verify first response timestamp is stored in _api_spans tuple (G4)."""
    # Create mock span
    mock_span = MagicMock()
    request_id = "test-request-789"
    arrival_time = time.monotonic()
    first_response_time = time.monotonic()

    # Store span with initial None for first_response_time
    serving_instance._store_api_span(request_id, mock_span, arrival_time)

    # Verify initial state
    span, arr_time, first_resp_time = serving_instance._get_api_span_info(request_id)
    assert span is mock_span
    assert arr_time == arrival_time
    assert first_resp_time is None

    # Update first response time
    serving_instance._update_first_response_time(request_id, first_response_time)

    # Retrieve and verify tuple updated
    span, arr_time, first_resp_time = serving_instance._get_api_span_info(request_id)
    assert span is mock_span
    assert arr_time == arrival_time  # Unchanged
    assert first_resp_time == first_response_time  # Updated


# P7: test_request_attributes_set_on_span
def test_request_attributes_set_on_span():
    """Verify all request attributes set on span with non-None values (G5)."""
    from vllm.tracing import SpanAttributes

    # Create mock serving instance
    mock_models = MagicMock()
    mock_models.model_config = MagicMock()
    mock_models.model_config.max_model_len = 4096

    serving = OpenAIServingChat(
        engine_client=AsyncMock(),
        models=mock_models,
        response_role="assistant",
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    # Create mock span that tracks set_attribute calls
    mock_span = MagicMock()
    mock_span.is_recording = MagicMock(return_value=True)
    set_attribute_calls = {}

    def capture_set_attribute(key, value):
        set_attribute_calls[key] = value

    mock_span.set_attribute = capture_set_attribute

    # Create sampling params with specific values
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        n=2,
    )

    # Create test data
    model_name = "test-model"
    prompt_token_ids = [1, 2, 3, 4, 5]  # 5 tokens

    # Call helper directly
    serving._set_api_span_request_attributes(
        mock_span,
        model_name,
        prompt_token_ids,
        sampling_params,
    )

    # Verify all attributes were set
    assert SpanAttributes.GEN_AI_RESPONSE_MODEL in set_attribute_calls
    assert set_attribute_calls[SpanAttributes.GEN_AI_RESPONSE_MODEL] == model_name

    assert SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS in set_attribute_calls
    assert set_attribute_calls[SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] == 5

    assert SpanAttributes.GEN_AI_REQUEST_TEMPERATURE in set_attribute_calls
    assert set_attribute_calls[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7

    assert SpanAttributes.GEN_AI_REQUEST_TOP_P in set_attribute_calls
    assert set_attribute_calls[SpanAttributes.GEN_AI_REQUEST_TOP_P] == 0.9

    assert SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS in set_attribute_calls
    assert set_attribute_calls[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 100

    assert SpanAttributes.GEN_AI_REQUEST_N in set_attribute_calls
    assert set_attribute_calls[SpanAttributes.GEN_AI_REQUEST_N] == 2


# P8: test_request_attributes_skip_none_values
def test_request_attributes_skip_none_values():
    """Verify only non-None attributes are set (G5)."""
    from vllm.tracing import SpanAttributes

    # Create mock serving instance
    mock_models = MagicMock()
    mock_models.model_config = MagicMock()
    mock_models.model_config.max_model_len = 4096

    serving = OpenAIServingChat(
        engine_client=AsyncMock(),
        models=mock_models,
        response_role="assistant",
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    # Create mock span
    mock_span = MagicMock()
    mock_span.is_recording = MagicMock(return_value=True)
    set_attribute_calls = {}

    def capture_set_attribute(key, value):
        set_attribute_calls[key] = value

    mock_span.set_attribute = capture_set_attribute

    # Create sampling params with default values
    sampling_params = SamplingParams()
    # Explicitly set these to None to test conditional logic
    sampling_params.temperature = None
    sampling_params.top_p = None
    sampling_params.max_tokens = None
    sampling_params.n = None

    # Create test data
    model_name = "test-model"
    prompt_token_ids = [1, 2, 3]

    # Call helper
    serving._set_api_span_request_attributes(
        mock_span,
        model_name,
        prompt_token_ids,
        sampling_params,
    )

    # Verify only always-set attributes were set
    assert SpanAttributes.GEN_AI_RESPONSE_MODEL in set_attribute_calls
    assert SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS in set_attribute_calls

    # Verify None attributes were NOT set
    assert SpanAttributes.GEN_AI_REQUEST_TEMPERATURE not in set_attribute_calls
    assert SpanAttributes.GEN_AI_REQUEST_TOP_P not in set_attribute_calls
    assert SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS not in set_attribute_calls
    assert SpanAttributes.GEN_AI_REQUEST_N not in set_attribute_calls


# P9: test_zero_overhead_when_span_none
def test_zero_overhead_when_span_none():
    """Verify no span operations when span is None (G6)."""
    # Create mock serving instance
    mock_models = MagicMock()
    mock_models.model_config = MagicMock()
    mock_models.model_config.max_model_len = 4096

    serving = OpenAIServingChat(
        engine_client=AsyncMock(),
        models=mock_models,
        response_role="assistant",
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    # Call helper with None span
    sampling_params = SamplingParams(temperature=0.7)
    model_name = "test-model"
    prompt_token_ids = [1, 2, 3]

    # Should not raise and should return immediately
    serving._set_api_span_request_attributes(
        None,  # span is None
        model_name,
        prompt_token_ids,
        sampling_params,
    )

    # If we get here without exception, test passes
    assert True


# P10: test_defensive_event_emission_failure
@pytest.mark.asyncio
async def test_defensive_event_emission_failure(mock_chat_serving):
    """Verify request succeeds despite event emission failure (G7).

    This test uses a mock span that raises exceptions on add_event,
    then drives the actual generator to verify the request completes successfully.
    """
    # Create mock span that raises on add_event
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_span.add_event = MagicMock(side_effect=Exception("OTEL error"))
    mock_span.end = MagicMock()  # end() should still work
    mock_span.set_status = MagicMock()

    # Store span
    request_id = "test-req-error-123"
    mock_chat_serving._store_api_span(request_id, mock_span, time.monotonic())

    request = create_mock_request(stream=True)
    request_metadata = RequestResponseMetadata(request_id=request_id)

    # Mock result generator
    async def mock_generator():
        output = MagicMock()
        output.prompt_token_ids = [1, 2, 3]
        output.encoder_prompt_token_ids = None
        output.num_cached_tokens = None
        output.outputs = [MagicMock(
            index=0,
            token_ids=[4, 5],
            text="test",
            finish_reason="stop",
            logprobs=None
        )]
        yield output

    # Call streaming generator - should NOT raise despite add_event failures
    chunks = []
    try:
        async for chunk in mock_chat_serving.chat_completion_stream_generator(
            request=request,
            result_generator=mock_generator(),
            request_id=request_id,
            model_name="test-model",
            conversation=[],
            tokenizer=None,
            request_metadata=request_metadata,
        ):
            chunks.append(chunk)
    except Exception as e:
        pytest.fail(f"Generator should not raise despite tracing failures: {e}")

    # Verify we got chunks (request completed successfully)
    assert len(chunks) > 0, "Request should complete successfully despite tracing failures"

    # Verify add_event was called (so it did fail)
    assert mock_span.add_event.called, "add_event should have been called"


# P11: test_defensive_attribute_setting_failure
def test_defensive_attribute_setting_failure():
    """Verify request succeeds despite attribute setting failure (G7)."""
    # Create mock serving instance
    mock_models = MagicMock()
    mock_models.model_config = MagicMock()
    mock_models.model_config.max_model_len = 4096

    serving = OpenAIServingChat(
        engine_client=AsyncMock(),
        models=mock_models,
        response_role="assistant",
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    # Create mock span that raises on set_attribute
    mock_span = MagicMock()
    mock_span.is_recording = MagicMock(return_value=True)
    mock_span.set_attribute = MagicMock(side_effect=Exception("OTEL error"))

    # Create test data
    sampling_params = SamplingParams(temperature=0.7)
    model_name = "test-model"
    prompt_token_ids = [1, 2, 3]

    # Call helper - should not raise
    try:
        serving._set_api_span_request_attributes(
            mock_span,
            model_name,
            prompt_token_ids,
            sampling_params,
        )
    except Exception:
        pytest.fail("Attribute setting failure should have been caught defensively")

    # If we get here without exception, test passes
    assert True


# P12: test_update_first_response_time_missing_request
def test_update_first_response_time_missing_request(serving_instance):
    """Verify _update_first_response_time handles missing request_id safely."""
    # Should not raise when request_id doesn't exist
    serving_instance._update_first_response_time("nonexistent-id", time.monotonic())

    # Verify dict is still empty
    assert len(serving_instance._api_spans) == 0


# P13: test_span_not_recording_early_exit
def test_span_not_recording_early_exit():
    """Verify early exit when span.is_recording() returns False (G6)."""
    # Create mock serving instance
    mock_models = MagicMock()
    mock_models.model_config = MagicMock()
    mock_models.model_config.max_model_len = 4096

    serving = OpenAIServingChat(
        engine_client=AsyncMock(),
        models=mock_models,
        response_role="assistant",
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    # Create mock span that is not recording
    mock_span = MagicMock()
    mock_span.is_recording = MagicMock(return_value=False)
    set_attribute_called = False

    def track_set_attribute(key, value):
        nonlocal set_attribute_called
        set_attribute_called = True

    mock_span.set_attribute = track_set_attribute

    # Call helper
    sampling_params = SamplingParams(temperature=0.7)
    serving._set_api_span_request_attributes(
        mock_span,
        "test-model",
        [1, 2, 3],
        sampling_params,
    )

    # Verify set_attribute was never called
    assert not set_attribute_called, "set_attribute should not be called when span not recording"
