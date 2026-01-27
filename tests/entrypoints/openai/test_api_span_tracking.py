# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for API span tracking infrastructure (PR #5)."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels


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


def test_api_span_dict_initialized(serving_instance):
    """Test that _api_spans dict and cache are initialized correctly."""
    assert hasattr(serving_instance, "_api_spans")
    assert isinstance(serving_instance._api_spans, dict)
    assert len(serving_instance._api_spans) == 0

    assert hasattr(serving_instance, "_cached_is_tracing_enabled")
    assert serving_instance._cached_is_tracing_enabled is None


def test_store_and_retrieve_api_span(serving_instance):
    """Test storing and retrieving API span info."""
    # Create mock span
    mock_span = MagicMock()
    request_id = "test-request-123"
    arrival_time = time.monotonic()

    # Store span
    serving_instance._store_api_span(request_id, mock_span, arrival_time)

    # Retrieve span
    span, arr_time, first_resp_time = serving_instance._get_api_span_info(request_id)

    # Verify all fields
    assert span is mock_span
    assert arr_time == arrival_time
    assert first_resp_time is None  # Should be None initially


def test_retrieve_missing_request_returns_none_tuple(serving_instance):
    """Test that retrieving missing request returns (None, None, None)."""
    result = serving_instance._get_api_span_info("nonexistent-id")

    assert result == (None, None, None)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_cleanup_removes_api_span(serving_instance):
    """Test that cleanup removes API span entry."""
    # Store span
    mock_span = MagicMock()
    request_id = "test-request-456"
    serving_instance._store_api_span(request_id, mock_span, time.monotonic())

    # Verify stored
    span, _, _ = serving_instance._get_api_span_info(request_id)
    assert span is mock_span

    # Cleanup
    serving_instance._cleanup_api_span(request_id)

    # Verify removed
    result = serving_instance._get_api_span_info(request_id)
    assert result == (None, None, None)
    assert len(serving_instance._api_spans) == 0


def test_cleanup_nonexistent_request_is_safe(serving_instance):
    """Test that cleanup on nonexistent request doesn't raise exception."""
    # Should not raise
    serving_instance._cleanup_api_span("nonexistent-id")

    # Verify dict is still empty
    assert len(serving_instance._api_spans) == 0


@pytest.mark.asyncio
async def test_tracing_enabled_cache_works(serving_instance):
    """Test that tracing enabled check is cached."""
    # Mock to return True
    serving_instance.engine_client.is_tracing_enabled = AsyncMock(return_value=True)

    # Call twice
    result1 = await serving_instance._get_is_tracing_enabled()
    result2 = await serving_instance._get_is_tracing_enabled()

    # Both should return True
    assert result1 is True
    assert result2 is True

    # Engine client should be called only ONCE (caching works)
    assert serving_instance.engine_client.is_tracing_enabled.call_count == 1

    # Cache should be set
    assert serving_instance._cached_is_tracing_enabled is True


@pytest.mark.asyncio
async def test_tracing_enabled_check_handles_errors(serving_instance):
    """Test that tracing enabled check handles errors gracefully."""
    # Mock to raise exception
    serving_instance.engine_client.is_tracing_enabled = AsyncMock(
        side_effect=Exception("Engine error")
    )

    # Call should not raise, should return False
    result = await serving_instance._get_is_tracing_enabled()
    assert result is False

    # Cache should be set to False (prevents retry)
    assert serving_instance._cached_is_tracing_enabled is False

    # Call again - should still return False without calling engine again
    result2 = await serving_instance._get_is_tracing_enabled()
    assert result2 is False

    # Engine should have been called only once (second call used cache)
    assert serving_instance.engine_client.is_tracing_enabled.call_count == 1


def test_multiple_requests_tracked_independently(serving_instance):
    """Test that multiple requests are tracked independently."""
    # Store spans for 3 different requests
    request_ids = ["req-1", "req-2", "req-3"]
    mock_spans = [MagicMock(), MagicMock(), MagicMock()]
    arrival_times = [time.monotonic(), time.monotonic(), time.monotonic()]

    for req_id, span, arr_time in zip(request_ids, mock_spans, arrival_times):
        serving_instance._store_api_span(req_id, span, arr_time)

    # Verify all 3 are stored
    assert len(serving_instance._api_spans) == 3

    # Retrieve each independently
    for i, req_id in enumerate(request_ids):
        span, arr_time, _ = serving_instance._get_api_span_info(req_id)
        assert span is mock_spans[i]
        assert arr_time == arrival_times[i]

    # Cleanup one request
    serving_instance._cleanup_api_span("req-2")

    # Verify req-2 is gone but others remain
    assert len(serving_instance._api_spans) == 2
    assert serving_instance._get_api_span_info("req-2") == (None, None, None)

    # Verify req-1 and req-3 still intact
    span1, _, _ = serving_instance._get_api_span_info("req-1")
    span3, _, _ = serving_instance._get_api_span_info("req-3")
    assert span1 is mock_spans[0]
    assert span3 is mock_spans[2]
