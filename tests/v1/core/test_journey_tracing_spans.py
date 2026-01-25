"""Tests for dual-stream journey tracing with OTEL spans.

This module tests the new dual-stream journey tracing architecture where:
- API layer creates parent spans (llm_request)
- Scheduler creates child spans (llm_core)
- Events are emitted directly to spans in real-time
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from tests.v1.core.utils import create_scheduler
from vllm.sampling_params import SamplingParams
from vllm.tracing import SpanAttributes
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.journey_events import RequestJourneyEventType
from vllm.v1.request import Request, RequestStatus

EOS_TOKEN_ID = 50256
_none_hash_initialized = False


class MockSpan:
    """Mock OTEL span for testing."""

    def __init__(self, name, kind, context=None, start_time=None):
        self.name = name
        self.kind = kind
        self.context = MagicMock()
        self.context.trace_id = 12345678901234567890
        self.context.span_id = 98765432109876543210
        self.parent_context = context
        self._is_recording = True
        self.attributes = {}
        self.events = []
        self.end_called = False
        self.end_time = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def add_event(self, name, attributes=None, timestamp=None):
        self.events.append({
            "name": name,
            "attributes": attributes or {},
            "timestamp": timestamp
        })

    def end(self, end_time=None):
        self.end_called = True
        self.end_time = end_time
        self._is_recording = False

    def is_recording(self):
        return self._is_recording


class MockTracer:
    """Mock OTEL tracer for testing."""

    def __init__(self):
        self.spans = []

    def start_span(self, name, kind=None, context=None, start_time=None):
        span = MockSpan(name, kind, context, start_time)
        self.spans.append(span)
        return span


def _create_request(
    prompt_len: int = 10,
    max_tokens: int = 10,
    trace_headers: dict | None = None
) -> Request:
    """Helper to create a test request."""
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    request_id = f"request-{time.time()}"
    prompt_token_ids = list(range(prompt_len))
    sampling_params = SamplingParams(max_tokens=max_tokens)
    block_hasher = get_request_block_hasher(16, sha256)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        block_hasher=block_hasher,
        trace_headers=trace_headers,
    )


def test_queued_scheduler_step_never_none():
    """Verify QUEUED event includes scheduler_step snapshot (never None)."""
    mock_tracer = MockTracer()

    with patch("vllm.tracing.init_tracer", return_value=mock_tracer):
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://localhost:4318/v1/traces"
        )

        # Create requests
        requests = [_create_request() for _ in range(3)]

        # Enqueue first request (before any schedule)
        scheduler.add_request(requests[0])

        # Verify span created
        spans = [s for s in mock_tracer.spans if s.name == "llm_core"]
        assert len(spans) == 1, "Should have created one core span"

        # Verify QUEUED event has scheduler_step=0 (not None)
        span = spans[0]
        queued_events = [e for e in span.events if e["name"] == "journey.QUEUED"]
        assert len(queued_events) == 1, "Should have one QUEUED event"

        queued_event = queued_events[0]
        assert SpanAttributes.JOURNEY_SCHEDULER_STEP in queued_event["attributes"]
        assert queued_event["attributes"][SpanAttributes.JOURNEY_SCHEDULER_STEP] == 0

        # Run schedule (increments counter to 1)
        scheduler.schedule()

        # Enqueue second request
        scheduler.add_request(requests[1])

        # Verify new span has QUEUED with scheduler_step=1
        spans = [s for s in mock_tracer.spans if s.name == "llm_core"]
        assert len(spans) == 2, "Should have two core spans"

        span2 = spans[1]
        queued_events = [e for e in span2.events if e["name"] == "journey.QUEUED"]
        assert len(queued_events) == 1
        assert queued_events[0]["attributes"][SpanAttributes.JOURNEY_SCHEDULER_STEP] == 1


def test_scheduler_core_span_with_trace_headers():
    """Verify scheduler creates child span when trace_headers provided."""
    mock_tracer = MockTracer()

    with patch("vllm.tracing.init_tracer", return_value=mock_tracer):
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://localhost:4318/v1/traces"
        )

        # Create request with trace_headers (simulating parent span context)
        trace_headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "tracestate": "congo=t61rcWkgMzE"
        }

        # Mock extract_trace_context to return a mock context
        mock_context = Mock()
        with patch("vllm.tracing.extract_trace_context", return_value=mock_context):
            request = _create_request(trace_headers=trace_headers)

            # Add request
            scheduler.add_request(request)

            # Verify span created
            spans = [s for s in mock_tracer.spans if s.name == "llm_core"]
            assert len(spans) == 1, "Should have created one core span"

            # Verify span has parent context (extracted from trace_headers)
            span = spans[0]
            assert span.parent_context == mock_context, "Span should have parent context"

            # Verify span has request_id attribute
            assert SpanAttributes.GEN_AI_REQUEST_ID in span.attributes
            assert span.attributes[SpanAttributes.GEN_AI_REQUEST_ID] == request.request_id


def test_scheduler_core_span_without_trace_headers():
    """Verify scheduler creates root span when no trace_headers provided."""
    mock_tracer = MockTracer()

    with patch("vllm.tracing.init_tracer", return_value=mock_tracer):
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://localhost:4318/v1/traces"
        )

        # Create request without trace_headers
        request = _create_request(trace_headers=None)

        # Mock extract_trace_context to return None
        with patch("vllm.tracing.extract_trace_context", return_value=None):
            # Add request
            scheduler.add_request(request)

            # Verify span created
            spans = [s for s in mock_tracer.spans if s.name == "llm_core"]
            assert len(spans) == 1, "Should have created one core span"

            # Verify span has no parent context (root span)
            span = spans[0]
            assert span.parent_context is None, "Span should be root span (no parent)"


def test_core_span_lifecycle():
    """Verify core span is created, receives events, and is closed properly."""
    mock_tracer = MockTracer()

    with patch("vllm.tracing.init_tracer", return_value=mock_tracer):
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://localhost:4318/v1/traces"
        )

        request = _create_request()

        # Add request - should create span and emit QUEUED
        scheduler.add_request(request)

        spans = [s for s in mock_tracer.spans if s.name == "llm_core"]
        assert len(spans) == 1
        span = spans[0]

        # Verify span is recording
        assert span.is_recording(), "Span should be recording"

        # Verify QUEUED event was emitted
        queued_events = [e for e in span.events if e["name"] == "journey.QUEUED"]
        assert len(queued_events) == 1

        # Schedule request - should emit SCHEDULED
        scheduler.schedule()

        scheduled_events = [e for e in span.events if e["name"] == "journey.SCHEDULED"]
        assert len(scheduled_events) == 1

        # Finish request - should emit FINISHED and close span
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)

        finished_events = [e for e in span.events if e["name"] == "journey.FINISHED"]
        assert len(finished_events) == 1

        # Verify span was closed
        assert span.end_called, "Span should be closed"
        assert not span.is_recording(), "Span should not be recording after close"


def test_cleanup_on_finish_with_spans():
    """Verify journey tracking state is cleaned up when request finishes."""
    mock_tracer = MockTracer()

    with patch("vllm.tracing.init_tracer", return_value=mock_tracer):
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://localhost:4318/v1/traces"
        )

        request = _create_request()

        # Add and schedule request
        scheduler.add_request(request)
        scheduler.schedule()

        # Verify state exists
        assert request.request_id in scheduler._journey_prefill_hiwater
        assert request.request_id in scheduler._core_spans

        # Finish request
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)

        # Verify state is cleaned up
        assert request.request_id not in scheduler._journey_prefill_hiwater
        assert request.request_id not in scheduler._first_token_emitted
        assert request.request_id not in scheduler._core_spans


def test_no_spans_when_tracer_not_configured():
    """Verify no spans created when tracer is None (no OTLP endpoint)."""
    # Don't provide otlp_traces_endpoint - tracer should be None
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        # No otlp_traces_endpoint
    )

    assert scheduler.tracer is None, "Tracer should be None when no endpoint configured"

    request = _create_request()

    # Add request - should not crash, should not create spans
    scheduler.add_request(request)

    # Verify no spans in _core_spans
    assert len(scheduler._core_spans) == 0, "No spans should be created without tracer"


def test_span_attributes_set_correctly():
    """Verify core span has correct attributes set."""
    mock_tracer = MockTracer()

    with patch("vllm.tracing.init_tracer", return_value=mock_tracer):
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://localhost:4318/v1/traces"
        )

        request = _create_request()
        scheduler.add_request(request)

        spans = [s for s in mock_tracer.spans if s.name == "llm_core"]
        assert len(spans) == 1
        span = spans[0]

        # Verify span has request_id attribute
        assert SpanAttributes.GEN_AI_REQUEST_ID in span.attributes
        assert span.attributes[SpanAttributes.GEN_AI_REQUEST_ID] == request.request_id


def test_event_attributes_include_monotonic_timestamp():
    """Verify all journey events include monotonic timestamp."""
    mock_tracer = MockTracer()

    with patch("vllm.tracing.init_tracer", return_value=mock_tracer):
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://localhost:4318/v1/traces"
        )

        request = _create_request()

        # Add request
        scheduler.add_request(request)

        spans = [s for s in mock_tracer.spans if s.name == "llm_core"]
        span = spans[0]

        # Check QUEUED event has monotonic timestamp
        queued_events = [e for e in span.events if e["name"] == "journey.QUEUED"]
        assert len(queued_events) == 1
        assert SpanAttributes.EVENT_TS_MONOTONIC in queued_events[0]["attributes"]

        # Schedule request
        scheduler.schedule()

        # Check SCHEDULED event has monotonic timestamp
        scheduled_events = [e for e in span.events if e["name"] == "journey.SCHEDULED"]
        assert len(scheduled_events) == 1
        assert SpanAttributes.EVENT_TS_MONOTONIC in scheduled_events[0]["attributes"]


def test_multiple_requests_separate_spans():
    """Verify each request gets its own span."""
    mock_tracer = MockTracer()

    with patch("vllm.tracing.init_tracer", return_value=mock_tracer):
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://localhost:4318/v1/traces"
        )

        # Create and add multiple requests
        requests = [_create_request() for _ in range(5)]
        for request in requests:
            scheduler.add_request(request)

        # Verify 5 spans created
        spans = [s for s in mock_tracer.spans if s.name == "llm_core"]
        assert len(spans) == 5, "Should have 5 core spans"

        # Verify each span has unique request_id
        request_ids = [span.attributes[SpanAttributes.GEN_AI_REQUEST_ID] for span in spans]
        assert len(set(request_ids)) == 5, "All request IDs should be unique"


def test_cleanup_on_abort_path():
    """Verify journey tracking state is cleaned up when request is aborted."""
    mock_tracer = MockTracer()

    with patch("vllm.tracing.init_tracer", return_value=mock_tracer):
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://localhost:4318/v1/traces"
        )

        request = _create_request()

        # Add and schedule request
        scheduler.add_request(request)
        scheduler.schedule()

        # Verify state exists before abort
        assert request.request_id in scheduler._journey_prefill_hiwater
        assert request.request_id in scheduler._core_spans

        # Abort request (using FINISHED_ABORTED status)
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

        # Verify state is cleaned up after abort
        assert request.request_id not in scheduler._journey_prefill_hiwater
        assert request.request_id not in scheduler._first_token_emitted
        assert request.request_id not in scheduler._core_spans

        # Verify span was closed
        spans = [s for s in mock_tracer.spans if s.name == "llm_core"]
        assert len(spans) == 1
        span = spans[0]
        assert span.end_called, "Span should be closed after abort"
        assert not span.is_recording(), "Span should not be recording after abort"

        # Verify FINISHED event was emitted with abort status
        finished_events = [e for e in span.events if e["name"] == "journey.FINISHED"]
        assert len(finished_events) == 1
        assert finished_events[0]["attributes"]["finish.status"] == "aborted"


def test_cleanup_on_natural_completion():
    """Verify journey tracking state is cleaned up on natural completion (EOS/max_tokens)."""
    mock_tracer = MockTracer()

    with patch("vllm.tracing.init_tracer", return_value=mock_tracer):
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://localhost:4318/v1/traces"
        )

        request = _create_request(prompt_len=10, max_tokens=10)

        # Add and schedule request
        scheduler.add_request(request)
        scheduler.schedule()

        # Verify state exists before completion
        assert request.request_id in scheduler._core_spans

        # Finish request with FINISHED_STOPPED (natural completion)
        # This simulates the path where request completes normally (EOS or max_tokens)
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)

        # Verify state is cleaned up after natural completion
        assert request.request_id not in scheduler._core_spans
        assert request.request_id not in scheduler._journey_prefill_hiwater
        assert request.request_id not in scheduler._first_token_emitted

        # Verify span was closed
        spans = [s for s in mock_tracer.spans if s.name == "llm_core"]
        assert len(spans) == 1
        assert spans[0].end_called, "Span should be closed after natural completion"
        assert not spans[0].is_recording(), "Span should not be recording after completion"

        # Verify FINISHED event was emitted with stopped status
        finished_events = [e for e in spans[0].events if e["name"] == "journey.FINISHED"]
        assert len(finished_events) == 1
        assert finished_events[0]["attributes"]["finish.status"] == "stopped"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
