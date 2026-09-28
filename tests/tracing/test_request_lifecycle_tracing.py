# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_TRACES_INSECURE,
)

from tests.tracing.conftest import FAKE_TRACE_SERVER_ADDRESS, FakeTraceService
from vllm.sampling_params import SamplingParams
from vllm.tracing import (
    SpanAttributes,
    SpanKind,
    init_tracer,
    is_otel_available,
    start_request_span,
)
from vllm.v1.request import Request, RequestStatus

# Skip everything if OTel is missing
pytestmark = pytest.mark.skipif(not is_otel_available(), reason="OTel required")


class TestRequestLifecycleTracing:
    """Focuses on request lifecycle spans and root llm_request hierarchy."""

    @pytest.fixture(autouse=True)
    def setup_tracing(self, monkeypatch):
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        init_tracer("test.request_lifecycle", FAKE_TRACE_SERVER_ADDRESS)

    def test_lifecycle_spans_nested_under_root_request(
        self, trace_service: FakeTraceService
    ):
        """Verify root span establishes valid hierarchy for lifecycle spans."""
        arrival_time_ns = time.time_ns()

        # 1. Start root llm_request span (as done in AsyncLLM/OutputProcessor)
        root_span, updated_headers = start_request_span(
            span_name="llm_request",
            start_time=arrival_time_ns,
            kind=SpanKind.SERVER,
        )
        assert root_span is not None
        assert updated_headers is not None
        assert "traceparent" in updated_headers

        # 2. Instantiate engine Request with updated_headers
        sampling_params = SamplingParams(max_tokens=10)
        req = Request(
            request_id="req-123",
            prompt_token_ids=[1, 2, 3],
            sampling_params=sampling_params,
            pooling_params=None,
            arrival_time=arrival_time_ns / 1e9,
            trace_headers=updated_headers,
        )

        # 3. Simulate Queue -> Prefill -> Decode lifecycle
        req.trace_end_queuing()
        req.prefill_start_time_ns = time.time_ns()
        time.sleep(0.01)

        req.trace_end_prefill()
        req.decode_start_time_ns = time.time_ns()
        time.sleep(0.01)

        req.trace_end_decode()

        # 4. End root llm_request span
        root_span.set_attributes(
            {
                SpanAttributes.GEN_AI_REQUEST_ID: "req-123",
                SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: 3,
                SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: 5,
            }
        )
        root_span.end()

        # 5. Verify all spans and hierarchy
        assert trace_service.wait_for_spans(count=4)
        spans = trace_service.get_all_spans()

        root = next(s for s in spans if s["name"] == "llm_request")
        queue_span = next(s for s in spans if s["name"] == "vllm.request.queue")
        prefill_span = next(s for s in spans if s["name"] == "vllm.request.prefill")
        decode_span = next(s for s in spans if s["name"] == "vllm.request.decode")

        # Check all child spans have root_span as parent
        root_span_id = root["span_id"]
        assert queue_span["parent_span_id"] == root_span_id
        assert prefill_span["parent_span_id"] == root_span_id
        assert decode_span["parent_span_id"] == root_span_id

        # Check all spans share the same trace_id
        trace_id = root["trace_id"]
        assert queue_span["trace_id"] == trace_id
        assert prefill_span["trace_id"] == trace_id
        assert decode_span["trace_id"] == trace_id

    def test_preemption_and_cleanup_spans(self, trace_service: FakeTraceService):
        """Verify preemption and cleanup gracefully close active spans."""
        arrival_time_ns = time.time_ns()
        root_span, updated_headers = start_request_span(
            span_name="llm_request",
            start_time=arrival_time_ns,
            kind=SpanKind.SERVER,
        )

        sampling_params = SamplingParams(max_tokens=10)
        req = Request(
            request_id="req-456",
            prompt_token_ids=[1, 2, 3],
            sampling_params=sampling_params,
            pooling_params=None,
            arrival_time=arrival_time_ns / 1e9,
            trace_headers=updated_headers,
        )

        # Start prefill then preempt
        req.prefill_start_time_ns = time.time_ns()
        req.trace_preempt()

        # Resume decode then finish with abort
        req.decode_start_time_ns = time.time_ns()
        req.status = RequestStatus.FINISHED_ABORTED
        req.trace_cleanup()

        root_span.end()

        assert trace_service.wait_for_spans(count=3)
        spans = trace_service.get_all_spans()

        prefill_span = next(s for s in spans if s["name"] == "vllm.request.prefill")
        decode_span = next(s for s in spans if s["name"] == "vllm.request.decode")

        assert prefill_span["attributes"].get("preempted") is True
        assert decode_span["attributes"].get("aborted") is True
