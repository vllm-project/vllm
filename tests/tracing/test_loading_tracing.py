# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import gc

import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_INSECURE

from tests.tracing.conftest import FAKE_TRACE_SERVER_ADDRESS, FakeTraceService
from vllm.tracing import init_tracer, instrument, is_otel_available

# Skip everything if OTel is missing
pytestmark = pytest.mark.skipif(not is_otel_available(), reason="OTel required")


class TestCoreInstrumentation:
    """Focuses on the @instrument decorator's ability to capture execution data."""

    @pytest.fixture(autouse=True)
    def setup_tracing(self, monkeypatch):
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        init_tracer("test.core", FAKE_TRACE_SERVER_ADDRESS)

    def test_decorator_captures_sync_and_async(self, trace_service: FakeTraceService):
        """Verify basic span creation for both sync and async functions."""

        @instrument(span_name="sync_task")
        def sync_task():
            return True

        @instrument(span_name="async_task")
        async def async_task():
            return True

        sync_task()
        asyncio.run(async_task())

        assert trace_service.wait_for_spans(count=2)
        span_names = [s["name"] for s in trace_service.get_all_spans()]
        assert "sync_task" in span_names
        assert "async_task" in span_names

    def test_nested_spans_hierarchy(self, trace_service: FakeTraceService):
        """Verify that nested calls create a parent-child relationship."""

        @instrument(span_name="child")
        def child():
            pass

        @instrument(span_name="parent")
        def parent():
            child()

        parent()

        assert trace_service.wait_for_spans(count=2)
        spans = trace_service.get_all_spans()
        parent_span = next(s for s in spans if s["name"] == "parent")
        child_span = next(s for s in spans if s["name"] == "child")

        assert child_span["parent_span_id"] == parent_span["span_id"]


class TestMeterProviderSetup:
    """init_tracer must install a MeterProvider so the SDK's per-tracer
    metrics instruments don't materialize a new ProxyMeter per call."""

    def test_no_proxy_meter_leak(self, monkeypatch, trace_service):
        """Repeated get_tracer calls (as done per request by
        instrument_otel) must not accumulate proxy objects."""
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        init_tracer("test.meter", FAKE_TRACE_SERVER_ADDRESS)

        assert type(metrics.get_meter_provider()).__name__ != "_ProxyMeterProvider"

        def count_proxies():
            gc.collect()
            return sum(
                1 for o in gc.get_objects() if "proxy" in type(o).__name__.lower()
            )

        before = count_proxies()
        for _ in range(100):
            trace.get_tracer("leak.probe")
        assert count_proxies() - before == 0


class TestInterProcessPropagation:
    """Test the propagation of trace context between processes."""

    def test_pickup_external_context(self, monkeypatch, trace_service):
        """Test that vLLM attaches to an existing trace ID if in environment."""
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")

        # Manually simulate an external parent trace ID
        fake_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        fake_parent_id = "00f067aa0ba902b7"
        monkeypatch.setenv("traceparent", f"00-{fake_trace_id}-{fake_parent_id}-01")

        init_tracer("test.external", FAKE_TRACE_SERVER_ADDRESS)

        @instrument(span_name="follower")
        def follower_func():
            pass

        follower_func()

        assert trace_service.wait_for_spans(count=1)
        span = trace_service.get_all_spans()[0]

        assert span["trace_id"] == fake_trace_id
        assert span["parent_span_id"] == fake_parent_id
