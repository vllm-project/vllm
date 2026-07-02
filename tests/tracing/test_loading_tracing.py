# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time

import pytest
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_INSECURE

from tests.tracing.conftest import FAKE_TRACE_SERVER_ADDRESS, FakeTraceService
from vllm.tracing import init_tracer, instrument, instrument_manual, is_otel_available

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


class TestManualSpanTimestamps:
    """Regression test for https://github.com/vllm-project/vllm/issues/46193:
    manual spans (e.g. the "Dynamo bytecode transform" span emitted from
    vllm/compilation/backends.py) must be anchored to wall-clock time, not a
    monotonic clock with an arbitrary epoch like time.perf_counter()."""

    @pytest.fixture(autouse=True)
    def setup_tracing(self, monkeypatch):
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        init_tracer("test.manual", FAKE_TRACE_SERVER_ADDRESS)

    def test_manual_span_uses_wall_clock_time(self, trace_service: FakeTraceService):
        from vllm.compilation import monitor
        from vllm.config import VllmConfig

        # Exercise the real monitor_torch_compile context manager so this
        # test fails if vllm/compilation/monitor.py regresses to only
        # tracking a monotonic (perf_counter) timestamp.
        with monitor.monitor_torch_compile(VllmConfig()):
            time.sleep(0.05)

        # Same arithmetic as vllm/compilation/backends.py: an elapsed
        # duration measured with time.perf_counter(), anchored to the
        # wall-clock reference point captured by monitor_torch_compile.
        dynamo_time = time.perf_counter() - monitor.torch_compile_start_time
        start_time_ns = int(monitor.torch_compile_start_time_wall * 1e9)
        end_time_ns = int((monitor.torch_compile_start_time_wall + dynamo_time) * 1e9)

        instrument_manual(
            "Dynamo bytecode transform",
            start_time_ns,
            end_time_ns,
            {"dynamo.time_seconds": dynamo_time},
        )

        assert trace_service.wait_for_spans(count=1)
        span = trace_service.get_all_spans()[0]

        now_ns = time.time_ns()
        one_day_ns = 24 * 60 * 60 * 1_000_000_000

        # The recorded timestamps must be close to "now", not close to the
        # Unix epoch (which is what happens if a perf_counter() value is
        # mistakenly used as a nanoseconds-since-epoch timestamp).
        assert abs(now_ns - span["start_time_unix_nano"]) < one_day_ns
        assert abs(now_ns - span["end_time_unix_nano"]) < one_day_ns

        duration_s = (span["end_time_unix_nano"] - span["start_time_unix_nano"]) / 1e9
        assert duration_s == pytest.approx(dynamo_time, abs=0.05)
