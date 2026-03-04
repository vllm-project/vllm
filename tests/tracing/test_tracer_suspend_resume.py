# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for tracer teardown and re-initialization across suspend/resume.

Validates that shutdown_otel_tracer() properly resets OTel global state
so the tracer can be re-initialized after CRIU restore. This is the same
class of problem as NCCL: gRPC TCP sockets become stale after restore,
and OTel's global "set once" lock prevents re-initialization without
an explicit reset.
"""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_INSECURE

from tests.tracing.conftest import FAKE_TRACE_SERVER_ADDRESS, FakeTraceService
from vllm.tracing import (
    init_tracer,
    instrument,
    is_otel_available,
    shutdown_tracer,
)
from vllm.tracing.otel import shutdown_otel_tracer

pytestmark = pytest.mark.skipif(not is_otel_available(), reason="OTel required")


def _flush_and_wait(svc: FakeTraceService, count: int = 1, timeout: float = 5) -> bool:
    """Force-flush the active TracerProvider then wait for spans."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=3000)
    return svc.wait_for_spans(count=count, timeout=timeout)


@pytest.fixture(autouse=True)
def _clean_otel_state(monkeypatch):
    """Ensure every test starts and ends with a clean OTel provider."""
    monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
    shutdown_otel_tracer()
    yield
    shutdown_otel_tracer()


class TestShutdownOtelTracer:
    """Unit tests for shutdown_otel_tracer() — the low-level OTel reset."""

    def test_shutdown_resets_provider(self, trace_service: FakeTraceService):
        """After shutdown, set_tracer_provider() must succeed again."""
        init_tracer("test.shutdown_reset", FAKE_TRACE_SERVER_ADDRESS)
        provider = trace.get_tracer_provider()
        assert hasattr(provider, "shutdown"), "Expected an SDK TracerProvider"

        shutdown_otel_tracer()

        # The global provider should be cleared
        assert trace._TRACER_PROVIDER is None

        # Re-init should not raise "Overriding of current TracerProvider
        # is not allowed"
        init_tracer("test.shutdown_reset.2", FAKE_TRACE_SERVER_ADDRESS)
        new_provider = trace.get_tracer_provider()
        assert new_provider is not provider

    def test_shutdown_is_idempotent(self, trace_service: FakeTraceService):
        """Calling shutdown twice must not raise."""
        init_tracer("test.idempotent", FAKE_TRACE_SERVER_ADDRESS)
        shutdown_otel_tracer()
        shutdown_otel_tracer()  # second call — should be a no-op

    def test_shutdown_without_init_is_safe(self):
        """Calling shutdown before any init must not raise."""
        shutdown_otel_tracer()  # no provider was ever set — should be a no-op


class TestShutdownTracer:
    """Tests for the public shutdown_tracer() wrapper in __init__.py."""

    def test_public_wrapper_delegates(self, trace_service: FakeTraceService):
        """shutdown_tracer() should reset state just like the OTel-specific
        function."""
        init_tracer("test.wrapper", FAKE_TRACE_SERVER_ADDRESS)
        assert trace._TRACER_PROVIDER is not None

        shutdown_tracer()
        assert trace._TRACER_PROVIDER is None

    def test_public_wrapper_safe_without_init(self):
        """shutdown_tracer() should be safe to call with no active provider."""
        shutdown_tracer()  # should not raise


class TestTracerReinitProducesSpans:
    """Integration test: spans export correctly after a shutdown → re-init
    cycle, simulating what happens across suspend/resume."""

    def test_spans_before_and_after_reinit(self, trace_service: FakeTraceService):
        """Verify that spans produced after re-init reach the collector."""
        init_tracer("test.reinit_spans", FAKE_TRACE_SERVER_ADDRESS)

        @instrument(span_name="before_shutdown")
        def before():
            pass

        before()
        assert _flush_and_wait(trace_service, count=1), (
            "No spans received before shutdown"
        )
        spans_before = trace_service.get_all_spans()
        assert any(s["name"] == "before_shutdown" for s in spans_before)

        # Simulate suspend: shutdown tracer
        shutdown_otel_tracer()

        # Clear the fake service to isolate post-reinit spans
        trace_service.clear()

        # Simulate resume: re-init tracer
        init_tracer("test.reinit_spans.resumed", FAKE_TRACE_SERVER_ADDRESS)

        @instrument(span_name="after_reinit")
        def after():
            pass

        after()
        assert _flush_and_wait(trace_service, count=1), (
            "No spans received after tracer re-init — gRPC channel may be stale"
        )
        spans_after = trace_service.get_all_spans()
        assert any(s["name"] == "after_reinit" for s in spans_after)

    def test_multiple_reinit_cycles(self, trace_service: FakeTraceService):
        """Three full init → shutdown cycles should all produce spans."""
        for i in range(3):
            trace_service.clear()
            init_tracer(f"test.cycle_{i}", FAKE_TRACE_SERVER_ADDRESS)

            @instrument(span_name=f"cycle_{i}")
            def work():
                pass

            work()
            assert _flush_and_wait(trace_service, count=1), (
                f"Cycle {i}: no spans received"
            )
            spans = trace_service.get_all_spans()
            assert any(s["name"] == f"cycle_{i}" for s in spans), (
                f"Cycle {i}: expected span 'cycle_{i}' not found in {spans}"
            )
            shutdown_otel_tracer()
