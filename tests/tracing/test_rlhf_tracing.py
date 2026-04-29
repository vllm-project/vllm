# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that RLHF API router endpoints emit OpenTelemetry spans."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_INSECURE

from tests.tracing.conftest import FAKE_TRACE_SERVER_ADDRESS, FakeTraceService
from vllm.tracing import init_tracer, is_otel_available

pytestmark = pytest.mark.skipif(not is_otel_available(), reason="OTel required")


def _make_mock_request(*, paused: bool = False):
    """Create a mock FastAPI Request with a mock EngineClient on app state."""
    engine = AsyncMock()
    engine.pause_generation = AsyncMock()
    engine.resume_generation = AsyncMock()
    engine.is_paused = AsyncMock(return_value=paused)

    app_state = MagicMock()
    app_state.engine_client = engine

    app = MagicMock()
    app.state = app_state

    request = MagicMock()
    request.app = app
    return request


class TestRLHFRouteTracing:
    """Verify that each RLHF API route emits an OTel span."""

    @pytest.fixture(autouse=True)
    def setup_tracing(self, monkeypatch):
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        init_tracer("test.rlhf", FAKE_TRACE_SERVER_ADDRESS)

    def test_pause_emits_span(self, trace_service: FakeTraceService):
        from vllm.entrypoints.serve.rlhf.api_router import pause_generation

        request = _make_mock_request()
        resp = asyncio.run(pause_generation(request))
        assert resp.status_code == 200

        assert trace_service.wait_for_spans(count=1)
        span_names = [s["name"] for s in trace_service.get_all_spans()]
        assert "POST /pause" in span_names

    def test_resume_emits_span(self, trace_service: FakeTraceService):
        from vllm.entrypoints.serve.rlhf.api_router import resume_generation

        request = _make_mock_request()
        resp = asyncio.run(resume_generation(request))
        assert resp.status_code == 200

        assert trace_service.wait_for_spans(count=1)
        span_names = [s["name"] for s in trace_service.get_all_spans()]
        assert "POST /resume" in span_names

    def test_is_paused_emits_span(self, trace_service: FakeTraceService):
        from vllm.entrypoints.serve.rlhf.api_router import is_paused

        request = _make_mock_request(paused=True)
        resp = asyncio.run(is_paused(request))
        assert resp.status_code == 200

        assert trace_service.wait_for_spans(count=1)
        span_names = [s["name"] for s in trace_service.get_all_spans()]
        assert "GET /is_paused" in span_names


class TestTraceContextMiddleware:
    """Verify the TraceContextMiddleware propagates W3C headers."""

    @pytest.fixture(autouse=True)
    def setup_tracing(self, monkeypatch):
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        init_tracer("test.middleware", FAKE_TRACE_SERVER_ADDRESS)

    def test_middleware_propagates_traceparent(
        self, trace_service: FakeTraceService
    ):
        """Span created inside middleware should inherit the incoming
        traceparent as its parent."""
        from opentelemetry import trace

        from vllm.tracing.middleware import TraceContextMiddleware

        fake_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        fake_parent_id = "00f067aa0ba902b7"
        traceparent = f"00-{fake_trace_id}-{fake_parent_id}-01"

        captured = {}

        async def downstream(scope, receive, send):
            """Fake ASGI app that creates a span."""
            tracer = trace.get_tracer("test")
            with tracer.start_as_current_span("inner-span"):
                span = trace.get_current_span()
                ctx = span.get_span_context()
                captured["trace_id"] = format(ctx.trace_id, "032x")
                captured["parent_span_id"] = span.parent.span_id if span.parent else None

        middleware = TraceContextMiddleware(downstream)

        scope = {
            "type": "http",
            "headers": [
                (b"traceparent", traceparent.encode()),
            ],
        }

        asyncio.run(middleware(scope, None, None))

        assert trace_service.wait_for_spans(count=1)
        span = trace_service.get_all_spans()[0]
        assert span["trace_id"] == fake_trace_id
        assert span["parent_span_id"] == fake_parent_id

    def test_middleware_noop_without_traceparent(self):
        """Without traceparent header, middleware is a passthrough."""
        from vllm.tracing.middleware import TraceContextMiddleware

        called = False

        async def downstream(scope, receive, send):
            nonlocal called
            called = True

        middleware = TraceContextMiddleware(downstream)
        scope = {"type": "http", "headers": []}

        asyncio.run(middleware(scope, None, None))
        assert called
