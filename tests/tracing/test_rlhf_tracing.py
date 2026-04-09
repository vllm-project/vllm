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
