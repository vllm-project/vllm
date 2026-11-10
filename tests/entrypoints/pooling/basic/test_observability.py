# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.pooling.base.observability import ObservabilityMixin
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.tracing import SpanAttributes
from vllm.tracing.otel import _IS_OTEL_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _IS_OTEL_AVAILABLE,
    reason="opentelemetry-sdk is required to exercise the OTEL mixin",
)


# ---------------------------------------------------------------------------
# Deterministic stand-ins so span attributes are stable across runs
# ---------------------------------------------------------------------------


class FakeRequest:
    """Stable ``__class__.__name__`` for ``GEN_AI_POOLING_REQUEST_CLASS``."""


class FakeIOProcessor:
    """Stable ``__class__.__name__`` for ``GEN_AI_POOLING_IO_PROCESSOR``."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def exporter():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(exporter):
    """A real ``TracerProvider`` wired to an in-memory exporter."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        yield provider
    finally:
        provider.shutdown()


@pytest.fixture
def mock_init_provider(monkeypatch, tracer_provider):
    """Replace ``init_otel_trace_provider`` in the observability module with a
    function that returns our test provider.

    The mock is returned so tests can assert on the call arguments (e.g. the
    OTLP endpoint forwarded from the config).
    """
    import vllm.entrypoints.pooling.base.observability as obs_mod

    mock_init = MagicMock(return_value=tracer_provider)
    monkeypatch.setattr(obs_mod, "init_otel_trace_provider", mock_init)
    return mock_init


def _make_mixin(
    *,
    tracing_enabled: bool,
    api_process_rank: int = 0,
    otlp_endpoint: str = "http://test-collector:4317",
) -> Any:
    """Build a mixin instance and run its real ``_init_observability``."""
    vllm_config = MagicMock()
    vllm_config.observability_config.is_tracing_enabled = tracing_enabled
    vllm_config.observability_config.otlp_traces_endpoint_str = otlp_endpoint

    renderer = MagicMock()
    renderer.api_process_rank = api_process_rank

    inst = ObservabilityMixin.__new__(ObservabilityMixin)
    inst.vllm_config = vllm_config
    inst.renderer = renderer
    inst._init_observability()
    return inst


@pytest.fixture
def mixin(mock_init_provider, tracer_provider) -> Any:
    """Construct the real OTEL mixin with a mocked provider factory."""
    inst = _make_mixin(tracing_enabled=True)

    assert inst.is_tracing_request is True
    assert inst.trace_provider is tracer_provider
    return inst


@pytest.fixture
def ctx():
    """A ``PoolingServeContext`` with deterministic timing fields.

    Timing layout (ns):
        arrival             = 1_000_000_000
        preprocessing_done  = 1_100_000_000  -> 100 ms
        engine_call_done    = 1_500_000_000  -> 400 ms
        postprocessing_done = 1_600_000_000  -> 100 ms
    """
    c = PoolingServeContext[Any](
        request=cast(Any, FakeRequest()),
        raw_request=None,
        model_name="test-model",
        pooling_params=None,
        request_id="pool-abc",
        lora_request=None,
        priorities=0,
        prompt_extras={},
    )
    c.arrival_time_ns = 1_000_000_000
    c.preprocessing_finished = 1_100_000_000
    c.engine_call_finished = 1_500_000_000
    c.postprocessing_finished = 1_600_000_000
    c.request_span_context = None
    c.entrypoint_tracer = None
    c.trace_headers = None
    return c


@pytest.fixture
def io_processor():
    return FakeIOProcessor()


# ---------------------------------------------------------------------------
# A. Provider / propagator construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_provider_and_propagator(self, mixin, tracer_provider):
        assert mixin.is_tracing_request is True
        assert mixin.trace_provider is tracer_provider
        assert mixin.propagator is not None

    def test_sets_scope_endpoint_attributes(self, mixin):
        assert mixin.scope_endpoint_attributes == {
            "vllm.process_name": "APIServer_0",
        }

    def test_forwards_endpoint_to_provider_factory(
        self, mock_init_provider, tracer_provider
    ):
        """The endpoint from the config must be passed to the factory."""
        inst = _make_mixin(
            tracing_enabled=True,
            otlp_endpoint="http://collector.example:4317",
        )

        assert inst.trace_provider is tracer_provider
        # The observability config field is what gets forwarded; assert on
        # the argument, not just the return value.
        mock_init_provider.assert_called_once_with("http://collector.example:4317")

    def test_uses_renderer_process_rank(self, mock_init_provider):
        """The renderer's process rank must appear in endpoint attributes."""
        inst = _make_mixin(tracing_enabled=True, api_process_rank=3)

        assert cast(Any, inst).scope_endpoint_attributes == {
            "vllm.process_name": "APIServer_3",
        }

    def test_skips_when_tracing_disabled(self):
        """No provider/propagator is created when tracing is off."""
        inst = _make_mixin(tracing_enabled=False)

        assert inst.is_tracing_request is False
        assert not hasattr(inst, "trace_provider")
        assert not hasattr(inst, "propagator")


# ---------------------------------------------------------------------------
# B. ``maybe_tracing`` — the top-level request span
# ---------------------------------------------------------------------------


class TestRequestSpan:
    @pytest.mark.asyncio
    async def test_attributes(self, mixin, exporter, ctx, io_processor):
        async with mixin.maybe_tracing(ctx, io_processor):
            assert ctx.request_span_context is not None
            assert ctx.entrypoint_tracer is not None

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        req = spans[0]

        assert req.name == "prefill"
        # start_time is forwarded as ns; exporter preserves the unit verbatim.
        assert req.start_time == pytest.approx(ctx.arrival_time_ns, rel=0, abs=1)
        assert req.end_time is not None
        assert req.end_time >= req.start_time

        attrs = req.attributes
        assert attrs[SpanAttributes.GEN_AI_REQUEST_ID] == "pool-abc"
        assert attrs[SpanAttributes.GEN_AI_POOLING_IO_PROCESSOR] == "FakeIOProcessor"
        assert attrs[SpanAttributes.GEN_AI_POOLING_REQUEST_CLASS] == "FakeRequest"

        # Latencies are reported in milliseconds.
        assert attrs[SpanAttributes.GEN_AI_LATENCY_PREPROCESSING] == pytest.approx(
            100.0
        )
        assert attrs[SpanAttributes.GEN_AI_LATENCY_ENGINE_CALL] == pytest.approx(400.0)
        assert attrs[SpanAttributes.GEN_AI_LATENCY_POSTPROCESSING] == pytest.approx(
            100.0
        )

    @pytest.mark.asyncio
    async def test_without_incoming_headers_starts_new_trace(
        self, mixin, exporter, ctx, io_processor
    ):
        """No incoming traceparent must produce a fresh (non-zero) trace_id."""
        async with mixin.maybe_tracing(ctx, io_processor):
            pass

        req = exporter.get_finished_spans()[0]
        assert req.context.trace_id != 0

    @pytest.mark.asyncio
    async def test_inherits_remote_trace_from_headers(
        self, mixin, exporter, ctx, io_processor
    ):
        """An incoming traceparent must be used as the parent trace context of
        the ``prefill`` span."""
        remote_trace_id = "0af7651916cd43dd8448eb211c80319c"
        remote_span_id = "b7ad6b7169203331"

        raw_request = MagicMock()
        raw_request.headers = {
            "traceparent": f"00-{remote_trace_id}-{remote_span_id}-01",
            "tracestate": "congo=t61rcWkgMzE",
        }

        ctx.raw_request = raw_request

        async with mixin.maybe_tracing(ctx, io_processor):
            pass

        req = exporter.get_finished_spans()[0]
        assert f"{req.context.trace_id:032x}" == remote_trace_id

    @pytest.mark.asyncio
    async def test_records_error(self, mixin, exporter, ctx, io_processor):
        """OTEL records exceptions as a generic ``"exception"`` event whose type
        is stored in ``attributes["exception.type"]`` — not as an event named
        after the exception class.
        """
        from opentelemetry.trace import StatusCode

        class BoomError(Exception):
            pass

        with pytest.raises(BoomError):
            async with mixin.maybe_tracing(ctx, io_processor):
                raise BoomError("boom")

        req = exporter.get_finished_spans()[0]
        assert req.status.status_code == StatusCode.ERROR
        assert any(
            e.name == "exception"
            and "BoomError" in e.attributes.get("exception.type", "")
            for e in req.events
        )
        assert req.end_time is not None


# ---------------------------------------------------------------------------
# C. Sync/async entrypoint span helpers
# ---------------------------------------------------------------------------


class TestStartSpan:
    # -- sync -------------------------------------------------------------

    def test_sync_with_tracer(self, mixin, exporter):
        tracer = mixin.trace_provider.get_tracer("t")

        with mixin.maybe_start_span(tracer, "op") as span:
            assert span is not None

        finished = exporter.get_finished_spans()[0]
        assert finished.name == "op"
        assert finished.attributes[SpanAttributes.GEN_AI_PROCESS_ID] == str(os.getpid())
        assert finished.attributes[SpanAttributes.GEN_AI_THREADING_ID] == str(
            threading.get_ident()
        )

    def test_sync_without_tracer_is_noop(self, mixin, exporter):
        with mixin.maybe_start_span(None, "op") as span:
            assert span is None
        assert exporter.get_finished_spans() == ()

    def test_sync_forwards_custom_attributes(self, mixin, exporter):
        tracer = mixin.trace_provider.get_tracer("t")

        with mixin.maybe_start_span(tracer, "op", attributes={"custom": "v"}):
            pass

        span = exporter.get_finished_spans()[0]
        assert span.attributes["custom"] == "v"
        assert SpanAttributes.GEN_AI_PROCESS_ID in span.attributes

    def test_sync_records_error(self, mixin, exporter):
        from opentelemetry.trace import StatusCode

        tracer = mixin.trace_provider.get_tracer("t")

        with pytest.raises(ValueError), mixin.maybe_start_span(tracer, "op"):
            raise ValueError("bad")

        finished = exporter.get_finished_spans()[0]
        assert finished.status.status_code == StatusCode.ERROR
        assert finished.end_time is not None

    # -- async ------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_async_with_tracer(self, mixin, exporter):
        tracer = mixin.trace_provider.get_tracer("t")

        async with mixin.maybe_start_span_async(tracer, "op") as span:
            assert span is not None

        finished = exporter.get_finished_spans()[0]
        assert finished.name == "op"
        assert SpanAttributes.GEN_AI_PROCESS_ID in finished.attributes

    @pytest.mark.asyncio
    async def test_async_without_tracer_is_noop(self, mixin, exporter):
        async with mixin.maybe_start_span_async(None, "op") as span:
            assert span is None
        assert exporter.get_finished_spans() == ()

    @pytest.mark.asyncio
    async def test_async_forwards_custom_attributes(self, mixin, exporter):
        tracer = mixin.trace_provider.get_tracer("t")

        async with mixin.maybe_start_span_async(tracer, "op", attributes={"k": "v"}):
            pass

        span = exporter.get_finished_spans()[0]
        assert span.attributes["k"] == "v"

    @pytest.mark.asyncio
    async def test_async_records_error(self, mixin, exporter):
        from opentelemetry.trace import StatusCode

        tracer = mixin.trace_provider.get_tracer("t")

        with pytest.raises(RuntimeError):
            async with mixin.maybe_start_span_async(tracer, "op"):
                raise RuntimeError("bad")

        finished = exporter.get_finished_spans()[0]
        assert finished.status.status_code == StatusCode.ERROR
        assert finished.end_time is not None


# ---------------------------------------------------------------------------
# D. Incoming trace header extraction / outgoing trace header injection
# ---------------------------------------------------------------------------


class TestTraceHeaders:
    @pytest.mark.asyncio
    async def test_extract_absent(self, mixin):
        assert await mixin._get_trace_headers({}) is None

    @pytest.mark.asyncio
    async def test_extract_present(self, mixin):
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "tracestate": "congo=t61rcWkgMzE",
        }
        extracted = await mixin._get_trace_headers(headers)
        assert extracted is not None
        # Be lenient about the exact wire format; just require the trace_id
        # to survive extraction.
        assert "0af7651916cd43dd8448eb211c80319c" in extracted["traceparent"]

    @pytest.mark.asyncio
    async def test_extract_traceparent_only(self, mixin):
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        }
        extracted = await mixin._get_trace_headers(headers)
        assert extracted is not None
        assert "traceparent" in extracted

    def test_inject_round_trip(self, mixin, exporter):
        """Injected traceparent must carry the span's trace_id."""
        tracer = mixin.trace_provider.get_tracer("t")

        with mixin.maybe_start_span(tracer, "op") as span:
            injected = mixin._get_span_trace_headers(span)

        assert "traceparent" in injected
        finished = exporter.get_finished_spans()[0]
        expected_trace_id = f"{finished.context.trace_id:032x}"
        assert expected_trace_id in injected["traceparent"]

    def test_inject_uses_span_id(self, mixin, exporter):
        """The span-id component must match the span's own span_id."""
        tracer = mixin.trace_provider.get_tracer("t")

        with mixin.maybe_start_span(tracer, "op") as span:
            injected = mixin._get_span_trace_headers(span)

        finished = exporter.get_finished_spans()[0]
        expected_span_id = f"{finished.context.span_id:016x}"
        assert expected_span_id in injected["traceparent"]


# ---------------------------------------------------------------------------
# E. Error paths shared across helpers
# ---------------------------------------------------------------------------


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_request_span_closes_on_error(
        self, mixin, exporter, ctx, io_processor
    ):
        from opentelemetry.trace import StatusCode

        with pytest.raises(RuntimeError):  # noqa: SIM117
            async with mixin.maybe_tracing(ctx, io_processor):
                raise RuntimeError("engine failure")

        req = exporter.get_finished_spans()[0]
        assert req.status.status_code == StatusCode.ERROR
        assert req.end_time is not None
        assert any(
            e.name == "exception"
            and "RuntimeError" in e.attributes.get("exception.type", "")
            for e in req.events
        )

    @pytest.mark.asyncio
    async def test_nested_span_error_propagates(
        self, mixin, exporter, ctx, io_processor
    ):
        """An error inside an entrypoint span must escape ``maybe_tracing``."""
        from opentelemetry.trace import StatusCode

        class Boom(Exception):
            pass

        with pytest.raises(Boom):
            async with mixin.maybe_tracing(ctx, io_processor):
                async with mixin.maybe_start_span_async(
                    ctx.entrypoint_tracer,
                    mixin.span_entrypoint_preprocessing,
                    context=ctx.request_span_context,
                ):
                    raise Boom("preprocessing")

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        assert set(by_name) == {"prefill", "preprocessing"}
        assert by_name["prefill"].status.status_code == StatusCode.ERROR
        assert by_name["preprocessing"].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# F. Composition — mirrors ``PoolingBaseServing.__call__`` span structure
# ---------------------------------------------------------------------------


class TestServingIntegration:
    @pytest.mark.asyncio
    async def test_span_tree_composition(self, mixin, exporter, ctx, io_processor):
        """All entrypoint spans must be children of the ``prefill`` span."""
        async with mixin.maybe_tracing(ctx, io_processor):
            async with mixin.maybe_start_span_async(
                ctx.entrypoint_tracer,
                mixin.span_entrypoint_preprocessing,
                context=ctx.request_span_context,
            ):
                pass

            async with mixin.maybe_start_span_async(
                ctx.entrypoint_tracer,
                mixin.span_entrypoint_engine_call,
                context=ctx.request_span_context,
            ) as span:
                if span is not None:
                    ctx.trace_headers = mixin._get_span_trace_headers(span)

            with mixin.maybe_start_span(
                ctx.entrypoint_tracer,
                mixin.span_entrypoint_postprocessing,
                context=ctx.request_span_context,
            ):
                pass

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        assert set(by_name) == {
            "prefill",
            "preprocessing",
            "engine_call",
            "postprocessing",
        }

        req = by_name["prefill"]
        for child_name in ("preprocessing", "engine_call", "postprocessing"):
            child = by_name[child_name]
            assert child.context.trace_id == req.context.trace_id
            assert child.parent is not None
            assert child.parent.span_id == req.context.span_id

    @pytest.mark.asyncio
    async def test_trace_headers_reach_engine_call(
        self, mixin, exporter, ctx, io_processor
    ):
        """``ctx.trace_headers`` must be injected before ``engine_client.encode``."""
        captured: dict = {}

        def fake_encode(**kwargs):
            captured.update(kwargs)

            async def _empty():
                if False:  # pragma: no cover
                    yield

            return _empty()

        async with mixin.maybe_tracing(ctx, io_processor):  # noqa: SIM117
            async with mixin.maybe_start_span_async(
                ctx.entrypoint_tracer,
                mixin.span_entrypoint_engine_call,
                context=ctx.request_span_context,
            ) as span:
                if span is not None:
                    ctx.trace_headers = mixin._get_span_trace_headers(span)
                # Sync call; not awaited. The returned generator is unused.
                _ = fake_encode(trace_headers=ctx.trace_headers)

        engine_span = next(
            s for s in exporter.get_finished_spans() if s.name == "engine_call"
        )
        assert "traceparent" in captured["trace_headers"]
        expected_trace_id = f"{engine_span.context.trace_id:032x}"
        assert expected_trace_id in captured["trace_headers"]["traceparent"]

    @pytest.mark.asyncio
    async def test_propagates_remote_trace_to_engine(
        self, mixin, exporter, ctx, io_processor
    ):
        """End-to-end: incoming traceparent appears in the engine_call headers."""
        remote_trace_id = "0af7651916cd43dd8448eb211c80319c"
        raw_request = MagicMock()
        raw_request.headers = {
            "traceparent": f"00-{remote_trace_id}-b7ad6b7169203331-01",
        }
        ctx.raw_request = raw_request

        captured: dict = {}

        def fake_encode(**kwargs):
            captured.update(kwargs)

            async def _empty():
                if False:  # pragma: no cover
                    yield

            return _empty()

        async with mixin.maybe_tracing(ctx, io_processor):  # noqa: SIM117
            async with mixin.maybe_start_span_async(
                ctx.entrypoint_tracer,
                mixin.span_entrypoint_engine_call,
                context=ctx.request_span_context,
            ) as span:
                if span is not None:
                    ctx.trace_headers = mixin._get_span_trace_headers(span)
                _ = fake_encode(trace_headers=ctx.trace_headers)

        assert remote_trace_id in captured["trace_headers"]["traceparent"]

    @pytest.mark.asyncio
    async def test_span_none_tracer_is_noop_in_composition(
        self, mixin, exporter, ctx, io_processor
    ):
        """A ``None`` tracer in the composition must not crash."""
        with mixin.maybe_start_span(None, "op") as span:
            assert span is None

        async with mixin.maybe_start_span_async(None, "op") as span_async:
            assert span_async is None

        assert exporter.get_finished_spans() == ()
