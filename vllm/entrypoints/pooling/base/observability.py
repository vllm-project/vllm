# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING

from fastapi import Request

from vllm.config import VllmConfig
from vllm.renderers import BaseRenderer
from vllm.tracing import (
    SpanAttributes,
    contains_trace_headers,
    extract_trace_headers,
)
from vllm.tracing.otel import _IS_OTEL_AVAILABLE, Span, Tracer, init_otel_trace_provider

from ..base.io_processor import PoolingIOProcessor
from ..typing import PoolingServeContext

TO_S = 1000_000_000
TO_MS = 1000_000


class _ObservabilityMixin:
    vllm_config: VllmConfig
    renderer: BaseRenderer

    scope_request: str = "request"
    span_request = "request"

    scope_entrypoint: str = "entrypoint"
    span_entrypoint_preprocessing = "preprocessing"
    span_entrypoint_engine_call = "engine_call"
    span_entrypoint_postprocessing = "postprocessing"

    def _init_observability(self):
        observability_config = self.vllm_config.observability_config
        self.is_tracing_request = observability_config.is_tracing_enabled

        if not self.is_tracing_request:
            return

        # Will raise ValueError if OpenTelemetry is not available.
        self.trace_provider = init_otel_trace_provider(
            observability_config.otlp_traces_endpoint_str
        )

    @asynccontextmanager
    async def maybe_tracing(
        self,
        ctx: PoolingServeContext,
        io_processor: PoolingIOProcessor,
        raw_request: Request | None = None,
    ):
        raw_trace_headers = await self._get_trace_headers(
            {} if raw_request is None else raw_request.headers
        )

        ctx.trace_headers = raw_trace_headers
        yield
        return

    @contextmanager
    def maybe_start_span(self, tracer: Tracer | None, *args, **kwargs):
        yield None
        return

    @asynccontextmanager
    async def maybe_start_span_async(self, tracer: Tracer | None, *args, **kwargs):
        yield None
        return

    async def _get_trace_headers(
        self, headers: Mapping[str, str]
    ) -> Mapping[str, str] | None:
        if not contains_trace_headers(headers):
            return None
        return extract_trace_headers(headers)

    def _get_span_trace_headers(self, span: Span) -> Mapping[str, str]:
        return {}


if TYPE_CHECKING or not _IS_OTEL_AVAILABLE:
    ObservabilityMixin = _ObservabilityMixin

else:
    from opentelemetry.sdk.trace import Span
    from opentelemetry.trace import (
        NonRecordingSpan,
        Status,
        StatusCode,
        Tracer,
        set_span_in_context,
    )

    class _ObservabilityMixinWithOpenTelemetry(_ObservabilityMixin):
        def __init__(self):
            super().__init__()

            if not self.is_tracing_request:
                return

            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            self.propagator = TraceContextTextMapPropagator()
            self.scope_endpoint_attributes = {
                "vllm.process_name": f"APIServer_{self.renderer.api_process_rank}",
            }

        @asynccontextmanager
        async def maybe_tracing(
            self,
            ctx: PoolingServeContext,
            io_processor: PoolingIOProcessor,
            raw_request: Request | None = None,
        ):
            raw_trace_headers = await self._get_trace_headers(
                {} if raw_request is None else raw_request.headers
            )

            if not self.is_tracing_request:
                ctx.trace_headers = raw_trace_headers
                yield
                return

            request_tracer = self.trace_provider.get_tracer(self.scope_request)
            trace_context = self.propagator.extract(
                raw_trace_headers if raw_trace_headers is not None else {}
            )
            request_span = request_tracer.start_span(
                self.span_request, context=trace_context, start_time=ctx.arrival_time_ns
            )
            request_span_context = get_span_context(request_span)
            ctx.request_span_context = request_span_context

            ctx.entrypoint_tracer = self.trace_provider.get_tracer(
                self.scope_entrypoint, attributes=self.scope_endpoint_attributes
            )

            try:
                yield

                request_span.set_attributes(
                    self._get_request_span_attributes(ctx, io_processor)
                )

            except Exception as e:
                request_span.set_status(get_status_error())
                request_span.record_exception(e)
                raise e
            finally:
                end_time = time.time_ns()
                request_span.end(end_time)

        @contextmanager
        def maybe_start_span(self, tracer: Tracer | None, *args, **kwargs):
            if tracer is None:
                yield None
                return

            start_time = time.time_ns()
            attributes = kwargs.pop("attributes", {})
            attributes[SpanAttributes.GEN_AI_PROCESS_ID] = str(os.getpid())
            attributes[SpanAttributes.GEN_AI_THREADING_ID] = str(threading.get_ident())

            span = tracer.start_span(
                *args, start_time=start_time, attributes=attributes, **kwargs
            )
            try:
                yield span
            except Exception as e:
                span.set_status(get_status_error())
                span.record_exception(e)
                raise
            finally:
                end_time = time.time_ns()
                span.end(end_time)

        @asynccontextmanager
        async def maybe_start_span_async(self, tracer: Tracer | None, *args, **kwargs):
            if tracer is None:
                yield None
                return

            start_time = time.time_ns()
            attributes = kwargs.pop("attributes", {})
            attributes[SpanAttributes.GEN_AI_PROCESS_ID] = str(os.getpid())
            attributes[SpanAttributes.GEN_AI_THREADING_ID] = str(threading.get_ident())

            span = tracer.start_span(
                *args, start_time=start_time, attributes=attributes, **kwargs
            )
            try:
                yield span
            except Exception as e:
                span.set_status(get_status_error())
                span.record_exception(e)
                raise
            finally:
                end_time = time.time_ns()
                span.end(end_time)

        def _get_request_span_attributes(
            self,
            ctx: PoolingServeContext,
            io_processor: PoolingIOProcessor,
        ) -> Mapping[str, str]:
            return {
                SpanAttributes.GEN_AI_REQUEST_ID: ctx.request_id,
                SpanAttributes.GEN_AI_POOLING_IO_PROCESSOR: (
                    io_processor.__class__.__name__
                ),
                SpanAttributes.GEN_AI_POOLING_REQUEST_CLASS: (
                    ctx.request.__class__.__name__
                ),
                SpanAttributes.GEN_AI_LATENCY_PREPROCESSING: (
                    ctx.preprocessing_finished - ctx.arrival_time_ns
                )
                / TO_MS,
                SpanAttributes.GEN_AI_LATENCY_ENGINE_CALL: (
                    ctx.engine_call_finished - ctx.preprocessing_finished
                )
                / TO_MS,
                SpanAttributes.GEN_AI_LATENCY_POSTPROCESSING: (
                    ctx.postprocessing_finished - ctx.engine_call_finished
                )
                / TO_MS,
            }

        def _get_span_trace_headers(self, span: Span) -> Mapping[str, str]:
            trace_headers: Mapping[str, str] = {}
            span_context = get_span_context(span)
            self.propagator.inject(trace_headers, context=span_context)
            return trace_headers

    def get_span_context(span: Span):
        return set_span_in_context(NonRecordingSpan(span.get_span_context()))

    def get_status_error():
        return Status(StatusCode.ERROR)

    ObservabilityMixin = _ObservabilityMixinWithOpenTelemetry
