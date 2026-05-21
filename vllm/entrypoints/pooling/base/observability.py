# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from abc import ABC
from collections.abc import Mapping
from contextlib import asynccontextmanager

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.tracing import (
    SpanAttributes,
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.tracing.otel import (
    TO_MS,
    get_span_context,
    get_status_error,
    init_otel_trace_provider,
    maybe_get_links,
    maybe_start_span,
    maybe_start_span_async,
)


class PoolingServingObservabilityMixin(ABC):
    engine_client: EngineClient

    _maybe_get_links = staticmethod(maybe_get_links)
    _maybe_start_span = staticmethod(maybe_start_span)
    _maybe_start_span_async = staticmethod(maybe_start_span_async)
    _get_span_context = staticmethod(get_span_context)

    def __init__(self, is_tracing_enabled):
        self.is_tracing_enabled = is_tracing_enabled
        if self.is_tracing_enabled:
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            assert otlp_endpoint is not None

            api_process_count = getattr(self.engine_client, "api_process_count", 1)
            api_process_rank = getattr(self.engine_client, "_api_process_rank", 0)
            process_title = f"APIServer_{api_process_rank}"

            self.propagator = TraceContextTextMapPropagator()

            self.trace_provider = init_otel_trace_provider(otlp_endpoint)
            self.scope_request = "vllm.request"
            self.scope_endpoint = "vllm.entrypoint"
            self.scope_endpoint_attributes = {
                "vllm.process_kind": "entrypoint",
                "vllm.process_count": f"{api_process_count}",
                "vllm.process_name": process_title,
            }

            self.span_request = "vllm.request"
            self.span_entrypoint_preprocessing = "vllm.entrypoint.preprocessing"
            self.span_entrypoint_engine_call = "vllm.entrypoint.engine_call"
            self.span_entrypoint_postprocessing = "vllm.entrypoint.postprocessing"

    async def _get_trace_headers(
        self,
        raw_request: Request | None = None,
    ) -> Mapping[str, str] | None:
        if raw_request is None:
            return None

        headers = raw_request.headers

        if self.is_tracing_enabled:
            return extract_trace_headers(headers)

        if contains_trace_headers(headers):
            log_tracing_disabled_warning()

        return None

    @asynccontextmanager
    async def _maybe_tracing(
        self,
        ctx: PoolingServeContext,
        io_processor: PoolingIOProcessor,
        raw_request: Request | None = None,
    ):
        raw_trace_headers = await self._get_trace_headers(raw_request)

        if not self.is_tracing_enabled:
            yield
            return

        request_tracer = self.trace_provider.get_tracer(self.scope_request)
        entrypoint_tracer = self.trace_provider.get_tracer(
            self.scope_endpoint, attributes=self.scope_endpoint_attributes
        )
        trace_context = self.propagator.extract(raw_trace_headers)

        request_span = request_tracer.start_span(
            self.span_request, context=trace_context, start_time=ctx.arrival_time
        )
        request_span_context = get_span_context(request_span)

        ctx.trace_headers = raw_trace_headers
        ctx.entrypoint_tracer = entrypoint_tracer
        ctx.request_span_context = request_span_context

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
            end_st = time.monotonic_ns()
            end_time = ctx.time_offset + end_st
            request_span.end(end_time)

    def _get_request_span_attributes(
        self,
        ctx: PoolingServeContext,
        io_processor: PoolingIOProcessor,
    ) -> Mapping[str, str]:
        return {
            SpanAttributes.GEN_AI_PROCESS_ID: str(os.getpid()),
            SpanAttributes.GEN_AI_REQUEST_ID: ctx.request_id,
            SpanAttributes.GEN_AI_POOLING_IO_PROCESSOR: io_processor.__class__.__name__,
            SpanAttributes.GEN_AI_POOLING_REQUEST_CLASS: ctx.request.__class__.__name__,
            SpanAttributes.GEN_AI_LATENCY_PREPROCESSING: (
                ctx.preprocessing_finished - ctx.arrival_time
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

    def _get_preprocessing_span_attributes(
        self, ctx: PoolingServeContext
    ) -> Mapping[str, str]:
        return dict()

    def _get_engine_call_span_attributes(
        self, ctx: PoolingServeContext
    ) -> Mapping[str, str]:
        return dict()

    def _get_postprocessing_span_attributes(
        self, ctx: PoolingServeContext
    ) -> Mapping[str, str]:
        return dict()
