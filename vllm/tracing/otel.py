# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import functools
import inspect
import os
import traceback
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from typing import Any

from vllm.logger import init_logger
from vllm.tracing.utils import TRACE_HEADERS, LoadingSpanAttributes

logger = init_logger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.context.context import Context
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as OTLPGrpcExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as OTLPHttpExporter,
    )
    from opentelemetry.propagate import inject
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import (
        SpanKind,  # noqa: F401
        Tracer,
        set_tracer_provider,
    )
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _IS_OTEL_AVAILABLE = True
    otel_import_error_traceback = None
except ImportError:
    _IS_OTEL_AVAILABLE = False
    otel_import_error_traceback = traceback.format_exc()
    trace = None  # type: ignore
    Context = Any  # type: ignore
    Tracer = Any  # type: ignore
    inject = None  # type: ignore
    Resource = None  # type: ignore
    SpanKind = Any  # type: ignore

_GLOBAL_TRACER_PROVIDER: Any = None


def is_otel_available() -> bool:
    return _IS_OTEL_AVAILABLE


def _get_tracer(name: str = __name__) -> Tracer:
    if _GLOBAL_TRACER_PROVIDER is not None:
        return _GLOBAL_TRACER_PROVIDER.get_tracer(name)
    return trace.get_tracer(name)


def init_otel_tracer(
    instrumenting_module_name: str,
    otlp_traces_endpoint: str,
    extra_attributes: dict[str, str] | None = None,
) -> Tracer:
    """Initializes the OpenTelemetry tracer provider."""
    if not _IS_OTEL_AVAILABLE:
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}"
        )

    # Store the endpoint in environment so child processes can inherit it
    os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = otlp_traces_endpoint

    resource_attrs = {}
    resource_attrs["service.name"] = instrumenting_module_name
    resource_attrs["vllm.instrumenting_module_name"] = instrumenting_module_name
    resource_attrs["vllm.process_id"] = str(os.getpid())
    if extra_attributes:
        resource_attrs.update(extra_attributes)
    resource = Resource.create(resource_attrs)

    trace_provider = TracerProvider(resource=resource)
    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    with suppress(Exception):
        set_tracer_provider(trace_provider)
    global _GLOBAL_TRACER_PROVIDER
    _GLOBAL_TRACER_PROVIDER = trace_provider

    atexit.register(trace_provider.shutdown)

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer


def get_span_exporter(endpoint: str):
    # Normalize endpoint: strip grpc:// prefix if present for OTLPGrpcExporter
    clean_endpoint = endpoint
    if clean_endpoint.startswith("grpc://"):
        clean_endpoint = clean_endpoint[len("grpc://") :]

    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        exporter = OTLPGrpcExporter(endpoint=clean_endpoint, insecure=True)
    elif protocol == "http/protobuf":
        exporter = OTLPHttpExporter(endpoint=clean_endpoint)
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")
    return exporter


def init_otel_worker_tracer(
    instrumenting_module_name: str,
    process_kind: str,
    process_name: str,
) -> Tracer:
    """Backend-specific initialization for OpenTelemetry in a worker process."""
    # Initialize the tracer if an OTLP endpoint is configured.
    # The endpoint is propagated via environment variable from the main process.
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not otlp_endpoint:
        return None

    extra_attrs = {
        "vllm.process_kind": process_kind,
        "vllm.process_name": process_name,
    }

    return init_otel_tracer(instrumenting_module_name, otlp_endpoint, extra_attrs)


def extract_trace_context(headers: Mapping[str, str] | None) -> Context | None:
    """Extracts context from HTTP headers."""
    if _IS_OTEL_AVAILABLE and headers:
        return TraceContextTextMapPropagator().extract(headers)
    return None


def instrument_otel(func, span_name, attributes, record_exception):
    """Internal wrapper logic for sync and async functions."""
    # Pre-calculate static code attributes once (these don't change)
    code_attrs = {
        LoadingSpanAttributes.CODE_FUNCTION: func.__qualname__,
        LoadingSpanAttributes.CODE_NAMESPACE: func.__module__,
        LoadingSpanAttributes.CODE_FILEPATH: func.__code__.co_filename,
        LoadingSpanAttributes.CODE_LINENO: str(func.__code__.co_firstlineno),
    }
    if attributes:
        code_attrs.update(attributes)

    final_span_name = span_name or func.__qualname__
    module_name = func.__module__

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        tracer = _get_tracer(module_name)
        ctx = _get_smart_context()
        with (
            tracer.start_as_current_span(
                final_span_name,
                context=ctx,
                attributes=code_attrs,
                record_exception=record_exception,
            ),
            propagate_trace_to_env(),
        ):
            return await func(*args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        tracer = _get_tracer(module_name)
        ctx = _get_smart_context()
        with (
            tracer.start_as_current_span(
                final_span_name,
                context=ctx,
                attributes=code_attrs,
                record_exception=record_exception,
            ),
            propagate_trace_to_env(),
        ):
            return func(*args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


def manual_instrument_otel(
    span_name: str,
    start_time: int,
    end_time: int | None = None,
    attributes: dict[str, Any] | None = None,
    context: Context | None = None,
    kind: Any = None,  # SpanKind, but typed as Any for when OTEL unavailable
):
    """Manually create and end a span with explicit timestamps."""
    if not _IS_OTEL_AVAILABLE:
        return

    tracer = _get_tracer(__name__)
    # Use provided context, or fall back to smart context detection
    ctx = context if context is not None else _get_smart_context()

    span_kwargs: dict[str, Any] = {
        "name": span_name,
        "context": ctx,
        "start_time": start_time,
    }
    if kind is not None:
        span_kwargs["kind"] = kind

    span = tracer.start_span(**span_kwargs)
    if attributes:
        span.set_attributes(attributes)
    if end_time is not None:
        span.end(end_time=end_time)
    else:
        span.end()


def start_request_span_otel(
    span_name: str,
    start_time: int,
    attributes: dict[str, Any] | None = None,
    context: Context | None = None,
    kind: Any = None,
) -> tuple[Any, dict[str, str] | None]:
    """Start an OpenTelemetry span and inject its context into a carrier dict.

    Returns:
        (span, trace_headers): The active span object and a dict of W3C
        trace headers (e.g. {'traceparent': ...}) representing this span's
        context for propagation to downstream workers/subsystems.
    """
    if not _IS_OTEL_AVAILABLE:
        return None, None

    tracer = _get_tracer(__name__)
    ctx = context if context is not None else _get_smart_context()

    span_kwargs: dict[str, Any] = {
        "name": span_name,
        "context": ctx,
        "start_time": start_time,
    }
    if kind is not None:
        span_kwargs["kind"] = kind

    span = tracer.start_span(**span_kwargs)
    if attributes:
        span.set_attributes(attributes)

    # Inject the new span's context into W3C trace headers
    carrier: dict[str, str] = {}
    span_ctx = trace.set_span_in_context(span)
    TraceContextTextMapPropagator().inject(carrier, context=span_ctx)

    return span, carrier


def _get_smart_context() -> Context | None:
    """Determines the parent context.
    1. If a Span is already active in this process, use it.
    2. If not, extract from os.environ, handling the case-sensitivity mismatch.
    """
    current_span = trace.get_current_span()
    if current_span.get_span_context().is_valid:
        return None

    carrier = {}

    if tp := os.environ.get("traceparent", os.environ.get("TRACEPARENT")):  # noqa: SIM112
        carrier["traceparent"] = tp

    if ts := os.environ.get("tracestate", os.environ.get("TRACESTATE")):  # noqa: SIM112
        carrier["tracestate"] = ts

    if not carrier:
        carrier = dict(os.environ)

    return TraceContextTextMapPropagator().extract(carrier)


@contextmanager
def propagate_trace_to_env():
    """Temporarily injects the current OTel context into os.environ.
    This ensures that any subprocesses (like vLLM workers) spawned
    within this context inherit the correct traceparent.
    """
    if not _IS_OTEL_AVAILABLE:
        yield
        return

    # Capture original state of relevant keys
    original_state = {k: os.environ.get(k) for k in TRACE_HEADERS}

    try:
        # inject() writes 'traceparent' and 'tracestate' to os.environ
        inject(os.environ)
        yield

    finally:
        # Restore original environment
        for key, original_value in original_state.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
