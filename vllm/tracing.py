# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import functools
import inspect
import os
import traceback
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

from vllm.logger import init_logger
from vllm.utils.func_utils import run_once

TRACE_HEADERS = ["traceparent", "tracestate"]

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


def is_otel_available() -> bool:
    return _IS_OTEL_AVAILABLE


def init_tracer(
    instrumenting_module_name: str,
    otlp_traces_endpoint: str,
    extra_attributes: dict[str, str] | None = None,
) -> Tracer | None:
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
    resource_attrs["vllm.instrumenting_module_name"] = instrumenting_module_name
    resource_attrs["vllm.process_id"] = str(os.getpid())
    if extra_attributes:
        resource_attrs.update(extra_attributes)
    resource = Resource.create(resource_attrs)

    trace_provider = TracerProvider(resource=resource)
    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    set_tracer_provider(trace_provider)

    atexit.register(trace_provider.shutdown)

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer


def get_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        exporter = OTLPGrpcExporter(endpoint=endpoint, insecure=True)
    elif protocol == "http/protobuf":
        exporter = OTLPHttpExporter(endpoint=endpoint)
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")
    return exporter


_worker_tracer_initialized = False


def maybe_init_worker_tracer(
    instrumenting_module_name: str = "vllm.worker",
    process_kind: str = "worker",
    process_name: str = "",
) -> Tracer | None:
    """Initialize tracer in sub-processes (workers) if context exists."""
    global _worker_tracer_initialized

    if _worker_tracer_initialized or not _IS_OTEL_AVAILABLE:
        return None

    # Initialize worker tracer if an OTLP endpoint is configured.
    # The endpoint is propagated via environment variable from the main process.
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")

    if not otlp_endpoint:
        return None

    extra_attrs = {}
    extra_attrs["vllm.process_kind"] = process_kind
    extra_attrs["vllm.process_name"] = process_name

    _worker_tracer_initialized = True
    return init_tracer(instrumenting_module_name, otlp_endpoint, extra_attrs)


def extract_trace_context(headers: Mapping[str, str] | None) -> Context | None:
    """Extracts context from HTTP headers."""
    if _IS_OTEL_AVAILABLE and headers:
        return TraceContextTextMapPropagator().extract(headers)
    return None


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    # Attribute names added until they are added to the semantic conventions:
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    # Time taken in the forward pass for this across all workers
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"
    # Time taken in the model execute function. This will include model
    # forward, block/sync across workers, cpu-gpu sync time and sampling time.
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


class LoadingSpanAttributes:
    """Custom attributes for code-level tracing."""

    CODE_NAMESPACE = "code.namespace"
    CODE_FUNCTION = "code.function"
    CODE_FILEPATH = "code.filepath"
    CODE_LINENO = "code.lineno"


def instrument(
    obj: Callable | None = None,
    *,
    span_name: str = "",
    attributes: dict[str, str] | None = None,
    record_exception: bool = True,
):
    """
    Decorator to instrument functions with OTel spans.

    Usage:
        @instrument
        def my_func(): ...

        @instrument(span_name="custom_name")
        def my_func(): ...
    """
    # If OTel is not available, return the original object immediately.
    if not _IS_OTEL_AVAILABLE:
        if obj is not None:
            return obj
        return lambda x: x

    # Handle factory usage: @instrument(span_name="foo")
    if obj is None:
        return functools.partial(
            instrument,
            span_name=span_name,
            attributes=attributes,
            record_exception=record_exception,
        )

    return _instrument_function(obj, span_name, attributes, record_exception)


def _instrument_function(func, span_name_fmt, user_attrs, record_exception):
    """Internal wrapper logic for sync and async functions."""

    # Pre-calculate static code attributes once (these don't change)
    code_attrs = {
        LoadingSpanAttributes.CODE_FUNCTION: func.__qualname__,
        LoadingSpanAttributes.CODE_NAMESPACE: func.__module__,
        LoadingSpanAttributes.CODE_FILEPATH: func.__code__.co_filename,
        LoadingSpanAttributes.CODE_LINENO: str(func.__code__.co_firstlineno),
    }
    if user_attrs:
        code_attrs.update(user_attrs)

    final_span_name = span_name_fmt or func.__qualname__
    module_name = func.__module__

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        tracer = trace.get_tracer(module_name)
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
        tracer = trace.get_tracer(module_name)
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


def _get_smart_context() -> Context | None:
    """
    Determines the parent context.
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
    """
    Temporarily injects the current OTel context into os.environ.
    This ensures that any subprocesses (like vLLM workers) spawned
    within this context inherit the correct traceparent.
    """
    if not _IS_OTEL_AVAILABLE:
        yield
        return

    # Capture original state of relevant keys
    keys_to_manage = ["traceparent", "tracestate"]
    original_state = {k: os.environ.get(k) for k in keys_to_manage}

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


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")
