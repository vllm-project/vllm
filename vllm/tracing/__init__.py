# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable
from typing import Any, TypeAlias

# Import the implementation details
from .otel import (
    SpanKind,
    extract_trace_context,
    init_otel_tracer,
    init_otel_worker_tracer,
    instrument_otel,
    is_otel_available,
    manual_instrument_otel,
    otel_import_error_traceback,
)
from .utils import (
    SpanAttributes,
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)

__all__ = [
    "instrument",
    "instrument_manual",
    "init_tracer",
    "maybe_init_worker_tracer",
    "is_tracing_available",
    "SpanAttributes",
    "SpanKind",
    "extract_trace_context",
    "extract_trace_headers",
    "log_tracing_disabled_warning",
    "contains_trace_headers",
    "otel_import_error_traceback",
]

BackendAvailableFunc: TypeAlias = Callable[[], bool]
InstrumentFunc: TypeAlias = Callable[..., Any]
InstrumentManualFunc: TypeAlias = Callable[..., Any]
InitTracerFunc: TypeAlias = Callable[..., Any]
InitWorkerTracerFunc: TypeAlias = Callable[..., Any]
_REGISTERED_TRACING_BACKENDS: dict[
    str,
    tuple[
        BackendAvailableFunc,
        InitTracerFunc,
        InitWorkerTracerFunc,
        InstrumentFunc,
        InstrumentManualFunc,
    ],
] = {
    "otel": (
        is_otel_available,
        init_otel_tracer,
        init_otel_worker_tracer,
        instrument_otel,
        manual_instrument_otel,
    ),
}


def init_tracer(
    instrumenting_module_name: str,
    otlp_traces_endpoint: str,
    extra_attributes: dict[str, str] | None = None,
):
    is_available, init_tracer_fn, _, _, _ = _REGISTERED_TRACING_BACKENDS["otel"]
    if is_available():
        return init_tracer_fn(
            instrumenting_module_name, otlp_traces_endpoint, extra_attributes
        )


def maybe_init_worker_tracer(
    instrumenting_module_name: str,
    process_kind: str,
    process_name: str,
):
    is_available, _, init_worker_tracer_fn, _, _ = _REGISTERED_TRACING_BACKENDS["otel"]
    if is_available():
        return init_worker_tracer_fn(
            instrumenting_module_name, process_kind, process_name
        )


def instrument(
    obj: Callable | None = None,
    *,
    span_name: str = "",
    attributes: dict[str, str] | None = None,
    record_exception: bool = True,
):
    """
    Generic decorator to instrument functions.
    """
    if obj is None:
        return functools.partial(
            instrument,
            span_name=span_name,
            attributes=attributes,
            record_exception=record_exception,
        )

    # Dispatch to OTel (and potentially others later)
    is_available, _, _, otel_instrument, _ = _REGISTERED_TRACING_BACKENDS["otel"]
    if is_available():
        return otel_instrument(
            func=obj,
            span_name=span_name,
            attributes=attributes,
            record_exception=record_exception,
        )
    else:
        return obj


def instrument_manual(
    span_name: str,
    start_time: int,
    end_time: int | None = None,
    attributes: dict[str, Any] | None = None,
    context: Any = None,
    kind: Any = None,
    error: BaseException | None = None,
):
    """Manually create a span with explicit timestamps.

    Args:
        span_name: Name of the span to create.
        start_time: Start time in nanoseconds since epoch.
        end_time: Optional end time in nanoseconds. If None, ends immediately.
        attributes: Optional dict of span attributes.
        context: Optional trace context (e.g., from extract_trace_context).
        kind: Optional SpanKind (e.g., SpanKind.SERVER).
    """
    is_available, _, _, _, manual_instrument_fn = _REGISTERED_TRACING_BACKENDS["otel"]
    if is_available():
        return manual_instrument_fn(
            span_name, start_time, end_time, attributes, context, kind, error
        )
    else:
        return None


def is_tracing_available() -> bool:
    """
    Returns True if any tracing backend (OTel, Profiler, etc.) is available.
    Use this to guard expensive tracing logic in the main code.
    """
    check_available = [
        is_available
        for is_available, _, _, _, _ in _REGISTERED_TRACING_BACKENDS.values()
    ]
    return any(check_available)
