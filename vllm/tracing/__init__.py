# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable

# Import the implementation details
from .otel import (
    extract_trace_context,
    init_otel_worker,
    init_tracer,
    instrument_otel,
    is_otel_available,
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
    "init_tracer",
    "is_tracing_available",
    "is_otel_available",
    "otel_import_error_traceback",
    "SpanAttributes",
    "extract_trace_context",
    "extract_trace_headers",
    "log_tracing_disabled_warning",
    "contains_trace_headers",
]


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
    return instrument_otel(
        func=obj,
        span_name=span_name,
        attributes=attributes,
        record_exception=record_exception,
    )


def is_tracing_available() -> bool:
    """
    Returns True if any tracing backend (OTel, Profiler, etc.) is available.
    Use this to guard expensive tracing logic in the main code.
    """

    return is_otel_available()


_WORKER_TRACING_INITIALIZED = False


def maybe_init_worker_tracer(
    instrumenting_module_name: str = "vllm.worker",
    process_kind: str = "worker",
    process_name: str = "",
):
    """
    Generic entry point to initialize tracing in sub-processes.
    Dispatches to OTel, Profiler, etc.
    """
    global _WORKER_TRACING_INITIALIZED

    if _WORKER_TRACING_INITIALIZED:
        return

    init_otel_worker(instrumenting_module_name, process_kind, process_name)

    _WORKER_TRACING_INITIALIZED = True
