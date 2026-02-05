# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable
from typing import Any, TypeAlias

# Import the implementation details
from .otel import (
    extract_trace_context,
    init_tracer,
    instrument_otel,
    is_otel_available,
    manual_instrument_otel,
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
    "SpanAttributes",
    "extract_trace_context",
    "extract_trace_headers",
    "log_tracing_disabled_warning",
    "contains_trace_headers",
]

BackendAvailableFunc: TypeAlias = Callable[[], bool]
InstrumentFunc: TypeAlias = Callable[..., Any]
InstrumentManualFunc: TypeAlias = Callable[..., Any]
_REGISTERED_TRACING_BACKENDS: dict[
    str, tuple[BackendAvailableFunc, InstrumentFunc, InstrumentManualFunc]
] = {
    "otel": (is_otel_available, instrument_otel, manual_instrument_otel),
}


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
    is_otel_available, otel_instrument, _ = _REGISTERED_TRACING_BACKENDS["otel"]
    if is_otel_available():
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
    attributes: dict[str, str] | None = None,
):
    is_otel_available, _, manual_instrument_otel = _REGISTERED_TRACING_BACKENDS["otel"]
    if is_otel_available():
        return manual_instrument_otel(span_name, start_time, end_time, attributes)
    else:
        return None


def is_tracing_available() -> bool:
    """
    Returns True if any tracing backend (OTel, Profiler, etc.) is available.
    Use this to guard expensive tracing logic in the main code.
    """
    check_available = [
        is_available for is_available, _, _ in _REGISTERED_TRACING_BACKENDS.values()
    ]
    return any(check_available)
