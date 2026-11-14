# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Final, TypeAlias

# Import the implementation details
from .otel import (
    SpanKind,
    end_otel_request_span,
    extract_trace_context,
    init_otel_tracer,
    init_otel_worker_tracer,
    instrument_otel,
    is_otel_available,
    manual_instrument_otel,
    otel_import_error_traceback,
    otel_span,
    start_otel_request_span,
    use_otel_span,
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
    "REQUEST_SPAN_NAME",
    "RequestSpan",
    "start_request_span",
    "end_request_span",
    "span",
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
    """Generic decorator to instrument functions."""
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
            span_name, start_time, end_time, attributes, context, kind
        )
    else:
        return None


def is_tracing_available() -> bool:
    """Returns True if any tracing backend (OTel, Profiler, etc.) is available.
    Use this to guard expensive tracing logic in the main code.
    """
    check_available = [
        is_available
        for is_available, _, _, _, _ in _REGISTERED_TRACING_BACKENDS.values()
    ]
    return any(check_available)


REQUEST_SPAN_NAME: Final[str] = "llm_request"


class RequestSpan:
    """The `llm_request` span of one request, opened when it arrives.

    Opening it early lets the work done before the request reaches the engine,
    such as rendering and loading `prompt_embeds`, be recorded as its
    children. The output processor ends it when the request finishes or is
    aborted; call `abandon` if the request never gets that far.
    """

    def __init__(
        self, span: Any = None, trace_headers: Mapping[str, str] | None = None
    ):
        self._span = span
        self.trace_headers = trace_headers
        """Headers to pass to the engine, naming this span when it is open."""

    def activate(self) -> AbstractContextManager[None]:
        """Make this span current, so stages run within become its children."""
        return nullcontext() if self._span is None else use_otel_span(self._span)

    def abandon(self, error: BaseException | str | None = None) -> None:
        """End the span of a request that will not reach the output processor."""
        if self._span is not None:
            end_otel_request_span(self.trace_headers, error=error)

    @contextmanager
    def guard(self) -> Iterator[None]:
        """`activate`, abandoning the span if an exception escapes."""
        try:
            with self.activate():
                yield
        except BaseException as exc:
            self.abandon(exc)
            raise


def start_request_span(trace_headers: Mapping[str, str] | None) -> RequestSpan:
    """Open the `llm_request` span for a request with these trace headers.

    When tracing is not configured, or the request is not sampled, the
    returned handle records nothing and passes `trace_headers` through.
    """
    if (opened := start_otel_request_span(REQUEST_SPAN_NAME, trace_headers)) is None:
        return RequestSpan(trace_headers=trace_headers)
    return RequestSpan(*opened)


def end_request_span(
    trace_headers: Mapping[str, str] | None,
    attributes: Mapping[str, Any] | None = None,
    error: BaseException | str | None = None,
) -> bool:
    """End the `llm_request` span opened for these trace headers.

    Returns `False` when no span was opened for them, in which case the caller
    should record the request some other way.
    """
    return end_otel_request_span(trace_headers, attributes, error)


def span(
    span_name: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    trace_headers: Mapping[str, str] | None = None,
) -> AbstractContextManager[Any]:
    """Record a stage of a traced request as a child span.

    The parent is the span named by `trace_headers` when given, otherwise the
    span in scope. Outside a recording request this records nothing and yields
    `None`. Attributes must never carry request content.
    """
    return otel_span(span_name, attributes, trace_headers)
