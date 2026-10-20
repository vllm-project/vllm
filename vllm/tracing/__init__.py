# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

# Import the implementation details
from .otel import (
    SpanKind,
    StatusCode,
    extract_trace_context,
    init_otel_tracer,
    init_otel_worker_tracer,
    instrument_otel,
    is_otel_available,
    manual_instrument_otel,
    otel_import_error_traceback,
)
from .otel import start_span_otel as start_span
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
    "StatusCode",
    "start_span",
    "RequestTrace",
    "get_request_trace",
    "use_request_trace",
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


class RequestTrace:
    """Frontend ownership of engine spans until final response processing."""

    def __init__(self):
        self.spans: list[Span] = []
        self.closed = False
        self.failed = False
        self.tool_attributes: dict[str, str | int | bool] | None = None

    def capture_tool_request(
        self,
        request: "ChatCompletionRequest | AnthropicMessagesRequest",
        api_surface: Literal["chat_completions", "messages"],
    ) -> None:
        """Keep only bounded native request metadata, before parser mutation."""
        if not any(
            field in request.model_fields_set
            and getattr(request, field, None) is not None
            for field in ("tools", "tool_choice", "parallel_tool_calls")
        ):
            return
        choice = request.tool_choice
        choice_type = (
            choice if isinstance(choice, str) else getattr(choice, "type", "unset")
        )
        choice_label = {"any": "required", "tool": "named", "function": "named"}.get(
            choice_type, choice_type
        )
        if choice_label not in ("unset", "auto", "none", "required", "named"):
            choice_label = "other"
        self.tool_attributes = {
            SpanAttributes.TOOL_CALLING_PRESENT: True,
            SpanAttributes.TOOL_CALLING_TOOL_COUNT: len(request.tools or ()),
            SpanAttributes.TOOL_CALLING_TOOL_CHOICE: choice_label,
            SpanAttributes.TOOL_CALLING_API_SURFACE: api_surface,
            SpanAttributes.TOOL_CALLING_RESPONSE_HAS_TOOL_CALLS: False,
            SpanAttributes.TOOL_CALLING_TOOL_CALLS_COUNT: 0,
            SpanAttributes.TOOL_CALLING_PARSER_STATUS: "interrupted",
        }

    def record_tool_response(
        self,
        *,
        tool_calls_count: int = 0,
        parse_failed: bool = False,
        completed: bool = True,
    ) -> None:
        if self.tool_attributes is None:
            return
        self.failed |= parse_failed
        # Messages finalization must not hide a failure surfaced by Chat parsing.
        if (
            parse_failed
            or self.tool_attributes[SpanAttributes.TOOL_CALLING_PARSER_STATUS]
            == "parse_failure"
        ):
            status = "parse_failure"
        elif not completed:
            status = "interrupted"
        else:
            status = "success" if tool_calls_count else "no_tool_calls"
        self.tool_attributes.update(
            {
                SpanAttributes.TOOL_CALLING_RESPONSE_HAS_TOOL_CALLS: tool_calls_count
                > 0,
                SpanAttributes.TOOL_CALLING_TOOL_CALLS_COUNT: tool_calls_count,
                SpanAttributes.TOOL_CALLING_PARSER_STATUS: status,
            }
        )

    def end(self, *, error: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        if (
            self.tool_attributes is not None
            and (error or self.failed)
            and self.tool_attributes[SpanAttributes.TOOL_CALLING_PARSER_STATUS]
            != "parse_failure"
        ):
            self.tool_attributes[SpanAttributes.TOOL_CALLING_PARSER_STATUS] = (
                "interrupted"
            )
        for span in self.spans:
            if span.is_recording():
                if self.tool_attributes is not None:
                    span.set_attributes(self.tool_attributes)
                if error or self.failed:
                    span.set_status(StatusCode.ERROR)
            span.end()
        self.spans.clear()


_REQUEST_TRACE: ContextVar[RequestTrace | None] = ContextVar(
    "vllm_request_trace", default=None
)


def get_request_trace() -> RequestTrace | None:
    owner = _REQUEST_TRACE.get()
    return owner if owner is not None and not owner.closed else None


@contextmanager
def use_request_trace(owner: RequestTrace):
    token = _REQUEST_TRACE.set(owner)
    try:
        yield
    finally:
        _REQUEST_TRACE.reset(token)


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
