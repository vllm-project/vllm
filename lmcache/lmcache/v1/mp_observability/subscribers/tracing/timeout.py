# SPDX-License-Identifier: Apache-2.0

"""OTel span for timeout errors.

Records each ``TIMEOUT_RAISED`` as a zero-duration ``timeout`` span with an
``exception`` event (type/message/stacktrace) and ERROR status, nested under
the request span when ``session_id`` matches an open one.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import SpanRegistry

logger = init_logger(__name__)

try:
    # Third Party
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    _tracer = trace.get_tracer("lmcache_mp.timeout")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class TimeoutTracingSubscriber(EventSubscriber):
    """Records each ``TIMEOUT_RAISED`` event as an OTel span.

    Args:
        registry: Shared span registry for finding the request span to nest
            under; a private one is used when omitted (spans are then roots).
    """

    def __init__(self, registry: SpanRegistry | None = None) -> None:
        self._registry = registry if registry is not None else SpanRegistry()

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {EventType.TIMEOUT_RAISED: self._on_timeout}

    def _on_timeout(self, event: Event) -> None:
        if not _HAS_OTEL:
            return
        parent_ctx = (
            self._registry.get_context(event.session_id, "request")
            if event.session_id
            else None
        )
        start_ns = int(event.timestamp * 1e9)
        message = str(event.metadata.get("message", ""))
        exception_type = str(event.metadata.get("exception_type", "TimeoutError"))

        span = _tracer.start_span("timeout", context=parent_ctx, start_time=start_ns)
        span.set_attribute("session_id", event.session_id)
        span.add_event(
            "exception",
            attributes={
                "exception.type": exception_type,
                "exception.message": message,
                "exception.stacktrace": str(event.metadata.get("stacktrace", "")),
            },
            timestamp=start_ns,
        )
        span.set_status(Status(StatusCode.ERROR, message))
        span.end(end_time=start_ns)
