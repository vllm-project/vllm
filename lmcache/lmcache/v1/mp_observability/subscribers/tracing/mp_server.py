# SPDX-License-Identifier: Apache-2.0

"""OTel tracing subscriber for MP server operations.

Creates a root ``"request"`` span per session wrapping all child spans.
Opens at ``MP_REQUEST_START``; closes at ``MP_REQUEST_END``, deferred until
any in-flight GPU store/retrieve callbacks complete.

Accepts an optional :class:`~lmcache.v1.mp_observability.subscribers.tracing\
.span_registry.SpanRegistry` so other subscribers can nest spans under the
root or any child span — see
``docs/design/observability/request-event-span.md`` for examples.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import SpanRegistry

logger = init_logger(__name__)

try:
    # Third Party
    from opentelemetry import trace

    _tracer = trace.get_tracer("lmcache_mp.server")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class MPServerTracingSubscriber(EventSubscriber):
    """Creates OTel spans from MP server events with a per-request root span.

    Each session gets one root ``"request"`` span that nests all child spans
    (``mp.lookup_prefetch``, ``mp.retrieve``, ``mp.store``) beneath it.

    The root span is opened eagerly at ``MP_REQUEST_START`` and closed at
    ``MP_REQUEST_END``, with deferral if GPU stores are still in flight.
    """

    # Maps START event types to span names
    _SPAN_NAMES: dict[EventType, str] = {
        EventType.MP_STORE_START: "mp.store",
        EventType.MP_RETRIEVE_START: "mp.retrieve",
        EventType.MP_LOOKUP_PREFETCH_START: "mp.lookup_prefetch",
    }

    _END_TO_START: dict[EventType, EventType] = {
        EventType.MP_STORE_END: EventType.MP_STORE_START,
        EventType.MP_RETRIEVE_END: EventType.MP_RETRIEVE_START,
        EventType.MP_LOOKUP_PREFETCH_END: EventType.MP_LOOKUP_PREFETCH_START,
    }

    # Logical registry names for child spans.  When a child span is opened it
    # is also stored under this name in the shared registry so that other
    # subscribers (or future sub-span subscribers) can look up its context as
    # a parent without coupling to this class.
    _LOGICAL_NAMES: dict[EventType, str] = {
        EventType.MP_STORE_START: "store",
        EventType.MP_RETRIEVE_START: "retrieve",
        EventType.MP_LOOKUP_PREFETCH_START: "lookup_prefetch",
    }

    def __init__(self, registry: SpanRegistry | None = None) -> None:
        # Shared span registry.  When a registry is provided, other subscribers
        # that share the same instance can look up the root "request" span or
        # any child span ("store", "retrieve", "lookup_prefetch") as a parent
        # context.  If None, a private registry is created so that the class
        # remains usable without a shared registry.
        self._registry = registry if registry is not None else SpanRegistry()

        # session_id -> (span, start_event_type) for pending child spans
        self._pending: dict[str, tuple[Any, EventType]] = {}

        # session_id -> number of MP_STORE_SUBMITTED events without a
        # matching MP_STORE_END; guards against REQUEST_END racing GPU stores
        self._pending_store_count: dict[str, int] = {}

        # session_id -> number of MP_RETRIEVE_SUBMITTED events without a
        # matching MP_RETRIEVE_END; guards against REQUEST_END racing GPU retrieves
        self._pending_retrieve_count: dict[str, int] = {}

        # session_id -> REQUEST_END timestamp saved when stores/retrieves are in
        # flight; the last MP_STORE_END / MP_RETRIEVE_END uses this to close the
        # root span
        self._deferred_session_end_ts: dict[str, float] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return the event-to-callback mapping for this subscriber."""
        return {
            EventType.MP_REQUEST_START: self._on_request_start,
            EventType.MP_STORE_SUBMITTED: self._on_store_submitted,
            EventType.MP_RETRIEVE_SUBMITTED: self._on_retrieve_submitted,
            EventType.MP_REQUEST_END: self._on_session_end,
            EventType.MP_STORE_START: self._on_start,
            EventType.MP_STORE_END: self._on_end,
            EventType.MP_RETRIEVE_START: self._on_start,
            EventType.MP_RETRIEVE_END: self._on_end,
            EventType.MP_LOOKUP_PREFETCH_START: self._on_start,
            EventType.MP_LOOKUP_PREFETCH_END: self._on_end,
        }

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """End all leaked spans on bus shutdown."""
        for key, (span, _) in self._pending.items():
            try:
                span.end()
            except Exception:
                pass
        self._pending.clear()

        sessions = (
            set(self._pending_store_count)
            | set(self._pending_retrieve_count)
            | set(self._deferred_session_end_ts)
            | self._registry.all_session_ids()
        )
        for sid in sessions:
            self._registry.clear_session(sid)
        self._pending_store_count.clear()
        self._pending_retrieve_count.clear()
        self._deferred_session_end_ts.clear()

    # ------------------------------------------------------------------
    # Root span handlers
    # ------------------------------------------------------------------

    def _on_request_start(self, event: Event) -> None:
        """Create the root span eagerly at true request arrival.

        Args:
            event: ``MP_REQUEST_START`` event with ``session_id`` set.
        """
        if not _HAS_OTEL:
            return
        self._get_or_create_request_span(event.session_id, event.timestamp)

    def _on_store_submitted(self, event: Event) -> None:
        """Increment the in-flight store counter for the session.

        Called CPU-synchronously before the GPU store is enqueued, ensuring
        the counter is non-zero before ``MP_REQUEST_END`` can arrive.

        Args:
            event: ``MP_STORE_SUBMITTED`` event.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        self._pending_store_count[sid] = self._pending_store_count.get(sid, 0) + 1

    def _on_retrieve_submitted(self, event: Event) -> None:
        """Increment the in-flight retrieve counter for the session.

        Called CPU-synchronously before the GPU retrieve is enqueued, ensuring
        the counter is non-zero before ``MP_REQUEST_END`` can arrive.

        Args:
            event: ``MP_RETRIEVE_SUBMITTED`` event.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        self._pending_retrieve_count[sid] = self._pending_retrieve_count.get(sid, 0) + 1

    def _on_session_end(self, event: Event) -> None:
        """Close the root span, or defer if GPU stores/retrieves are still in flight.

        Args:
            event: ``MP_REQUEST_END`` event carrying the logical end timestamp.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        if (
            self._pending_store_count.get(sid, 0) == 0
            and self._pending_retrieve_count.get(sid, 0) == 0
        ):
            self._close_request_span(sid, event.timestamp)
        else:
            # Stores/retrieves still in flight — save timestamp; the last
            # MP_STORE_END / MP_RETRIEVE_END closes the root span
            self._deferred_session_end_ts[sid] = event.timestamp

    # ------------------------------------------------------------------
    # Child span handlers
    # ------------------------------------------------------------------

    def _on_start(self, event: Event) -> None:
        """Create a child span nested under the root span.

        Falls back to ``_get_or_create_request_span`` if ``MP_REQUEST_START`` was
        never emitted (e.g. store-only path with no lookup).

        Args:
            event: One of ``MP_STORE_START``, ``MP_RETRIEVE_START``, or
                ``MP_LOOKUP_PREFETCH_START``.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        _, root_ctx = self._get_or_create_request_span(sid, event.timestamp)

        span_name = self._SPAN_NAMES[event.event_type]
        span = _tracer.start_span(
            span_name,
            context=root_ctx,
            start_time=int(event.timestamp * 1e9),
        )
        for k, v in event.metadata.items():
            span.set_attribute(k, str(v))
        span.set_attribute("session_id", sid)

        key = f"{sid}:{event.event_type.value}"
        self._pending[key] = (span, event.event_type)

        # Also register in the shared registry so other subscribers can use
        # this span as a parent context for nested sub-spans.
        logical = self._LOGICAL_NAMES.get(event.event_type)
        if logical:
            self._registry.open(sid, logical, span, trace.set_span_in_context(span))

    def _on_end(self, event: Event) -> None:
        """Close a pending child span and handle store-count deferral.

        On ``MP_STORE_END``: decrements the store counter; if it reaches zero
        and a deferred session-end timestamp exists, closes the root span.

        Args:
            event: One of ``MP_STORE_END``, ``MP_RETRIEVE_END``, or
                ``MP_LOOKUP_PREFETCH_END``.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        start_type = self._END_TO_START[event.event_type]
        key = f"{sid}:{start_type.value}"
        entry = self._pending.pop(key, None)
        if entry is None:
            logger.debug(
                "No pending span for %s session=%s",
                event.event_type.value,
                sid,
            )
        else:
            span, _ = entry
            for k, v in event.metadata.items():
                span.set_attribute(k, str(v))
            span.end(end_time=int(event.timestamp * 1e9))

        # Remove the child span from the shared registry now that it is closed.
        logical = self._LOGICAL_NAMES.get(start_type)
        if logical:
            self._registry.pop(sid, logical)

        if event.event_type == EventType.MP_LOOKUP_PREFETCH_END:
            root_entry = self._registry.get(sid, "request")
            if root_entry is not None:
                root_span, _ = root_entry
                hit_tokens = int(event.metadata.get("hit_tokens", 0))
                requested_tokens = int(event.metadata.get("requested_tokens", 0))
                hit_rate = (
                    hit_tokens / requested_tokens if requested_tokens > 0 else 0.0
                )
                root_span.set_attribute("hit_tokens", hit_tokens)
                root_span.set_attribute("requested_tokens", requested_tokens)
                root_span.set_attribute("hit_rate", hit_rate)

        if event.event_type == EventType.MP_STORE_END:
            if (count := self._pending_store_count.get(sid, 0)) > 0:
                self._pending_store_count[sid] = count - 1
        elif event.event_type == EventType.MP_RETRIEVE_END:
            if (count := self._pending_retrieve_count.get(sid, 0)) > 0:
                self._pending_retrieve_count[sid] = count - 1

        if (
            sid in self._deferred_session_end_ts
            and self._pending_store_count.get(sid, 0) == 0
            and self._pending_retrieve_count.get(sid, 0) == 0
        ):
            deferred_ts = self._deferred_session_end_ts.pop(sid)
            self._close_request_span(sid, deferred_ts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_request_span(
        self, session_id: str, ts: float
    ) -> tuple[Any, Any]:
        """Return the root span and its OTel context, creating them if absent.

        The span is stored in the registry under ``span_name="request"`` so
        that other subscribers sharing the same :class:`SpanRegistry` can
        retrieve the parent context via
        ``registry.get_context(session_id, "request")``.

        Args:
            session_id: The request session identifier.
            ts: Wall-clock timestamp (``time.time()``) to use as span start
                if the root is created now.

        Returns:
            ``(root_span, root_otel_context)`` tuple.
        """
        entry = self._registry.get(session_id, "request")
        if entry is not None:
            return entry
        root_span = _tracer.start_span(
            "request",
            start_time=int(ts * 1e9),
        )
        root_span.set_attribute("session_id", session_id)
        root_ctx = trace.set_span_in_context(root_span)
        self._registry.open(session_id, "request", root_span, root_ctx)
        return root_span, root_ctx

    def _close_request_span(self, session_id: str, end_ts: float) -> None:
        """End the root span and clean up all per-session state.

        Args:
            session_id: The request session identifier.
            end_ts: Wall-clock timestamp to stamp as the span end time.
        """
        entry = self._registry.pop(session_id, "request")
        if entry is not None:
            root_span, _ = entry
            try:
                root_span.end(end_time=int(end_ts * 1e9))
            except Exception:
                pass
        self._pending_store_count.pop(session_id, None)
        self._pending_retrieve_count.pop(session_id, None)
        self._deferred_session_end_ts.pop(session_id, None)
