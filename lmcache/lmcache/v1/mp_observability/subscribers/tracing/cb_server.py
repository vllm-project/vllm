# SPDX-License-Identifier: Apache-2.0

"""OTel tracing subscriber for Cache Blending (CB) operations.

Creates a root ``"cb.request"`` span per session wrapping all CB child spans.
Opens at ``CB_REQUEST_START``; closes at ``CB_REQUEST_END``, deferred until
any in-flight GPU store/retrieve callbacks complete **and** until
``CB_STORE_FINAL_SUBMITTED`` has been received (if a retrieve was submitted).

On the HIT path the lifecycle is:

  CB_RETRIEVE_SUBMITTED → [retrieve GPU op] → CB_RETRIEVE_END
      → [inference, ~hundreds of ms, pending_gpu_ops==0 here]
  CB_STORE_FINAL_SUBMITTED → CB_REQUEST_END → [store GPU op] → CB_STORE_FINAL_END

During inference ``_pending_gpu_ops`` is 0, so a stray ``CB_REQUEST_END``
(e.g. from a second CB lookup call that misses) would
otherwise close the root span prematurely.  ``_waiting_for_store_final``
bridges this gap: it is populated by ``CB_RETRIEVE_SUBMITTED`` and cleared by
``CB_STORE_FINAL_SUBMITTED``, so the root span cannot close until both
conditions hold simultaneously:

* ``_pending_gpu_ops[sid] == 0``
* ``sid not in _waiting_for_store_final``

``cb.request`` is the trace root: blend_v3 owns the CB lookup end-to-end (prefix
+ non-prefix legs, direct against the storage manager), so a CB request never
opens an MP ``"request"`` span to nest under. The optional
:class:`~lmcache.v1.mp_observability.subscribers.tracing.span_registry\
.SpanRegistry` is used to nest the CB child spans under ``cb.request``.
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

    _tracer = trace.get_tracer("lmcache_mp.blend")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class BlendTracingSubscriber(EventSubscriber):
    """Creates OTel spans from CB (Cache Blending) START/END event pairs.

    Each session gets one root ``"cb.request"`` span that nests all child
    spans (``cb.lookup``, ``cb.store_pre_computed``, ``cb.retrieve``,
    ``cb.store_final``).  The root is opened at ``CB_REQUEST_START`` and
    closed at ``CB_REQUEST_END``, with deferral if GPU ops are still in
    flight.

    When a shared :class:`SpanRegistry` is provided, ``"cb.request"`` is
    nested under the MP server ``"request"`` span for the same session.
    """

    # Maps each START event to its span name (used for creation and registry key).
    _SPAN_DEFS: dict[EventType, str] = {
        EventType.CB_STORE_PRE_COMPUTED_START: "cb.store_pre_computed",
        EventType.CB_LOOKUP_START: "cb.lookup",
        EventType.CB_RETRIEVE_START: "cb.retrieve",
        EventType.CB_STORE_FINAL_START: "cb.store_final",
        # V3 lookup sub-spans (nest under cb.lookup, see _SPAN_PARENTS).
        EventType.CB_FINGERPRINT_MATCH_START: "cb.fingerprint_match",
        EventType.CB_PREFIX_LOOKUP_START: "cb.prefix_lookup",
        EventType.CB_COORDINATOR_MATCH_START: "cb.coordinator_match",
        EventType.CB_SPARSE_PREFETCH_START: "cb.sparse_prefetch",
        # V3 retrieve sub-span (nests under cb.retrieve).
        EventType.CB_SCATTER_START: "cb.scatter",
    }

    # Child span -> parent span name for nesting; absent => nest under the
    # cb.request root (the default for the top-level lookup/retrieve/store spans).
    _SPAN_PARENTS: dict[str, str] = {
        "cb.fingerprint_match": "cb.lookup",
        "cb.prefix_lookup": "cb.lookup",
        "cb.coordinator_match": "cb.lookup",
        "cb.sparse_prefetch": "cb.lookup",
        "cb.scatter": "cb.retrieve",
    }

    _END_TO_START: dict[EventType, EventType] = {
        EventType.CB_STORE_PRE_COMPUTED_END: EventType.CB_STORE_PRE_COMPUTED_START,
        EventType.CB_LOOKUP_END: EventType.CB_LOOKUP_START,
        EventType.CB_RETRIEVE_END: EventType.CB_RETRIEVE_START,
        EventType.CB_STORE_FINAL_END: EventType.CB_STORE_FINAL_START,
        EventType.CB_FINGERPRINT_MATCH_END: EventType.CB_FINGERPRINT_MATCH_START,
        EventType.CB_PREFIX_LOOKUP_END: EventType.CB_PREFIX_LOOKUP_START,
        EventType.CB_COORDINATOR_MATCH_END: EventType.CB_COORDINATOR_MATCH_START,
        EventType.CB_SPARSE_PREFETCH_END: EventType.CB_SPARSE_PREFETCH_START,
        EventType.CB_SCATTER_END: EventType.CB_SCATTER_START,
    }

    # END events that correspond to a SUBMITTED sentinel (decrement ops counter)
    _GPU_OP_END_EVENTS: frozenset[EventType] = frozenset(
        {
            EventType.CB_STORE_PRE_COMPUTED_END,
            EventType.CB_RETRIEVE_END,
            EventType.CB_STORE_FINAL_END,
        }
    )

    def __init__(self, registry: SpanRegistry | None = None) -> None:
        self._registry = registry if registry is not None else SpanRegistry()

        # session_id -> (span, start_event_type) for pending child spans
        self._pending: dict[str, tuple[Any, EventType]] = {}

        # session_id -> number of in-flight GPU ops (SUBMITTED without matching END)
        self._pending_gpu_ops: dict[str, int] = {}

        # session_id -> REQUEST_END timestamp saved when GPU ops are in flight
        # or when a store_final is still expected
        self._deferred_session_end_ts: dict[str, float] = {}

        # Sessions where CB_RETRIEVE_SUBMITTED was seen but CB_STORE_FINAL_SUBMITTED
        # has not yet arrived.  Prevents premature root-span closure during the
        # inference window when _pending_gpu_ops is transiently 0.
        self._waiting_for_store_final: set[str] = set()

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return the event-to-callback mapping for this subscriber."""
        return {
            # Root span lifecycle
            EventType.CB_REQUEST_START: self._on_request_start,
            EventType.CB_STORE_PRE_COMPUTED_SUBMITTED: self._on_submitted,
            EventType.CB_RETRIEVE_SUBMITTED: self._on_submitted,
            EventType.CB_STORE_FINAL_SUBMITTED: self._on_submitted,
            EventType.CB_REQUEST_END: self._on_session_end,
            # Child spans
            EventType.CB_STORE_PRE_COMPUTED_START: self._on_start,
            EventType.CB_STORE_PRE_COMPUTED_END: self._on_end,
            EventType.CB_LOOKUP_START: self._on_start,
            EventType.CB_LOOKUP_END: self._on_end,
            EventType.CB_RETRIEVE_START: self._on_start,
            EventType.CB_RETRIEVE_END: self._on_end,
            EventType.CB_STORE_FINAL_START: self._on_start,
            EventType.CB_STORE_FINAL_END: self._on_end,
            # V3 lookup sub-spans (nested under cb.lookup)
            EventType.CB_FINGERPRINT_MATCH_START: self._on_start,
            EventType.CB_FINGERPRINT_MATCH_END: self._on_end,
            EventType.CB_PREFIX_LOOKUP_START: self._on_start,
            EventType.CB_PREFIX_LOOKUP_END: self._on_end,
            EventType.CB_COORDINATOR_MATCH_START: self._on_start,
            EventType.CB_COORDINATOR_MATCH_END: self._on_end,
            EventType.CB_SPARSE_PREFETCH_START: self._on_start,
            EventType.CB_SPARSE_PREFETCH_END: self._on_end,
            # V3 retrieve sub-span (nested under cb.retrieve, GPU-timed)
            EventType.CB_SCATTER_START: self._on_start,
            EventType.CB_SCATTER_END: self._on_end,
            # Point events
            EventType.CB_FINGERPRINTS_REGISTERED: self._on_point,
            EventType.CB_CHUNKS_EVICTED: self._on_point,
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
            set(self._pending_gpu_ops)
            | set(self._deferred_session_end_ts)
            | self._registry.all_session_ids()
        )
        for sid in sessions:
            self._registry.clear_session(sid)
        self._pending_gpu_ops.clear()
        self._deferred_session_end_ts.clear()
        self._waiting_for_store_final.clear()

    # ------------------------------------------------------------------
    # Root span handlers
    # ------------------------------------------------------------------

    def _on_request_start(self, event: Event) -> None:
        """Create the ``"cb.request"`` root span.

        blend_v3 owns the CB lookup end-to-end, so a CB request never opens an MP
        ``"request"`` span — ``cb.request`` is the trace root.

        Args:
            event: ``CB_REQUEST_START`` event with ``session_id`` set.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        if self._registry.get_context(sid, "cb.request") is not None:
            logger.warning("CB_REQUEST_START fired twice for session=%s; ignoring", sid)
            return
        root_span = _tracer.start_span(
            "cb.request",
            start_time=int(event.timestamp * 1e9),
        )
        root_span.set_attribute("session_id", sid)
        self._registry.open(
            sid, "cb.request", root_span, trace.set_span_in_context(root_span)
        )

    def _on_submitted(self, event: Event) -> None:
        """Increment the in-flight GPU-ops counter and update store-final tracking.

        ``CB_RETRIEVE_SUBMITTED`` marks the session as waiting for a store_final,
        bridging the inference gap where ``_pending_gpu_ops`` is transiently 0.
        ``CB_STORE_FINAL_SUBMITTED`` clears that marker (store_final has arrived).

        Args:
            event: One of ``CB_STORE_PRE_COMPUTED_SUBMITTED``,
                ``CB_RETRIEVE_SUBMITTED``, or ``CB_STORE_FINAL_SUBMITTED``.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        self._pending_gpu_ops[sid] = self._pending_gpu_ops.get(sid, 0) + 1
        if event.event_type == EventType.CB_RETRIEVE_SUBMITTED:
            self._waiting_for_store_final.add(sid)
        elif event.event_type == EventType.CB_STORE_FINAL_SUBMITTED:
            self._waiting_for_store_final.discard(sid)

    def _on_session_end(self, event: Event) -> None:
        """Close the root span, or defer if GPU ops are in flight or store_final
        is pending.

        Defers when either condition holds:
        * ``_pending_gpu_ops[sid] > 0`` — a GPU callback is still in flight.
        * ``sid in _waiting_for_store_final`` — retrieve completed but
          ``CB_STORE_FINAL_SUBMITTED`` has not yet arrived (inference window).

        Always overwrites any previously saved deferred timestamp so the
        ``CB_REQUEST_END`` from ``cb_store_final`` (the correct logical end)
        supersedes an earlier one from a concurrent lookup miss.

        Args:
            event: ``CB_REQUEST_END`` event carrying the logical end timestamp.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        if (
            self._pending_gpu_ops.get(sid, 0) == 0
            and sid not in self._waiting_for_store_final
        ):
            self._close_request_span(sid, event.timestamp)
        else:
            self._deferred_session_end_ts[sid] = event.timestamp

    # ------------------------------------------------------------------
    # Child span handlers
    # ------------------------------------------------------------------

    def _on_start(self, event: Event) -> None:
        """Create a child span nested under the ``"cb.request"`` root span.

        Args:
            event: One of the CB ``*_START`` event types.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        span_name = self._SPAN_DEFS[event.event_type]

        # Nest under the mapped parent span (e.g. cb.fingerprint_match under
        # cb.lookup); top-level spans and any orphan fall back to cb.request.
        parent_name = self._SPAN_PARENTS.get(span_name, "cb.request")
        parent_ctx = self._registry.get_context(sid, parent_name)
        if parent_ctx is None and parent_name != "cb.request":
            parent_ctx = self._registry.get_context(sid, "cb.request")
        span = _tracer.start_span(
            span_name,
            context=parent_ctx,
            start_time=int(event.timestamp * 1e9),
        )
        span.set_attribute("session_id", sid)
        for k, v in event.metadata.items():
            span.set_attribute(k, str(v))

        key = f"{sid}:{event.event_type.value}"
        self._pending[key] = (span, event.event_type)

        self._registry.open(sid, span_name, span, trace.set_span_in_context(span))

    def _on_end(self, event: Event) -> None:
        """Close a pending child span and handle GPU-ops counter deferral.

        For GPU-backed END events (store_pre_computed, retrieve, store_final),
        decrements the in-flight counter; if it reaches zero and a deferred
        session-end timestamp exists, closes the root span using the GPU
        callback timestamp so that ``cb.request`` end_time reflects when
        inference and the GPU copy actually completed.

        Args:
            event: One of the CB ``*_END`` event types.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        start_type = self._END_TO_START[event.event_type]
        key = f"{sid}:{start_type.value}"
        entry = self._pending.pop(key, None)
        if entry is None:
            logger.debug(
                "No pending CB span for %s session=%s",
                event.event_type.value,
                sid,
            )
        else:
            span, _ = entry
            for k, v in event.metadata.items():
                span.set_attribute(k, str(v))
            span.end(end_time=int(event.timestamp * 1e9))

        self._registry.pop(sid, self._SPAN_DEFS[start_type])

        if event.event_type == EventType.CB_LOOKUP_END:
            root_entry = self._registry.get(sid, "cb.request")
            if root_entry is not None:
                root_span, _ = root_entry
                hit_tokens = int(event.metadata.get("hit_tokens", 0))
                requested_tokens = int(event.metadata.get("requested_tokens", 0))
                prefix_hit_tokens = int(event.metadata.get("prefix_hit_tokens", 0))
                seg_prefix_hit_tokens = int(
                    event.metadata.get("segmented_prefix_hit_tokens", 0)
                )
                non_prefix_hit_tokens = int(
                    event.metadata.get("non_prefix_hit_tokens", 0)
                )
                denom = requested_tokens or 1  # avoid /0; rates are 0 when requested=0
                root_span.set_attribute("hit_tokens", hit_tokens)
                root_span.set_attribute("requested_tokens", requested_tokens)
                # hit_rate numerator = prefix + segmented-prefix tail + non-prefix
                # reuse (hit_tokens).
                root_span.set_attribute("hit_rate", hit_tokens / denom)
                root_span.set_attribute(
                    "prefix_hits", int(event.metadata.get("prefix_hits", 0))
                )
                root_span.set_attribute("prefix_hit_tokens", prefix_hit_tokens)
                root_span.set_attribute(
                    "segmented_prefix_hit_tokens", seg_prefix_hit_tokens
                )
                root_span.set_attribute("non_prefix_hit_tokens", non_prefix_hit_tokens)
                # Per-component hit rates (sum to hit_rate).
                root_span.set_attribute("prefix_hit_rate", prefix_hit_tokens / denom)
                root_span.set_attribute(
                    "segmented_prefix_hit_rate", seg_prefix_hit_tokens / denom
                )
                root_span.set_attribute(
                    "non_prefix_hit_rate", non_prefix_hit_tokens / denom
                )

        if event.event_type in self._GPU_OP_END_EVENTS:
            if (count := self._pending_gpu_ops.get(sid, 0)) > 0:
                if count == 1:
                    self._pending_gpu_ops.pop(sid)
                else:
                    self._pending_gpu_ops[sid] = count - 1
            if (
                sid in self._deferred_session_end_ts
                and self._pending_gpu_ops.get(sid, 0) == 0
                and sid not in self._waiting_for_store_final
            ):
                self._deferred_session_end_ts.pop(sid)
                # Use the GPU callback timestamp so cb.request end_time reflects
                # when inference + the GPU copy actually finished, not when
                # cb_store_final was submitted on the CPU.
                self._close_request_span(sid, event.timestamp)

    def _on_point(self, event: Event) -> None:
        """Emit an instant span for point events (no paired END).

        Args:
            event: ``CB_FINGERPRINTS_REGISTERED`` or ``CB_CHUNKS_EVICTED``.
        """
        if not _HAS_OTEL:
            return
        sid = event.session_id
        ts_ns = int(event.timestamp * 1e9)
        parent_ctx = self._registry.get_context(sid, "cb.request")
        span = _tracer.start_span(
            event.event_type.value,
            context=parent_ctx,
            start_time=ts_ns,
        )
        span.set_attribute("session_id", sid)
        for k, v in event.metadata.items():
            span.set_attribute(k, str(v))
        span.end(end_time=ts_ns)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _close_request_span(self, session_id: str, end_ts: float) -> None:
        """End the ``"cb.request"`` root span and clean up per-session state.

        Args:
            session_id: The request session identifier.
            end_ts: Wall-clock timestamp to stamp as the span end time.
        """
        entry = self._registry.pop(session_id, "cb.request")
        if entry is not None:
            root_span, _ = entry
            try:
                root_span.end(end_time=int(end_ts * 1e9))
            except Exception:
                pass
        self._pending_gpu_ops.pop(session_id, None)
        self._deferred_session_end_ts.pop(session_id, None)
        self._waiting_for_store_final.discard(session_id)
