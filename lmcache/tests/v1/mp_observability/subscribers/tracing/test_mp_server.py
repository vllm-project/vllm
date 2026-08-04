# SPDX-License-Identifier: Apache-2.0

"""Tests for MPServerTracingSubscriber."""

# Standard
from unittest.mock import patch
import time

# Third Party
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.tracing import (
    MPServerTracingSubscriber,
)
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import (
    SpanRegistry,
)
import lmcache.v1.mp_observability.subscribers.tracing.mp_server as mp_server_module


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def registry():
    return SpanRegistry()


@pytest.fixture
def subscriber(registry, bus):
    sub = MPServerTracingSubscriber(registry)
    bus.register_subscriber(sub)
    return sub


class TestMPServerTracingSubscriber:
    def test_subscriptions_cover_all_mp_server_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.MP_REQUEST_START in subs
        assert EventType.MP_STORE_SUBMITTED in subs
        assert EventType.MP_RETRIEVE_SUBMITTED in subs
        assert EventType.MP_REQUEST_END in subs
        assert EventType.MP_STORE_START in subs
        assert EventType.MP_STORE_END in subs
        assert EventType.MP_RETRIEVE_START in subs
        assert EventType.MP_RETRIEVE_END in subs
        assert EventType.MP_LOOKUP_PREFETCH_START in subs
        assert EventType.MP_LOOKUP_PREFETCH_END in subs

    # ------------------------------------------------------------------
    # Root span creation
    # ------------------------------------------------------------------

    def test_root_span_created_on_request_start(self, bus, registry, subscriber):
        bus.start()
        bus.publish(Event(event_type=EventType.MP_REQUEST_START, session_id="req-root"))
        time.sleep(0.15)
        assert registry.get("req-root", "request") is not None
        bus.stop()

    def test_no_root_span_before_any_event(self, registry):
        assert registry.get("any-session", "request") is None

    # ------------------------------------------------------------------
    # Session-end closes root immediately when no stores in flight
    # ------------------------------------------------------------------

    def test_session_end_closes_root_immediately_when_no_store(
        self, bus, registry, subscriber
    ):
        bus.start()
        now = time.time()
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id="req-lookup-only",
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id="req-lookup-only",
                timestamp=now + 0.001,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-lookup-only",
                timestamp=now + 0.010,
                metadata={"found_count": 4},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id="req-lookup-only",
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get("req-lookup-only", "request") is None
        assert "req-lookup-only" not in subscriber._pending_store_count
        assert len(subscriber._pending) == 0

    # ------------------------------------------------------------------
    # Deferred close: SESSION_END races GPU store
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_store_finishes(self, bus, registry, subscriber):
        bus.start()
        now = time.time()
        sid = "req-deferred"

        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        # SESSION_END arrives before STORE_END
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Root should still be open (store in flight)
        assert registry.get(sid, "request") is not None
        assert sid in subscriber._deferred_session_end_ts

        # Now GPU store completes
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.050,
                metadata={"stored_count": 2, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        # Root should now be closed
        assert registry.get(sid, "request") is None
        assert sid not in subscriber._deferred_session_end_ts
        assert sid not in subscriber._pending_store_count

    # ------------------------------------------------------------------
    # Multiple stores: root stays open until all complete
    # ------------------------------------------------------------------

    def test_multiple_stores_all_must_finish(self, bus, registry, subscriber):
        bus.start()
        now = time.time()
        sid = "req-multi-store"

        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.002,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # count=2 — still open
        assert registry.get(sid, "request") is not None

        # First store ends — count=1, still open
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.030,
                metadata={"stored_count": 1, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        assert registry.get(sid, "request") is not None

        # Second store ends — count=0, closes now
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.040,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.060,
                metadata={"stored_count": 1, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "request") is None
        assert sid not in subscriber._pending_store_count

    # ------------------------------------------------------------------
    # Lazy root creation on store-only path (no lookup)
    # ------------------------------------------------------------------

    def test_lazy_root_on_store_only_path(self, bus, registry, subscriber):
        bus.start()
        now = time.time()
        sid = "req-store-only"

        # No MP_REQUEST_START — root created lazily on MP_STORE_START
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)

        assert registry.get(sid, "request") is not None

        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.020,
                metadata={"stored_count": 3, "device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.025,
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "request") is None

    # ------------------------------------------------------------------
    # Retrieve deferral: SESSION_END races GPU retrieve
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_retrieve_finishes(
        self, bus, registry, subscriber
    ):
        bus.start()
        now = time.time()
        sid = "req-deferred-retrieve"

        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        # SESSION_END arrives before RETRIEVE_END (the race condition)
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Root should still be open (retrieve in flight)
        assert registry.get(sid, "request") is not None
        assert sid in subscriber._deferred_session_end_ts

        # Now GPU retrieve completes
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id=sid,
                timestamp=now + 0.050,
                metadata={"retrieved_count": 4, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        # Root should now be closed
        assert registry.get(sid, "request") is None
        assert sid not in subscriber._deferred_session_end_ts
        assert sid not in subscriber._pending_retrieve_count

    def test_session_end_deferred_until_both_store_and_retrieve_finish(
        self, bus, registry, subscriber
    ):
        """Root span stays open until both a store and a retrieve finish."""
        bus.start()
        now = time.time()
        sid = "req-store-and-retrieve"

        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.002,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Both in flight — root still open
        assert registry.get(sid, "request") is not None

        # Store finishes first — retrieve still pending, root stays open
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.030,
                metadata={"stored_count": 1, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        assert registry.get(sid, "request") is not None

        # Retrieve finishes — now both counters are zero → root closes
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=sid,
                timestamp=now + 0.040,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id=sid,
                timestamp=now + 0.060,
                metadata={"retrieved_count": 2, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "request") is None
        assert sid not in subscriber._deferred_session_end_ts

    # ------------------------------------------------------------------
    # Child spans registered in shared registry for sub-span parenting
    # ------------------------------------------------------------------

    def test_retrieve_span_registered_in_registry_while_open(
        self, bus, registry, subscriber
    ):
        """An open mp.retrieve span is accessible in the registry under
        ``span_name="retrieve"`` so that a future sub-span subscriber can
        look it up as a parent."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id="req-reg",
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        assert registry.get("req-reg", "retrieve") is not None
        bus.stop()

    def test_retrieve_span_deregistered_after_close(self, bus, registry, subscriber):
        """Registry entry is cleaned up when mp.retrieve ends so stale
        contexts do not leak into subsequent requests."""
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id="req-reg2",
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id="req-reg2",
                metadata={"device": f"{torch_device_type}:0", "retrieved_count": 1},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert registry.get("req-reg2", "retrieve") is None

    # ------------------------------------------------------------------
    # Existing lifecycle tests (unchanged behaviour)
    # ------------------------------------------------------------------

    def test_store_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id="req-1",
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id="req-1",
                metadata={"device": f"{torch_device_type}:0", "stored_count": 5},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_retrieve_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id="req-2",
                metadata={"device": f"{torch_device_type}:1"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id="req-2",
                metadata={"device": f"{torch_device_type}:1", "retrieved_count": 3},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_lookup_prefetch_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id="req-3",
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-3",
                metadata={"found_count": 10},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_unmatched_end_does_not_crash(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id="orphan",
                metadata={"stored_count": 1, "device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_unmatched_start_cleaned_on_shutdown(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id="leaked",
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()
        subscriber.shutdown()
        assert len(subscriber._pending) == 0

    def test_multiple_concurrent_sessions(self, bus, subscriber):
        bus.start()
        for i in range(5):
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=f"req-{i}",
                    metadata={"device": f"{torch_device_type}:0"},
                )
            )
        for i in range(5):
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_END,
                    session_id=f"req-{i}",
                    metadata={"device": f"{torch_device_type}:0", "stored_count": i},
                )
            )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0


class TestHitRateAttributes:
    """Verify hit_tokens / requested_tokens / hit_rate are set on the root span.

    Uses a real OTel TracerProvider backed by InMemorySpanExporter so that span
    attributes are actually recorded.  The module-level ``_tracer`` is patched
    for the duration of each test; ``_HAS_OTEL`` is forced True so that the
    handlers don't early-return.
    """

    @pytest.fixture
    def exporter(self):
        """Real OTel provider with in-memory exporter; patches module tracer."""
        exp = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exp))
        real_tracer = provider.get_tracer("lmcache_mp.server")
        with (
            patch.object(mp_server_module, "_tracer", real_tracer),
            patch.object(mp_server_module, "_HAS_OTEL", True),
        ):
            yield exp
        exp.shutdown()

    def _root_span(self, exporter: InMemorySpanExporter, sid: str):
        """Return the finished root span for *sid*, or None."""
        for span in exporter.get_finished_spans():
            if span.name == "request" and span.attributes.get("session_id") == sid:
                return span
        return None

    def test_hit_rate_attrs_set_on_root_span(self, exporter):
        """LP_END with hit_tokens=512, requested_tokens=1024 → hit_rate=0.5."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(MPServerTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "req-hr-normal"

        bus.publish(
            Event(event_type=EventType.MP_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "hit_tokens": 512,
                    "requested_tokens": 1024,
                    "found_count": 2,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = self._root_span(exporter, sid)
        assert root is not None
        assert root.attributes["hit_tokens"] == 512
        assert root.attributes["requested_tokens"] == 1024
        assert abs(root.attributes["hit_rate"] - 0.5) < 1e-9

    def test_hit_rate_zero_when_requested_tokens_is_zero(self, exporter):
        """LP_END with requested_tokens=0 yields hit_rate=0.0 without error."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(MPServerTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "req-hr-zero"

        bus.publish(
            Event(event_type=EventType.MP_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=sid,
                timestamp=now + 0.001,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"hit_tokens": 0, "requested_tokens": 0, "found_count": 0},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = self._root_span(exporter, sid)
        assert root is not None
        assert root.attributes["hit_tokens"] == 0
        assert root.attributes["requested_tokens"] == 0
        assert root.attributes["hit_rate"] == 0.0

    def test_no_hit_rate_attrs_on_store_only_path(self, exporter):
        """Store-only request (no LP_END) leaves root span without hit rate attrs."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(MPServerTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "req-hr-store-only"

        bus.publish(
            Event(
                event_type=EventType.MP_STORE_SUBMITTED, session_id=sid, timestamp=now
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id=sid,
                timestamp=now + 0.020,
                metadata={"stored_count": 3, "device": f"{torch_device_type}:0"},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.025,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = self._root_span(exporter, sid)
        assert root is not None
        assert "hit_tokens" not in root.attributes
        assert "hit_rate" not in root.attributes
