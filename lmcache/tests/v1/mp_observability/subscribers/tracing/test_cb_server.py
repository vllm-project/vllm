# SPDX-License-Identifier: Apache-2.0

"""Tests for BlendTracingSubscriber."""

# Standard
from unittest.mock import patch
import time

# Third Party
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.tracing.cb_server import (
    BlendTracingSubscriber,
)
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import (
    SpanRegistry,
)
import lmcache.v1.mp_observability.subscribers.tracing.cb_server as cb_server_module


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def registry():
    return SpanRegistry()


@pytest.fixture
def subscriber(registry, bus):
    sub = BlendTracingSubscriber(registry)
    bus.register_subscriber(sub)
    return sub


class TestBlendTracingSubscriber:
    def test_subscriptions_cover_all_cb_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.CB_REQUEST_START in subs
        assert EventType.CB_REQUEST_END in subs
        assert EventType.CB_STORE_PRE_COMPUTED_SUBMITTED in subs
        assert EventType.CB_RETRIEVE_SUBMITTED in subs
        assert EventType.CB_STORE_FINAL_SUBMITTED in subs
        assert EventType.CB_LOOKUP_START in subs
        assert EventType.CB_LOOKUP_END in subs
        assert EventType.CB_STORE_PRE_COMPUTED_START in subs
        assert EventType.CB_STORE_PRE_COMPUTED_END in subs
        assert EventType.CB_RETRIEVE_START in subs
        assert EventType.CB_RETRIEVE_END in subs
        assert EventType.CB_STORE_FINAL_START in subs
        assert EventType.CB_STORE_FINAL_END in subs
        assert EventType.CB_FINGERPRINTS_REGISTERED in subs
        assert EventType.CB_CHUNKS_EVICTED in subs

    # ------------------------------------------------------------------
    # Root span creation
    # ------------------------------------------------------------------

    def test_root_span_created_on_request_start(self, bus, registry, subscriber):
        bus.start()
        bus.publish(Event(event_type=EventType.CB_REQUEST_START, session_id="req-root"))
        time.sleep(0.15)
        assert registry.get("req-root", "cb.request") is not None
        bus.stop()

    def test_no_root_span_before_any_event(self, registry):
        assert registry.get("any-session", "cb.request") is None

    # ------------------------------------------------------------------
    # Session end closes root immediately when no GPU ops in flight
    # ------------------------------------------------------------------

    def test_session_end_closes_root_immediately_when_no_gpu_ops(
        self, bus, registry, subscriber
    ):
        bus.start()
        now = time.time()
        sid = "req-lookup-only"

        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"num_tokens": 64},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "num_tokens": 64,
                    "fingerprint_hits": 2,
                    "storage_hits": 2,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "cb.request") is None
        assert sid not in subscriber._pending_gpu_ops
        assert len(subscriber._pending) == 0

    # ------------------------------------------------------------------
    # Deferred close: SESSION_END races GPU store_pre_computed
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_store_pre_computed_finishes(
        self, bus, registry, subscriber
    ):
        bus.start()
        now = time.time()
        sid = "req-deferred-store-pre"

        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"instance_id": 0},
            )
        )
        # SESSION_END arrives before GPU store completes
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Root should still be open
        assert registry.get(sid, "cb.request") is not None
        assert sid in subscriber._deferred_session_end_ts

        # GPU store completes
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"instance_id": 0, "num_tokens": 64},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id=sid,
                timestamp=now + 0.050,
                metadata={"instance_id": 0, "stored_chunks": 4, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "cb.request") is None
        assert sid not in subscriber._deferred_session_end_ts
        assert sid not in subscriber._pending_gpu_ops

    # ------------------------------------------------------------------
    # Deferred close: SESSION_END races GPU retrieve
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_retrieve_finishes(
        self, bus, registry, subscriber
    ):
        bus.start()
        now = time.time()
        sid = "req-deferred-retrieve"

        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"instance_id": 1},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        assert registry.get(sid, "cb.request") is not None
        assert sid in subscriber._deferred_session_end_ts

        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"instance_id": 1, "num_chunks": 3},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id=sid,
                timestamp=now + 0.050,
                metadata={"instance_id": 1, "num_chunks": 3, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "cb.request") is None
        assert sid not in subscriber._deferred_session_end_ts
        assert sid not in subscriber._pending_gpu_ops

    # ------------------------------------------------------------------
    # Deferred close: SESSION_END races GPU store_final
    # ------------------------------------------------------------------

    def test_session_end_deferred_until_store_final_finishes(
        self, bus, registry, subscriber
    ):
        bus.start()
        now = time.time()
        sid = "req-deferred-store-final"

        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"instance_id": 2},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        assert registry.get(sid, "cb.request") is not None
        assert sid in subscriber._deferred_session_end_ts

        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"instance_id": 2, "num_tokens": 256},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_END,
                session_id=sid,
                timestamp=now + 0.060,
                metadata={"instance_id": 2, "stored_chunks": 16, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "cb.request") is None
        assert sid not in subscriber._deferred_session_end_ts
        assert sid not in subscriber._pending_gpu_ops

    # ------------------------------------------------------------------
    # Multiple GPU ops: all must finish before root closes
    # ------------------------------------------------------------------

    def test_multiple_gpu_ops_all_must_finish(self, bus, registry, subscriber):
        bus.start()
        now = time.time()
        sid = "req-multi-gpu-ops"

        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_START,
                session_id=sid,
                timestamp=now,
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"instance_id": 0},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_SUBMITTED,
                session_id=sid,
                timestamp=now + 0.002,
                metadata={"instance_id": 0},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.005,
            )
        )
        time.sleep(0.15)

        # Both in flight — root still open
        assert registry.get(sid, "cb.request") is not None

        # Store finishes first — retrieve still pending, root stays open
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={"instance_id": 0, "num_tokens": 64},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id=sid,
                timestamp=now + 0.030,
                metadata={"instance_id": 0, "stored_chunks": 4, "success": True},
            )
        )
        time.sleep(0.15)
        assert registry.get(sid, "cb.request") is not None

        # Retrieve finishes — all done, root closes
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id=sid,
                timestamp=now + 0.040,
                metadata={"instance_id": 0, "num_chunks": 4},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id=sid,
                timestamp=now + 0.060,
                metadata={"instance_id": 0, "num_chunks": 4, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert registry.get(sid, "cb.request") is None
        assert sid not in subscriber._deferred_session_end_ts

    # ------------------------------------------------------------------
    # Child span lifecycles
    # ------------------------------------------------------------------

    def test_lookup_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id="req-lookup",
                metadata={"num_tokens": 128},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id="req-lookup",
                metadata={
                    "num_tokens": 128,
                    "fingerprint_hits": 3,
                    "storage_hits": 2,
                    "stale_chunks": 1,
                    "no_gpu_context": False,
                },
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_store_pre_computed_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id="req-sp",
                metadata={"instance_id": 0, "num_tokens": 64},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id="req-sp",
                metadata={"instance_id": 0, "stored_chunks": 4, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_retrieve_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id="req-ret",
                metadata={"instance_id": 1, "num_chunks": 3},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id="req-ret",
                metadata={"instance_id": 1, "num_chunks": 3, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_store_final_span_lifecycle(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_START,
                session_id="req-sf",
                metadata={"instance_id": 2, "num_tokens": 512},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_END,
                session_id="req-sf",
                metadata={"instance_id": 2, "stored_chunks": 32, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    # ------------------------------------------------------------------
    # Point events
    # ------------------------------------------------------------------

    def test_fingerprints_registered_no_crash(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_FINGERPRINTS_REGISTERED,
                session_id="req-fp",
                metadata={"num_chunks": 8, "num_tokens": 256},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_chunks_evicted_no_crash(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_CHUNKS_EVICTED,
                session_id="req-ev",
                metadata={"num_chunks": 2},
            )
        )
        time.sleep(0.15)
        bus.stop()

    # ------------------------------------------------------------------
    # Error resilience
    # ------------------------------------------------------------------

    def test_unmatched_end_does_not_crash(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id="orphan",
                metadata={"stored_chunks": 2, "success": True},
            )
        )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0

    def test_unmatched_start_cleaned_on_shutdown(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id="leaked",
                metadata={"instance_id": 0, "num_tokens": 64},
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
                    event_type=EventType.CB_LOOKUP_START,
                    session_id=f"req-{i}",
                    metadata={"num_tokens": 100},
                )
            )
        for i in range(5):
            bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id=f"req-{i}",
                    metadata={
                        "num_tokens": 100,
                        "fingerprint_hits": 2,
                        "storage_hits": 2,
                        "stale_chunks": 0,
                        "no_gpu_context": False,
                    },
                )
            )
        time.sleep(0.15)
        bus.stop()
        assert len(subscriber._pending) == 0


class TestCBHitRateAttributes:
    """Verify hit_tokens / requested_tokens / hit_rate are set on cb.request root span.

    Uses a real OTel TracerProvider backed by InMemorySpanExporter so that span
    attributes are actually recorded.  The module-level ``_tracer`` is patched
    for the duration of each test; ``_HAS_OTEL`` is forced True.
    """

    @pytest.fixture
    def exporter(self):
        """Real OTel provider with in-memory exporter; patches module tracer."""
        exp = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exp))
        real_tracer = provider.get_tracer("lmcache_mp.blend")
        with (
            patch.object(cb_server_module, "_tracer", real_tracer),
            patch.object(cb_server_module, "_HAS_OTEL", True),
        ):
            yield exp
        exp.shutdown()

    def _root_span(self, exporter: InMemorySpanExporter, sid: str):
        """Return the finished cb.request root span for *sid*, or None."""
        for span in exporter.get_finished_spans():
            if span.name == "cb.request" and span.attributes.get("session_id") == sid:
                return span
        return None

    def test_hit_rate_attrs_set_on_root_span(self, exporter):
        """CB_LOOKUP_END with hit_tokens=512, requested_tokens=1024 → hit_rate=0.5."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(BlendTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "cb-hr-normal"

        bus.publish(
            Event(event_type=EventType.CB_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"num_tokens": 1024},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "num_tokens": 1024,
                    "fingerprint_hits": 4,
                    "storage_hits": 2,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                    "hit_tokens": 512,
                    "requested_tokens": 1024,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
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
        """CB_LOOKUP_END with requested_tokens=0 yields hit_rate=0.0 without error."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(BlendTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "cb-hr-zero"

        bus.publish(
            Event(event_type=EventType.CB_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"num_tokens": 0},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "num_tokens": 0,
                    "fingerprint_hits": 0,
                    "storage_hits": 0,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                    "hit_tokens": 0,
                    "requested_tokens": 0,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
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

    def test_total_miss_hit_rate(self, exporter):
        """CB_LOOKUP_END with storage_hits=0 but requested_tokens>0 → hit_rate=0.0."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(BlendTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "cb-hr-miss"

        bus.publish(
            Event(event_type=EventType.CB_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"num_tokens": 1024},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "num_tokens": 1024,
                    "fingerprint_hits": 0,
                    "storage_hits": 0,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                    "hit_tokens": 0,
                    "requested_tokens": 1024,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = self._root_span(exporter, sid)
        assert root is not None
        assert root.attributes["hit_tokens"] == 0
        assert root.attributes["requested_tokens"] == 1024
        assert root.attributes["hit_rate"] == 0.0

    def test_prefix_hits_attr_set_on_root_span(self, exporter):
        """CB_LOOKUP_END with prefix_hits=2 stamps prefix_hits=2 on root span."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(BlendTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "cb-prefix-hits"

        bus.publish(
            Event(event_type=EventType.CB_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"num_tokens": 512},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "num_tokens": 512,
                    "fingerprint_hits": 0,
                    "prefix_hits": 2,
                    "storage_hits": 2,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                    "hit_tokens": 512,
                    "requested_tokens": 512,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = self._root_span(exporter, sid)
        assert root is not None
        assert root.attributes["prefix_hits"] == 2
        assert root.attributes["hit_tokens"] == 512
        assert root.attributes["hit_rate"] == 1.0

    def test_prefix_hits_defaults_to_zero_when_absent(self, exporter):
        """CB_LOOKUP_END without prefix_hits stamps prefix_hits=0 (backward compat)."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        registry = SpanRegistry()
        bus.register_subscriber(BlendTracingSubscriber(registry))
        bus.start()
        now = time.time()
        sid = "cb-prefix-absent"

        bus.publish(
            Event(event_type=EventType.CB_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"num_tokens": 256},
            )
        )
        # Omit prefix_hits to simulate an older server payload
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "num_tokens": 256,
                    "fingerprint_hits": 1,
                    "storage_hits": 1,
                    "stale_chunks": 0,
                    "no_gpu_context": False,
                    "hit_tokens": 256,
                    "requested_tokens": 256,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.020,
            )
        )
        time.sleep(0.15)
        bus.stop()

        root = self._root_span(exporter, sid)
        assert root is not None
        assert root.attributes["prefix_hits"] == 0

    def test_hit_rate_includes_prefix_and_non_prefix(self, exporter):
        """V3 hit_rate numerator = prefix_hit_tokens + non_prefix_hit_tokens;
        both are also recorded as separate attributes on the root span."""
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(BlendTracingSubscriber(SpanRegistry()))
        bus.start()
        now = time.time()
        sid = "cb-hr-split"
        bus.publish(
            Event(event_type=EventType.CB_REQUEST_START, session_id=sid, timestamp=now)
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=sid,
                timestamp=now + 0.001,
                metadata={"num_tokens": 1024},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=sid,
                timestamp=now + 0.010,
                metadata={
                    "prefix_hit_tokens": 256,
                    "non_prefix_hit_tokens": 256,
                    "hit_tokens": 512,  # = prefix + non_prefix (set by blend_v3)
                    "requested_tokens": 1024,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=sid,
                timestamp=now + 0.011,
            )
        )
        time.sleep(0.2)
        bus.stop()

        root = self._root_span(exporter, sid)
        assert root is not None
        assert root.attributes["prefix_hit_tokens"] == 256
        assert root.attributes["non_prefix_hit_tokens"] == 256
        assert root.attributes["hit_tokens"] == 512
        assert root.attributes["hit_rate"] == 0.5


class TestCBLookupSubspans:
    """V3 lookup sub-spans (cb.fingerprint_match / cb.sparse_prefetch) nest under
    cb.lookup, not the cb.request root. The prefix lookup has no cb.* span (it is
    traced by mp.lookup_prefetch); prefix_chunks rides on cb.lookup."""

    @pytest.fixture
    def exporter(self):
        exp = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exp))
        real_tracer = provider.get_tracer("lmcache_mp.blend")
        with (
            patch.object(cb_server_module, "_tracer", real_tracer),
            patch.object(cb_server_module, "_HAS_OTEL", True),
        ):
            yield exp
        exp.shutdown()

    def _spans_by_name(self, exporter: InMemorySpanExporter, sid: str):
        return {
            s.name: s
            for s in exporter.get_finished_spans()
            if s.attributes.get("session_id") == sid
        }

    def test_lookup_subspans_nest_under_cb_lookup(self, exporter):
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(BlendTracingSubscriber(SpanRegistry()))
        bus.start()
        now = time.time()
        sid = "cb-subspans"
        seq = [
            (EventType.CB_REQUEST_START, {}),
            (EventType.CB_LOOKUP_START, {"num_tokens": 1024}),
            (EventType.CB_FINGERPRINT_MATCH_START, {}),
            (EventType.CB_FINGERPRINT_MATCH_END, {"matches": 7}),
            (EventType.CB_SPARSE_PREFETCH_START, {"n_chunks": 5}),
            (EventType.CB_SPARSE_PREFETCH_END, {"found_keys": 5}),
            (EventType.CB_LOOKUP_END, {"num_tokens": 1024, "prefix_chunks": 2}),
            (EventType.CB_REQUEST_END, {}),
        ]
        for i, (et, md) in enumerate(seq):
            bus.publish(
                Event(
                    event_type=et,
                    session_id=sid,
                    timestamp=now + i * 0.001,
                    metadata=md,
                )
            )
        time.sleep(0.3)
        bus.stop()

        spans = self._spans_by_name(exporter, sid)
        for name in (
            "cb.lookup",
            "cb.fingerprint_match",
            "cb.sparse_prefetch",
        ):
            assert name in spans, f"missing span {name}; have {sorted(spans)}"
        # No cb.prefix_lookup span (traced by mp.lookup_prefetch instead).
        assert "cb.prefix_lookup" not in spans
        lookup_id = spans["cb.lookup"].context.span_id
        for name in ("cb.fingerprint_match", "cb.sparse_prefetch"):
            assert spans[name].parent is not None, f"{name} has no parent span"
            assert spans[name].parent.span_id == lookup_id, (
                f"{name} should nest under cb.lookup"
            )
        # cb.lookup itself nests under the cb.request root (unchanged behavior).
        assert spans["cb.lookup"].parent.span_id == spans["cb.request"].context.span_id
        # sub-span metadata propagates as attributes.
        assert spans["cb.fingerprint_match"].attributes.get("matches") == "7"
        # prefix coverage rides on cb.lookup (no dedicated prefix span).
        assert spans["cb.lookup"].attributes.get("prefix_chunks") == "2"
        assert spans["cb.sparse_prefetch"].attributes.get("found_keys") == "5"

    def test_coordinator_match_span_nests_under_cb_lookup(self, exporter):
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(BlendTracingSubscriber(SpanRegistry()))
        bus.start()
        now = time.time()
        sid = "cb-coord-match"
        seq = [
            (EventType.CB_REQUEST_START, {}),
            (EventType.CB_LOOKUP_START, {"num_tokens": 1024}),
            (EventType.CB_COORDINATOR_MATCH_START, {}),
            (EventType.CB_COORDINATOR_MATCH_END, {"matches": 3, "timed_out": False}),
            (EventType.CB_LOOKUP_END, {"num_tokens": 1024, "prefix_chunks": 2}),
            (EventType.CB_REQUEST_END, {}),
        ]
        for i, (et, md) in enumerate(seq):
            bus.publish(
                Event(
                    event_type=et,
                    session_id=sid,
                    timestamp=now + i * 0.001,
                    metadata=md,
                )
            )
        time.sleep(0.3)
        bus.stop()

        spans = self._spans_by_name(exporter, sid)
        assert "cb.coordinator_match" in spans, (
            f"missing cb.coordinator_match; have {sorted(spans)}"
        )
        assert (
            spans["cb.coordinator_match"].parent.span_id
            == spans["cb.lookup"].context.span_id
        ), "cb.coordinator_match should nest under cb.lookup"
        assert spans["cb.coordinator_match"].attributes.get("matches") == "3"
        assert spans["cb.coordinator_match"].attributes.get("timed_out") == "False"

    def test_scatter_span_nests_under_cb_retrieve(self, exporter):
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(BlendTracingSubscriber(SpanRegistry()))
        bus.start()
        now = time.time()
        sid = "cb-scatter"
        seq = [
            (EventType.CB_REQUEST_START, {}),
            (EventType.CB_RETRIEVE_START, {"num_chunks": 5}),
            (
                EventType.CB_SCATTER_START,
                {
                    "scattered_tokens": 1280,
                    "n_prefix": 1,
                    "n_shifted": 4,
                    "dropped": 0,
                },
            ),
            (EventType.CB_SCATTER_END, {}),
            (EventType.CB_RETRIEVE_END, {}),
            (EventType.CB_REQUEST_END, {}),
        ]
        for i, (et, md) in enumerate(seq):
            bus.publish(
                Event(
                    event_type=et,
                    session_id=sid,
                    timestamp=now + i * 0.001,
                    metadata=md,
                )
            )
        time.sleep(0.3)
        bus.stop()

        spans = self._spans_by_name(exporter, sid)
        assert "cb.scatter" in spans, f"missing cb.scatter; have {sorted(spans)}"
        assert "cb.retrieve" in spans
        assert (
            spans["cb.scatter"].parent.span_id == spans["cb.retrieve"].context.span_id
        ), "cb.scatter should nest under cb.retrieve"
        assert spans["cb.scatter"].attributes.get("scattered_tokens") == "1280"
        assert spans["cb.scatter"].attributes.get("n_shifted") == "4"
        assert spans["cb.scatter"].attributes.get("dropped") == "0"
