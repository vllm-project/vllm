# SPDX-License-Identifier: Apache-2.0

"""Tests for EventBus, EventSubscriber, and singleton management."""

# Standard
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventBusConfig,
    EventSubscriber,
    get_event_bus,
    init_event_bus,
)
import lmcache.v1.mp_observability.event_bus as _bus_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type: EventType = EventType.L1_READ_FINISHED,
    session_id: str = "s1",
    **metadata,
) -> Event:
    return Event(event_type=event_type, session_id=session_id, metadata=metadata)


class _RecordingSubscriber(EventSubscriber):
    """Test subscriber that records events."""

    def __init__(self, event_types: list[EventType] | None = None):
        if event_types is None:
            event_types = [EventType.L1_READ_FINISHED]
        self._event_types = event_types
        self.events: list[Event] = []
        self.shutdown_called = False

    def get_subscriptions(self):
        return {et: self._on_event for et in self._event_types}

    def _on_event(self, event: Event) -> None:
        self.events.append(event)

    def shutdown(self) -> None:
        self.shutdown_called = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_global_bus():
    """Save and restore the global singleton so tests don't leak state."""
    saved = _bus_module._global_bus
    yield
    _bus_module._global_bus = saved


@pytest.fixture
def bus():
    """Enabled EventBus with small queue for fast tests."""
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


# ---------------------------------------------------------------------------
# Subscribe / publish basics
# ---------------------------------------------------------------------------


class TestSubscription:
    def test_subscribe_adds_callback(self, bus):
        cb = MagicMock()
        bus.subscribe(EventType.L1_READ_FINISHED, cb)
        assert cb in bus._subscribers[EventType.L1_READ_FINISHED]

    def test_multiple_callbacks_per_event_type(self, bus):
        cb1, cb2 = MagicMock(), MagicMock()
        bus.subscribe(EventType.L1_READ_FINISHED, cb1)
        bus.subscribe(EventType.L1_READ_FINISHED, cb2)
        assert len(bus._subscribers[EventType.L1_READ_FINISHED]) == 2

    def test_register_subscriber(self, bus):
        sub = _RecordingSubscriber()
        bus.register_subscriber(sub)
        assert sub in bus._registered_subscribers
        assert len(bus._subscribers[EventType.L1_READ_FINISHED]) == 1


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_launches_thread(self, bus):
        bus.start()
        assert bus._thread is not None
        assert bus._thread.is_alive()
        bus.stop()

    def test_disabled_noop(self):
        b = EventBus(EventBusConfig(enabled=False))
        b.start()
        assert b._thread is None

    def test_thread_is_daemon(self, bus):
        bus.start()
        assert bus._thread.daemon is True
        bus.stop()

    def test_thread_name(self, bus):
        bus.start()
        assert bus._thread.name == "EventBus"
        bus.stop()

    def test_double_start_is_idempotent(self, bus):
        bus.start()
        thread = bus._thread
        bus.start()
        assert bus._thread is thread
        bus.stop()

    def test_double_stop(self, bus):
        bus.start()
        bus.stop()
        bus.stop()

    def test_stop_without_start(self, bus):
        bus.stop()


# ---------------------------------------------------------------------------
# Event dispatch
# ---------------------------------------------------------------------------


class TestEventDispatch:
    def test_event_reaches_subscriber(self, bus):
        sub = _RecordingSubscriber()
        bus.register_subscriber(sub)
        bus.start()

        bus.publish(_make_event())
        time.sleep(0.15)
        bus.stop()

        assert len(sub.events) == 1
        assert sub.events[0].event_type == EventType.L1_READ_FINISHED

    def test_publish_stamps_timestamp(self, bus):
        sub = _RecordingSubscriber()
        bus.register_subscriber(sub)
        bus.start()

        before = time.time()
        bus.publish(_make_event())
        after = time.time()
        time.sleep(0.15)
        bus.stop()

        assert len(sub.events) == 1
        assert before <= sub.events[0].timestamp <= after

    def test_events_dispatched_in_order(self, bus):
        sub = _RecordingSubscriber()
        bus.register_subscriber(sub)
        bus.start()

        for i in range(5):
            bus.publish(_make_event(session_id=str(i)))
        time.sleep(0.15)
        bus.stop()

        assert len(sub.events) == 5
        for i, ev in enumerate(sub.events):
            assert ev.session_id == str(i)

    def test_event_only_dispatched_to_matching_subscribers(self, bus):
        l1_sub = _RecordingSubscriber([EventType.L1_READ_FINISHED])
        sm_sub = _RecordingSubscriber([EventType.SM_WRITE_RESERVED])
        bus.register_subscriber(l1_sub)
        bus.register_subscriber(sm_sub)
        bus.start()

        bus.publish(_make_event(event_type=EventType.L1_READ_FINISHED))
        bus.publish(_make_event(event_type=EventType.SM_WRITE_RESERVED))
        time.sleep(0.15)
        bus.stop()

        assert len(l1_sub.events) == 1
        assert l1_sub.events[0].event_type == EventType.L1_READ_FINISHED
        assert len(sm_sub.events) == 1
        assert sm_sub.events[0].event_type == EventType.SM_WRITE_RESERVED

    def test_multiple_subscribers_same_event(self, bus):
        sub1 = _RecordingSubscriber()
        sub2 = _RecordingSubscriber()
        bus.register_subscriber(sub1)
        bus.register_subscriber(sub2)
        bus.start()

        bus.publish(_make_event())
        time.sleep(0.15)
        bus.stop()

        assert len(sub1.events) == 1
        assert len(sub2.events) == 1

    def test_metadata_preserved(self, bus):
        sub = _RecordingSubscriber()
        bus.register_subscriber(sub)
        bus.start()

        bus.publish(_make_event(count=42, name="test"))
        time.sleep(0.15)
        bus.stop()

        assert sub.events[0].metadata["count"] == 42
        assert sub.events[0].metadata["name"] == "test"

    def test_disabled_bus_drops_events(self):
        b = EventBus(EventBusConfig(enabled=False))
        sub = _RecordingSubscriber()
        b.register_subscriber(sub)
        b.publish(_make_event())
        assert len(b._queue) == 0

    def test_late_registered_subscriber(self, bus):
        bus.start()
        time.sleep(0.05)

        sub = _RecordingSubscriber()
        bus.register_subscriber(sub)

        bus.publish(_make_event())
        time.sleep(0.15)
        bus.stop()

        assert len(sub.events) >= 1

    def test_stop_drains_remaining_events(self, bus):
        sub = _RecordingSubscriber()
        bus.register_subscriber(sub)
        # Don't start the drain thread — events accumulate in the queue
        for _ in range(3):
            bus.publish(_make_event())
        assert len(bus._queue) == 3

        # stop() should do a final drain
        bus.stop()
        assert len(sub.events) == 3


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------


class TestExceptionIsolation:
    def test_bad_callback_doesnt_block_good(self, bus):
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        bus.subscribe(EventType.L1_READ_FINISHED, bad_cb)

        good_sub = _RecordingSubscriber()
        bus.register_subscriber(good_sub)
        bus.start()

        bus.publish(_make_event())
        time.sleep(0.15)
        bus.stop()

        assert len(good_sub.events) == 1

    def test_shutdown_exception_isolated(self, bus):
        bad_sub = _RecordingSubscriber()
        bad_sub.shutdown = MagicMock(side_effect=RuntimeError("shutdown boom"))
        good_sub = _RecordingSubscriber()

        bus.register_subscriber(bad_sub)
        bus.register_subscriber(good_sub)
        bus.start()
        bus.stop()

        assert good_sub.shutdown_called


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    def test_events_discarded_when_queue_full(self):
        b = EventBus(EventBusConfig(enabled=True, max_queue_size=5))
        # Don't start — nothing drains
        for _ in range(10):
            b.publish(_make_event())

        assert len(b._queue) == 5
        assert b._discard_count == 5


# ---------------------------------------------------------------------------
# Self-monitoring accessors
# ---------------------------------------------------------------------------


class TestSelfMonitoring:
    def test_queue_depth_reports_current_size(self, bus):
        # Don't start the drain thread so events accumulate.
        assert bus.queue_depth() == 0
        for _ in range(3):
            bus.publish(_make_event())
        assert bus.queue_depth() == 3

    def test_oldest_event_lag_zero_when_empty(self, bus):
        assert bus.oldest_event_lag_seconds() == 0.0

    def test_oldest_event_lag_grows_with_age(self, bus):
        # Don't start drain — let events sit in the queue.
        bus.publish(_make_event())
        time.sleep(0.05)
        lag = bus.oldest_event_lag_seconds()
        assert lag >= 0.05
        # Drain via stop() so the bus shuts down cleanly.
        bus.stop()

    def test_oldest_event_lag_returns_zero_after_drain(self, bus):
        sub = _RecordingSubscriber()
        bus.register_subscriber(sub)
        bus.start()
        bus.publish(_make_event())
        time.sleep(0.15)
        # Drained, queue should be empty.
        assert bus.oldest_event_lag_seconds() == 0.0
        bus.stop()

    def test_dropped_events_count_starts_zero(self, bus):
        assert bus.dropped_events_count() == 0

    def test_dropped_events_count_increments_on_drop(self):
        b = EventBus(EventBusConfig(enabled=True, max_queue_size=2))
        # Don't start — nothing drains.
        for _ in range(5):
            b.publish(_make_event())
        assert b.dropped_events_count() == 3

    def test_subscriber_exception_counts_starts_empty(self, bus):
        assert bus.subscriber_exception_counts() == {}

    def test_subscriber_exception_counts_tracks_bound_methods(self, bus):
        class _BadSub(EventSubscriber):
            def get_subscriptions(self):
                return {EventType.L1_READ_FINISHED: self._on_event}

            def _on_event(self, event):
                raise RuntimeError("boom")

        bus.register_subscriber(_BadSub())
        bus.start()
        bus.publish(_make_event())
        bus.publish(_make_event())
        time.sleep(0.15)
        bus.stop()

        counts = bus.subscriber_exception_counts()
        assert counts.get("_BadSub") == 2

    def test_subscriber_exception_counts_returns_copy(self, bus):
        class _BadSub(EventSubscriber):
            def get_subscriptions(self):
                return {EventType.L1_READ_FINISHED: self._on_event}

            def _on_event(self, event):
                raise RuntimeError("boom")

        bus.register_subscriber(_BadSub())
        bus.start()
        bus.publish(_make_event())
        time.sleep(0.15)
        bus.stop()

        snapshot = bus.subscriber_exception_counts()
        snapshot["_BadSub"] = 99
        assert bus.subscriber_exception_counts().get("_BadSub") == 1

    def test_subscriber_exception_counts_uses_qualname_for_free_function(self, bus):
        def free_callback(event):
            raise RuntimeError("boom")

        bus.subscribe(EventType.L1_READ_FINISHED, free_callback)
        bus.start()
        bus.publish(_make_event())
        time.sleep(0.15)
        bus.stop()

        counts = bus.subscriber_exception_counts()
        # Free functions are labeled by __qualname__, which here lives
        # inside the test method.
        assert any("free_callback" in name for name in counts)

    def test_good_subscriber_not_counted(self, bus):
        class _BadSub(EventSubscriber):
            def get_subscriptions(self):
                return {EventType.L1_READ_FINISHED: self._on_event}

            def _on_event(self, event):
                raise RuntimeError("boom")

        good = _RecordingSubscriber()
        bus.register_subscriber(_BadSub())
        bus.register_subscriber(good)
        bus.start()
        bus.publish(_make_event())
        time.sleep(0.15)
        bus.stop()

        counts = bus.subscriber_exception_counts()
        assert "_BadSub" in counts
        assert "_RecordingSubscriber" not in counts


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestGlobalSingleton:
    def test_get_returns_instance(self):
        bus = get_event_bus()
        assert isinstance(bus, EventBus)

    def test_init_replaces_global(self):
        old = get_event_bus()
        new = init_event_bus(EventBusConfig(enabled=False))
        assert get_event_bus() is new
        assert get_event_bus() is not old

    def test_default_singleton_is_disabled(self):
        bus = get_event_bus()
        assert bus._config.enabled is False

    def test_init_with_none_uses_defaults(self):
        bus = init_event_bus()
        assert bus._config.enabled is True
        assert bus._config.max_queue_size == 10_000


# ---------------------------------------------------------------------------
# Block allocation event
# ---------------------------------------------------------------------------


class TestBlockAllocationEvent:
    def test_publish_block_allocation_event(self, bus):
        """Verify MP_VLLM_BLOCK_ALLOCATION events are delivered to subscribers."""
        sub = _RecordingSubscriber(event_types=[EventType.MP_VLLM_BLOCK_ALLOCATION])
        bus.register_subscriber(sub)
        bus.start()

        # First Party
        from lmcache.v1.multiprocess.custom_types import BlockAllocationRecord

        records = [
            BlockAllocationRecord(
                req_id="req-1",
                new_block_ids=[0, 1, 2],
                new_token_ids=[10, 20, 30],
            ),
        ]
        bus.publish(
            _make_event(
                event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                session_id="",
                records=records,
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert len(sub.events) == 1
        evt = sub.events[0]
        assert evt.event_type == EventType.MP_VLLM_BLOCK_ALLOCATION
        assert len(evt.metadata["records"]) == 1
        assert evt.metadata["records"][0].req_id == "req-1"
        assert evt.metadata["records"][0].new_block_ids == [0, 1, 2]

    def test_block_allocation_not_delivered_to_other_subscriber(self, bus):
        """Verify block allocation events are not delivered to unrelated subscribers."""
        sub = _RecordingSubscriber(event_types=[EventType.L1_READ_FINISHED])
        bus.register_subscriber(sub)
        bus.start()

        bus.publish(
            _make_event(
                event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                session_id="",
            )
        )
        time.sleep(0.15)
        bus.stop()

        assert len(sub.events) == 0


_REGISTER_GAUGE = "lmcache.v1.mp_observability.event_bus.register_gauge"


class TestSelfMonitoringGauges:
    def test_init_registers_both_gauges(self):
        with patch(_REGISTER_GAUGE) as mock_register:
            EventBus(EventBusConfig(enabled=True))
            names = [c.args[1] for c in mock_register.call_args_list]
            assert set(names) == {
                "lmcache_mp.event_bus.queue_depth",
                "lmcache_mp.event_bus.drain_lag_seconds",
            }

    def test_queue_depth_callback_reflects_bus(self):
        captured: dict[str, object] = {}

        def _capture(_meter, name, _desc, func):
            captured[name] = func

        with patch(_REGISTER_GAUGE, side_effect=_capture):
            bus = EventBus(EventBusConfig(enabled=True))
            bus.publish(_make_event(session_id="s1"))
            bus.publish(_make_event(session_id="s2"))
            assert captured["lmcache_mp.event_bus.queue_depth"]() == 2
