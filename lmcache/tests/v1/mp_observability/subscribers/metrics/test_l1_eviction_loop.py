# SPDX-License-Identifier: Apache-2.0

"""Tests for L1EvictionLoopSubscriber."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l1_eviction_loop import (
    L1EvictionLoopSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import (
    counter_delta,
    read_counters,
)

_DRAIN_WAIT = 0.15


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = L1EvictionLoopSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    """Capture counters before the test; yield a callable that returns deltas."""
    before = read_counters()

    def get_delta() -> dict[str, int]:
        return counter_delta(before, read_counters())

    return get_delta


def _tick(triggered: bool, usage: float = 0.5) -> Event:
    return Event(
        event_type=EventType.L1_EVICTION_LOOP_TICK,
        metadata={"usage": usage, "watermark": 0.8, "triggered": triggered},
    )


class TestL1EvictionLoopSubscriber:
    def test_subscribes_only_to_eviction_loop_tick(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L1_EVICTION_LOOP_TICK in subs
        assert len(subs) == 1

    def test_below_watermark_increments_only_ticks(self, bus, subscriber, snapshot):
        bus.start()
        for _ in range(5):
            bus.publish(_tick(triggered=False))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l1_eviction_loop_ticks"] == 5
        assert delta.get("lmcache_mp.l1_eviction_loop_triggered", 0) == 0

    def test_triggered_increments_both_counters(self, bus, subscriber, snapshot):
        bus.start()
        for _ in range(3):
            bus.publish(_tick(triggered=True))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l1_eviction_loop_ticks"] == 3
        assert delta["lmcache_mp.l1_eviction_loop_triggered"] == 3

    def test_mixed_ticks(self, bus, subscriber, snapshot):
        """Ratio of triggered to ticks reflects how often eviction fired."""
        bus.start()
        for _ in range(4):
            bus.publish(_tick(triggered=False))
        for _ in range(6):
            bus.publish(_tick(triggered=True))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l1_eviction_loop_ticks"] == 10
        assert delta["lmcache_mp.l1_eviction_loop_triggered"] == 6

    def test_missing_metadata_uses_safe_defaults(self, bus, subscriber, snapshot):
        """A tick event with empty metadata still increments ``ticks`` (and
        leaves ``triggered`` unchanged) without crashing."""
        bus.start()
        bus.publish(Event(event_type=EventType.L1_EVICTION_LOOP_TICK, metadata={}))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l1_eviction_loop_ticks"] == 1
        assert delta.get("lmcache_mp.l1_eviction_loop_triggered", 0) == 0
