# SPDX-License-Identifier: Apache-2.0

"""Tests for L1MetricsSubscriber."""

# Standard
from types import SimpleNamespace
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l1 import (
    L1MetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

_DRAIN_WAIT = 0.15


def _make_keys(count: int, cache_salt: str = "") -> list:
    """Create a list of placeholder key objects with a cache_salt attribute."""
    return [SimpleNamespace(cache_salt=cache_salt, id=i) for i in range(count)]


def _make_event(event_type: EventType, keys: list) -> Event:
    return Event(event_type=event_type, metadata={"keys": keys})


def _read_counters_by_attrs() -> dict[str, dict[tuple, int]]:
    """Snapshot counter values keyed by (metric_name, frozenset(attrs))."""
    data = _reader.get_metrics_data()
    result: dict[str, dict[tuple, int]] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue
                    key = tuple(sorted(dict(dp.attributes).items()))
                    result.setdefault(metric.name, {})[key] = int(dp.value)
    return result


@pytest.fixture
def bus():
    b = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
    return b


@pytest.fixture
def subscriber(bus):
    sub = L1MetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


class TestL1MetricsSubscriber:
    def test_read_finished_increments_counter(self, bus, subscriber):
        bus.start()
        bus.publish(_make_event(EventType.L1_READ_FINISHED, _make_keys(5)))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

    def test_write_finished_increments_counter(self, bus, subscriber):
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, _make_keys(3)))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

    def test_write_finished_and_read_reserved_increments_write_counter(
        self, bus, subscriber
    ):
        bus.start()
        bus.publish(
            _make_event(EventType.L1_WRITE_FINISHED_AND_READ_RESERVED, _make_keys(7))
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

    def test_evicted_increments_counter(self, bus, subscriber):
        bus.start()
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, _make_keys(4)))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

    def test_no_subscription_for_reserved_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L1_READ_RESERVED not in subs
        assert EventType.L1_WRITE_RESERVED not in subs

    def test_subscriptions_cover_expected_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L1_READ_FINISHED in subs
        assert EventType.L1_WRITE_FINISHED in subs
        assert EventType.L1_WRITE_FINISHED_AND_READ_RESERVED in subs
        assert EventType.L1_KEYS_EVICTED in subs

    def test_multiple_events_accumulate(self, bus, subscriber):
        bus.start()
        for _ in range(10):
            bus.publish(_make_event(EventType.L1_READ_FINISHED, _make_keys(2)))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

    def test_does_not_crash_on_empty_keys(self, bus, subscriber):
        bus.start()
        bus.publish(_make_event(EventType.L1_READ_FINISHED, []))
        time.sleep(_DRAIN_WAIT)
        bus.stop()


class TestL1CacheSaltTagging:
    def test_read_carries_cache_salt(self, bus, subscriber):
        before = _read_counters_by_attrs().get("lmcache_mp.l1_read", {})
        bus.start()
        bus.publish(
            _make_event(
                EventType.L1_READ_FINISHED, _make_keys(5, cache_salt="tenant-a")
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get("lmcache_mp.l1_read", {})
        key = (("cache_salt", "tenant-a"),)
        assert after.get(key, 0) >= before.get(key, 0) + 5

    def test_different_cache_salts_accumulate_independently(self, bus, subscriber):
        before = _read_counters_by_attrs().get("lmcache_mp.l1_write", {})
        bus.start()
        bus.publish(
            _make_event(
                EventType.L1_WRITE_FINISHED, _make_keys(3, cache_salt="tenant-a")
            )
        )
        bus.publish(
            _make_event(
                EventType.L1_WRITE_FINISHED, _make_keys(7, cache_salt="tenant-b")
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get("lmcache_mp.l1_write", {})
        key_a = (("cache_salt", "tenant-a"),)
        key_b = (("cache_salt", "tenant-b"),)
        assert after.get(key_a, 0) >= before.get(key_a, 0) + 3
        assert after.get(key_b, 0) >= before.get(key_b, 0) + 7

    def test_mixed_salts_in_single_event(self, bus, subscriber):
        """A single event with keys from two tenants splits correctly."""
        mixed_keys = _make_keys(3, cache_salt="salt-x") + _make_keys(
            2, cache_salt="salt-y"
        )
        before = _read_counters_by_attrs().get("lmcache_mp.l1_read", {})
        bus.start()
        bus.publish(_make_event(EventType.L1_READ_FINISHED, mixed_keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get("lmcache_mp.l1_read", {})
        key_x = (("cache_salt", "salt-x"),)
        key_y = (("cache_salt", "salt-y"),)
        assert after.get(key_x, 0) >= before.get(key_x, 0) + 3
        assert after.get(key_y, 0) >= before.get(key_y, 0) + 2

    def test_empty_cache_salt_omits_attribute(self, bus, subscriber):
        """Empty salt produces a dimensionless counter (no cache_salt attr)."""
        before = _read_counters_by_attrs().get("lmcache_mp.l1_evicted", {})
        bus.start()
        bus.publish(
            _make_event(EventType.L1_KEYS_EVICTED, _make_keys(4, cache_salt=""))
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get("lmcache_mp.l1_evicted", {})
        key = ()  # no attributes
        assert after.get(key, 0) >= before.get(key, 0) + 4
