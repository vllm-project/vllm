# SPDX-License-Identifier: Apache-2.0

"""Tests for L1LifecycleSubscriber.

Uses ``InMemoryMetricReader`` to read back actual OTel histogram values
and verify lifecycle tracking with sampling.

NOTE: OTel only allows one MeterProvider per process. If these tests run
in the same process as other test files, the provider is already set. We
re-read from the same global reader.
"""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l1_lifecycle import (
    L1LifecycleSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keys(prefix: str, count: int) -> list:
    return [f"{prefix}-{i}" for i in range(count)]


def _make_event(event_type: EventType, keys: list) -> Event:
    return Event(event_type=event_type, metadata={"keys": keys})


def _get_histogram_count(name: str) -> int:
    data = _reader.get_metrics_data()
    result: dict[str, list] = {}
    if data is None:
        return 0
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                result[metric.name] = list(metric.data.data_points)
    dps = result.get(name, [])
    return sum(dp.count for dp in dps)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = L1LifecycleSubscriber(sample_rate=1.0)
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def sampled_subscriber(bus):
    """Subscriber with 0% sample rate — nothing should be tracked."""
    sub = L1LifecycleSubscriber(sample_rate=1e-9)
    bus.register_subscriber(sub)
    return sub


# ---------------------------------------------------------------------------
# Tests: Shadow map and lifecycle
# ---------------------------------------------------------------------------


class TestL1Lifecycle:
    def test_write_populates_shadow(self, bus, subscriber):
        keys = ["life-a", "life-b"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        assert "life-a" in subscriber._shadow
        assert "life-b" in subscriber._shadow

    def test_eviction_records_lifetime(self, bus, subscriber):
        count_before = _get_histogram_count("lmcache_mp.l1_chunk_lifetime")
        keys = ["lt-1"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        count_after = _get_histogram_count("lmcache_mp.l1_chunk_lifetime")
        assert count_after == count_before + 1

    def test_eviction_records_idle(self, bus, subscriber):
        count_before = _get_histogram_count("lmcache_mp.l1_chunk_idle_before_evict")
        keys = ["idle-1"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        count_after = _get_histogram_count("lmcache_mp.l1_chunk_idle_before_evict")
        assert count_after == count_before + 1

    def test_eviction_removes_from_shadow(self, bus, subscriber):
        keys = ["rm-1"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        assert "rm-1" not in subscriber._shadow

    def test_eviction_without_write_no_crash(self, bus, subscriber):
        """Evicting a key that was never written should not crash."""
        bus.start()
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, ["ghost-1"]))
        time.sleep(_DRAIN_WAIT)
        bus.stop()


# ---------------------------------------------------------------------------
# Tests: Reuse gap
# ---------------------------------------------------------------------------


class TestL1ReuseGap:
    def test_read_after_write_records_reuse_gap(self, bus, subscriber):
        count_before = _get_histogram_count("lmcache_mp.l1_chunk_reuse_gap")
        keys = ["rg-1"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_READ_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        count_after = _get_histogram_count("lmcache_mp.l1_chunk_reuse_gap")
        assert count_after == count_before + 1

    def test_rewrite_records_reuse_gap(self, bus, subscriber):
        count_before = _get_histogram_count("lmcache_mp.l1_chunk_reuse_gap")
        keys = ["rg-2"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        count_after = _get_histogram_count("lmcache_mp.l1_chunk_reuse_gap")
        assert count_after == count_before + 1

    def test_read_untracked_key_no_gap(self, bus, subscriber):
        """Reading a key never written should not record a gap."""
        count_before = _get_histogram_count("lmcache_mp.l1_chunk_reuse_gap")
        bus.start()
        bus.publish(_make_event(EventType.L1_READ_FINISHED, ["nowrite-1"]))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        count_after = _get_histogram_count("lmcache_mp.l1_chunk_reuse_gap")
        assert count_after == count_before


# ---------------------------------------------------------------------------
# Tests: Evict-reuse gap
# ---------------------------------------------------------------------------


class TestL1EvictReuseGap:
    def test_rewrite_after_eviction_records_gap(self, bus, subscriber):
        count_before = _get_histogram_count("lmcache_mp.l1_chunk_evict_reuse_gap")
        keys = ["erg-1"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        count_after = _get_histogram_count("lmcache_mp.l1_chunk_evict_reuse_gap")
        assert count_after == count_before + 1

    def test_evicted_key_tracked_in_evicted_at(self, bus, subscriber):
        keys = ["erg-2"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        assert "erg-2" in subscriber._evicted_at

    def test_rewrite_clears_evicted_at(self, bus, subscriber):
        keys = ["erg-3"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        assert "erg-3" not in subscriber._evicted_at


# ---------------------------------------------------------------------------
# Tests: Sampling
# ---------------------------------------------------------------------------


class TestL1Sampling:
    def test_sample_rate_1_tracks_all(self, bus, subscriber):
        """With sample_rate=1.0, all keys should be tracked."""
        keys = _make_keys("samp", 10)
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        assert len(subscriber._shadow) >= 10

    def test_sample_rate_zero_tracks_none(self, bus, sampled_subscriber):
        """With near-zero sample rate, no keys should be tracked."""
        keys = _make_keys("nosamp", 100)
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        assert len(sampled_subscriber._shadow) == 0

    def test_deterministic_sampling_is_consistent(self, bus):
        """Same key always gets the same sampling decision."""
        sub = L1LifecycleSubscriber(sample_rate=0.5)
        bus.register_subscriber(sub)
        keys = ["det-1"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        tracked_first = "det-1" in sub._shadow
        # Evict and re-write — should get same decision
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        tracked_second = "det-1" in sub._shadow
        assert tracked_first == tracked_second

    def test_unsampled_key_ignored_on_eviction(self, bus, sampled_subscriber):
        """Evicting an unsampled key should not record lifetime."""
        count_before = _get_histogram_count("lmcache_mp.l1_chunk_lifetime")
        keys = ["skip-ev-1"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        count_after = _get_histogram_count("lmcache_mp.l1_chunk_lifetime")
        assert count_after == count_before


# ---------------------------------------------------------------------------
# Tests: Sweep stale evictions
# ---------------------------------------------------------------------------


class TestL1SweepStaleEvictions:
    def test_sweep_removes_old_entries(self, bus):
        sub = L1LifecycleSubscriber(sample_rate=1.0, max_evict_reuse_wait=0.1)
        bus.register_subscriber(sub)

        keys = ["sweep-1"]
        bus.start()
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, keys))
        time.sleep(_DRAIN_WAIT)
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, keys))
        time.sleep(_DRAIN_WAIT)

        # Wait longer than max_evict_reuse_wait
        time.sleep(0.2)

        # Trigger sweep via another write
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, ["sweep-trigger"]))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert "sweep-1" not in sub._evicted_at


# ---------------------------------------------------------------------------
# Tests: Subscriptions
# ---------------------------------------------------------------------------


class TestL1Subscriptions:
    def test_subscriptions_cover_expected_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L1_READ_FINISHED in subs
        assert EventType.L1_WRITE_FINISHED in subs
        assert EventType.L1_WRITE_FINISHED_AND_READ_RESERVED in subs
        assert EventType.L1_KEYS_EVICTED in subs

    def test_no_subscription_for_reserved_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L1_READ_RESERVED not in subs
        assert EventType.L1_WRITE_RESERVED not in subs

    def test_does_not_crash_on_empty_keys(self, bus, subscriber):
        bus.start()
        bus.publish(_make_event(EventType.L1_READ_FINISHED, []))
        bus.publish(_make_event(EventType.L1_WRITE_FINISHED, []))
        bus.publish(_make_event(EventType.L1_KEYS_EVICTED, []))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
