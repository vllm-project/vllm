# SPDX-License-Identifier: Apache-2.0

"""Tests for L2MetricsSubscriber.

Uses ``InMemoryMetricReader`` to read back actual OTel counter values
and assert exact counts after publishing known events through the EventBus.

OTel only allows one MeterProvider per process, so we use a module-scoped
provider and assert on counter **deltas** between before/after snapshots.
"""

# Standard
from collections import Counter
from types import SimpleNamespace
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l2 import (
    L2MetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keys(count: int, cache_salt: str = "") -> list:
    return [SimpleNamespace(cache_salt=cache_salt, id=i) for i in range(count)]


def _salt_counts(keys: list) -> dict[str, int]:
    return dict(Counter(k.cache_salt for k in keys))


def _read_counters() -> dict[str, int]:
    """Snapshot all counter values from the module-level reader, summed
    across attribute combinations.  A counter with multiple labeled data
    points (e.g. ``l2_name="fs"`` and ``l2_name="nixl"``) reports the
    aggregate; tests that need per-label values use ``_read_counters_by_attrs``.
    """
    data = _reader.get_metrics_data()
    result: dict[str, int] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                total = 0
                any_value = False
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue  # skip histogram data points
                    total += int(dp.value)
                    any_value = True
                if any_value:
                    result[metric.name] = total
    return result


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


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Compute the difference between two counter snapshots."""
    all_keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in all_keys}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = L2MetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    """Capture counters before the test; yield a callable that returns deltas."""
    before = _read_counters()

    def get_delta() -> dict[str, int]:
        return _counter_delta(before, _read_counters())

    return get_delta


# ---------------------------------------------------------------------------
# Store events
# ---------------------------------------------------------------------------


class TestL2StoreMetrics:
    def test_store_submitted_counts(self, bus, subscriber, snapshot):
        bus.start()
        keys_a = _make_keys(10)
        keys_b = _make_keys(5)
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                metadata={
                    "adapter_index": 0,
                    "key_count": 10,
                    "key_count_per_salt": _salt_counts(keys_a),
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                metadata={
                    "adapter_index": 1,
                    "key_count": 5,
                    "key_count_per_salt": _salt_counts(keys_b),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_store_submitted"] == 2
        assert delta["lmcache_mp.l2_store_submitted_objects"] == 15

    def test_store_completed_success(self, bus, subscriber, snapshot):
        bus.start()
        keys = _make_keys(8)
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_COMPLETED,
                metadata={
                    "adapter_index": 0,
                    "succeeded_count": 8,
                    "failed_count": 0,
                    "key_count_per_salt": _salt_counts(keys),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_store_completed"] == 1
        assert delta["lmcache_mp.l2_store_completed_objects"] == 8

    def test_store_completed_with_failures(self, bus, subscriber, snapshot):
        bus.start()
        succeeded = _make_keys(3)
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_COMPLETED,
                metadata={
                    "adapter_index": 0,
                    "succeeded_count": 3,
                    "failed_count": 7,
                    "key_count_per_salt": _salt_counts(succeeded),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_store_completed"] == 1
        assert delta["lmcache_mp.l2_store_completed_objects"] == 3

    def test_store_full_lifecycle(self, bus, subscriber, snapshot):
        """Simulate warmup: submit 20 keys, all succeed."""
        bus.start()
        keys = _make_keys(20)
        sc = _salt_counts(keys)
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                metadata={
                    "adapter_index": 0,
                    "key_count": 20,
                    "key_count_per_salt": sc,
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_COMPLETED,
                metadata={
                    "adapter_index": 0,
                    "succeeded_count": 20,
                    "failed_count": 0,
                    "key_count_per_salt": sc,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_store_submitted"] == 1
        assert delta["lmcache_mp.l2_store_submitted_objects"] == 20
        assert delta["lmcache_mp.l2_store_completed"] == 1
        assert delta["lmcache_mp.l2_store_completed_objects"] == 20


# ---------------------------------------------------------------------------
# Prefetch events
# ---------------------------------------------------------------------------


class TestL2PrefetchMetrics:
    def test_lookup_submitted_counts(self, bus, subscriber, snapshot):
        bus.start()
        keys = _make_keys(12)
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={
                    "request_id": 1,
                    "key_count": 12,
                    "adapter_count": 2,
                    "key_count_per_salt": _salt_counts(keys),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_lookup"] == 1
        assert delta["lmcache_mp.l2_prefetch_lookup_objects"] == 12

    def test_lookup_completed_counts_hits(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                metadata={"request_id": 1, "prefix_hit_count": 10},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_hit"] == 10

    def test_load_submitted_counts(self, bus, subscriber, snapshot):
        bus.start()
        keys = _make_keys(10)
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                metadata={
                    "request_id": 1,
                    "key_count": 10,
                    "adapter_count": 2,
                    "key_count_per_salt": _salt_counts(keys),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_load_submitted"] == 2
        assert delta["lmcache_mp.l2_prefetch_load_submitted_objects"] == 10

    def test_load_completed_counts(self, bus, subscriber, snapshot):
        bus.start()
        loaded = _make_keys(9)
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                metadata={
                    "request_id": 1,
                    "loaded_count": 9,
                    "failed_count": 1,
                    "key_count_per_salt": _salt_counts(loaded),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_load_completed"] == 9

    def test_prefetch_full_lifecycle(self, bus, subscriber, snapshot):
        """Simulate query: lookup 20 keys, 18 prefix hits, all 18 load OK."""
        bus.start()
        lookup_keys = _make_keys(20)
        load_keys = _make_keys(18)
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={
                    "request_id": 42,
                    "key_count": 20,
                    "adapter_count": 1,
                    "key_count_per_salt": _salt_counts(lookup_keys),
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                metadata={"request_id": 42, "prefix_hit_count": 18},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                metadata={
                    "request_id": 42,
                    "key_count": 18,
                    "adapter_count": 1,
                    "key_count_per_salt": _salt_counts(load_keys),
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                metadata={
                    "request_id": 42,
                    "loaded_count": 18,
                    "failed_count": 0,
                    "key_count_per_salt": _salt_counts(load_keys),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_lookup"] == 1
        assert delta["lmcache_mp.l2_prefetch_lookup_objects"] == 20
        assert delta["lmcache_mp.l2_prefetch_hit"] == 18
        assert delta["lmcache_mp.l2_prefetch_load_submitted"] == 1
        assert delta["lmcache_mp.l2_prefetch_load_submitted_objects"] == 18
        assert delta["lmcache_mp.l2_prefetch_load_completed"] == 18


# ---------------------------------------------------------------------------
# Eviction events
# ---------------------------------------------------------------------------


class TestL2EvictionMetrics:
    def test_eviction_counts(self, bus, subscriber, snapshot):
        bus.start()
        keys = _make_keys(7)
        bus.publish(
            Event(
                event_type=EventType.L2_KEYS_EVICTED,
                metadata={
                    "key_count": 7,
                    "key_count_per_salt": _salt_counts(keys),
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_evicted_objects"] == 7


# ---------------------------------------------------------------------------
# Subscription wiring
# ---------------------------------------------------------------------------


class TestL2MetricsSubscriptions:
    def test_subscriptions_cover_all_l2_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L2_STORE_SUBMITTED in subs
        assert EventType.L2_STORE_COMPLETED in subs
        assert EventType.L2_LOAD_TASK_COMPLETED in subs
        assert EventType.L2_PREFETCH_LOOKUP_SUBMITTED in subs
        assert EventType.L2_PREFETCH_LOOKUP_COMPLETED in subs
        assert EventType.L2_PREFETCH_LOAD_SUBMITTED in subs
        assert EventType.L2_PREFETCH_LOAD_COMPLETED in subs
        assert EventType.L2_KEYS_EVICTED in subs
        assert len(subs) == 8


# ---------------------------------------------------------------------------
# l2_name-labeled counters (for per-backend IOPS via rate())
# ---------------------------------------------------------------------------


class TestL2NameLabeledCounters:
    def test_store_completed_carries_l2_name(self, bus, subscriber):
        bus.start()
        before = _read_counters_by_attrs().get("lmcache_mp.l2_store_completed", {})
        keys = _make_keys(5)
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_COMPLETED,
                metadata={
                    "adapter_index": 0,
                    "task_id": 1,
                    "l2_name": "fs",
                    "succeeded_count": 5,
                    "failed_count": 0,
                    "key_count_per_salt": _salt_counts(keys),
                    "total_bytes": 1_000,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get("lmcache_mp.l2_store_completed", {})
        fs_key = (("l2_name", "fs"),)
        assert after.get(fs_key, 0) == before.get(fs_key, 0) + 1

    def test_load_task_completed_carries_l2_name(self, bus, subscriber):
        bus.start()
        before = _read_counters_by_attrs().get("lmcache_mp.l2_load_completed", {})
        bus.publish(
            Event(
                event_type=EventType.L2_LOAD_TASK_COMPLETED,
                metadata={
                    "request_id": 7,
                    "adapter_index": 1,
                    "task_id": 42,
                    "l2_name": "nixl_store",
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get("lmcache_mp.l2_load_completed", {})
        nixl_key = (("l2_name", "nixl_store"),)
        assert after.get(nixl_key, 0) == before.get(nixl_key, 0) + 1

    def test_different_l2_names_accumulate_independently(self, bus, subscriber):
        bus.start()
        before = _read_counters_by_attrs().get("lmcache_mp.l2_load_completed", {})
        # 3x fs, 2x nixl_store completions.
        for _ in range(3):
            bus.publish(
                Event(
                    event_type=EventType.L2_LOAD_TASK_COMPLETED,
                    metadata={
                        "request_id": 1,
                        "adapter_index": 0,
                        "task_id": 1,
                        "l2_name": "fs",
                    },
                )
            )
        for _ in range(2):
            bus.publish(
                Event(
                    event_type=EventType.L2_LOAD_TASK_COMPLETED,
                    metadata={
                        "request_id": 1,
                        "adapter_index": 1,
                        "task_id": 1,
                        "l2_name": "nixl_store",
                    },
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get("lmcache_mp.l2_load_completed", {})
        fs_key = (("l2_name", "fs"),)
        nixl_key = (("l2_name", "nixl_store"),)
        assert after.get(fs_key, 0) == before.get(fs_key, 0) + 3
        assert after.get(nixl_key, 0) == before.get(nixl_key, 0) + 2


# ---------------------------------------------------------------------------
# Accumulation across multiple events
# ---------------------------------------------------------------------------


class TestL2MetricsAccumulation:
    def test_multiple_store_events_accumulate(self, bus, subscriber, snapshot):
        bus.start()
        keys = _make_keys(3)
        sc = _salt_counts(keys)
        for _ in range(5):
            bus.publish(
                Event(
                    event_type=EventType.L2_STORE_SUBMITTED,
                    metadata={
                        "adapter_index": 0,
                        "key_count": 3,
                        "key_count_per_salt": sc,
                    },
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.L2_STORE_COMPLETED,
                    metadata={
                        "adapter_index": 0,
                        "succeeded_count": 3,
                        "failed_count": 0,
                        "key_count_per_salt": sc,
                    },
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_store_submitted"] == 5
        assert delta["lmcache_mp.l2_store_submitted_objects"] == 15
        assert delta["lmcache_mp.l2_store_completed"] == 5
        assert delta["lmcache_mp.l2_store_completed_objects"] == 15

    def test_multiple_prefetch_events_accumulate(self, bus, subscriber, snapshot):
        bus.start()
        lookup_sc = _salt_counts(_make_keys(10))
        load_sc = _salt_counts(_make_keys(8))
        loaded_sc = _salt_counts(_make_keys(7))
        for i in range(3):
            bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                    metadata={
                        "request_id": i,
                        "key_count": 10,
                        "adapter_count": 1,
                        "key_count_per_salt": lookup_sc,
                    },
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                    metadata={"request_id": i, "prefix_hit_count": 8},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                    metadata={
                        "request_id": i,
                        "key_count": 8,
                        "adapter_count": 1,
                        "key_count_per_salt": load_sc,
                    },
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                    metadata={
                        "request_id": i,
                        "loaded_count": 7,
                        "failed_count": 1,
                        "key_count_per_salt": loaded_sc,
                    },
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert delta["lmcache_mp.l2_prefetch_lookup"] == 3
        assert delta["lmcache_mp.l2_prefetch_lookup_objects"] == 30
        assert delta["lmcache_mp.l2_prefetch_hit"] == 24
        assert delta["lmcache_mp.l2_prefetch_load_submitted"] == 3
        assert delta["lmcache_mp.l2_prefetch_load_submitted_objects"] == 24
        assert delta["lmcache_mp.l2_prefetch_load_completed"] == 21


# ---------------------------------------------------------------------------
# cache_salt-labeled counters
# ---------------------------------------------------------------------------


class TestL2CacheSaltTagging:
    def test_store_submitted_groups_by_salt(self, bus, subscriber):
        """Store submitted object counter groups by cache_salt."""
        before = _read_counters_by_attrs().get(
            "lmcache_mp.l2_store_submitted_objects", {}
        )
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                metadata={
                    "adapter_index": 0,
                    "key_count": 7,
                    "key_count_per_salt": {"tenant-a": 3, "tenant-b": 4},
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get(
            "lmcache_mp.l2_store_submitted_objects", {}
        )
        key_a = (("cache_salt", "tenant-a"),)
        key_b = (("cache_salt", "tenant-b"),)
        assert after.get(key_a, 0) >= before.get(key_a, 0) + 3
        assert after.get(key_b, 0) >= before.get(key_b, 0) + 4

    def test_store_completed_groups_by_salt(self, bus, subscriber):
        """Store completed object counter groups by cache_salt."""
        before = _read_counters_by_attrs().get(
            "lmcache_mp.l2_store_completed_objects", {}
        )
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_COMPLETED,
                metadata={
                    "adapter_index": 0,
                    "task_id": 1,
                    "l2_name": "fs",
                    "succeeded_count": 7,
                    "failed_count": 0,
                    "key_count_per_salt": {"tenant-a": 3, "tenant-b": 4},
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get(
            "lmcache_mp.l2_store_completed_objects", {}
        )
        key_a = (("cache_salt", "tenant-a"),)
        key_b = (("cache_salt", "tenant-b"),)
        assert after.get(key_a, 0) >= before.get(key_a, 0) + 3
        assert after.get(key_b, 0) >= before.get(key_b, 0) + 4

    def test_prefetch_lookup_submitted_groups_by_salt(self, bus, subscriber):
        """Prefetch lookup submitted object counter groups by cache_salt."""
        before = _read_counters_by_attrs().get(
            "lmcache_mp.l2_prefetch_lookup_objects", {}
        )
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={
                    "request_id": 1,
                    "key_count": 5,
                    "adapter_count": 1,
                    "key_count_per_salt": {"tenant-d": 5},
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get(
            "lmcache_mp.l2_prefetch_lookup_objects", {}
        )
        key = (("cache_salt", "tenant-d"),)
        assert after.get(key, 0) >= before.get(key, 0) + 5

    def test_prefetch_load_submitted_groups_by_salt(self, bus, subscriber):
        """Prefetch load submitted object counter groups by cache_salt."""
        before = _read_counters_by_attrs().get(
            "lmcache_mp.l2_prefetch_load_submitted_objects", {}
        )
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                metadata={
                    "request_id": 1,
                    "key_count": 6,
                    "adapter_count": 1,
                    "key_count_per_salt": {"tenant-e": 6},
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get(
            "lmcache_mp.l2_prefetch_load_submitted_objects", {}
        )
        key = (("cache_salt", "tenant-e"),)
        assert after.get(key, 0) >= before.get(key, 0) + 6

    def test_prefetch_load_completed_groups_by_salt(self, bus, subscriber):
        """Prefetch load completed counter groups by cache_salt."""
        before = _read_counters_by_attrs().get(
            "lmcache_mp.l2_prefetch_load_completed", {}
        )
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                metadata={
                    "request_id": 1,
                    "loaded_count": 5,
                    "failed_count": 0,
                    "key_count_per_salt": {"tenant-c": 5},
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get(
            "lmcache_mp.l2_prefetch_load_completed", {}
        )
        key = (("cache_salt", "tenant-c"),)
        assert after.get(key, 0) >= before.get(key, 0) + 5

    def test_evicted_groups_by_salt(self, bus, subscriber):
        """Eviction counter groups by cache_salt."""
        before = _read_counters_by_attrs().get("lmcache_mp.l2_evicted_objects", {})
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_KEYS_EVICTED,
                metadata={
                    "key_count": 9,
                    "key_count_per_salt": {"tenant-a": 4, "tenant-b": 5},
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        after = _read_counters_by_attrs().get("lmcache_mp.l2_evicted_objects", {})
        key_a = (("cache_salt", "tenant-a"),)
        key_b = (("cache_salt", "tenant-b"),)
        assert after.get(key_a, 0) >= before.get(key_a, 0) + 4
        assert after.get(key_b, 0) >= before.get(key_b, 0) + 5
