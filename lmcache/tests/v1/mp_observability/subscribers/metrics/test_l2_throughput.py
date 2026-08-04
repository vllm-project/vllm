# SPDX-License-Identifier: Apache-2.0

"""Tests for L2ThroughputSubscriber.

Uses ``InMemoryMetricReader`` (via ``otel_setup``) to read back OTel
histogram values.  Most tests invoke handlers directly to stay
deterministic; the end-to-end test drives through ``EventBus``.
"""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l2_throughput import (
    L2ThroughputSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

_DRAIN_WAIT = 0.15
_STORE_METRIC = "lmcache_mp.l2_store_throughput"
_LOAD_METRIC = "lmcache_mp.l2_load_throughput"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_submitted(
    task_id: int,
    t: float,
    adapter_index: int = 0,
    l2_name: str = "fs",
    total_bytes: int = 0,
) -> Event:
    return Event(
        event_type=EventType.L2_STORE_SUBMITTED,
        timestamp=t,
        metadata={
            "adapter_index": adapter_index,
            "task_id": task_id,
            "l2_name": l2_name,
            "key_count": 1,
            "total_bytes": total_bytes,
        },
    )


def _store_completed(
    task_id: int,
    t: float,
    adapter_index: int = 0,
    l2_name: str = "fs",
    bytes_transferred: int | None = None,
) -> Event:
    # ``bytes_transferred`` is populated by fast-path-aware adapters (see
    # ``store_controller._finalize_store``).  Absence means the subscriber
    # falls back to the submitted-bytes accounting it cached at SUBMITTED
    # time.
    metadata: dict = {
        "adapter_index": adapter_index,
        "task_id": task_id,
        "l2_name": l2_name,
        "succeeded_count": 1,
        "failed_count": 0,
    }
    if bytes_transferred is not None:
        metadata["bytes_transferred"] = bytes_transferred
    return Event(
        event_type=EventType.L2_STORE_COMPLETED,
        timestamp=t,
        metadata=metadata,
    )


def _load_submitted(
    request_id: int,
    t: float,
    adapter_index: int = 0,
    l2_name: str = "fs",
    total_bytes: int = 0,
    task_id: int = 0,
) -> Event:
    return Event(
        event_type=EventType.L2_LOAD_TASK_SUBMITTED,
        timestamp=t,
        metadata={
            "request_id": request_id,
            "adapter_index": adapter_index,
            "task_id": task_id,
            "l2_name": l2_name,
            "key_count": 1,
            "total_bytes": total_bytes,
        },
    )


def _load_completed(
    request_id: int,
    t: float,
    adapter_index: int = 0,
    l2_name: str = "fs",
    task_id: int = 0,
) -> Event:
    # L2_LOAD_TASK_COMPLETED does not carry total_bytes; bytes are
    # looked up from the subscriber's SUBMITTED-time cache.
    return Event(
        event_type=EventType.L2_LOAD_TASK_COMPLETED,
        timestamp=t,
        metadata={
            "request_id": request_id,
            "adapter_index": adapter_index,
            "task_id": task_id,
            "l2_name": l2_name,
        },
    )


def _read_histograms() -> dict[str, list]:
    data = _reader.get_metrics_data()
    result: dict[str, list] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                result[metric.name] = list(metric.data.data_points)
    return result


def _total_count(name: str) -> int:
    dps = _read_histograms().get(name, [])
    return sum(dp.count for dp in dps)


def _sum(name: str) -> float:
    dps = _read_histograms().get(name, [])
    return sum(dp.sum for dp in dps)


def _attrs_of_nonzero_dps(name: str) -> list[dict]:
    dps = _read_histograms().get(name, [])
    return [dict(dp.attributes) for dp in dps if dp.count > 0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def subscriber():
    return L2ThroughputSubscriber()


# ---------------------------------------------------------------------------
# Subscription surface
# ---------------------------------------------------------------------------


class TestSubscriptions:
    def test_covers_expected_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L2_STORE_SUBMITTED in subs
        assert EventType.L2_STORE_COMPLETED in subs
        assert EventType.L2_LOAD_TASK_SUBMITTED in subs
        assert EventType.L2_LOAD_TASK_COMPLETED in subs

    def test_does_not_subscribe_to_request_level_load_events(self, subscriber):
        # The request-level L2_PREFETCH_LOAD_* events aggregate across
        # adapters; per-adapter throughput requires the task-level events.
        subs = subscriber.get_subscriptions()
        assert EventType.L2_PREFETCH_LOAD_SUBMITTED not in subs
        assert EventType.L2_PREFETCH_LOAD_COMPLETED not in subs


# ---------------------------------------------------------------------------
# Happy-path throughput recording
# ---------------------------------------------------------------------------


class TestStoreThroughput:
    def test_records_gbps_with_l2_name(self, subscriber):
        count_before = _total_count(_STORE_METRIC)
        sum_before = _sum(_STORE_METRIC)

        # 2 GB in 0.1 s -> 20 GB/s
        subscriber._on_store_submitted(
            _store_submitted(
                task_id=1,
                t=1000.0,
                adapter_index=0,
                l2_name="nixl",
                total_bytes=2_000_000_000,
            )
        )
        subscriber._on_store_completed(
            _store_completed(task_id=1, t=1000.1, adapter_index=0, l2_name="nixl")
        )

        assert _total_count(_STORE_METRIC) == count_before + 1
        assert _sum(_STORE_METRIC) - sum_before == pytest.approx(20.0, rel=1e-6)

        attrs = _attrs_of_nonzero_dps(_STORE_METRIC)
        assert any(a.get("l2_name") == "nixl" for a in attrs)

    def test_drains_pending_dict_on_completed(self, subscriber):
        subscriber._on_store_submitted(
            _store_submitted(task_id=7, t=0.0, total_bytes=1_000)
        )
        assert (0, 7) in subscriber._pending_store

        subscriber._on_store_completed(_store_completed(task_id=7, t=0.1))
        assert (0, 7) not in subscriber._pending_store


class TestLoadThroughput:
    def test_records_gbps_with_l2_name(self, subscriber):
        count_before = _total_count(_LOAD_METRIC)
        sum_before = _sum(_LOAD_METRIC)

        # 500 MB in 0.05 s -> 10 GB/s
        subscriber._on_load_submitted(
            _load_submitted(
                request_id=42,
                t=2000.0,
                adapter_index=1,
                l2_name="mooncake_store",
                total_bytes=500_000_000,
            )
        )
        subscriber._on_load_completed(
            _load_completed(
                request_id=42,
                t=2000.05,
                adapter_index=1,
                l2_name="mooncake_store",
            )
        )

        assert _total_count(_LOAD_METRIC) == count_before + 1
        assert _sum(_LOAD_METRIC) - sum_before == pytest.approx(10.0, rel=1e-6)

        attrs = _attrs_of_nonzero_dps(_LOAD_METRIC)
        assert any(a.get("l2_name") == "mooncake_store" for a in attrs)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_completed_without_submitted_is_noop(self, subscriber):
        count_before = _total_count(_STORE_METRIC)
        subscriber._on_store_completed(_store_completed(task_id=999, t=1.0))
        assert _total_count(_STORE_METRIC) == count_before

    def test_zero_bytes_is_noop(self, subscriber):
        count_before = _total_count(_STORE_METRIC)
        subscriber._on_store_submitted(
            _store_submitted(task_id=3, t=0.0, total_bytes=0)
        )
        subscriber._on_store_completed(_store_completed(task_id=3, t=0.1))
        assert _total_count(_STORE_METRIC) == count_before

    def test_nonpositive_duration_is_noop(self, subscriber):
        count_before = _total_count(_STORE_METRIC)
        subscriber._on_store_submitted(
            _store_submitted(task_id=4, t=5.0, total_bytes=10**9)
        )
        subscriber._on_store_completed(_store_completed(task_id=4, t=5.0))
        assert _total_count(_STORE_METRIC) == count_before

    def test_missing_correlation_fields_skips(self, subscriber):
        # Missing task_id on the submitted event.
        subscriber._on_store_submitted(
            Event(
                event_type=EventType.L2_STORE_SUBMITTED,
                timestamp=1.0,
                metadata={"adapter_index": 0, "l2_name": "fs"},
            )
        )
        assert len(subscriber._pending_store) == 0

    def test_same_task_id_different_adapters_do_not_collide(self, subscriber):
        # Task IDs are only unique per-adapter; the compound key must
        # keep different adapters separate.
        count_before = _total_count(_STORE_METRIC)

        subscriber._on_store_submitted(
            _store_submitted(
                task_id=1,
                t=100.0,
                adapter_index=0,
                l2_name="fs",
                total_bytes=1_000_000_000,
            )
        )
        subscriber._on_store_submitted(
            _store_submitted(
                task_id=1,
                t=100.0,
                adapter_index=1,
                l2_name="nixl",
                total_bytes=1_000_000_000,
            )
        )
        assert (0, 1) in subscriber._pending_store
        assert (1, 1) in subscriber._pending_store

        subscriber._on_store_completed(
            _store_completed(task_id=1, t=100.1, adapter_index=0, l2_name="fs")
        )
        subscriber._on_store_completed(
            _store_completed(task_id=1, t=100.1, adapter_index=1, l2_name="nixl")
        )

        assert _total_count(_STORE_METRIC) == count_before + 2
        assert (0, 1) not in subscriber._pending_store
        assert (1, 1) not in subscriber._pending_store

    def test_store_and_load_dicts_are_independent(self, subscriber):
        # Store and load use structurally similar compound keys but
        # different pending dicts; no cross-pollination.
        subscriber._on_store_submitted(
            _store_submitted(task_id=5, t=0.0, total_bytes=10**9)
        )
        subscriber._on_load_submitted(
            _load_submitted(request_id=5, t=0.0, adapter_index=0, total_bytes=10**9)
        )

        subscriber._on_store_completed(_store_completed(task_id=5, t=0.1))
        assert (0, 5) not in subscriber._pending_store
        # Load pending dict untouched — different key space (request_id).
        assert (5, 0) in subscriber._pending_load


# ---------------------------------------------------------------------------
# bytes_transferred override (fast-path-aware adapters)
# ---------------------------------------------------------------------------


class TestStoreBytesTransferred:
    """Store path prefers ``bytes_transferred`` from the COMPLETED event
    over the submitted-bytes cache when the adapter populates it."""

    def test_uses_bytes_transferred_when_present(self, subscriber):
        count_before = _total_count(_STORE_METRIC)
        sum_before = _sum(_STORE_METRIC)

        # Submitted = 10 GB but adapter actually transferred only 1 GB
        # (rest were fast-pathed duplicates).  Over 0.25 s -> 4 GB/s,
        # not 40 GB/s.
        subscriber._on_store_submitted(
            _store_submitted(task_id=10, t=0.0, total_bytes=10_000_000_000)
        )
        subscriber._on_store_completed(
            _store_completed(task_id=10, t=0.25, bytes_transferred=1_000_000_000)
        )

        assert _total_count(_STORE_METRIC) == count_before + 1
        assert _sum(_STORE_METRIC) - sum_before == pytest.approx(4.0, rel=1e-6)

    def test_zero_bytes_transferred_drops_sample(self, subscriber):
        count_before = _total_count(_STORE_METRIC)
        sum_before = _sum(_STORE_METRIC)

        # Adapter fast-pathed every key -- no real work, drop the sample
        # even though submitted bytes and dt look reasonable.
        subscriber._on_store_submitted(
            _store_submitted(task_id=11, t=0.0, total_bytes=10_000_000_000)
        )
        subscriber._on_store_completed(
            _store_completed(task_id=11, t=0.1, bytes_transferred=0)
        )

        assert _total_count(_STORE_METRIC) == count_before
        assert _sum(_STORE_METRIC) == sum_before
        # Pending dict still drained even when sample dropped.
        assert (0, 11) not in subscriber._pending_store

    def test_absent_field_falls_back_to_submitted_bytes(self, subscriber):
        # Adapters that don't track per-task transfer bytes omit the
        # field; behavior matches the load path -- submitted bytes / dt.
        count_before = _total_count(_STORE_METRIC)
        sum_before = _sum(_STORE_METRIC)

        # 2 GB in 0.1 s -> 20 GB/s
        subscriber._on_store_submitted(
            _store_submitted(task_id=12, t=0.0, total_bytes=2_000_000_000)
        )
        subscriber._on_store_completed(
            _store_completed(task_id=12, t=0.1)  # no bytes_transferred
        )

        assert _total_count(_STORE_METRIC) == count_before + 1
        assert _sum(_STORE_METRIC) - sum_before == pytest.approx(20.0, rel=1e-6)


# ---------------------------------------------------------------------------
# End-to-end: drive through EventBus
# ---------------------------------------------------------------------------


class TestEventBusIntegration:
    def test_store_pair_via_bus_records_metric(self):
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        sub = L2ThroughputSubscriber()
        bus.register_subscriber(sub)

        count_before = _total_count(_STORE_METRIC)
        bus.start()
        bus.publish(
            _store_submitted(
                task_id=77,
                t=100.0,
                adapter_index=2,
                l2_name="fs",
                total_bytes=4_000_000_000,
            )
        )
        bus.publish(
            _store_completed(task_id=77, t=100.2, adapter_index=2, l2_name="fs")
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert _total_count(_STORE_METRIC) == count_before + 1
        attrs = _attrs_of_nonzero_dps(_STORE_METRIC)
        assert any(a.get("l2_name") == "fs" for a in attrs)
