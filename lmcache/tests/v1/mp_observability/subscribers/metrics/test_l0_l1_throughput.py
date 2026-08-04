# SPDX-License-Identifier: Apache-2.0

"""Tests for L0L1ThroughputSubscriber.

Uses ``InMemoryMetricReader`` (via ``otel_setup``) to read back OTel
histogram values.  Most tests invoke handlers directly to stay
deterministic; the end-to-end test drives through ``EventBus``.
"""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l0_l1_throughput import (
    L0L1ThroughputSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

_DRAIN_WAIT = 0.15
_STORE_METRIC = "lmcache_mp.l0_l1_store_throughput"
_LOAD_METRIC = "lmcache_mp.l0_l1_load_throughput"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_event(
    event_type: EventType,
    session_id: str,
    t: float,
    engine_id: int = 0,
    device: str = f"{torch_device_type}:0",
    model_name: str = "test-model",
) -> Event:
    return Event(
        event_type=event_type,
        timestamp=t,
        session_id=session_id,
        metadata={
            "engine_id": engine_id,
            "device": device,
            "model_name": model_name,
        },
    )


def _end_event(
    event_type: EventType,
    session_id: str,
    t: float,
    total_bytes: int,
    engine_id: int = 0,
    device: str = f"{torch_device_type}:0",
    model_name: str = "test-model",
) -> Event:
    return Event(
        event_type=event_type,
        timestamp=t,
        session_id=session_id,
        metadata={
            "engine_id": engine_id,
            "device": device,
            "model_name": model_name,
            "total_bytes": total_bytes,
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
    return L0L1ThroughputSubscriber()


# ---------------------------------------------------------------------------
# Subscription surface
# ---------------------------------------------------------------------------


class TestSubscriptions:
    def test_covers_expected_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.MP_STORE_START in subs
        assert EventType.MP_STORE_END in subs
        assert EventType.MP_RETRIEVE_START in subs
        assert EventType.MP_RETRIEVE_END in subs

    def test_does_not_subscribe_to_l1_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L1_WRITE_RESERVED not in subs
        assert EventType.L1_READ_RESERVED not in subs


# ---------------------------------------------------------------------------
# Happy-path throughput recording
# ---------------------------------------------------------------------------


class TestStoreThroughput:
    def test_records_gbps_with_attrs(self, subscriber):
        count_before = _total_count(_STORE_METRIC)
        sum_before = _sum(_STORE_METRIC)

        # 2 GB in 0.1 s → 20 GB/s
        subscriber._on_store_start(
            _start_event(
                EventType.MP_STORE_START,
                "req-1",
                1000.0,
                engine_id=7,
                device=f"{torch_device_type}:3",
            )
        )
        subscriber._on_store_end(
            _end_event(
                EventType.MP_STORE_END,
                "req-1",
                1000.1,
                total_bytes=2_000_000_000,
                engine_id=7,
                device=f"{torch_device_type}:3",
            )
        )

        assert _total_count(_STORE_METRIC) == count_before + 1
        observed_delta = _sum(_STORE_METRIC) - sum_before
        assert observed_delta == pytest.approx(20.0, rel=1e-6)

        attrs = _attrs_of_nonzero_dps(_STORE_METRIC)
        assert any(
            a.get("engine_id") == "7"
            and a.get("device") == f"{torch_device_type}:3"
            and a.get("model_name") == "test-model"
            for a in attrs
        )

    def test_drains_pending_dict_on_end(self, subscriber):
        subscriber._on_store_start(
            _start_event(EventType.MP_STORE_START, "req-drain", 0.0)
        )
        assert ("req-drain", f"{torch_device_type}:0") in subscriber._pending_store

        subscriber._on_store_end(
            _end_event(EventType.MP_STORE_END, "req-drain", 0.1, total_bytes=1_000)
        )
        assert ("req-drain", f"{torch_device_type}:0") not in subscriber._pending_store


class TestLoadThroughput:
    def test_records_gbps_with_attrs(self, subscriber):
        count_before = _total_count(_LOAD_METRIC)
        sum_before = _sum(_LOAD_METRIC)

        # 500 MB in 0.05 s → 10 GB/s
        subscriber._on_retrieve_start(
            _start_event(
                EventType.MP_RETRIEVE_START,
                "req-r1",
                2000.0,
                engine_id=2,
                device=f"{torch_device_type}:0",
            )
        )
        subscriber._on_retrieve_end(
            _end_event(
                EventType.MP_RETRIEVE_END,
                "req-r1",
                2000.05,
                total_bytes=500_000_000,
                engine_id=2,
                device=f"{torch_device_type}:0",
            )
        )

        assert _total_count(_LOAD_METRIC) == count_before + 1
        observed_delta = _sum(_LOAD_METRIC) - sum_before
        assert observed_delta == pytest.approx(10.0, rel=1e-6)

        attrs = _attrs_of_nonzero_dps(_LOAD_METRIC)
        assert any(
            a.get("engine_id") == "2"
            and a.get("device") == f"{torch_device_type}:0"
            and a.get("model_name") == "test-model"
            for a in attrs
        )


# ---------------------------------------------------------------------------
# Edge cases — must not crash or emit nonsense
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_end_without_start_is_noop(self, subscriber):
        count_before = _total_count(_STORE_METRIC)
        subscriber._on_store_end(
            _end_event(EventType.MP_STORE_END, "orphan", 1.0, total_bytes=10**9)
        )
        assert _total_count(_STORE_METRIC) == count_before

    def test_zero_bytes_is_noop(self, subscriber):
        count_before = _total_count(_STORE_METRIC)
        subscriber._on_store_start(
            _start_event(EventType.MP_STORE_START, "req-zb", 0.0)
        )
        subscriber._on_store_end(
            _end_event(EventType.MP_STORE_END, "req-zb", 0.1, total_bytes=0)
        )
        assert _total_count(_STORE_METRIC) == count_before

    def test_nonpositive_duration_is_noop(self, subscriber):
        count_before = _total_count(_STORE_METRIC)
        subscriber._on_store_start(
            _start_event(EventType.MP_STORE_START, "req-dt", 5.0)
        )
        # END at same timestamp → dt == 0 → skip.
        subscriber._on_store_end(
            _end_event(EventType.MP_STORE_END, "req-dt", 5.0, total_bytes=10**9)
        )
        assert _total_count(_STORE_METRIC) == count_before

    def test_missing_session_id_skips_tracking(self, subscriber):
        # session_id="" (default) must not populate pending map.
        subscriber._on_store_start(
            Event(
                event_type=EventType.MP_STORE_START,
                timestamp=1.0,
                session_id="",
                metadata={"engine_id": 1, "device": f"{torch_device_type}:0"},
            )
        )
        assert len(subscriber._pending_store) == 0

    def test_missing_device_skips_tracking(self, subscriber):
        # Without "device" in metadata the correlation key is ambiguous
        # across GPUs — event must be dropped.
        subscriber._on_store_start(
            Event(
                event_type=EventType.MP_STORE_START,
                timestamp=1.0,
                session_id="req-no-dev",
                metadata={"engine_id": 1},
            )
        )
        assert len(subscriber._pending_store) == 0

    def test_store_and_load_dicts_are_independent(self, subscriber):
        # Same session_id used for both paths — must not cross-pollinate.
        subscriber._on_store_start(
            _start_event(EventType.MP_STORE_START, "req-dual", 0.0)
        )
        subscriber._on_retrieve_start(
            _start_event(EventType.MP_RETRIEVE_START, "req-dual", 0.0)
        )

        # Finish store only.
        subscriber._on_store_end(
            _end_event(EventType.MP_STORE_END, "req-dual", 0.1, total_bytes=10**9)
        )
        assert ("req-dual", f"{torch_device_type}:0") not in subscriber._pending_store
        assert ("req-dual", f"{torch_device_type}:0") in subscriber._pending_load

    def test_same_session_id_different_devices_do_not_collide(self, subscriber):
        # One MP server handles multiple GPUs; TP replicas of the same
        # request_id must not stomp on each other's pending START timestamp.
        count_before = _total_count(_STORE_METRIC)

        # Two concurrent STARTs, same session_id, different devices.
        subscriber._on_store_start(
            _start_event(
                EventType.MP_STORE_START,
                "tp-req",
                100.0,
                engine_id=0,
                device=f"{torch_device_type}:0",
            )
        )
        subscriber._on_store_start(
            _start_event(
                EventType.MP_STORE_START,
                "tp-req",
                100.0,
                engine_id=1,
                device=f"{torch_device_type}:1",
            )
        )
        assert ("tp-req", f"{torch_device_type}:0") in subscriber._pending_store
        assert ("tp-req", f"{torch_device_type}:1") in subscriber._pending_store

        # Both END independently — both must record.
        subscriber._on_store_end(
            _end_event(
                EventType.MP_STORE_END,
                "tp-req",
                100.1,
                total_bytes=1_000_000_000,
                engine_id=0,
                device=f"{torch_device_type}:0",
            )
        )
        subscriber._on_store_end(
            _end_event(
                EventType.MP_STORE_END,
                "tp-req",
                100.1,
                total_bytes=1_000_000_000,
                engine_id=1,
                device=f"{torch_device_type}:1",
            )
        )

        assert _total_count(_STORE_METRIC) == count_before + 2
        assert ("tp-req", f"{torch_device_type}:0") not in subscriber._pending_store
        assert ("tp-req", f"{torch_device_type}:1") not in subscriber._pending_store


# ---------------------------------------------------------------------------
# End-to-end: drive through EventBus
# ---------------------------------------------------------------------------


class TestEventBusIntegration:
    def test_store_pair_via_bus_records_metric(self):
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        sub = L0L1ThroughputSubscriber()
        bus.register_subscriber(sub)

        count_before = _total_count(_STORE_METRIC)
        bus.start()
        bus.publish(
            _start_event(
                EventType.MP_STORE_START,
                "bus-req",
                100.0,
                engine_id=9,
                device=f"{torch_device_type}:1",
            )
        )
        bus.publish(
            _end_event(
                EventType.MP_STORE_END,
                "bus-req",
                100.2,
                total_bytes=4_000_000_000,
                engine_id=9,
                device=f"{torch_device_type}:1",
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert _total_count(_STORE_METRIC) == count_before + 1
        attrs = _attrs_of_nonzero_dps(_STORE_METRIC)
        assert any(
            a.get("engine_id") == "9" and a.get("device") == f"{torch_device_type}:1"
            for a in attrs
        )
