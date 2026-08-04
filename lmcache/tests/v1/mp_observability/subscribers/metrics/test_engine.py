# SPDX-License-Identifier: Apache-2.0

"""Tests for EngineMetricsSubscriber."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.engine import (
    EngineMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

_DRAIN_WAIT = 0.15
_METRIC = "lmcache_mp.num_chunks_loaded"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _retrieve_end(
    retrieved_count: int,
    engine_id: int = 0,
    model_name: str = "test-model",
    cache_salt: str = "",
    device: str = f"{torch_device_type}:0",
) -> Event:
    return Event(
        event_type=EventType.MP_RETRIEVE_END,
        session_id="req-1",
        metadata={
            "retrieved_count": retrieved_count,
            "device": device,
            "engine_id": engine_id,
            "model_name": model_name,
            "cache_salt": cache_salt,
            "total_bytes": 0,
        },
    )


def _attrs(
    worker_id: str, model_name: str = "test-model", cache_salt: str = ""
) -> tuple:
    """Build the sorted-tuple attribute key the subscriber emits."""
    return tuple(
        sorted(
            {
                "worker_id": worker_id,
                "model_name": model_name,
                "cache_salt": cache_salt,
            }.items()
        )
    )


def _read_counter_by_attrs() -> dict[tuple, int]:
    data = _reader.get_metrics_data()
    result: dict[tuple, int] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != _METRIC:
                    continue
                for dp in metric.data.data_points:
                    if not hasattr(dp, "value"):
                        continue
                    key = tuple(sorted(dict(dp.attributes).items()))
                    result[key] = int(dp.value)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def subscriber():
    return EngineMetricsSubscriber()


# ---------------------------------------------------------------------------
# Subscription surface
# ---------------------------------------------------------------------------


class TestSubscriptions:
    def test_subscribes_to_retrieve_end_only(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.MP_RETRIEVE_END in subs
        # Store path is not of interest here; the counter is load-only.
        assert EventType.MP_STORE_END not in subs


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestNumChunksLoaded:
    def test_single_retrieve_adds_retrieved_count(self, subscriber):
        before = _read_counter_by_attrs()
        subscriber._on_retrieve_end(_retrieve_end(retrieved_count=8, engine_id=3))
        after = _read_counter_by_attrs()

        key = _attrs(worker_id="3")
        assert after.get(key, 0) == before.get(key, 0) + 8

    def test_different_workers_are_independent(self, subscriber):
        before = _read_counter_by_attrs()
        subscriber._on_retrieve_end(_retrieve_end(retrieved_count=5, engine_id=0))
        subscriber._on_retrieve_end(_retrieve_end(retrieved_count=7, engine_id=1))
        subscriber._on_retrieve_end(_retrieve_end(retrieved_count=3, engine_id=0))
        after = _read_counter_by_attrs()

        worker_0 = _attrs(worker_id="0")
        worker_1 = _attrs(worker_id="1")
        assert after.get(worker_0, 0) == before.get(worker_0, 0) + 8
        assert after.get(worker_1, 0) == before.get(worker_1, 0) + 7

    def test_carries_model_name_and_cache_salt(self, subscriber):
        before = _read_counter_by_attrs()
        subscriber._on_retrieve_end(
            _retrieve_end(
                retrieved_count=4,
                engine_id=2,
                model_name="llama-3.1-8b",
                cache_salt="tenant-A",
            )
        )
        after = _read_counter_by_attrs()
        key = _attrs(worker_id="2", model_name="llama-3.1-8b", cache_salt="tenant-A")
        assert after.get(key, 0) == before.get(key, 0) + 4

    def test_different_models_or_salts_accumulate_independently(self, subscriber):
        before = _read_counter_by_attrs()
        # Same worker, different (model, salt) pairs.
        subscriber._on_retrieve_end(
            _retrieve_end(
                retrieved_count=5,
                engine_id=0,
                model_name="model-A",
                cache_salt="salt-1",
            )
        )
        subscriber._on_retrieve_end(
            _retrieve_end(
                retrieved_count=3,
                engine_id=0,
                model_name="model-A",
                cache_salt="salt-2",
            )
        )
        subscriber._on_retrieve_end(
            _retrieve_end(
                retrieved_count=7,
                engine_id=0,
                model_name="model-B",
                cache_salt="salt-1",
            )
        )
        after = _read_counter_by_attrs()
        a1 = _attrs(worker_id="0", model_name="model-A", cache_salt="salt-1")
        a2 = _attrs(worker_id="0", model_name="model-A", cache_salt="salt-2")
        b1 = _attrs(worker_id="0", model_name="model-B", cache_salt="salt-1")
        assert after.get(a1, 0) == before.get(a1, 0) + 5
        assert after.get(a2, 0) == before.get(a2, 0) + 3
        assert after.get(b1, 0) == before.get(b1, 0) + 7


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_count_is_noop(self, subscriber):
        before = _read_counter_by_attrs()
        subscriber._on_retrieve_end(_retrieve_end(retrieved_count=0, engine_id=4))
        after = _read_counter_by_attrs()
        key = _attrs(worker_id="4")
        assert after.get(key, 0) == before.get(key, 0)

    def test_missing_engine_id_still_records_without_attr(self, subscriber):
        # Some future emission site may forget engine_id; we should
        # record the count anyway (so operators notice the total) but
        # drop the worker_id attribute.
        before = _read_counter_by_attrs()
        subscriber._on_retrieve_end(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                metadata={"retrieved_count": 2},
            )
        )
        after = _read_counter_by_attrs()
        empty_key: tuple = ()
        assert after.get(empty_key, 0) == before.get(empty_key, 0) + 2


# ---------------------------------------------------------------------------
# End-to-end via EventBus
# ---------------------------------------------------------------------------


class TestEventBusIntegration:
    def test_retrieve_end_via_bus_increments_counter(self):
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        sub = EngineMetricsSubscriber()
        bus.register_subscriber(sub)

        before = _read_counter_by_attrs()
        bus.start()
        bus.publish(_retrieve_end(retrieved_count=12, engine_id=9))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        after = _read_counter_by_attrs()

        key = _attrs(worker_id="9")
        assert after.get(key, 0) == before.get(key, 0) + 12
