# SPDX-License-Identifier: Apache-2.0

"""Tests for L2FailureMetricsSubscriber.

Verifies that ``L2_PREFETCH_FAILED`` events produce the expected
``lmcache_mp.l2_prefetch_failure`` counter with ``reason`` and
``model_name`` attributes. Uses the shared ``InMemoryMetricReader`` to
assert on counter deltas.
"""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l2_failures import (
    L2FailureMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.counter_helpers import (
    counter_delta,
    counter_value,
    make_key,
    read_tagged_counters,
)

_DRAIN_WAIT = 0.15


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = L2FailureMetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    before = read_tagged_counters()

    def get_delta():
        return counter_delta(before, read_tagged_counters())

    return get_delta


class TestL2PrefetchFailure:
    def test_reason_l1_oom(self, bus, subscriber, snapshot):
        bus.start()
        keys = [make_key("llama-7b", i) for i in range(4)]
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={"reason": "l1_oom", "keys": keys},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="l1_oom",
                model_name="llama-7b",
            )
            == 4
        )

    def test_reason_not_found(self, bus, subscriber, snapshot):
        bus.start()
        keys = [make_key("mistral-7b", i) for i in range(2)]
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={"reason": "not_found", "keys": keys},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="not_found",
                model_name="mistral-7b",
            )
            == 2
        )

    def test_multi_model_buckets_separately(self, bus, subscriber, snapshot):
        bus.start()
        keys = [
            make_key("llama-7b", 1),
            make_key("mistral-7b", 2),
            make_key("mistral-7b", 3),
        ]
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={"reason": "not_found", "keys": keys},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="not_found",
                model_name="llama-7b",
            )
            == 1
        )
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="not_found",
                model_name="mistral-7b",
            )
            == 2
        )

    def test_different_reasons_counted_separately(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={
                    "reason": "l1_oom",
                    "keys": [make_key("llama-7b", 0), make_key("llama-7b", 1)],
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={
                    "reason": "not_found",
                    "keys": [make_key("llama-7b", 2)],
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="l1_oom",
                model_name="llama-7b",
            )
            == 2
        )
        assert (
            counter_value(
                delta,
                "lmcache_mp.l2_prefetch_failure",
                reason="not_found",
                model_name="llama-7b",
            )
            == 1
        )


class TestL2FailureSubscriptions:
    def test_subscribes_only_to_prefetch_failed(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L2_PREFETCH_FAILED in subs
        assert len(subs) == 1

    def test_no_subscription_for_normal_l2_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L2_STORE_SUBMITTED not in subs
        assert EventType.L2_PREFETCH_LOAD_COMPLETED not in subs


class TestL2FailureEdgeCases:
    def test_empty_keys_is_noop(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_FAILED,
                metadata={"reason": "l1_oom", "keys": []},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        for (name, _attrs), val in delta.items():
            assert not (name == "lmcache_mp.l2_prefetch_failure" and val != 0), (
                f"Unexpected emission for empty keys: {name}={val}"
            )
