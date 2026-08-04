# SPDX-License-Identifier: Apache-2.0

"""Tests for L1FailureMetricsSubscriber.

Verifies that ``L1_ALLOCATION_FAILED`` and ``L1_READ_FAILED`` events
produce the expected OTel counters with correct ``during``, ``reason``,
and ``model_name`` attributes. Uses the shared ``InMemoryMetricReader``
to assert on counter deltas keyed by (metric_name, attributes).
"""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l1_failures import (
    L1FailureMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.counter_helpers import (
    counter_delta,
    counter_value,
    make_key,
    read_tagged_counters,
)

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = L1FailureMetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    before = read_tagged_counters()

    def get_delta():
        return counter_delta(before, read_tagged_counters())

    return get_delta


class TestL1AllocationFailure:
    def test_during_l1_store_single_model(self, bus, subscriber, snapshot):
        bus.start()
        keys = [make_key("llama-7b", i) for i in range(3)]
        bus.publish(
            Event(
                event_type=EventType.L1_ALLOCATION_FAILED,
                metadata={"during": "l1_store", "keys": keys},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l1_allocation_failure",
                during="l1_store",
                model_name="llama-7b",
            )
            == 3
        )

    def test_during_l2_prefetch_single_model(self, bus, subscriber, snapshot):
        bus.start()
        keys = [make_key("mistral-7b", i) for i in range(5)]
        bus.publish(
            Event(
                event_type=EventType.L1_ALLOCATION_FAILED,
                metadata={"during": "l2_prefetch", "keys": keys},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l1_allocation_failure",
                during="l2_prefetch",
                model_name="mistral-7b",
            )
            == 5
        )

    def test_multi_model_in_one_event_buckets_separately(
        self, bus, subscriber, snapshot
    ):
        """A single event with keys from multiple models should emit
        per-model increments."""
        bus.start()
        keys = [
            make_key("llama-7b", 1),
            make_key("llama-7b", 2),
            make_key("mistral-7b", 3),
        ]
        bus.publish(
            Event(
                event_type=EventType.L1_ALLOCATION_FAILED,
                metadata={"during": "l1_store", "keys": keys},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l1_allocation_failure",
                during="l1_store",
                model_name="llama-7b",
            )
            == 2
        )
        assert (
            counter_value(
                delta,
                "lmcache_mp.l1_allocation_failure",
                during="l1_store",
                model_name="mistral-7b",
            )
            == 1
        )

    def test_accumulates_across_events(self, bus, subscriber, snapshot):
        bus.start()
        for _ in range(4):
            bus.publish(
                Event(
                    event_type=EventType.L1_ALLOCATION_FAILED,
                    metadata={
                        "during": "l1_store",
                        "keys": [make_key("llama-7b", 0)],
                    },
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l1_allocation_failure",
                during="l1_store",
                model_name="llama-7b",
            )
            == 4
        )


class TestL1ReadFailure:
    def test_during_l2_store_not_found(self, bus, subscriber, snapshot):
        bus.start()
        keys = [make_key("llama-7b", i) for i in range(2)]
        bus.publish(
            Event(
                event_type=EventType.L1_READ_FAILED,
                metadata={
                    "during": "l2_store",
                    "reason": "not_found",
                    "keys": keys,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l1_read_failure",
                during="l2_store",
                reason="not_found",
                model_name="llama-7b",
            )
            == 2
        )

    def test_during_l1_retrieve_write_locked(self, bus, subscriber, snapshot):
        bus.start()
        keys = [make_key("llama-7b", i) for i in range(3)]
        bus.publish(
            Event(
                event_type=EventType.L1_READ_FAILED,
                metadata={
                    "during": "l1_retrieve",
                    "reason": "write_locked",
                    "keys": keys,
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l1_read_failure",
                during="l1_retrieve",
                reason="write_locked",
                model_name="llama-7b",
            )
            == 3
        )

    def test_different_reasons_counted_separately(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L1_READ_FAILED,
                metadata={
                    "during": "l2_store",
                    "reason": "not_found",
                    "keys": [make_key("llama-7b", 0)],
                },
            )
        )
        bus.publish(
            Event(
                event_type=EventType.L1_READ_FAILED,
                metadata={
                    "during": "l2_store",
                    "reason": "write_locked",
                    "keys": [make_key("llama-7b", 1), make_key("llama-7b", 2)],
                },
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta,
                "lmcache_mp.l1_read_failure",
                during="l2_store",
                reason="not_found",
                model_name="llama-7b",
            )
            == 1
        )
        assert (
            counter_value(
                delta,
                "lmcache_mp.l1_read_failure",
                during="l2_store",
                reason="write_locked",
                model_name="llama-7b",
            )
            == 2
        )


class TestL1FailureSubscriptions:
    def test_subscriptions_cover_both_failure_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L1_ALLOCATION_FAILED in subs
        assert EventType.L1_READ_FAILED in subs
        assert len(subs) == 2

    def test_no_subscription_for_normal_l1_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.L1_READ_FINISHED not in subs
        assert EventType.L1_WRITE_FINISHED not in subs


class TestL1FailureEdgeCases:
    def test_empty_keys_list_is_noop(self, bus, subscriber, snapshot):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.L1_ALLOCATION_FAILED,
                metadata={"during": "l1_store", "keys": []},
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        # No keys → no emissions
        for (name, _attrs), val in delta.items():
            assert not (name == "lmcache_mp.l1_allocation_failure" and val != 0), (
                f"Unexpected emission for empty keys: {name}={val}"
            )
