# SPDX-License-Identifier: Apache-2.0

"""Tests for TimeoutMetricsSubscriber.

Verifies that ``TIMEOUT_RAISED`` events increment the ``lmcache_mp.timeouts``
OTel counter, tagged by ``exception_type``. Uses the shared
``InMemoryMetricReader`` to assert on counter deltas keyed by attributes.
"""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.timeout import (
    TimeoutMetricsSubscriber,
)
from tests.v1.mp_observability.subscribers.metrics.counter_helpers import (
    counter_delta,
    counter_value,
    read_tagged_counters,
)

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = TimeoutMetricsSubscriber()
    bus.register_subscriber(sub)
    return sub


@pytest.fixture
def snapshot():
    before = read_tagged_counters()

    def get_delta():
        return counter_delta(before, read_tagged_counters())

    return get_delta


def _publish_timeout(bus, exception_type: str = "LMCacheTimeoutError") -> None:
    bus.publish(
        Event(
            event_type=EventType.TIMEOUT_RAISED,
            metadata={
                "message": "boom",
                "exception_type": exception_type,
                "stacktrace": "trace",
            },
        )
    )


class TestTimeoutMetricsSubscriber:
    def test_subscription_covers_timeout_event(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.TIMEOUT_RAISED in subs
        assert len(subs) == 1

    def test_single_timeout_increments_counter(self, bus, subscriber, snapshot):
        bus.start()
        _publish_timeout(bus)
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta, "lmcache_mp.timeouts", exception_type="LMCacheTimeoutError"
            )
            == 1
        )

    def test_accumulates_across_events(self, bus, subscriber, snapshot):
        bus.start()
        for _ in range(5):
            _publish_timeout(bus)
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta, "lmcache_mp.timeouts", exception_type="LMCacheTimeoutError"
            )
            == 5
        )

    def test_different_types_counted_separately(self, bus, subscriber, snapshot):
        bus.start()
        _publish_timeout(bus, "LMCacheTimeoutError")
        _publish_timeout(bus, "MyTimeout")
        _publish_timeout(bus, "MyTimeout")
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        delta = snapshot()
        assert (
            counter_value(
                delta, "lmcache_mp.timeouts", exception_type="LMCacheTimeoutError"
            )
            == 1
        )
        assert (
            counter_value(delta, "lmcache_mp.timeouts", exception_type="MyTimeout") == 2
        )
