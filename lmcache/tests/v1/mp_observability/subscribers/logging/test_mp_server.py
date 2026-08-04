# SPDX-License-Identifier: Apache-2.0

"""Tests for MPServerLoggingSubscriber."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.logging.mp_server import (
    MPServerLoggingSubscriber,
)


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = MPServerLoggingSubscriber()
    bus.register_subscriber(sub)
    return sub


class TestMPServerLoggingSubscriber:
    def test_subscriptions_cover_all_mp_server_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.MP_STORE_START in subs
        assert EventType.MP_STORE_END in subs
        assert EventType.MP_RETRIEVE_START in subs
        assert EventType.MP_RETRIEVE_END in subs
        assert EventType.MP_LOOKUP_PREFETCH_START in subs
        assert EventType.MP_LOOKUP_PREFETCH_END in subs

    def test_store_start_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id="req-1",
                metadata={"device": f"{torch_device_type}:0"},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_store_end_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id="req-1",
                metadata={"device": f"{torch_device_type}:0", "stored_count": 5},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_start_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id="req-2",
                metadata={"device": f"{torch_device_type}:1"},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_retrieve_end_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_END,
                session_id="req-2",
                metadata={"device": f"{torch_device_type}:1", "retrieved_count": 3},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_lookup_prefetch_start_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id="req-3",
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_lookup_prefetch_end_logs(self, bus, subscriber):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-3",
                metadata={"found_count": 10},
            )
        )
        time.sleep(0.15)
        bus.stop()

    def test_multiple_events_no_crash(self, bus, subscriber):
        bus.start()
        for i in range(10):
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=f"req-{i}",
                    metadata={"device": f"{torch_device_type}:0"},
                )
            )
            bus.publish(
                Event(
                    event_type=EventType.MP_STORE_END,
                    session_id=f"req-{i}",
                    metadata={"device": f"{torch_device_type}:0", "stored_count": i},
                )
            )
        time.sleep(0.15)
        bus.stop()
