# SPDX-License-Identifier: Apache-2.0

"""Tests for TimeoutLoggingSubscriber."""

# Standard
import logging
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.logging.timeout import (
    TimeoutLoggingSubscriber,
)

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15
_LOGGER_NAME = "lmcache.v1.mp_observability.subscribers.logging.timeout"


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = TimeoutLoggingSubscriber()
    bus.register_subscriber(sub)
    return sub


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestTimeoutLoggingSubscriber:
    def test_subscription_covers_timeout_event(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.TIMEOUT_RAISED in subs
        assert len(subs) == 1

    def test_logs_warning_with_message_and_stacktrace(self, bus, subscriber):
        handler = _CaptureHandler()
        lg = logging.getLogger(_LOGGER_NAME)
        lg.addHandler(handler)
        try:
            bus.start()
            bus.publish(
                Event(
                    event_type=EventType.TIMEOUT_RAISED,
                    metadata={
                        "message": "boom",
                        "exception_type": "LMCacheTimeoutError",
                        "stacktrace": "the-stacktrace",
                    },
                )
            )
            time.sleep(_DRAIN_WAIT)
            bus.stop()
        finally:
            lg.removeHandler(handler)

        warnings = [r for r in handler.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "boom" in message
        assert "LMCacheTimeoutError" in message
        assert "the-stacktrace" in message
