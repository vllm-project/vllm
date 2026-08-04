# SPDX-License-Identifier: Apache-2.0

"""Tests for TimeoutTracingSubscriber.

The subscriber's module-level ``_tracer`` is patched with a mock so the test
can assert on the span shape (name, exception event, ERROR status, parent
context) without depending on the single process-wide OTel TracerProvider.
"""

# Standard
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.tracing.span_registry import SpanRegistry
from lmcache.v1.mp_observability.subscribers.tracing.timeout import (
    TimeoutTracingSubscriber,
)
import lmcache.v1.mp_observability.subscribers.tracing.timeout as timeout_module

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def registry():
    return SpanRegistry()


def _publish(bus, session_id: str = "", message: str = "boom") -> None:
    bus.publish(
        Event(
            event_type=EventType.TIMEOUT_RAISED,
            session_id=session_id,
            metadata={
                "message": message,
                "exception_type": "LMCacheTimeoutError",
                "stacktrace": "the-stacktrace",
            },
        )
    )


class TestTimeoutTracingSubscriber:
    def test_subscription_covers_timeout_event(self, registry):
        sub = TimeoutTracingSubscriber(registry)
        subs = sub.get_subscriptions()
        assert EventType.TIMEOUT_RAISED in subs
        assert len(subs) == 1

    def test_creates_span_with_exception_event_and_error_status(self, bus, registry):
        fake_span = MagicMock()
        fake_tracer = MagicMock()
        fake_tracer.start_span.return_value = fake_span
        with (
            patch.object(timeout_module, "_tracer", fake_tracer),
            patch.object(timeout_module, "_HAS_OTEL", True),
        ):
            bus.register_subscriber(TimeoutTracingSubscriber(registry))
            bus.start()
            _publish(bus, message="boom")
            time.sleep(_DRAIN_WAIT)
            bus.stop()

        fake_tracer.start_span.assert_called_once()
        assert fake_tracer.start_span.call_args.args[0] == "timeout"

        # Exception recorded as an "exception" span event with the OTel
        # exception semantic-convention attributes.
        fake_span.add_event.assert_called_once()
        assert fake_span.add_event.call_args.args[0] == "exception"
        attrs = fake_span.add_event.call_args.kwargs["attributes"]
        assert attrs["exception.type"] == "LMCacheTimeoutError"
        assert attrs["exception.message"] == "boom"
        assert attrs["exception.stacktrace"] == "the-stacktrace"

        fake_span.set_status.assert_called_once()
        fake_span.end.assert_called_once()

    def test_nests_under_request_span_when_session_matches(self, bus, registry):
        sentinel_ctx = object()
        registry.open("req-1", "request", MagicMock(), sentinel_ctx)
        fake_tracer = MagicMock()
        fake_tracer.start_span.return_value = MagicMock()
        with (
            patch.object(timeout_module, "_tracer", fake_tracer),
            patch.object(timeout_module, "_HAS_OTEL", True),
        ):
            bus.register_subscriber(TimeoutTracingSubscriber(registry))
            bus.start()
            _publish(bus, session_id="req-1")
            time.sleep(_DRAIN_WAIT)
            bus.stop()

        assert fake_tracer.start_span.call_args.kwargs["context"] is sentinel_ctx

    def test_standalone_span_when_no_session(self, bus, registry):
        fake_tracer = MagicMock()
        fake_tracer.start_span.return_value = MagicMock()
        with (
            patch.object(timeout_module, "_tracer", fake_tracer),
            patch.object(timeout_module, "_HAS_OTEL", True),
        ):
            bus.register_subscriber(TimeoutTracingSubscriber(registry))
            bus.start()
            _publish(bus, session_id="")
            time.sleep(_DRAIN_WAIT)
            bus.stop()

        assert fake_tracer.start_span.call_args.kwargs["context"] is None
