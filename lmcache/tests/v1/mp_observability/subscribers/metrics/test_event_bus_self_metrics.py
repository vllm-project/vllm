# SPDX-License-Identifier: Apache-2.0

"""Tests for ``EventBusSelfMetricsSubscriber``."""

# Standard
from unittest.mock import MagicMock, patch
import time

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventBusConfig,
    EventSubscriber,
)
from lmcache.v1.mp_observability.subscribers.metrics.event_bus import (
    EventBusSelfMetricsSubscriber,
)

_PATCH_TARGET = "lmcache.v1.mp_observability.subscribers.metrics.event_bus.metrics"


def _make_mock_metrics() -> tuple[MagicMock, MagicMock]:
    """Build a (mock_metrics_module, mock_meter) pair for patching OTel."""
    mock_meter = MagicMock()
    mock_metrics = MagicMock()
    mock_metrics.get_meter.return_value = mock_meter
    mock_metrics.Observation = lambda value, attrs=None: (value, attrs)
    return mock_metrics, mock_meter


def _counter_callbacks(meter: MagicMock) -> dict:
    return {
        c.args[0]: c.kwargs["callbacks"][0]
        for c in meter.create_observable_counter.call_args_list
    }


class TestRegistration:
    def test_registers_two_counters(self):
        bus = EventBus(EventBusConfig(enabled=True))
        mock_metrics, meter = _make_mock_metrics()
        with patch(_PATCH_TARGET, mock_metrics):
            EventBusSelfMetricsSubscriber(bus)
            assert set(_counter_callbacks(meter)) == {
                "lmcache_mp.event_bus.dropped_events_total",
                "lmcache_mp.event_bus.subscriber_exceptions",
            }

    def test_dropped_counter_callback_reflects_drops(self):
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=2))
        for _ in range(5):
            bus.publish(Event(event_type=EventType.L1_READ_FINISHED, session_id="s"))

        mock_metrics, meter = _make_mock_metrics()
        with patch(_PATCH_TARGET, mock_metrics):
            EventBusSelfMetricsSubscriber(bus)
            cb = _counter_callbacks(meter)["lmcache_mp.event_bus.dropped_events_total"]
            assert cb(None) == [(3, None)]

    def test_exceptions_counter_emits_per_subscriber(self):
        class _BadSub(EventSubscriber):
            def get_subscriptions(self):
                return {EventType.L1_READ_FINISHED: self._on_event}

            def _on_event(self, event):
                raise RuntimeError("boom")

        bus = EventBus(EventBusConfig(enabled=True))
        bus.register_subscriber(_BadSub())
        bus.start()
        bus.publish(Event(event_type=EventType.L1_READ_FINISHED, session_id="s1"))
        bus.publish(Event(event_type=EventType.L1_READ_FINISHED, session_id="s2"))
        time.sleep(0.15)
        bus.stop()

        mock_metrics, meter = _make_mock_metrics()
        with patch(_PATCH_TARGET, mock_metrics):
            EventBusSelfMetricsSubscriber(bus)
            cb = _counter_callbacks(meter)["lmcache_mp.event_bus.subscriber_exceptions"]
            assert (2, {"subscriber_name": "_BadSub"}) in cb(None)

    def test_exceptions_counter_empty_when_no_failures(self):
        bus = EventBus(EventBusConfig(enabled=True))
        mock_metrics, meter = _make_mock_metrics()
        with patch(_PATCH_TARGET, mock_metrics):
            EventBusSelfMetricsSubscriber(bus)
            cb = _counter_callbacks(meter)["lmcache_mp.event_bus.subscriber_exceptions"]
            assert cb(None) == []

    def test_no_event_subscriptions(self):
        bus = EventBus(EventBusConfig(enabled=True))
        mock_metrics, _ = _make_mock_metrics()
        with patch(_PATCH_TARGET, mock_metrics):
            sub = EventBusSelfMetricsSubscriber(bus)
        assert sub.get_subscriptions() == {}
