# SPDX-License-Identifier: Apache-2.0

"""Tests for ExtraStatsLoggingSubscriber."""

# Standard
from contextlib import contextmanager
import logging
import time

# Third Party
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.logging.extra_stats import (
    ExtraStatsLoggingSubscriber,
)

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15
_LOGGER_NAME = "lmcache.v1.mp_observability.subscribers.logging.extra_stats"
_INTERVAL = 0.05
_WAIT = 0.06


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == logging.INFO]


@contextmanager
def _capture_logs():
    handler = _CaptureHandler()
    lg = logging.getLogger(_LOGGER_NAME)
    old_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)


def _start(
    event_type: EventType,
    session: str,
    t: float,
    device: str = f"{torch_device_type}:0",
) -> Event:
    return Event(
        event_type=event_type,
        timestamp=t,
        session_id=session,
        metadata={"device": device, "engine_id": 0, "model_name": "test-model"},
    )


def _end(
    event_type: EventType,
    session: str,
    t: float,
    device: str = f"{torch_device_type}:0",
    total_bytes: int = 0,
    num_tokens: int = 0,
) -> Event:
    return Event(
        event_type=event_type,
        timestamp=t,
        session_id=session,
        metadata={
            "device": device,
            "engine_id": 0,
            "model_name": "test-model",
            "total_bytes": total_bytes,
            "num_tokens": num_tokens,
        },
    )


def _tick() -> Event:
    return Event(
        event_type=EventType.L1_EVICTION_LOOP_TICK,
        metadata={"usage": 0.1, "watermark": 0.8, "triggered": False},
    )


class TestExtraStatsLoggingSubscriber:
    def test_subscription_surface(self):
        subs = ExtraStatsLoggingSubscriber(1.0).get_subscriptions()
        assert set(subs) == {
            EventType.MP_STORE_START,
            EventType.MP_STORE_END,
            EventType.MP_RETRIEVE_START,
            EventType.MP_RETRIEVE_END,
            EventType.L1_EVICTION_LOOP_TICK,
        }

    @pytest.mark.parametrize("interval", [0.0, -1.0])
    def test_invalid_interval_raises(self, interval):
        with pytest.raises(ValueError):
            ExtraStatsLoggingSubscriber(interval)

    def test_store_window_logs_tokens_size_and_throughput(self):
        subs = ExtraStatsLoggingSubscriber(_INTERVAL).get_subscriptions()
        with _capture_logs() as handler:
            subs[EventType.MP_STORE_START](
                _start(EventType.MP_STORE_START, "req-1", 100.0)
            )
            subs[EventType.MP_STORE_END](
                _end(
                    EventType.MP_STORE_END,
                    "req-1",
                    100.5,
                    total_bytes=5_000_000_000,
                    num_tokens=24576,
                )
            )
            time.sleep(_WAIT)
            subs[EventType.L1_EVICTION_LOOP_TICK](_tick())

        window_lines = [m for m in handler.messages() if "last" in m]
        assert len(window_lines) == 1
        assert f"[{torch_device_type}:0]" in window_lines[0]
        assert (
            "store ops=1 tokens=24576 size=5.00GB avg_copy=10.00GB/s" in window_lines[0]
        )
        assert "retrieve ops=0 tokens=0 size=0.00GB avg_copy=n/a" in window_lines[0]

    def test_retrieve_window_logs_tokens_size_and_throughput(self):
        subs = ExtraStatsLoggingSubscriber(_INTERVAL).get_subscriptions()
        with _capture_logs() as handler:
            subs[EventType.MP_RETRIEVE_START](
                _start(EventType.MP_RETRIEVE_START, "req-1", 200.0)
            )
            subs[EventType.MP_RETRIEVE_END](
                _end(
                    EventType.MP_RETRIEVE_END,
                    "req-1",
                    200.25,
                    total_bytes=2_000_000_000,
                    num_tokens=4096,
                )
            )
            time.sleep(_WAIT)
            subs[EventType.L1_EVICTION_LOOP_TICK](_tick())

        window_lines = [m for m in handler.messages() if "last" in m]
        assert len(window_lines) == 1
        assert (
            "retrieve ops=1 tokens=4096 size=2.00GB avg_copy=8.00GB/s"
            in window_lines[0]
        )

    def test_groups_by_device(self):
        subs = ExtraStatsLoggingSubscriber(_INTERVAL).get_subscriptions()
        with _capture_logs() as handler:
            for device in (f"{torch_device_type}:0", f"{torch_device_type}:1"):
                subs[EventType.MP_STORE_START](
                    _start(EventType.MP_STORE_START, "req-1", 100.0, device=device)
                )
                subs[EventType.MP_STORE_END](
                    _end(
                        EventType.MP_STORE_END,
                        "req-1",
                        100.5,
                        device=device,
                        total_bytes=1_000_000_000,
                        num_tokens=1024,
                    )
                )
            time.sleep(_WAIT)
            subs[EventType.L1_EVICTION_LOOP_TICK](_tick())

        messages = handler.messages()
        assert (
            len(
                [m for m in messages if f"[{torch_device_type}:0]" in m and "last" in m]
            )
            == 1
        )
        assert (
            len(
                [m for m in messages if f"[{torch_device_type}:1]" in m and "last" in m]
            )
            == 1
        )
        assert (
            len(
                [
                    m
                    for m in messages
                    if f"[{torch_device_type}:0]" in m and "cumulative" in m
                ]
            )
            == 1
        )
        assert (
            len(
                [
                    m
                    for m in messages
                    if f"[{torch_device_type}:1]" in m and "cumulative" in m
                ]
            )
            == 1
        )

    def test_no_flush_before_interval(self):
        subs = ExtraStatsLoggingSubscriber(60.0).get_subscriptions()
        with _capture_logs() as handler:
            subs[EventType.MP_STORE_START](
                _start(EventType.MP_STORE_START, "req-1", 100.0)
            )
            subs[EventType.MP_STORE_END](
                _end(
                    EventType.MP_STORE_END,
                    "req-1",
                    100.5,
                    total_bytes=1_000_000_000,
                    num_tokens=1024,
                )
            )
            subs[EventType.L1_EVICTION_LOOP_TICK](_tick())

        assert handler.messages() == []

    def test_idle_window_emits_nothing(self):
        subs = ExtraStatsLoggingSubscriber(_INTERVAL).get_subscriptions()
        with _capture_logs() as handler:
            time.sleep(_WAIT)
            subs[EventType.L1_EVICTION_LOOP_TICK](_tick())

        assert handler.messages() == []

    def test_end_without_start_still_counts_tokens(self):
        subs = ExtraStatsLoggingSubscriber(_INTERVAL).get_subscriptions()
        with _capture_logs() as handler:
            subs[EventType.MP_STORE_END](
                _end(
                    EventType.MP_STORE_END,
                    "req-1",
                    100.5,
                    total_bytes=1_000_000_000,
                    num_tokens=1024,
                )
            )
            time.sleep(_WAIT)
            subs[EventType.L1_EVICTION_LOOP_TICK](_tick())

        window_lines = [m for m in handler.messages() if "last" in m]
        assert len(window_lines) == 1
        assert "store ops=1 tokens=1024 size=1.00GB avg_copy=n/a" in window_lines[0]

    def test_failed_op_counts_zero_tokens(self):
        subs = ExtraStatsLoggingSubscriber(_INTERVAL).get_subscriptions()
        with _capture_logs() as handler:
            subs[EventType.MP_STORE_START](
                _start(EventType.MP_STORE_START, "req-1", 100.0)
            )
            subs[EventType.MP_STORE_END](_end(EventType.MP_STORE_END, "req-1", 100.5))
            time.sleep(_WAIT)
            subs[EventType.L1_EVICTION_LOOP_TICK](_tick())

        window_lines = [m for m in handler.messages() if "last" in m]
        assert len(window_lines) == 1
        assert "store ops=1 tokens=0 size=0.00GB avg_copy=n/a" in window_lines[0]

    def test_window_resets_while_cumulative_persists(self):
        subs = ExtraStatsLoggingSubscriber(_INTERVAL).get_subscriptions()
        with _capture_logs() as handler:
            subs[EventType.MP_STORE_START](
                _start(EventType.MP_STORE_START, "req-1", 100.0)
            )
            subs[EventType.MP_STORE_END](
                _end(
                    EventType.MP_STORE_END,
                    "req-1",
                    100.5,
                    total_bytes=2_000_000_000,
                    num_tokens=1000,
                )
            )
            time.sleep(_WAIT)
            subs[EventType.L1_EVICTION_LOOP_TICK](_tick())

            subs[EventType.MP_STORE_START](
                _start(EventType.MP_STORE_START, "req-2", 200.0)
            )
            subs[EventType.MP_STORE_END](
                _end(
                    EventType.MP_STORE_END,
                    "req-2",
                    200.5,
                    total_bytes=4_000_000_000,
                    num_tokens=500,
                )
            )
            time.sleep(_WAIT)
            subs[EventType.L1_EVICTION_LOOP_TICK](_tick())

        window_lines = [m for m in handler.messages() if "last" in m]
        cumulative_lines = [m for m in handler.messages() if "cumulative" in m]
        assert len(window_lines) == 2
        assert len(cumulative_lines) == 2
        assert "store ops=1 tokens=500 size=4.00GB avg_copy=8.00GB/s" in window_lines[1]
        assert (
            "store ops=2 tokens=1500 size=6.00GB avg_copy=6.00GB/s"
            in cumulative_lines[1]
        )

    def test_stale_pending_start_is_pruned(self):
        subs = ExtraStatsLoggingSubscriber(_INTERVAL).get_subscriptions()
        with _capture_logs() as handler:
            stale_ts = time.time() - 200.0
            subs[EventType.MP_STORE_START](
                _start(EventType.MP_STORE_START, "req-1", stale_ts)
            )
            time.sleep(_WAIT)
            subs[EventType.L1_EVICTION_LOOP_TICK](_tick())

            subs[EventType.MP_STORE_END](
                _end(
                    EventType.MP_STORE_END,
                    "req-1",
                    time.time(),
                    total_bytes=1_000_000_000,
                    num_tokens=1024,
                )
            )
            time.sleep(_WAIT)
            subs[EventType.L1_EVICTION_LOOP_TICK](_tick())

        window_lines = [m for m in handler.messages() if "last" in m]
        assert len(window_lines) == 1
        assert "store ops=1 tokens=1024 size=1.00GB avg_copy=n/a" in window_lines[0]

    def test_end_to_end_via_event_bus(self):
        bus = EventBus(EventBusConfig(enabled=True, max_queue_size=100))
        bus.register_subscriber(ExtraStatsLoggingSubscriber(_INTERVAL))
        with _capture_logs() as handler:
            bus.start()
            bus.publish(_start(EventType.MP_STORE_START, "req-1", 100.0))
            bus.publish(
                _end(
                    EventType.MP_STORE_END,
                    "req-1",
                    100.5,
                    total_bytes=1_000_000_000,
                    num_tokens=1024,
                )
            )
            time.sleep(_WAIT + _DRAIN_WAIT)
            bus.publish(_tick())
            time.sleep(_DRAIN_WAIT)
            bus.stop()

        messages = handler.messages()
        assert any("last" in m and "tokens=1024" in m for m in messages)
        assert any("cumulative" in m and "tokens=1024" in m for m in messages)
