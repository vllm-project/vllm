# SPDX-License-Identifier: Apache-2.0

"""Tests for the C++ EventRecorder (record_event_on_stream / drain)."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache import torch_dev, torch_device_type

pytestmark = pytest.mark.cuda

if torch_device_type != "cuda" or not torch_dev.is_available():
    pytest.skip("requires available CUDA runtime", allow_module_level=True)

lmc_ops = pytest.importorskip("lmcache.c_ops", reason="lmcache.c_ops not built")
if not hasattr(lmc_ops, "record_event_on_stream"):
    pytest.skip("record_event_on_stream not available", allow_module_level=True)

# Third Party
cupy = pytest.importorskip("cupy", reason="cupy required")  # noqa: E402

# First Party
from lmcache.v1.mp_observability.event import Event, EventType  # noqa: E402
from lmcache.v1.mp_observability.event_bus import (  # noqa: E402
    EventBus,
    EventBusConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def stream():
    """Return a synchronised CuPy external stream."""
    s = cupy.cuda.Stream()
    yield s
    s.synchronize()


@pytest.fixture()
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


# ---------------------------------------------------------------------------
# Tests: C++ record / drain API
# ---------------------------------------------------------------------------


class TestRecordAndDrain:
    """Low-level tests for lmc_ops.record_event_on_stream / drain."""

    def test_drain_empty(self):
        events = lmc_ops.drain_recorded_events()
        assert events == []

    def test_single_event(self, stream):
        lmc_ops.record_event_on_stream(
            stream.ptr,
            "mp.store.start",
            "sess-1",
            {"device": f"{torch_device_type}:0"},
            {},
        )
        stream.synchronize()

        events = lmc_ops.drain_recorded_events()
        assert len(events) == 1
        name, sid, ts, str_meta, int_meta = events[0]
        assert name == "mp.store.start"
        assert sid == "sess-1"
        assert ts > 0.0
        assert str_meta == {"device": f"{torch_device_type}:0"}
        assert int_meta == {}

    def test_int_metadata_preserved(self, stream):
        lmc_ops.record_event_on_stream(
            stream.ptr,
            "mp.store.end",
            "sess-2",
            {"device": f"{torch_device_type}:0"},
            {"stored_count": 42},
        )
        stream.synchronize()

        events = lmc_ops.drain_recorded_events()
        assert len(events) == 1
        _, _, _, str_meta, int_meta = events[0]
        assert str_meta == {"device": f"{torch_device_type}:0"}
        assert int_meta == {"stored_count": 42}

    def test_multiple_events_ordered(self, stream):
        for i in range(5):
            lmc_ops.record_event_on_stream(
                stream.ptr,
                f"mp.test.{i}",
                f"sess-{i}",
                {},
                {"idx": i},
            )
        stream.synchronize()

        events = lmc_ops.drain_recorded_events()
        assert len(events) == 5
        for i, (name, sid, ts, _, int_meta) in enumerate(events):
            assert name == f"mp.test.{i}"
            assert sid == f"sess-{i}"
            assert int_meta["idx"] == i
            assert ts > 0.0

    def test_drain_clears_buffer(self, stream):
        lmc_ops.record_event_on_stream(stream.ptr, "mp.store.start", "s", {}, {})
        stream.synchronize()

        first = lmc_ops.drain_recorded_events()
        assert len(first) == 1

        second = lmc_ops.drain_recorded_events()
        assert second == []

    def test_timestamps_monotonic(self, stream):
        for _ in range(3):
            lmc_ops.record_event_on_stream(stream.ptr, "mp.store.start", "s", {}, {})
        stream.synchronize()

        events = lmc_ops.drain_recorded_events()
        timestamps = [e[2] for e in events]
        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# Tests: EventBus integration (publish_on_stream -> drain)
# ---------------------------------------------------------------------------


class TestEventBusIntegration:
    """Verify EventBus.publish_on_stream routes through C++ and events
    are dispatched to subscribers via the drain thread."""

    def test_publish_on_stream_reaches_subscriber(self, stream, bus):
        received: list[Event] = []
        bus.subscribe(EventType.MP_STORE_START, received.append)
        bus.start()

        bus.publish_on_stream(
            stream,
            Event(
                event_type=EventType.MP_STORE_START,
                session_id="req-1",
                metadata={"device": f"{torch_device_type}:0"},
            ),
        )
        stream.synchronize()
        time.sleep(0.3)
        bus.stop()

        assert len(received) == 1
        evt = received[0]
        assert evt.event_type is EventType.MP_STORE_START
        assert evt.session_id == "req-1"
        assert evt.metadata["device"] == f"{torch_device_type}:0"
        assert evt.timestamp > 0.0

    def test_int_metadata_roundtrip(self, stream, bus):
        received: list[Event] = []
        bus.subscribe(EventType.MP_STORE_END, received.append)
        bus.start()

        bus.publish_on_stream(
            stream,
            Event(
                event_type=EventType.MP_STORE_END,
                session_id="req-2",
                metadata={"device": f"{torch_device_type}:0", "stored_count": 10},
            ),
        )
        stream.synchronize()
        time.sleep(0.3)
        bus.stop()

        assert len(received) == 1
        assert received[0].metadata["stored_count"] == 10
        assert received[0].metadata["device"] == f"{torch_device_type}:0"

    def test_disabled_bus_skips_recording(self, stream):
        disabled_bus = EventBus(EventBusConfig(enabled=False))
        disabled_bus.publish_on_stream(
            stream,
            Event(
                event_type=EventType.MP_STORE_START,
                session_id="req-3",
                metadata={"device": f"{torch_device_type}:0"},
            ),
        )
        stream.synchronize()

        events = lmc_ops.drain_recorded_events()
        assert events == []
