# SPDX-License-Identifier: Apache-2.0

"""Tests for ``StorageTraceRecorder`` and the file format round-trip.

These tests exercise the full publish → encode → write → read path
without requiring a real ``StorageManager`` or any GPU code.
"""

# Standard
from dataclasses import dataclass
import os
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.trace.decorator import (
    is_tracing_enabled,
    publish_call_event,
    set_tracing_enabled,
)
from lmcache.v1.mp_observability.trace.format import (
    FORMAT_VERSION,
    MAGIC,
    TRACE_SCHEMA_VERSION,
)
from lmcache.v1.mp_observability.trace.reader import TraceReader
from lmcache.v1.mp_observability.trace.recorder import StorageTraceRecorder
import lmcache.v1.mp_observability.event_bus as _bus_module


@pytest.fixture(autouse=True)
def restore_global_bus():
    saved = _bus_module._global_bus
    yield
    _bus_module._global_bus = saved
    set_tracing_enabled(False)


@pytest.fixture
def trace_path(tmp_path):
    return str(tmp_path / "test.lct")


@dataclass
class _FakeAdapterCfg:
    name: str = "noop"


@dataclass
class _FakeStorageManagerCfg:
    chunk_size: int = 256
    adapters: tuple = ()


def _flush(bus: EventBus) -> None:
    """Wait for the drain thread to process pending events."""
    # The bus drains every 100ms; sleep slightly longer to be safe.
    time.sleep(0.25)
    # Force a final drain in case the wake event was missed.
    bus._drain_all()


class TestHeader:
    def test_header_roundtrip(self, trace_path):
        rec = StorageTraceRecorder(trace_path)
        try:
            rec.attach_storage_config(_FakeStorageManagerCfg())
        finally:
            rec.close()

        with TraceReader(trace_path) as r:
            h = r.header
            assert h.magic == MAGIC
            assert h.format_version == FORMAT_VERSION
            assert h.trace_schema_version == TRACE_SCHEMA_VERSION
            assert h.level == "storage"
            assert h.t_mono_start > 0
            assert h.t_wall_start > 0
            assert h.sm_config_json  # populated by attach_storage_config
            assert len(h.sm_config_digest) == 64  # sha256 hex

    def test_header_without_attach(self, trace_path):
        """When ``attach_storage_config`` is never called, the file is
        still readable; config fields are empty."""
        rec = StorageTraceRecorder(trace_path)
        rec.close()
        with TraceReader(trace_path) as r:
            assert r.header.sm_config_json == ""
            assert r.header.sm_config_digest == ""


class TestGateLifecycle:
    def test_init_flips_gate_on(self, trace_path):
        assert is_tracing_enabled() is False
        rec = StorageTraceRecorder(trace_path)
        try:
            assert is_tracing_enabled() is True
        finally:
            rec.close()
        assert is_tracing_enabled() is False

    def test_close_idempotent(self, trace_path):
        rec = StorageTraceRecorder(trace_path)
        rec.close()
        rec.close()  # must not raise
        assert is_tracing_enabled() is False


class TestRecordRoundtrip:
    def test_records_via_eventbus(self, trace_path):
        bus = EventBus(EventBusConfig(enabled=True))
        _bus_module._global_bus = bus
        bus.start()

        rec = StorageTraceRecorder(trace_path)
        bus.register_subscriber(rec)
        try:
            publish_call_event(
                "lmcache.X.method",
                {
                    "keys": [ObjectKey(chunk_hash=b"\x01", model_name="m", kv_rank=1)],
                    "mode": "new",
                },
            )
            publish_call_event(
                "lmcache.X.method2",
                {"extra_count": 7},
            )
            _flush(bus)
        finally:
            bus.stop()  # invokes recorder.shutdown()

        with TraceReader(trace_path) as r:
            records = list(r.records())
        assert len(records) == 2
        assert records[0].qualname == "lmcache.X.method"
        # Decoder yields msgpack-friendly form; codec decode happens in
        # the replay driver (PR2). We assert structure only.
        assert "keys" in records[0].args
        assert records[0].args["mode"] == "new"
        assert records[1].args["extra_count"] == 7
        assert records[0].t_mono >= 0
        assert records[1].t_mono >= records[0].t_mono

    def test_layout_desc_roundtrip(self, trace_path):
        """End-to-end: a publish carrying a MemoryLayoutDesc round-trips
        through codec encode → file → reader → codec decode."""
        # First Party
        from lmcache.v1.mp_observability.trace import codecs

        bus = EventBus(EventBusConfig(enabled=True))
        _bus_module._global_bus = bus
        bus.start()

        rec = StorageTraceRecorder(trace_path)
        bus.register_subscriber(rec)
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, 3])],
            dtypes=[torch.float16],
        )
        try:
            publish_call_event("lmcache.X.method", {"layout_desc": layout})
            _flush(bus)
        finally:
            bus.stop()

        with TraceReader(trace_path) as r:
            records = list(r.records())
        assert len(records) == 1
        decoded_args = codecs.decode_args(records[0].args)
        assert decoded_args["layout_desc"] == layout


class TestReaderRobustness:
    def test_truncated_file_warns(self, trace_path):
        bus = EventBus(EventBusConfig(enabled=True))
        _bus_module._global_bus = bus
        bus.start()

        rec = StorageTraceRecorder(trace_path)
        bus.register_subscriber(rec)
        try:
            publish_call_event("a.b", {"x": 1})
            _flush(bus)
        finally:
            bus.stop()

        # Truncate the file mid-frame.
        size = os.path.getsize(trace_path)
        with open(trace_path, "r+b") as fh:
            fh.truncate(size - 3)

        with TraceReader(trace_path) as r:
            # Header was full; truncation hit the record body.
            records = list(r.records())
        # No assert on record count — it may be 0 or 1 depending on
        # how big the record was; the point is no exception escapes.
        assert isinstance(records, list)

    def test_bad_magic_rejected(self, trace_path):
        # Build a structurally valid Header frame but with wrong magic
        # so the reader's magic check (rather than msgspec's schema
        # check) fires.
        # First Party
        from lmcache.v1.mp_observability.trace.format import (
            FORMAT_VERSION,
            Header,
            encode_header,
        )

        bad = Header(
            magic=b"WRNG",
            format_version=FORMAT_VERSION,
            level="storage",
            trace_schema_version=TRACE_SCHEMA_VERSION,
            t_mono_start=0.0,
            t_wall_start=0.0,
            sm_config_json="",
            sm_config_digest="",
        )
        frame = encode_header(bad)
        with open(trace_path, "wb") as fh:
            fh.write(len(frame).to_bytes(4, "big") + frame)
        with pytest.raises(ValueError, match="bad magic"):
            TraceReader(trace_path)


class TestDroppedCount:
    def test_unencodable_arg_drops_record(self, trace_path):
        """An arg whose type has no codec is dropped, not crashed on."""
        bus = EventBus(EventBusConfig(enabled=True))
        _bus_module._global_bus = bus
        bus.start()

        rec = StorageTraceRecorder(trace_path)
        bus.register_subscriber(rec)

        class Unknown:
            pass

        try:
            publish_call_event("a.b", {"v": Unknown()})
            publish_call_event("a.c", {"v": 1})
            _flush(bus)
        finally:
            bus.stop()

        assert rec.dropped_count == 1
        with TraceReader(trace_path) as r:
            records = list(r.records())
        assert len(records) == 1
        assert records[0].qualname == "a.c"


class TestEvent:
    def test_event_construction(self):
        # Sanity: the event constructor accepts our metadata shape.
        ev = Event(
            event_type=EventType.TRACE_CALL,
            metadata={"qualname": "x", "args": {}},
        )
        assert ev.event_type == EventType.TRACE_CALL


class TestShutdownContract:
    def test_bus_stop_closes_recorder(self, trace_path):
        """``EventBus.stop()`` alone must flush, fsync, and close the
        recorder — without an explicit ``close()`` call.  Regression
        guard for the server shutdown chain (event_bus.stop →
        subscriber.shutdown → recorder.close)."""
        bus = EventBus(EventBusConfig(enabled=True))
        _bus_module._global_bus = bus
        bus.start()

        rec = StorageTraceRecorder(trace_path)
        bus.register_subscriber(rec)
        publish_call_event("a.b", {"x": 1})
        _flush(bus)

        # Only the bus is stopped; the recorder is not closed directly.
        bus.stop()

        # Gate must be off and the record must be readable from disk
        # (i.e. the file was flushed + closed by shutdown()).
        assert is_tracing_enabled() is False
        with TraceReader(trace_path) as r:
            records = list(r.records())
        assert len(records) == 1
        assert records[0].qualname == "a.b"
