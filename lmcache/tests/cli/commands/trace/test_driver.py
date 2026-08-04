# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for :class:`StorageReplayDriver`.

Each test records a short sequence of StorageManager operations into a
binary trace file via the real ``EventBus`` + ``StorageTraceRecorder``
stack, then constructs a fresh StorageManager and drives replay over
that trace.  These tests avoid any GPU/vLLM dependencies — the whole
flow runs in-process on CPU memory.
"""

# Future
from __future__ import annotations

# Standard
from typing import Callable
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev
from lmcache.cli.commands.trace._dispatch import (
    CallDispatcher,
    ReplayContext,
    build_default_dispatcher,
)
from lmcache.cli.commands.trace._driver import StorageReplayDriver
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchRequestSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.trace.decorator import set_tracing_enabled
from lmcache.v1.mp_observability.trace.recorder import StorageTraceRecorder
import lmcache.v1.mp_observability.event_bus as _bus_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _should_use_lazy() -> bool:
    """Lazy allocator requires CUDA.  CPU-only hosts (our primary replay
    target) must use eager allocation."""
    return torch_dev.is_available()


def _make_sm_config() -> StorageManagerConfig:
    """Build a small StorageManagerConfig suitable for CPU testing."""
    memory = L1MemoryManagerConfig(
        size_in_bytes=64 * 1024 * 1024,
        use_lazy=_should_use_lazy(),
        init_size_in_bytes=32 * 1024 * 1024,
        align_bytes=0x1000,
    )
    l1 = L1ManagerConfig(
        memory_config=memory,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    return StorageManagerConfig(
        l1_manager_config=l1,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )


def _make_key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=i.to_bytes(4, "big"),
        model_name="test",
        kv_rank=0,
    )


def _make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([16, 16])],
        dtypes=[torch.float16],
    )


@pytest.fixture(autouse=True)
def restore_global_bus():
    saved = _bus_module._global_bus
    yield
    _bus_module._global_bus = saved
    set_tracing_enabled(False)


@pytest.fixture
def trace_path(tmp_path):
    return str(tmp_path / "run.lct")


def _flush(bus: EventBus) -> None:
    time.sleep(0.25)
    bus._drain_all()


# ---------------------------------------------------------------------------
# Helpers: record a scripted sequence into a trace file
# ---------------------------------------------------------------------------


def _record_sequence(
    trace_path: str,
    sm_config: StorageManagerConfig,
    script: Callable[[StorageManager], None],
) -> None:
    """Record whatever ``script(sm)`` does into *trace_path*.

    ``script`` receives a live StorageManager and should call traced
    methods on it.  This helper handles the bus / recorder lifecycle.
    """
    bus = EventBus(EventBusConfig(enabled=True))
    _bus_module._global_bus = bus
    bus.start()

    sm = StorageManager(sm_config)
    rec = StorageTraceRecorder(trace_path)
    rec.attach_storage_config(sm_config)
    bus.register_subscriber(rec)
    try:
        script(sm)
        _flush(bus)
    finally:
        bus.stop()
        sm.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecordReplayRoundtrip:
    def test_reserve_and_finish_write_replay(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(i) for i in range(3)]

        def script(sm: StorageManager) -> None:
            reserved = sm.reserve_write(keys, layout, mode="new")
            assert len(reserved) == 3
            sm.finish_write(keys)

        _record_sequence(trace_path, sm_config, script)

        with StorageReplayDriver(_make_sm_config(), trace_path) as driver:
            result = driver.run()

        assert result.records_failed == 0
        assert result.records_replayed >= 2  # reserve_write + finish_write
        assert result.header_level == "storage"
        # Same config used on both sides → digest matches.
        assert result.header_digest == result.replay_config_digest

    def test_full_prefetch_cycle_replay(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(i) for i in range(3)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")
            sm.finish_write(keys)
            handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, layout))
            assert handle is not None
            with sm.read_prefetched_results(keys) as objs:
                assert objs is not None
                assert len(objs) == 3

        _record_sequence(trace_path, sm_config, script)

        with StorageReplayDriver(_make_sm_config(), trace_path) as driver:
            result = driver.run()

        # Everything replayed, no failures.
        assert result.records_failed == 0
        assert result.records_skipped == 0
        qns = result.stats.summary().keys()
        # Every op from the script appears in stats.
        expected_substrings = [
            "reserve_write",
            "finish_write",
            "submit_prefetch_task",
            "read_prefetched_results.__enter__",
            "read_prefetched_results.__exit__",
        ]
        for sub in expected_substrings:
            assert any(sub in qn for qn in qns), (
                f"missing qualname containing {sub!r}: saw {sorted(qns)}"
            )

    def test_on_record_callback_fires_per_record(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(0)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")
            sm.finish_write(keys)

        _record_sequence(trace_path, sm_config, script)

        seen: list[tuple[str, bool]] = []

        def on_record(qualname: str, latency_s: float, failed: bool) -> None:
            seen.append((qualname, failed))

        with StorageReplayDriver(_make_sm_config(), trace_path) as driver:
            result = driver.run(on_record=on_record)

        assert len(seen) == result.records_replayed
        assert all(not failed for _, failed in seen)


class TestMismatchHandling:
    def test_unknown_qualname_is_skipped(self, trace_path):
        """An empty dispatcher with no matching handlers skips every
        record without raising."""
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(0)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")

        _record_sequence(trace_path, sm_config, script)

        empty = CallDispatcher()
        with StorageReplayDriver(
            _make_sm_config(), trace_path, dispatcher=empty
        ) as driver:
            result = driver.run()
        assert result.records_replayed == 0
        assert result.records_skipped >= 1

    def test_handler_failure_counted(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(0)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")

        _record_sequence(trace_path, sm_config, script)

        d = build_default_dispatcher()
        # Replace the reserve_write handler with one that raises.
        failing = CallDispatcher()
        for qn in d.registered_qualnames():
            if qn.endswith(".reserve_write"):

                def _boom(_ctx: ReplayContext, _a: dict) -> None:
                    raise RuntimeError("boom")

                failing.register(qn, _boom)
            else:
                # Reuse the default handler for anything else; none
                # is expected in this script but registering keeps
                # the test robust against future decorator additions.
                # First Party
                from lmcache.cli.commands.trace._dispatch import (
                    _call_sm_method,
                    _enter_read_prefetched,
                    _exit_read_prefetched,
                )

                if qn.endswith(".__enter__"):
                    failing.register(qn, _enter_read_prefetched)
                elif qn.endswith(".__exit__"):
                    failing.register(qn, _exit_read_prefetched)
                else:
                    failing.register(qn, _call_sm_method(qn.split(".")[-1]))

        with StorageReplayDriver(
            _make_sm_config(), trace_path, dispatcher=failing
        ) as driver:
            result = driver.run()

        assert result.records_failed >= 1


class TestPacing:
    def test_replay_does_not_regress_past_monotonic(self, trace_path):
        """Replay never runs *before* the recorded offset.

        Records have t_mono=0 and a positive value; the driver always
        honors the recorded gap — there is no as-fast-as-possible
        mode, because async read/write dependencies inside
        ``StorageManager`` make it unsafe to collapse the recorded
        inter-call intervals.
        """
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(0)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")
            time.sleep(0.05)  # force a gap
            sm.finish_write(keys)

        _record_sequence(trace_path, sm_config, script)

        start = time.monotonic()
        with StorageReplayDriver(_make_sm_config(), trace_path) as driver:
            result = driver.run()
        elapsed = time.monotonic() - start

        assert result.records_failed == 0
        # Replay should have slept ≈ 50ms at minimum.  Use a generous
        # bound to avoid flakes under load.
        assert elapsed >= 0.04
