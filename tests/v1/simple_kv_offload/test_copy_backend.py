# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the DMA copy backend: host-side address/size
staging (``prepare_copy_blocks``), the DMA-only timing bracket in
``DmaCopyBackend._copy_loop``, and the ``_EventPairPool`` allocator.

None of these tests touch a real CUDA/ROCm device: ``prepare_copy_blocks``
is pure NumPy, and ``_copy_loop`` is driven with fake streams/events so the
call ordering can be asserted without wall-clock timing or GPU hardware.
"""

from __future__ import annotations

import ctypes
import queue
import time
from typing import Any
from unittest import mock

import numpy as np
import pytest

from vllm.v1.simple_kv_offload import copy_backend as copy_backend_mod
from vllm.v1.simple_kv_offload.copy_backend import (
    _EVENT_POOL_INITIAL_SIZE,
    DmaCopyBackend,
    DmaCopyEvent,
    _EventPairPool,
)
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    BatchMemcpyParams,
    PreparedBatchCopy,
    _CUmemcpyAttributes,
    prepare_copy_blocks,
)
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch_params(
    src_bases: list[int],
    dst_bases: list[int],
    bpb: list[int],
) -> BatchMemcpyParams:
    """Build a ``BatchMemcpyParams`` directly (no ``build_params()``, which
    unconditionally resolves the CUDA/ROCm batch-memcpy symbol and would fail
    on a machine with no accelerator). ``prepare_copy_blocks`` only reads the
    NumPy fields (``src_bases``/``dst_bases``/``bpb``/``num_layers``); the
    ctypes fields below are never touched by it, so dummy values are fine.
    """
    return BatchMemcpyParams(
        src_bases=np.array(src_bases, dtype=np.uint64),
        dst_bases=np.array(dst_bases, dtype=np.uint64),
        bpb=np.array(bpb, dtype=np.uint64),
        num_layers=len(src_bases),
        attrs=_CUmemcpyAttributes(srcAccessOrder=1),
        attrs_idx=ctypes.c_size_t(0),
        fail_idx=ctypes.c_size_t(0),
        stream_handle=0,
    )


class _FakeStream:
    """Records ``wait_event`` calls without touching any real device."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def wait_event(self, event: Any) -> None:
        self._calls.append("wait_event")


class _FakeEvent:
    """Records ``record`` calls; ``elapsed_time`` returns a fixed value."""

    def __init__(self, tag: str, calls: list[str], elapsed_ms: float = 0.0) -> None:
        self.tag = tag
        self._calls = calls
        self._elapsed_ms = elapsed_ms

    def record(self, stream: Any) -> None:
        self._calls.append(f"{self.tag}.record")

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, other: _FakeEvent) -> float:
        return self._elapsed_ms


class _FakeEventPool:
    """Fake ``_EventPairPool``: ``acquire()`` returns fresh fake events,
    ``release()`` is a no-op (release is only invoked later by worker code,
    never from within ``_copy_loop`` itself)."""

    def __init__(self, calls: list[str], elapsed_ms: float = 0.0) -> None:
        self._calls = calls
        self._elapsed_ms = elapsed_ms

    def acquire(self) -> tuple[_FakeEvent, _FakeEvent]:
        return (
            _FakeEvent("start", self._calls, self._elapsed_ms),
            _FakeEvent("end", self._calls, self._elapsed_ms),
        )

    def release(self, pair: Any) -> None:
        pass


def _run_copy_loop_once(
    *,
    src_blocks: list[int],
    dst_blocks: list[int],
    is_store: bool,
    event_idx: int,
    events_list: list[DmaCopyEvent],
    wait_event: Any,
    event_pool: Any,
) -> None:
    """Feed a single work item through ``DmaCopyBackend._copy_loop`` and let
    it drain (terminates on the ``None`` sentinel), all on CPU."""
    import torch

    q: queue.SimpleQueue = queue.SimpleQueue()
    q.put(
        (
            src_blocks,
            dst_blocks,
            object(),  # params: opaque, only forwarded to the (monkeypatched)
            # prepare/submit functions in these tests.
            is_store,
            event_idx,
            events_list,
            wait_event,
        )
    )
    q.put(None)
    fake_stream = _FakeStream(getattr(event_pool, "_calls", []))
    # These tests drive the loop with fake streams/events on any host; on a
    # CUDA machine current_platform.set_device would reject the cpu device,
    # so stub it out -- device binding is irrelevant to the ordering under
    # test.
    with mock.patch.object(copy_backend_mod.current_platform, "set_device"):
        DmaCopyBackend._copy_loop(
            q, torch.device("cpu"), fake_stream, fake_stream, event_pool
        )


def _patch_prepare_submit(
    monkeypatch, calls: list[str], prepared, *, sleep_s: float = 0.0
):
    def fake_prepare(src, dst, params):
        calls.append("prepare")
        if sleep_s:
            time.sleep(sleep_s)
        return prepared

    def fake_submit(prepared_arg, params):
        calls.append("submit")

    monkeypatch.setattr(copy_backend_mod, "prepare_copy_blocks", fake_prepare)
    monkeypatch.setattr(copy_backend_mod, "submit_prepared_copy", fake_submit)


_SAMPLE_PREPARED = PreparedBatchCopy(
    src_all=np.array([1], dtype=np.uint64),
    dst_all=np.array([2], dtype=np.uint64),
    sz_all=np.array([8], dtype=np.uint64),
    total_transfers=1,
    total_bytes=8,
)


# ---------------------------------------------------------------------------
# 1. prepare_copy_blocks correctness
# ---------------------------------------------------------------------------


def test_prepare_copy_blocks_address_math():
    """addresses = base + id*bpb per layer; sz_all repeats per layer;
    total_transfers = n*num_layers; total_bytes is an exact Python int."""
    params = _make_batch_params(
        src_bases=[100, 1_000_000], dst_bases=[5_000, 9_000_000], bpb=[16, 32]
    )
    src_ids = [0, 1, 2]
    dst_ids = [3, 4, 5]

    prepared = prepare_copy_blocks(src_ids, dst_ids, params)

    assert prepared is not None
    n = len(src_ids)
    expected_src = []
    expected_dst = []
    expected_sz = []
    for layer in range(params.num_layers):
        base_s = int(params.src_bases[layer])
        base_d = int(params.dst_bases[layer])
        b = int(params.bpb[layer])
        for s_id, d_id in zip(src_ids, dst_ids):
            expected_src.append(base_s + s_id * b)
            expected_dst.append(base_d + d_id * b)
            expected_sz.append(b)

    np.testing.assert_array_equal(
        prepared.src_all, np.array(expected_src, dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        prepared.dst_all, np.array(expected_dst, dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        prepared.sz_all, np.array(expected_sz, dtype=np.uint64)
    )
    assert prepared.total_transfers == n * params.num_layers
    assert prepared.total_bytes == sum(expected_sz)
    assert isinstance(prepared.total_bytes, int)


def test_prepare_copy_blocks_empty_returns_none():
    params = _make_batch_params(src_bases=[0], dst_bases=[0], bpb=[16])
    assert prepare_copy_blocks([], [], params) is None


# ---------------------------------------------------------------------------
# 2. REQUIRED-ORDER: prepare -> [wait_event] -> start.record -> submit ->
#    end.record. Proves host prep and the compute barrier sit outside the
#    timed bracket, without relying on wall-clock timing.
# ---------------------------------------------------------------------------


def test_copy_loop_required_order_with_wait_event(monkeypatch):
    calls: list[str] = []
    _patch_prepare_submit(monkeypatch, calls, _SAMPLE_PREPARED)

    event_pool = _FakeEventPool(calls)
    events_list: list[DmaCopyEvent] = []
    fake_wait_event = object()

    _run_copy_loop_once(
        src_blocks=[0],
        dst_blocks=[0],
        is_store=True,
        event_idx=0,
        events_list=events_list,
        wait_event=fake_wait_event,
        event_pool=event_pool,
    )

    assert calls == ["prepare", "wait_event", "start.record", "submit", "end.record"]
    assert len(events_list) == 1
    assert events_list[0].num_bytes == 8
    assert events_list[0].is_store is True


def test_copy_loop_no_wait_event_skips_stream_wait(monkeypatch):
    """Loads pass wait_event=None; the stream.wait_event call must be
    skipped entirely (not just a no-op call)."""
    calls: list[str] = []
    _patch_prepare_submit(monkeypatch, calls, _SAMPLE_PREPARED)

    event_pool = _FakeEventPool(calls)
    events_list: list[DmaCopyEvent] = []

    _run_copy_loop_once(
        src_blocks=[0],
        dst_blocks=[0],
        is_store=False,
        event_idx=0,
        events_list=events_list,
        wait_event=None,
        event_pool=event_pool,
    )

    assert calls == ["prepare", "start.record", "submit", "end.record"]
    assert "wait_event" not in calls


# ---------------------------------------------------------------------------
# 3. DMA-only timing excludes host prep, even when prep is artificially slow.
# ---------------------------------------------------------------------------


def test_dma_time_excludes_prep_delay(monkeypatch):
    calls: list[str] = []
    # Inject an artificial delay into prepare; the *mocked* elapsed_time is
    # what determines the recorded seconds, so if the recorded value equals
    # elapsed_ms/1000 regardless of this delay, timing is provably decoupled
    # from prep cost (no reliance on comparing against wall-clock prep time).
    _patch_prepare_submit(monkeypatch, calls, _SAMPLE_PREPARED, sleep_s=0.05)

    mock_elapsed_ms = 3.5
    event_pool = _FakeEventPool(calls, elapsed_ms=mock_elapsed_ms)
    events_list: list[DmaCopyEvent] = []

    _run_copy_loop_once(
        src_blocks=[0],
        dst_blocks=[0],
        is_store=False,
        event_idx=0,
        events_list=events_list,
        wait_event=None,
        event_pool=event_pool,
    )

    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )
    worker._record_copy_event(events_list[0])

    assert worker._transfer_stats.load.time == pytest.approx(mock_elapsed_ms / 1000.0)
    assert worker._transfer_stats.load.bytes == _SAMPLE_PREPARED.total_bytes


# ---------------------------------------------------------------------------
# 4. Empty batch: prepare returns None -> submit must not run, but a
#    DmaCopyEvent(num_bytes=0) is still appended for event_idx accounting.
# ---------------------------------------------------------------------------


def test_copy_loop_empty_batch_still_appends_event(monkeypatch):
    calls: list[str] = []

    def fake_prepare(src, dst, params):
        calls.append("prepare")
        return None

    def fake_submit(prepared, params):
        pytest.fail("submit_prepared_copy must not be called for an empty batch")

    monkeypatch.setattr(copy_backend_mod, "prepare_copy_blocks", fake_prepare)
    monkeypatch.setattr(copy_backend_mod, "submit_prepared_copy", fake_submit)

    event_pool = _FakeEventPool(calls)
    events_list: list[DmaCopyEvent] = []

    _run_copy_loop_once(
        src_blocks=[],
        dst_blocks=[],
        is_store=False,
        event_idx=7,
        events_list=events_list,
        wait_event=None,
        event_pool=event_pool,
    )

    assert calls == ["prepare", "start.record", "end.record"]
    assert len(events_list) == 1
    assert events_list[0].event_idx == 7
    assert events_list[0].num_bytes == 0


# ---------------------------------------------------------------------------
# 5. _EventPairPool: preallocation, reuse, growth, release.
# ---------------------------------------------------------------------------


class _CountingEvent:
    """Stand-in for torch.Event that just counts construction calls."""

    _instances = 0

    def __init__(self, **kwargs: Any) -> None:
        type(self)._instances += 1


def test_event_pool_preallocates_exact_size(monkeypatch):
    _CountingEvent._instances = 0
    monkeypatch.setattr(copy_backend_mod.torch, "Event", _CountingEvent)

    pool = _EventPairPool()

    assert _CountingEvent._instances == 2 * _EVENT_POOL_INITIAL_SIZE
    assert len(pool._pairs) == _EVENT_POOL_INITIAL_SIZE


def test_event_pool_acquire_release_reuse_no_new_allocation():
    pool = _EventPairPool()
    initial_size = len(pool._pairs)

    for _ in range(50):
        pair = pool.acquire()
        pool.release(pair)

    assert len(pool._pairs) == initial_size


def test_event_pool_exhaustion_allocates_exactly_one_new_pair(monkeypatch):
    pool = _EventPairPool()
    acquired = [pool.acquire() for _ in range(_EVENT_POOL_INITIAL_SIZE)]
    assert len(pool._pairs) == 0

    _CountingEvent._instances = 0
    monkeypatch.setattr(copy_backend_mod.torch, "Event", _CountingEvent)

    extra_pair = pool.acquire()

    assert _CountingEvent._instances == 2
    assert extra_pair not in acquired


def test_event_pool_release_returns_pair():
    pool = _EventPairPool()
    initial_size = len(pool._pairs)

    pair = pool.acquire()
    assert len(pool._pairs) == initial_size - 1

    pool.release(pair)
    assert len(pool._pairs) == initial_size
