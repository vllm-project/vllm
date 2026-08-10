# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for DiskBackend._do_load cross-batch buffer reuse.

``_do_load`` stages blocks disk -> pinned CPU buffer -> GPU via an async H2D
(``cuMemcpyBatchAsync``). The pinned buffer pool (``num_buffer_slots``, default
2) is reused across transfers, so trailing H2D events must be drained before
returning -- otherwise a later batch's ``preadv`` could overwrite a buffer the
previous batch's H2D is still reading, corrupting the GPU KV cache. The async
H2D is mocked here; no CUDA or NVMe is required.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.simple_kv_offload import disk_backend
from vllm.v1.simple_kv_offload.disk_backend import DiskBackend


@pytest.fixture
def fake_load_env(monkeypatch):
    """DiskBackend._do_load wired with a simulated async H2D race detector."""
    backend = DiskBackend()
    backend._load_params = object()  # copy_blocks is mocked
    backend._num_buffer_slots = 2
    backend._total_block_bytes = 4096

    in_flight: set[int] = set()
    corruptions: list[int] = []
    pending: list[int] = []  # FIFO of slots launched by copy_blocks, claimed by Event()

    def fake_copy_blocks(src_blocks, _dst_blocks, _params):
        # In _do_load the source is the pinned buffer; the H2D reads it.
        in_flight.add(src_blocks[0])
        pending.append(src_blocks[0])

    class FakeEvent:
        def __init__(self):
            # None means "not a buffer-slot copy" (e.g. _load_loop's own
            # completion event, issued with no preceding copy_blocks call).
            self._slot = pending.pop(0) if pending else None

        def record(self, _stream):
            pass

        def synchronize(self):
            in_flight.discard(self._slot)

    def fake_readv(buf_slot, _file_offset):
        # preadv overwrites the pinned buffer; races any in-flight H2D.
        if buf_slot in in_flight:
            corruptions.append(buf_slot)

    backend._readv_slot = fake_readv
    monkeypatch.setattr(disk_backend, "copy_blocks", fake_copy_blocks)
    monkeypatch.setattr(torch, "Event", FakeEvent)
    return backend, corruptions


def test_do_load_safe_within_one_batch(fake_load_env):
    # Per-slot prev.synchronize() guards buffer reuse within a single call.
    backend, corruptions = fake_load_env
    backend._do_load([10, 11, 12, 13], [0, 1, 2, 3], stream=object())
    assert corruptions == []


def test_do_load_drains_inflight_before_next_batch(fake_load_env):
    # Trailing H2Ds of one batch must drain before the next reuses the buffer.
    backend, corruptions = fake_load_env
    backend._do_load([10, 11], [0, 1], stream=object())
    backend._do_load([20, 21], [2, 3], stream=object())
    assert corruptions == [], (
        f"preadv overwrote slot(s) {corruptions} while a prior batch's H2D "
        "was still reading them; _do_load must drain trailing H2D events "
        "before returning (as _do_store does) or track them across calls."
    )


def test_launch_copy_wires_through_load_loop(fake_load_env, monkeypatch):
    # launch_copy's queued item must actually reach _do_load via _load_loop,
    # with the resulting event recorded under its event_idx.
    backend, corruptions = fake_load_env
    monkeypatch.setattr(disk_backend.current_platform, "set_device", lambda _d: None)
    events_list: list[tuple[int, torch.Event]] = []
    backend.launch_copy(
        [10, 11], [0, 1], is_store=False, event_idx=7, events_list=events_list
    )
    backend._load_queue.put(None)  # sentinel: return after the one queued item
    backend._load_loop(device=object(), stream=object())
    assert corruptions == []
    assert [idx for idx, _ in events_list] == [7]
