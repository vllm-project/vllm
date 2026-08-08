# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for DiskBackend._do_load cross-batch buffer reuse.

``_do_load`` stages blocks disk -> pinned CPU buffer -> GPU via an async H2D
(``cuMemcpyBatchAsync``). The pinned buffer pool (``num_buffer_slots``, default
2) is reused across transfers. Pending H2D events are tracked in a *local* list
reset on every call and, unlike ``_do_store``, there is no final drain, so a
later batch's ``preadv`` can overwrite a buffer the previous batch's trailing
H2D is still reading -- silently corrupting the GPU KV cache. The async H2D is
mocked here; no CUDA or NVMe is required.
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
    pending = [None]  # slot launched by copy_blocks, claimed by next Event()

    def fake_copy_blocks(src_blocks, _dst_blocks, _params):
        # In _do_load the source is the pinned buffer; the H2D reads it.
        in_flight.add(src_blocks[0])
        pending[0] = src_blocks[0]

    class FakeEvent:
        def __init__(self):
            self._slot = pending[0]

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
