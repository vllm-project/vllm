# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid store/load correctness and bounded out-of-order SSD restores."""

import os
import threading
from contextlib import contextmanager

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.hybrid_backend import HybridBackend

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

BLOCK_BYTES = 2 * 4096


@contextmanager
def make_backend(tmp_path, num_blocks, hot_rows):
    gpu = {
        str(i): (torch.arange(2 * num_blocks * 4096, device="cuda") + i * 17)
        .remainder(127)
        .to(torch.int8)
        .view(2 * num_blocks, 4096)
        for i in range(2)
    }
    backend = HybridBackend()
    try:
        backend.init(
            gpu,
            gpu["0"].device,
            torch.cuda.Stream(),
            torch.cuda.Stream(),
            str(tmp_path / "kv"),
            num_blocks,
            BLOCK_BYTES,
            num_hot_slots=hot_rows,
            max_load_slots=4,
        )
        yield backend, gpu
    finally:
        backend.shutdown()


def transfer(backend, src, dst, *, is_store, hot_blocks=None):
    fence = torch.Event()
    fence.record(torch.cuda.current_stream())
    events: list[tuple[int, torch.Event]] = []
    backend.launch_copy(
        src, dst, is_store, 7, events, wait_event=fence, hot_blocks=hot_blocks
    )
    backend.synchronize()
    assert len(events) == 1
    assert events[0][0] == 7 and events[0][1].query()


@pytest.mark.parametrize("hot_rows", [0, 4])
def test_store_load_and_slot_reuse(tmp_path, monkeypatch, hot_rows):
    """Restore beyond host capacity; hot hits avoid reads and leases are returned."""
    num_blocks = 100
    reads = []
    preadv = os.preadv

    def track_read(fd, views, offset):
        reads.append(offset // BLOCK_BYTES)
        return preadv(fd, views, offset)

    monkeypatch.setattr(os, "preadv", track_read)
    with make_backend(tmp_path, num_blocks, hot_rows) as (backend, gpu):
        src = list(range(num_blocks))
        disk = list(reversed(src))
        dst = list(range(num_blocks, 2 * num_blocks))
        hot = disk[-2:] if hot_rows else []
        expected = {name: tensor[:num_blocks].cpu() for name, tensor in gpu.items()}
        transfer(backend, src, disk, is_store=True, hot_blocks=hot)

        # Repeat to check both lease release and staging reuse without readmission.
        for _ in range(2):
            reads.clear()
            for tensor in gpu.values():
                tensor[num_blocks:].zero_()
            transfer(backend, disk, dst, is_store=False)
            assert sorted(reads) == sorted(set(disk) - set(hot))
            for name, tensor in gpu.items():
                assert torch.equal(tensor[num_blocks:].cpu(), expected[name])
            assert backend._peak_load_slots == (4 if hot_rows else 2)
            assert len(backend._free_hot_rows) == hot_rows - len(hot)
            assert not any(backend._hot_refs)

        # Reusing a logical SSD slot must invalidate its previous hot copy.
        transfer(backend, [0], [disk[-1]], is_store=True)
        reads.clear()
        transfer(backend, [disk[-1]], [dst[-1]], is_store=False)
        assert reads == [disk[-1]]
        for name, tensor in gpu.items():
            assert torch.equal(tensor[dst[-1]].cpu(), expected[name][0])


@pytest.mark.parametrize("hot_rows", [0, 1])
def test_ready_reads_bypass_blocked_read(tmp_path, monkeypatch, hot_rows):
    """A slow first read must not hold ready reads or their staging slot hostage."""
    first_started = threading.Event()
    later_started = threading.Event()
    release_first = threading.Event()
    preadv = os.preadv

    def gated_read(fd, views, offset):
        if offset == 0:
            first_started.set()
            assert release_first.wait(10), "test did not release the first read"
        elif offset == 3 * BLOCK_BYTES:
            # With two staging slots and slot 0 blocked, reaching block 3
            # requires H2D for blocks 1 and 2 to finish before slot reuse.
            later_started.set()
        return preadv(fd, views, offset)

    with make_backend(tmp_path, 5, hot_rows) as (backend, gpu):
        expected = {name: tensor[:5].cpu() for name, tensor in gpu.items()}
        transfer(
            backend,
            list(range(5)),
            list(range(5)),
            is_store=True,
            hot_blocks=[4] if hot_rows else [],
        )
        for tensor in gpu.values():
            tensor[5:].zero_()
        fence = torch.Event()
        fence.record(torch.cuda.current_stream())
        monkeypatch.setattr(os, "preadv", gated_read)
        events: list[tuple[int, torch.Event]] = []
        backend.launch_copy(
            list(range(5)), list(range(5, 10)), False, 9, events, wait_event=fence
        )
        try:
            assert first_started.wait(10), "first read was not submitted"
            assert later_started.wait(10), "ready reads stalled behind the first read"
            backend.check_error()
            assert not events, "completion was published with a read still pending"
            for name, tensor in gpu.items():
                assert torch.equal(tensor[6:8].cpu(), expected[name][1:3])
                assert not tensor[5].count_nonzero().item()
        finally:
            release_first.set()
        backend.synchronize()
        assert len(events) == 1 and events[0][0] == 9 and events[0][1].query()
        for name, tensor in gpu.items():
            assert torch.equal(tensor[5:].cpu(), expected[name])
