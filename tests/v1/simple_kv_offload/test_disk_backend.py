# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backend unit tests for the padded-slot DMA path.

Verifies that store-then-load round-trips preserve bytes when per-tensor bpb
is not 4096-aligned, and that each slot uses a single iovec segment.

"""

from __future__ import annotations

import time

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

from vllm.v1.simple_kv_offload.disk_backend import (
    _DEFAULT_DIRECT_IO_ALIGNMENT,
    DiskBackend,
)

NUM_BLOCKS = 4
# k: 4096 bytes/block (aligned), v: 128 bytes/block (non-aligned)
# total_bpb = 4224, padded_total = 8192 -> exercises trailing pad
K_BPB = 4096
V_BPB = 128


def _make_disk_backend(
    tmp_path,
    *,
    direct_io_alignment=_DEFAULT_DIRECT_IO_ALIGNMENT,
    use_page_cache=False,
):
    gpu = {
        "k": torch.zeros(NUM_BLOCKS, K_BPB, dtype=torch.int8, device="cuda"),
        "v": torch.zeros(NUM_BLOCKS, V_BPB, dtype=torch.int8, device="cuda"),
    }
    low_pri, _ = torch.cuda.Stream.priority_range()
    backend = DiskBackend()
    backend.init(
        gpu_caches=gpu,
        device=gpu["k"].device,
        load_stream=torch.cuda.Stream(priority=low_pri),
        store_stream=torch.cuda.Stream(priority=low_pri),
        disk_path=str(tmp_path / "disk.bin"),
        num_disk_slots=NUM_BLOCKS,
        num_buffer_slots=2,
        use_page_cache=use_page_cache,
        direct_io_alignment=direct_io_alignment,
    )
    return backend, gpu


def _wait_events(events, timeout=10.0):
    deadline = time.time() + timeout
    while not events and time.time() < deadline:
        time.sleep(0.0005)
    assert events, "background copy was never enqueued"
    events[0][1].synchronize()


@pytest.mark.parametrize(
    ("direct_io_alignment", "use_page_cache", "expected_padded_total"),
    [
        (4096, False, 8192),
        (512, False, 4608),
        (4096, True, 4224),
    ],
)
def test_padded_slot_round_trip_preserves_bytes(
    tmp_path, direct_io_alignment, use_page_cache, expected_padded_total
):
    """Store then load back through the padded slot path."""
    backend, gpu = _make_disk_backend(
        tmp_path,
        direct_io_alignment=direct_io_alignment,
        use_page_cache=use_page_cache,
    )
    try:
        assert backend._padded_total == expected_padded_total
        assert backend._padded_total % backend._effective_alignment == 0
        assert (
            backend._store_buffer_caches["k"].data_ptr() % backend._effective_alignment
            == 0
        )
        assert (
            backend._load_buffer_caches["k"].data_ptr() % backend._effective_alignment
            == 0
        )

        # Fill with distinct non-zero data
        gpu["k"].copy_(
            torch.arange(NUM_BLOCKS * K_BPB, dtype=torch.int8)
            .view(NUM_BLOCKS, K_BPB)
            .cuda()
        )
        gpu["v"].copy_(
            torch.arange(NUM_BLOCKS * V_BPB, dtype=torch.int8)
            .view(NUM_BLOCKS, V_BPB)
            .cuda()
        )
        expected_k = gpu["k"].clone()
        expected_v = gpu["v"].clone()

        # Store block 0 -> disk slot 0
        store_events: list[tuple[int, torch.Event]] = []
        backend.launch_copy(
            src_blocks=[0],
            dst_blocks=[0],
            is_store=True,
            event_idx=0,
            events_list=store_events,
        )
        _wait_events(store_events)

        # Zero GPU tensors to prove load copies back
        gpu["k"].zero_()
        gpu["v"].zero_()

        # Load block 0 -> GPU slot 0
        load_events: list[tuple[int, torch.Event]] = []
        backend.launch_copy(
            src_blocks=[0],
            dst_blocks=[0],
            is_store=False,
            event_idx=0,
            events_list=load_events,
        )
        _wait_events(load_events)

        assert torch.all(gpu["k"][0] == expected_k[0])
        assert torch.all(gpu["v"][0] == expected_v[0])

        # Single iovec per slot (padded-slot invariant)
        assert len(backend._store_slot_views[0]) == 1
        assert len(backend._load_slot_views[0]) == 1
    finally:
        backend.shutdown()
