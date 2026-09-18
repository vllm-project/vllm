# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the Triton ``swap_blocks_batch`` fast-path kernel."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_offload.cpu import swap_blocks_triton
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    CALIBRATION_NS,
    measure_load_paths,
    pick_min_n,
    swap_blocks_batch,
)


def _addrs(buffers: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([b.data_ptr() for b in buffers], dtype=torch.int64)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton swap fast path requires CUDA"
)
def test_triton_swap_copies_source_bytes():
    # 8-byte-aligned, sub-threshold sizes covering 8 KiB chunk boundaries and
    # odd tail-mask lengths, with enough descriptors to take the Triton path.
    sizes = [8, 4096, 8192, 8200, 16384, 4088] * 8
    src = [torch.randint(256, (s,), dtype=torch.uint8, device="cuda") for s in sizes]
    dst = [torch.zeros_like(s) for s in src]
    sizes_t = torch.tensor(sizes, dtype=torch.int64)

    swap_blocks_batch(_addrs(src), _addrs(dst), sizes_t.clone(), bytes_per_chunk=8192)
    torch.accelerator.synchronize()

    for s, t in zip(src, dst):
        assert torch.equal(t, s)  # kernel copied the source bytes verbatim


def test_batches_below_min_n_use_dma(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        swap_blocks_triton.ops,
        "swap_blocks_batch",
        lambda *args, **kwargs: calls.append(args[0].numel()),
    )
    addrs = torch.zeros(48, dtype=torch.int64)

    swap_blocks_batch(addrs, addrs, addrs, bytes_per_chunk=8192, min_n=64)

    assert calls == [48]


@pytest.mark.parametrize(
    "dma_ms, triton_ms, expected",
    [
        # Triton takes over at N=64.
        ([1, 2, 4, 8, 16], [3, 3, 3, 4, 5], 64),
        # Triton is ahead everywhere: never go below the smallest probed N.
        ([4, 5, 6, 8, 16], [3, 3, 3, 4, 5], 16),
        # Triton never catches up.
        ([1, 2, 3, 4, 5], [3, 3, 4, 5, 6], None),
        # A win at N=32 doesn't count when N=64 loses again.
        ([1, 4, 3, 8, 16], [3, 3, 4, 4, 5], 128),
    ],
)
def test_pick_min_n(dma_ms, triton_ms, expected) -> None:
    assert pick_min_n(dma_ms, triton_ms) == expected


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton swap fast path requires CUDA"
)
def test_measure_load_paths_times_every_probed_n() -> None:
    copy_size = 4096
    host = torch.zeros((2 * CALIBRATION_NS[-1], copy_size), dtype=torch.int8)
    host = host.pin_memory()

    measured = measure_load_paths(copy_size, 4096, host, torch.device("cuda:0"))

    assert measured is not None
    dma_ms, triton_ms = measured
    assert len(dma_ms) == len(triton_ms) == len(CALIBRATION_NS)
    assert all(t > 0 for t in dma_ms + triton_ms)
    # Too small a host region: leave the defaults alone.
    assert (
        measure_load_paths(copy_size, 4096, host[:16], torch.device("cuda:0")) is None
    )
