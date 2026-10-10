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
    "ratios, default, expected",
    [
        # Triton takes over at N=64.
        ([3.0, 1.5, 0.75, 0.5, 0.3], None, 64),
        # Triton is ahead everywhere: never go below the smallest probed N.
        ([0.75, 0.6, 0.5, 0.5, 0.3], None, 16),
        # Triton never catches up.
        ([3.0, 1.5, 1.4, 1.3, 1.2], 16, None),
        # A win at N=32 doesn't count when N=64 loses again.
        ([3.0, 0.75, 1.3, 0.5, 0.3], 16, 128),
        # Within the margin the default decides: DMA when it never uses Triton,
        # Triton from its own min_n otherwise.
        ([3.0, 1.5, 0.98, 0.5, 0.3], None, 128),
        ([1.02, 0.98, 0.75, 0.5, 0.3], 16, 16),
        ([1.02, 0.98, 0.75, 0.5, 0.3], 64, 64),
    ],
)
def test_pick_min_n(ratios, default, expected) -> None:
    assert pick_min_n(ratios, default) == expected


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton swap fast path requires CUDA"
)
def test_measure_load_paths_times_every_probed_n() -> None:
    copy_size = 4096
    host = torch.zeros((2 * CALIBRATION_NS[-1], copy_size), dtype=torch.int8)
    host = host.pin_memory()

    ratios = measure_load_paths(copy_size, 4096, host, torch.device("cuda:0"))

    assert ratios is not None
    assert len(ratios) == len(CALIBRATION_NS)
    assert all(r > 0 for r in ratios)
    # Too small a host region: leave the defaults alone.
    assert (
        measure_load_paths(copy_size, 4096, host[:16], torch.device("cuda:0")) is None
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton swap fast path requires CUDA"
)
def test_measure_load_paths_reads_strided_host_rows() -> None:
    # A worker's view of the shared offload region: one chunk per row, with
    # the other tensors' (and workers') bytes in between.
    copy_size, row_stride = 4096, 3 * 4096
    rows = 2 * CALIBRATION_NS[-1]
    region = torch.zeros(rows * row_stride, dtype=torch.int8).pin_memory()
    host = torch.as_strided(region, (rows, copy_size), (row_stride, 1))
    assert not host.is_contiguous()

    ratios = measure_load_paths(copy_size, 4096, host, torch.device("cuda:0"))

    assert ratios is not None
    assert len(ratios) == len(CALIBRATION_NS)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton swap fast path requires CUDA"
)
def test_measure_load_paths_avoids_default_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On the default stream the DMA arm would time one cudaMemcpyAsync per
    # copy instead of cuMemcpyBatchAsync, which real loads use.
    streams = []
    monkeypatch.setattr(
        swap_blocks_triton.ops,
        "swap_blocks_batch",
        lambda *args, **kwargs: streams.append(torch.cuda.current_stream()),
    )
    monkeypatch.setattr(
        swap_blocks_triton,
        "swap_blocks_batch",
        lambda *args, **kwargs: streams.append(torch.cuda.current_stream()),
    )
    host = torch.zeros((2 * CALIBRATION_NS[-1], 4096), dtype=torch.int8)

    assert torch.cuda.current_stream() == torch.cuda.default_stream()
    measure_load_paths(4096, 4096, host.pin_memory(), torch.device("cuda:0"))

    assert streams
    assert all(s != torch.cuda.default_stream() for s in streams)
