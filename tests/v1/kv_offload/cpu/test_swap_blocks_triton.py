# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the Triton ``swap_blocks_batch`` fast-path kernel."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_offload.cpu.swap_blocks_triton import swap_blocks_batch


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


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton swap fast path requires CUDA"
)
@pytest.mark.parametrize("gpu_to_cpu", [False, True])
def test_triton_swap_host_device(gpu_to_cpu: bool):
    # Production shape: one side is pinned host memory dereferenced over UVA.
    # gpu_to_cpu=True is the store direction (kernel writes to host memory).
    sizes = [4096, 8192, 8200, 16384, 4088] * 8

    def make(device_side: bool, s: int) -> torch.Tensor:
        t = torch.randint(256, (s,), dtype=torch.uint8, device="cpu").pin_memory()
        return t.to("cuda") if device_side else t

    src = [make(gpu_to_cpu, s) for s in sizes]
    dst = [make(not gpu_to_cpu, s).zero_() for s in sizes]
    sizes_t = torch.tensor(sizes, dtype=torch.int64)

    swap_blocks_batch(_addrs(src), _addrs(dst), sizes_t, bytes_per_chunk=8192)
    torch.accelerator.synchronize()

    for s, t in zip(src, dst):
        assert torch.equal(t.cpu(), s.cpu())
