# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``ops.swap_blocks_batch`` C++ (cuMemcpyBatchAsync) path."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform


def _addrs(buffers: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([b.data_ptr() for b in buffers], dtype=torch.int64)


def _run_batch(sizes: list[int]) -> None:
    src = [torch.randint(256, (s,), dtype=torch.uint8, device="cuda") for s in sizes]
    dst = [torch.zeros_like(s) for s in src]
    ops.swap_blocks_batch(
        _addrs(src), _addrs(dst), torch.tensor(sizes, dtype=torch.int64)
    )
    torch.accelerator.synchronize()
    for s, d in zip(src, dst):
        assert torch.equal(d, s)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="swap_blocks_batch requires CUDA"
)
def test_swap_blocks_batch_default_stream():
    # cuMemcpyBatchAsync rejects the legacy default stream; the op must fall
    # back to per-copy transfers instead of raising.
    _run_batch([8, 4096, 8192])


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="swap_blocks_batch requires CUDA"
)
def test_swap_blocks_batch_dedicated_stream():
    # A dedicated non-default stream exercises the cuMemcpyBatchAsync fast path.
    with torch.cuda.stream(torch.Stream()):
        _run_batch([8, 4096, 8192])
