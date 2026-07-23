# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable pool of (src_ptrs, dst_ptrs, sizes) tensor triples.

Used by ECCPUWorker to batch swap_blocks_batch descriptors without
per-step allocation overhead.
"""

from typing import NamedTuple

import torch

from vllm.platforms import current_platform

# CUDA/ROCm cache_kernels.cu requires int64 pointers; the XPU DMA engine
# requires uint64 (see vllm._custom_ops.swap_blocks_batch).
_PTR_DTYPE = torch.uint64 if current_platform.is_xpu() else torch.int64


class DescriptorBuffers(NamedTuple):
    src_ptrs: torch.Tensor
    dst_ptrs: torch.Tensor
    sizes: torch.Tensor


class DescriptorBufferPool:
    """Pool of descriptor buffer triples for swap_blocks_batch.

    Each buffer is a `DescriptorBuffers` namedtuple of three 1-D tensors
    (dtype `_PTR_DTYPE`, platform-dependent) of equal length. Buffers are
    recycled across steps; if a returned buffer is too small it is discarded
    and a fresh one allocated.
    """

    def __init__(self) -> None:
        # LIFO stack of idle buffer triples.
        self._pool: list[DescriptorBuffers] = []

    def acquire(self, n: int) -> DescriptorBuffers:
        """Get a buffer triple with capacity >= *n*."""
        if self._pool:
            bufs = self._pool.pop()
            if bufs.src_ptrs.numel() >= n:
                return bufs
        return DescriptorBuffers(
            torch.empty(n, dtype=_PTR_DTYPE),
            torch.empty(n, dtype=_PTR_DTYPE),
            torch.empty(n, dtype=_PTR_DTYPE),
        )

    def release(self, bufs: DescriptorBuffers) -> None:
        """Return a buffer triple to the pool for reuse."""
        self._pool.append(bufs)
