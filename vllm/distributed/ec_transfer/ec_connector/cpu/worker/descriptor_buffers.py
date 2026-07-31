# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable pool of (src_ptrs, dst_ptrs, sizes) int64 tensor triples.

Used by ECCPUWorker to batch swap_blocks_batch descriptors without
per-step allocation overhead.
"""

from typing import NamedTuple

import torch


class DescriptorBuffers(NamedTuple):
    src_ptrs: torch.Tensor
    dst_ptrs: torch.Tensor
    sizes: torch.Tensor


class DescriptorBufferPool:
    """Pool of descriptor buffer triples for swap_blocks_batch.

    Each buffer is a `DescriptorBuffers` namedtuple of three 1-D int64
    tensors of equal length. Buffers are recycled across steps; if a
    returned buffer is too small it is discarded and a fresh one allocated.
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
            torch.empty(n, dtype=torch.int64),
            torch.empty(n, dtype=torch.int64),
            torch.empty(n, dtype=torch.int64),
        )

    def release(self, bufs: DescriptorBuffers) -> None:
        """Return a buffer triple to the pool for reuse."""
        self._pool.append(bufs)
