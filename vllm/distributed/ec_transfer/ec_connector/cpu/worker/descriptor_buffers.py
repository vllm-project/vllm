# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable pool of (src_ptrs, dst_ptrs, sizes) tensor triples.

Used by ECCPUWorker to batch swap_blocks_batch descriptors without
per-step allocation overhead.
"""

from typing import NamedTuple

import numpy as np
import torch

from vllm.platforms import current_platform

# CUDA/ROCm cache_kernels.cu requires int64 pointers; the XPU DMA engine
# requires uint64 (see vllm._custom_ops.swap_blocks_batch).
_PTR_DTYPE = torch.uint64 if current_platform.is_xpu() else torch.int64


class DescriptorBuffers(NamedTuple):
    src_ptrs: torch.Tensor
    dst_ptrs: torch.Tensor
    sizes: torch.Tensor
    # Numpy aliases of the three tensors, written via add_copies().
    src_np: np.ndarray
    dst_np: np.ndarray
    sizes_np: np.ndarray

    def add_copies(
        self,
        first: int,
        sources: np.ndarray,
        destinations: np.ndarray,
        byte_counts: np.ndarray,
    ) -> None:
        """Store a batch of copies in these buffers, the first at *first*.

        One copy is one source address, one destination address, and a number
        of bytes. `swap_blocks_batch` performs every copy held in the buffers
        in a single call, so a caller fills in all of a step's copies before
        asking for any of them to run, and passes *first* to say where its own
        copies begin.

        Whole batches go in at once because storing a single number into a
        torch tensor costs microseconds. Adding copies one at a time can take
        longer than performing them.

        TODO(torch>=2.14): write the tensors directly and drop these numpy
        aliases once the minimum supported torch is 2.14. They exist only
        because torch's setitem unpacks the value as a signed long long before
        2.14 (pytorch#191458), rejecting XPU USM addresses >= 2**63, while the
        two's-complement rewrite those versions accept is in turn rejected by
        uint64 from 2.14 on. Numpy casts against the array's own type and so
        works on either. Byte counts are small enough to need none of this
        care, and use their alias only to match the addresses.
        """
        last = first + len(sources)
        self.src_np[first:last] = sources
        self.dst_np[first:last] = destinations
        self.sizes_np[first:last] = byte_counts


class DescriptorBufferPool:
    """Pool of descriptor buffer triples for swap_blocks_batch.

    Each buffer is a `DescriptorBuffers` namedtuple of three 1-D tensors
    (dtype `_PTR_DTYPE`, platform-dependent) of equal length, paired with
    numpy aliases used to fill them. Buffers are recycled across steps; if a
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
        src, dst, sizes = (torch.empty(n, dtype=_PTR_DTYPE) for _ in range(3))
        return DescriptorBuffers(
            src, dst, sizes, src.numpy(), dst.numpy(), sizes.numpy()
        )

    def release(self, bufs: DescriptorBuffers) -> None:
        """Return a buffer triple to the pool for reuse."""
        self._pool.append(bufs)
