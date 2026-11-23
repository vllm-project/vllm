# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched DMA and host descriptor lifetime for HiSparse workers."""

from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from vllm.v1.hisparse.backend import HiSparseBackend


class DMADescriptors(NamedTuple):
    src: torch.Tensor
    dst: torch.Tensor
    sizes: torch.Tensor
    src_np: np.ndarray
    dst_np: np.ndarray
    sizes_np: np.ndarray


class HiSparseDMA:
    """Retain host pointer arrays until the device stream finishes the copy."""

    def __init__(self, backend: HiSparseBackend) -> None:
        self.backend = backend
        self._free: list[DMADescriptors] = []
        self._pending: deque[tuple[torch.Event, DMADescriptors]] = deque()

    def acquire(self, size: int) -> DMADescriptors:
        for index, descriptors in enumerate(self._free):
            if descriptors.src.numel() >= size:
                return self._free.pop(index)
        if self._free:
            # Replace an undersized allocation when batches grow across steps.
            self._free.pop()
        src, dst, sizes = (
            torch.empty(size, dtype=torch.int64, device="cpu") for _ in range(3)
        )
        return DMADescriptors(
            src,
            dst,
            sizes,
            src.numpy(),
            dst.numpy(),
            sizes.numpy(),
        )

    def submit(
        self,
        stream: torch.Stream,
        descriptors: DMADescriptors,
        count: int,
        *,
        host_write_event: torch.Event | None = None,
    ) -> torch.Event:
        completion = torch.Event()
        with stream:
            self.backend.swap_blocks_batch(
                descriptors.src[:count],
                descriptors.dst[:count],
                descriptors.sizes[:count],
            )
            if host_write_event is not None:
                host_write_event.record(stream)
            completion.record(stream)
        self._pending.append((completion, descriptors))
        return completion

    def release_completed(self) -> None:
        while self._pending and self._pending[0][0].query():
            _, descriptors = self._pending.popleft()
            self._free.append(descriptors)
