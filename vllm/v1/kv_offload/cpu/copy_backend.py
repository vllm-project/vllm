# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend plumbing for CPU KV-cache copies.

The worker describes a transfer as packed strided runs.  Each backend expands
those runs into the representation required by its existing copy API.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CopyRun
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    THRESHOLD_BYTES,
    swap_blocks_batch,
)

_DescriptorBuffers = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class CopyRunDescriptor(NamedTuple):
    """A logical copy run before backend-specific tensor materialization."""

    src_base: int
    dst_base: int
    fragment_size: int
    num_fragments: int
    src_stride: int
    dst_stride: int


class CopyBackend(ABC):
    """Interface for a backend that consumes logical copy run descriptors."""

    @abstractmethod
    def submit(self, run_descs: list[CopyRunDescriptor]) -> None:
        """Submit logical runs, materializing backend-specific descriptors."""

    @abstractmethod
    def finish_transfer(self) -> None:
        """Release resources held by the oldest submitted transfer."""

    @abstractmethod
    def clear(self) -> None:
        """Release all reusable and in-flight resources."""


def _new_descriptor_buffers(num_copy_ops: int) -> _DescriptorBuffers:
    # CUDA cache_kernels.cu requires int64; XPU DMA engine requires uint64.
    ptr_dtype = torch.uint64 if current_platform.is_xpu() else torch.int64
    return (
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=PIN_MEMORY),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=PIN_MEMORY),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=PIN_MEMORY),
    )


class _BatchCopyBackend(CopyBackend):
    """Base for backends using the existing fragment descriptor API."""

    def __init__(self) -> None:
        self._buffer_pool: list[_DescriptorBuffers] = []
        self._inflight_buffers: deque[_DescriptorBuffers] = deque()

    @staticmethod
    def _estimate_max_copy_ops(run_descs: list[CopyRunDescriptor]) -> int:
        return sum(run.num_fragments for run in run_descs)

    @staticmethod
    def _expand_run_desc(
        run_descs: list[CopyRunDescriptor],
        buffers: _DescriptorBuffers,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand logical runs into the legacy fragment descriptor buffers."""
        src_buffer, dst_buffer, size_buffer = buffers
        src = src_buffer.numpy().view(np.uint64)
        dst = dst_buffer.numpy().view(np.uint64)
        sizes = size_buffer.numpy()
        cursor = 0
        for run in run_descs:
            num_fragments = run.num_fragments
            if num_fragments == 0:
                continue
            end = cursor + num_fragments
            fragment_ids = np.arange(num_fragments, dtype=np.uint64)
            src[cursor:end] = run.src_base + fragment_ids * run.src_stride
            dst[cursor:end] = run.dst_base + fragment_ids * run.dst_stride
            sizes[cursor:end] = run.fragment_size
            cursor = end
        return src_buffer[:cursor], dst_buffer[:cursor], size_buffer[:cursor]

    @abstractmethod
    def _copy(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        sizes: torch.Tensor,
    ) -> None:
        """Invoke the concrete fragment copy implementation."""

    def submit(self, run_descs: list[CopyRunDescriptor]) -> None:
        num_copy_ops = self._estimate_max_copy_ops(run_descs)
        buffers = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else _new_descriptor_buffers(num_copy_ops)
        )
        if buffers[0].numel() < num_copy_ops:
            buffers = _new_descriptor_buffers(num_copy_ops)
        self._inflight_buffers.append(buffers)
        try:
            src, dst, sizes = self._expand_run_desc(run_descs, buffers)
            if src.numel() > 0:
                self._copy(src, dst, sizes)
        except Exception:
            assert self._inflight_buffers.pop() is buffers
            self._buffer_pool.append(buffers)
            raise

    def finish_transfer(self) -> None:
        self._buffer_pool.append(self._inflight_buffers.popleft())

    def clear(self) -> None:
        self._inflight_buffers.clear()
        self._buffer_pool.clear()


class BatchDMABackend(_BatchCopyBackend):
    """Existing C++ batch DMA implementation."""

    def __init__(self, *, is_src_access_order_any: bool) -> None:
        super().__init__()
        self._is_src_access_order_any = is_src_access_order_any

    def _copy(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        sizes: torch.Tensor,
    ) -> None:
        ops.swap_blocks_batch(
            src,
            dst,
            sizes,
            is_src_access_order_any=self._is_src_access_order_any,
        )


class BatchTritonBackend(_BatchCopyBackend):
    """Existing Triton batch fallback for small CPU-to-GPU fragments."""

    def __init__(self, *, bytes_per_chunk: int, is_src_access_order_any: bool) -> None:
        super().__init__()
        self._bytes_per_chunk = bytes_per_chunk
        self._is_src_access_order_any = is_src_access_order_any

    def _copy(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        sizes: torch.Tensor,
    ) -> None:
        swap_blocks_batch(
            src,
            dst,
            sizes,
            is_src_access_order_any=self._is_src_access_order_any,
            bytes_per_chunk=self._bytes_per_chunk,
        )


class CopyBackendAdapter:
    """Resolve one backend and forward transfers in submission order."""

    @staticmethod
    def resolve(
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        host_memory_is_pinned: bool = True,
        copy_runs: list[list[tuple[CopyRun, ...]]] | None = None,
    ) -> CopyBackend:
        """Resolve a concrete backend for one transfer direction."""
        is_src_access_order_any = not gpu_to_cpu
        if gpu_to_cpu:
            return BatchDMABackend(is_src_access_order_any=is_src_access_order_any)
        # The Triton kernel dereferences CPU pointers on the GPU, which is only
        # valid for pinned host memory. Fall back to DMA when unsupported.
        if (
            not host_memory_is_pinned
            or not HAS_TRITON
            or current_platform.is_xpu()
            or current_platform.is_rocm()
        ):
            return BatchDMABackend(is_src_access_order_any=is_src_access_order_any)

        page_sizes = [r.page_size_bytes for g in layer_refs_per_group for r in g]
        if (
            not page_sizes
            or max(page_sizes) >= THRESHOLD_BYTES
            or any(size % 8 for size in page_sizes)
        ):
            return BatchDMABackend(is_src_access_order_any=is_src_access_order_any)
        # The existing Triton fragment kernel addresses memory in 8-byte
        # words. Canonical runs that cannot be represented in whole words must
        # stay on the DMA path after the adapter expands them.
        if copy_runs is not None and any(
            run.fragment_size % 8 or run.local_stride % 8 or run.canonical_stride % 8
            for group in copy_runs
            for ref_runs in group
            for run in ref_runs
        ):
            return BatchDMABackend(is_src_access_order_any=is_src_access_order_any)

        chunk = min(triton.next_power_of_2(max(page_sizes)), 8192)
        return BatchTritonBackend(
            bytes_per_chunk=chunk,
            is_src_access_order_any=is_src_access_order_any,
        )

    def __init__(
        self,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        host_memory_is_pinned: bool = True,
        copy_runs: list[list[tuple[CopyRun, ...]]] | None = None,
    ) -> None:
        self.backend = self.resolve(
            layer_refs_per_group,
            gpu_to_cpu,
            host_memory_is_pinned,
            copy_runs,
        )
        self._inflight_jobs: deque[int] = deque()

    def submit(
        self,
        job_id: int,
        run_descs: list[CopyRunDescriptor],
    ) -> None:
        self.backend.submit(run_descs)
        self._inflight_jobs.append(job_id)

    def finish_transfer(self, job_id: int) -> None:
        if not self._inflight_jobs or self._inflight_jobs[0] != job_id:
            raise RuntimeError("copy transfers must finish in submission order")
        self._inflight_jobs.popleft()
        self.backend.finish_transfer()

    def clear(self) -> None:
        self._inflight_jobs.clear()
        self.backend.clear()
