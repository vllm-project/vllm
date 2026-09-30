# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend plumbing for CPU KV-cache copies.

The worker describes a transfer as packed strided runs.  The initial adapter
still expands those runs for the existing fragment batch APIs; later backends
can consume the same descriptor without changing the worker.
"""

import functools
from enum import Enum, auto
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


class CopyBackend(Enum):
    """Copy entry points understood by the adapter."""

    BATCH_DMA = auto()
    BATCH_TRITON = auto()


def _new_descriptor_buffers(num_copy_ops: int) -> _DescriptorBuffers:
    # CUDA cache_kernels.cu requires int64; XPU DMA engine requires uint64.
    ptr_dtype = torch.uint64 if current_platform.is_xpu() else torch.int64
    return (
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=PIN_MEMORY),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=PIN_MEMORY),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=PIN_MEMORY),
    )


class CopyBackendAdapter:
    """Own backend selection and resources for one transfer direction."""

    @staticmethod
    def resolve(
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        host_memory_is_pinned: bool = True,
        copy_runs: list[list[tuple[CopyRun, ...]]] | None = None,
    ):
        """Resolve a stable backend and callable for one transfer direction."""
        if gpu_to_cpu:
            return CopyBackend.BATCH_DMA, ops.swap_blocks_batch
        # The Triton kernel dereferences CPU pointers on the GPU, which is only
        # valid for pinned host memory. Fall back to DMA when unsupported.
        if (
            not host_memory_is_pinned
            or not HAS_TRITON
            or current_platform.is_xpu()
            or current_platform.is_rocm()
        ):
            return CopyBackend.BATCH_DMA, ops.swap_blocks_batch

        page_sizes = [r.page_size_bytes for g in layer_refs_per_group for r in g]
        if (
            not page_sizes
            or max(page_sizes) >= THRESHOLD_BYTES
            or any(size % 8 for size in page_sizes)
        ):
            return CopyBackend.BATCH_DMA, ops.swap_blocks_batch
        # The existing Triton fragment kernel addresses memory in 8-byte
        # words. Canonical runs that cannot be represented in whole words must
        # stay on the DMA path after the adapter expands them.
        if copy_runs is not None and any(
            run.fragment_size % 8 or run.local_stride % 8 or run.canonical_stride % 8
            for group in copy_runs
            for ref_runs in group
            for run in ref_runs
        ):
            return CopyBackend.BATCH_DMA, ops.swap_blocks_batch

        chunk = min(triton.next_power_of_2(max(page_sizes)), 8192)
        return CopyBackend.BATCH_TRITON, functools.partial(
            swap_blocks_batch, bytes_per_chunk=chunk
        )

    def __init__(
        self,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        host_memory_is_pinned: bool = True,
        copy_runs: list[list[tuple[CopyRun, ...]]] | None = None,
    ) -> None:
        self.backend, self._backend_copy = self.resolve(
            layer_refs_per_group,
            gpu_to_cpu,
            host_memory_is_pinned,
            copy_runs,
        )
        self._buffer_pool: list[_DescriptorBuffers] = []
        self._inflight_buffers: dict[int, _DescriptorBuffers] = {}

    def _expand_run_desc(
        self,
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

    @staticmethod
    def _estimate_max_copy_ops(run_descs: list[CopyRunDescriptor]) -> int:
        """Count legacy fragment descriptors needed by logical runs."""
        return sum(run.num_fragments for run in run_descs)

    def submit(
        self,
        job_id: int,
        run_descs: list[CopyRunDescriptor],
        *,
        is_src_access_order_any: bool,
    ) -> None:
        num_copy_ops = self._estimate_max_copy_ops(run_descs)
        buffers = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else _new_descriptor_buffers(num_copy_ops)
        )
        if buffers[0].numel() < num_copy_ops:
            buffers = _new_descriptor_buffers(num_copy_ops)
        self._inflight_buffers[job_id] = buffers
        try:
            src, dst, sizes = self._expand_run_desc(run_descs, buffers)
            if src.numel() > 0:
                self._backend_copy(
                    src,
                    dst,
                    sizes,
                    is_src_access_order_any=is_src_access_order_any,
                )
        except Exception:
            self.finish_transfer(job_id)
            raise

    def finish_transfer(self, job_id: int) -> None:
        buffers = self._inflight_buffers.pop(job_id)
        self._buffer_pool.append(buffers)

    def clear(self) -> None:
        self._inflight_buffers.clear()
        self._buffer_pool.clear()
