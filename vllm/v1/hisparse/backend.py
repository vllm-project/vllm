# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device operations used by the common HiSparse runtime and connector."""

from __future__ import annotations

import os
from functools import cache
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.hisparse.runtime import HiSparseRuntime
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

_HISPARSE_OP_NAMES = (
    "hisparse_resolve_residency",
    "hisparse_invalidate_written_slots",
    "hisparse_gather_plan",
    "hisparse_gather_compact",
)


class HiSparseBackend:
    supports_shared_host_pool = False
    stats_dtype = torch.uint64

    def check_available(self) -> None:
        raise NotImplementedError

    def pin_tensor(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError

    def unpin_tensor(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError

    def allocate_host_pool(
        self, size: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Return (pool, registered range, tensors to unpin on release)."""
        from vllm.v1.hisparse.runtime import allocate_pinned_host_pool

        pool, registered = allocate_pinned_host_pool(size)
        return pool, registered, [registered]

    def create_cache_ops(self, runtime: HiSparseRuntime) -> Any:
        raise NotImplementedError

    def validate_kv_cache_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Validate device cache formats before allocating or pinning storage."""

    def prepare_kv_cache_views(
        self,
        kv_cache_config: KVCacheConfig,
        kv_caches: dict[str, Any],
        forward_context: dict[str, Any],
    ) -> None:
        """Adapt device views before runtime binding; retain allocation ownership.

        Called once per cache initialization, inside the common rollback scope.
        Backends must not put per-step metadata or transfers in this hook.
        """

    def create_ipc_event(self) -> Any:
        """Create an event that can be exported to another process."""
        raise NotImplementedError

    def export_event(self, event: Any) -> Any:
        raise NotImplementedError

    def import_event(self, device: torch.device, handle: Any) -> Any:
        raise NotImplementedError

    def gather_resident_rows(
        self, source: torch.Tensor, destination: torch.Tensor, row_ids: torch.Tensor
    ) -> None:
        """Copy resident rows into staging without changing their payload format."""
        rows = row_ids.to(torch.long).clamp_min(0)
        block_size = source.shape[1]
        payload = source[rows // block_size, rows % block_size]
        if payload.dtype != destination.dtype:
            payload = payload.contiguous().view(destination.dtype)
        destination.copy_(payload)

    def is_current_stream_capturing(self) -> bool:
        raise NotImplementedError

    def swap_blocks_batch(
        self, src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor
    ) -> None:
        """Enqueue device-to-host copies on the current stream."""
        raise NotImplementedError


class CUDAHiSparseBackend(HiSparseBackend):
    supports_shared_host_pool = True

    def check_available(self) -> None:
        names = (
            "hisparse_resolve_residency",
            "hisparse_invalidate_written_slots",
            "hisparse_gather_plan",
            "hisparse_gather_compact",
        )
        if not all(hasattr(torch.ops._C_cache_ops, name) for name in names):
            raise RuntimeError(
                "HiSparse requires its compiled _C_cache_ops CUDA kernels. "
                "Rebuild vLLM from source with csrc/hisparse_kernels.cu."
            )

    def pin_tensor(self, tensor: torch.Tensor) -> None:
        from vllm.utils.host_memory import pin_tensor

        pin_tensor(tensor)

    def unpin_tensor(self, tensor: torch.Tensor) -> None:
        from vllm.utils.host_memory import unpin_tensor

        unpin_tensor(tensor)

    def create_cache_ops(self, runtime: HiSparseRuntime) -> Any:
        return torch.ops._C_cache_ops

    def create_ipc_event(self) -> Any:
        return torch.cuda.Event(interprocess=True)

    def export_event(self, event: Any) -> Any:
        return event.ipc_handle()

    def import_event(self, device: torch.device, handle: Any) -> Any:
        return torch.cuda.Event.from_ipc_handle(device, handle)

    def is_current_stream_capturing(self) -> bool:
        return torch.cuda.is_current_stream_capturing()

    def swap_blocks_batch(
        self, src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor
    ) -> None:
        from vllm import _custom_ops as ops

        ops.swap_blocks_batch(src, dst, sizes)


class _XPUHiSparseMockOps:
    """WIP bring-up mock: every top-k row resolves invalid; outputs are wrong."""

    def hisparse_resolve_residency(
        self,
        host_cache: torch.Tensor,
        hot_cache: torch.Tensor,
        hot_block_table: torch.Tensor,
        global_indices: torch.Tensor,
        hot_indices: torch.Tensor,
        device_global_indices: torch.Tensor,
        lru_slots: torch.Tensor,
        request_state_indices: torch.Tensor | None,
        region_stride: int,
        max_union_rows: int,
        stats: torch.Tensor | None = None,
        attention_indices: torch.Tensor | None = None,
        attention_block_stride: int = 0,
        request_ids: torch.Tensor | None = None,
        source_block_table: torch.Tensor | None = None,
        source_block_size: int = 0,
        valid_counts: torch.Tensor | None = None,
        swap_host_physical_rows: torch.Tensor | None = None,
        swap_device_physical_rows: torch.Tensor | None = None,
        swap_counts: torch.Tensor | None = None,
        resident_block_table: torch.Tensor | None = None,
        resident_block_size: int = 0,
        num_valid_rows: torch.Tensor | None = None,
    ) -> None:
        hot_indices.fill_(-1)
        for t in (
            attention_indices,
            swap_host_physical_rows,
            swap_device_physical_rows,
        ):
            if t is not None:
                t.fill_(-1)
        for t in (valid_counts, swap_counts):
            if t is not None:
                t.zero_()

    def hisparse_gather_compact(
        self,
        host_cache: torch.Tensor,
        hot_cache: torch.Tensor,
        miss_global_indices: torch.Tensor,
        miss_hot_indices: torch.Tensor,
        miss_counts: torch.Tensor,
    ) -> None:
        pass

    def hisparse_gather_plan(
        self,
        host_cache: torch.Tensor,
        hot_cache: torch.Tensor,
        global_indices: torch.Tensor,
        hot_indices: torch.Tensor,
        miss_mask: torch.Tensor,
    ) -> None:
        # The staging cache comes from torch.empty; avoid feeding NaNs to attention.
        hot_cache.zero_()

    def hisparse_invalidate_written_slots(
        self,
        device_global_indices_ptrs: torch.Tensor,
        request_state_indices: torch.Tensor,
        req_id_per_token: torch.Tensor,
        written_slots: torch.Tensor,
        num_state_rows: int,
        region_stride: int,
    ) -> None:
        pass


class XPUHiSparseBackend(HiSparseBackend):
    supports_shared_host_pool = False

    def _has_native_ops(self) -> bool:
        return all(hasattr(torch.ops._C_cache_ops, name) for name in _HISPARSE_OP_NAMES)

    def _use_mock_ops(self) -> bool:
        return os.environ.get("VLLM_XPU_HISPARSE_MOCK_OPS") == "1"

    def check_available(self) -> None:
        if self._has_native_ops():
            return
        if self._use_mock_ops():
            logger.warning_once(
                "Using HiSparse XPU mock ops; generated outputs are WRONG."
            )
            return
        raise RuntimeError(
            "HiSparse on XPU requires the hisparse_* ops from vllm-xpu-kernels."
        )

    def pin_tensor(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError("Shared host pools are not supported on XPU yet.")

    def unpin_tensor(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError("Shared host pools are not supported on XPU yet.")

    def allocate_host_pool(
        self, size: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        # Kernels read the pool directly, so it must be USM host memory.
        pool = torch.empty(size, dtype=torch.int8, pin_memory=True)
        return pool, pool, []

    def create_cache_ops(self, runtime: HiSparseRuntime) -> Any:
        if self._has_native_ops():
            return torch.ops._C_cache_ops
        return _XPUHiSparseMockOps()

    def create_ipc_event(self) -> Any:
        return torch.xpu.Event(interprocess=True)

    def export_event(self, event: Any) -> Any:
        return event.ipc_handle()

    def import_event(self, device: torch.device, handle: Any) -> Any:
        return torch.xpu.Event.from_ipc_handle(device, handle)

    def is_current_stream_capturing(self) -> bool:
        return torch.xpu.is_current_stream_capturing()

    def swap_blocks_batch(
        self, src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor
    ) -> None:
        from vllm import _custom_ops as ops

        ops.swap_blocks_batch(src, dst, sizes)


def get_hisparse_backend() -> HiSparseBackend:
    backend_cls = current_platform.get_hisparse_backend_cls()
    if backend_cls is None:
        raise RuntimeError("The current platform does not support HiSparse.")
    return _get_hisparse_backend(backend_cls)


@cache
def _get_hisparse_backend(backend_cls: str) -> HiSparseBackend:
    return resolve_obj_by_qualname(backend_cls)()
