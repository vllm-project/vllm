# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device operations used by the common HiSparse runtime and connector."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.hisparse.runtime import HiSparseRuntime
    from vllm.v1.kv_cache_interface import KVCacheConfig


class HiSparseBackend:
    supports_shared_host_pool = False
    stats_dtype = torch.uint64

    def check_available(self) -> None:
        raise NotImplementedError

    def pin_tensor(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError

    def unpin_tensor(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError

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
        from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

        pin_tensor(tensor)

    def unpin_tensor(self, tensor: torch.Tensor) -> None:
        cudart = torch.cuda.cudart()
        error = cudart.cudaHostUnregister(tensor.data_ptr())
        if error.value != 0:
            cudart.cudaGetLastError()
            raise RuntimeError(f"cudaHostUnregister failed (code={error.value}).")

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


def get_hisparse_backend() -> HiSparseBackend:
    backend_cls = current_platform.get_hisparse_backend_cls()
    if backend_cls is None:
        raise RuntimeError("The current platform does not support HiSparse.")
    return _get_hisparse_backend(backend_cls)


@cache
def _get_hisparse_backend(backend_cls: str) -> HiSparseBackend:
    return resolve_obj_by_qualname(backend_cls)()
