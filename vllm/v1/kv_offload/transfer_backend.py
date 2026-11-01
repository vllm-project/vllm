# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-operation seam for KV offloading workers (transfer contract v0).

The offloading worker machinery (`cpu/gpu_worker.py`) is device-agnostic
except for a few device-bound operations: the block-copy kernel, the
descriptor pointer dtype, device-tensor validation, and host-memory pinning.
`KVTransferBackend` concentrates them behind one injectable interface so
platform plugins can supply their own transfer paths (e.g. NPU) without
touching the offloading core.

Streams and events are intentionally NOT part of this interface: they are
already resolved through `current_platform` attribute delegation to
`torch.<device_type>`, which every conforming platform provides.

Error semantics:
- `host_register`/`host_unregister` are best-effort optimizations. They must
  not raise; returning False / no-op degrades transfers to pageable copies.
- Copy-op failures surface through the submitted job's completion.

Publication rule: a transfer becomes observable only through the worker's
`get_finished()` after its end event completes; a successful submit never
implies readiness.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch

from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import CanonicalKVCacheRef

# Uniform signature shared by every copy op returned by `select_copy_op`:
#     copy_op(src, dst, sizes, is_src_access_order_any=...) -> None
# `src`/`dst` are 1-D pointer tensors and `sizes` a 1-D byte-count tensor,
# all of length equal to the number of copy descriptors.
CopyOp = Callable[..., None]


class KVTransferBackend(ABC):
    """Pluggable device operations for KV offloading transfers."""

    @abstractmethod
    def select_copy_op(
        self,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        host_memory_is_pinned: bool = True,
    ) -> CopyOp:
        """Return the batched block-copy callable for this platform.

        The returned callable must accept
        `(src, dst, sizes, is_src_access_order_any=...)` and schedule the
        copies asynchronously on the current stream context.
        """

    @abstractmethod
    def descriptor_ptr_dtype(self) -> torch.dtype:
        """Pointer element dtype for copy-descriptor buffers.

        CUDA's copy kernel requires int64; other DMA engines may require
        uint64. The buffers are reinterpreted bit-exactly, so either width
        may address the full address space on the matching platform.
        """

    @abstractmethod
    def validate_device_tensor(self, tensor: torch.Tensor) -> None:
        """Raise if `tensor` is not a device tensor the copy op can address."""

    @abstractmethod
    def host_register(self, ptr: int, num_bytes: int) -> bool:
        """Register a host range as device-accessible pinned memory.

        Returns:
            True if the range was registered and later needs
            `host_unregister`.

        """

    @abstractmethod
    def host_unregister(self, ptr: int) -> None:
        """Release a host range previously registered by `host_register`."""


_REGISTRY: dict[str, KVTransferBackend] = {}


def register_kv_transfer_backend(device_type: str, backend: KVTransferBackend) -> None:
    """Register a transfer backend for a platform device type.

    Platform plugins (e.g. vllm-ascend for NPU) call this during plugin
    initialization. Registration takes precedence over the built-in
    CUDA-alike/XPU backend.
    """
    _REGISTRY[device_type] = backend


def unregister_kv_transfer_backend(device_type: str) -> None:
    _REGISTRY.pop(device_type, None)


def get_kv_transfer_backend() -> KVTransferBackend | None:
    """Resolve the transfer backend for the current platform, or None.

    Resolution order: explicit registration for `current_platform.device_type`,
    then the built-in backend on CUDA-alike and XPU platforms.
    """
    backend = _REGISTRY.get(current_platform.device_type)
    if backend is not None:
        return backend
    if current_platform.is_cuda_alike() or current_platform.is_xpu():
        from vllm.v1.kv_offload.cpu.gpu_worker import PlatformTransferBackend

        return PlatformTransferBackend()
    return None


def resolve_kv_transfer_backend() -> KVTransferBackend:
    """Like `get_kv_transfer_backend`, but raises when unsupported."""
    backend = get_kv_transfer_backend()
    if backend is None:
        raise RuntimeError(
            "KV offloading transfers are not supported on platform "
            f"{current_platform.device_type!r}; register a KVTransferBackend "
            "via register_kv_transfer_backend() to enable them."
        )
    return backend


def registered_backend_device_types() -> Sequence[str]:
    return tuple(_REGISTRY)
