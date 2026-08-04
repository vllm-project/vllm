# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific platform primitives."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.device_ops import DeviceOps
    from lmcache.v1.platform.base.event_ipc import EventIPCBackend
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class CudaDeviceSpec(DeviceSpec):
    """CUDA device specification for the detection registry."""

    _event_backend_cache: "EventIPCBackend | None" = None

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def torch_module_name(self) -> str:
        return "cuda"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.cuda.device_ops import CudaDeviceOps

        return CudaDeviceOps

    @property
    def event_ipc_backend(self) -> "EventIPCBackend":
        """Return the CUDA event IPC backend."""
        backend = self._event_backend_cache
        if backend is None:
            # Third Party
            import torch

            # First Party
            from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend

            backend = DefaultEventIPCBackend(
                event_module=torch.cuda,
                device_type=self.device_type,
            )
            self._event_backend_cache = backend
        return backend

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        return CudaPinMemoryBackend

    @property
    def ipc_wrapper_cls(self) -> type[DeviceIPCWrapper] | None:
        # First Party
        from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

        return CudaIPCWrapper

    def is_available(self) -> bool:
        """Check CUDA availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def create_cache_context(self, *args: Any, **kwargs: Any) -> "BaseCacheContext":
        # First Party
        from lmcache.v1.platform.cuda.cache_context import GPUCacheContext

        return GPUCacheContext(*args, **kwargs)
