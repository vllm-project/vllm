# SPDX-License-Identifier: Apache-2.0

# Standard
import ctypes

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator


class CuFileMemoryAllocator(GPUMemoryAllocator):
    def __init__(self, size: int, device=None) -> None:
        # HACK(Jiayi): cufile import is buggy on some hardware
        # (e.g., without GPUDirect), so it's temporarily put here.
        # Third Party
        from cufile.bindings import cuFileBufDeregister, cuFileBufRegister

        self.cuFileBufDeregister = cuFileBufDeregister
        if device is None:
            # TODO(Serapheim): Ideally we'd get the device from the upper
            # layer - for now just use the current device.
            if torch_dev.is_available():
                device = f"{torch_device_type}:{torch_dev.current_device()}"
            else:
                device = "cpu:0"
        super().__init__(size, device, align_bytes=4096)
        self.base_pointer = self.tensor.data_ptr()
        cuFileBufRegister(ctypes.c_void_p(self.base_pointer), size, flags=0)

    def __del__(self) -> None:
        if hasattr(self, "base_pointer") and hasattr(self, "cuFileBufDeregister"):
            try:
                self.cuFileBufDeregister(ctypes.c_void_p(self.base_pointer))
            except Exception:
                pass

    def __str__(self) -> str:
        return "CuFileMemoryAllocator"
