# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator


class HipFileMemoryAllocator(GPUMemoryAllocator):
    def __init__(self, size: int, device=None) -> None:
        # HACK: hipfile import is placed here to avoid import errors on
        # hardware without GPUDirect Storage / hipFile support.
        # Third Party
        from hipfile import Buffer

        if device is None:
            if torch_dev.is_available():
                # TODO: On ROCm, PyTorch still uses the CUDA API internally
                device = f"{torch_device_type}:{torch_dev.current_device()}"
            else:
                device = "cpu:0"

        super().__init__(size, device, align_bytes=4096)
        self.base_pointer = self.tensor.data_ptr()
        self.hipfile_buffer = Buffer(self.base_pointer, size, flags=0)
        self.hipfile_buffer.register()

    def __del__(self) -> None:
        try:
            self.hipfile_buffer.deregister()
        except Exception:
            pass

    def __str__(self) -> str:
        return "HipFileMemoryAllocator"
