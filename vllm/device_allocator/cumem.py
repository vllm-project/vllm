# cumem-based pytorch pluggable allocator
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
from contextlib import contextmanager
from typing import Dict, Optional

import torch
from vllm_allocator_adaptor import (HandleType, create_and_map,
                                    unmap_and_release,
                                    use_memory_pool_with_allocator)

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.utils import is_pin_memory_available

libcudart = CudaRTLibrary()


class CuMemAllocator:

    def __init__(self):
        self.pointer_to_handle: Dict[int, HandleType] = {}
        self.pointer_to_cpu_backup_tensor: Dict[int,
                                                Optional[torch.Tensor]] = {}

    def python_malloc_callback(self, allocation_handle: HandleType) -> None:
        py_d_mem = allocation_handle[2]
        self.pointer_to_handle[py_d_mem] = allocation_handle
        self.pointer_to_cpu_backup_tensor[py_d_mem] = None
        return

    def python_free_callback(self, ptr: int) -> HandleType:
        cpu_backup_tensor = self.pointer_to_cpu_backup_tensor.pop(ptr)
        if cpu_backup_tensor is not None:
            del cpu_backup_tensor
        return self.pointer_to_handle.pop(ptr)

    def offload(self):
        for ptr, handle in self.pointer_to_handle.items():
            size_in_bytes = handle[1]
            cpu_backup_tensor = torch.empty(
                size_in_bytes,
                dtype=torch.uint8,
                device='cpu',
                pin_memory=is_pin_memory_available())
            cpu_ptr = cpu_backup_tensor.data_ptr()
            libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
            self.pointer_to_cpu_backup_tensor[ptr] = cpu_backup_tensor
        self.unmap()

    def restore(self):
        self.remap()
        for ptr, cpu_backup_tensor in self.pointer_to_cpu_backup_tensor.items(
        ):
            size_in_bytes = cpu_backup_tensor.numel()
            cpu_ptr = cpu_backup_tensor.data_ptr()
            libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
        self.pointer_to_cpu_backup_tensor = {
            ptr: None
            for ptr in self.pointer_to_cpu_backup_tensor
        }

    def unmap(self):
        for handle in self.pointer_to_handle.values():
            unmap_and_release(handle)

    def remap(self):
        for handle in self.pointer_to_handle.values():
            create_and_map(handle)

    @contextmanager
    def use_memory_pool(self):
        with use_memory_pool_with_allocator(self.python_malloc_callback,
                                            self.python_free_callback):
            yield
