# cumem-based pytorch pluggable allocator
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Optional

import torch
from vllm_allocator_adaptor import (HandleType, create_and_map,
                                    unmap_and_release,
                                    use_memory_pool_with_allocator)

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.utils import is_pin_memory_available

libcudart = CudaRTLibrary()

# an enum of two modes: offload and discard
# offload: move the data from GPU to CPU when sleeping
# discard: discard the data when sleeping
# the default mode is offload


class CuMemMode(Enum):
    OFFLOAD = 1
    DISCARD = 2


class CuMemAllocator:
    """
    A singleton class that manages a memory pool for CUDA tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.

    Inside the `use_memory_pool(mode)` context, all tensors created will
    be allocated in the memory pool, and has the same mode as the
    mode passed to the context.

    Why it needs to be a singleton?
    When allocated tensors are garbage collected, PyTorch will call
    the free callback, which will call the `python_free_callback` method.
    The C-extension uses a global variable to store the function of an
    instance of this class. If we create multiple instances of this class,
    the global variable will be overwritten and the free callback will
    not work as expected.
    """
    instance: "CuMemAllocator" = None

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        if CuMemAllocator.instance is None:
            CuMemAllocator.instance = CuMemAllocator()
        return CuMemAllocator.instance

    def __init__(self):
        self.pointer_to_handle: Dict[int, HandleType] = {}
        self.pointer_to_cpu_backup_tensor: Dict[int,
                                                Optional[torch.Tensor]] = {}
        self.pointer_to_mode: Dict[int, CuMemMode] = {}
        self.current_mode = CuMemMode.OFFLOAD

    def python_malloc_callback(self, allocation_handle: HandleType) -> None:
        py_d_mem = allocation_handle[2]
        self.pointer_to_handle[py_d_mem] = allocation_handle
        self.pointer_to_cpu_backup_tensor[py_d_mem] = None
        self.pointer_to_mode[py_d_mem] = self.current_mode
        return

    def python_free_callback(self, ptr: int) -> HandleType:
        cpu_backup_tensor = self.pointer_to_cpu_backup_tensor.pop(ptr)
        if cpu_backup_tensor is not None:
            del cpu_backup_tensor
        return self.pointer_to_handle.pop(ptr)

    def sleep(self):
        for ptr, mode in self.pointer_to_mode.items():
            handle = self.pointer_to_handle[ptr]
            if mode == CuMemMode.OFFLOAD:
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(
                    size_in_bytes,
                    dtype=torch.uint8,
                    device='cpu',
                    pin_memory=is_pin_memory_available())
                cpu_ptr = cpu_backup_tensor.data_ptr()
                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
                self.pointer_to_cpu_backup_tensor[ptr] = cpu_backup_tensor
            unmap_and_release(handle)

    def wake_up(self):
        for ptr, mode in self.pointer_to_mode.items():
            handle = self.pointer_to_handle[ptr]
            create_and_map(handle)
            if mode == CuMemMode.OFFLOAD:
                cpu_backup_tensor = self.pointer_to_cpu_backup_tensor.pop(ptr)
                if cpu_backup_tensor is not None:
                    size_in_bytes = cpu_backup_tensor.numel(
                    ) * cpu_backup_tensor.element_size()
                    cpu_ptr = cpu_backup_tensor.data_ptr()
                    libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)

        self.pointer_to_cpu_backup_tensor = {
            ptr: None
            for ptr in self.pointer_to_cpu_backup_tensor
        }

    @contextmanager
    def use_memory_pool(self, mode: CuMemMode = CuMemMode.OFFLOAD):
        old_mode = self.current_mode
        self.current_mode = mode
        with use_memory_pool_with_allocator(self.python_malloc_callback,
                                            self.python_free_callback):
            yield
            self.current_mode = old_mode

    def get_current_usage(self):
        sum_bytes = 0
        for ptr, handle in self.pointer_to_handle.items():
            sum_bytes += handle[1]
        return sum_bytes
