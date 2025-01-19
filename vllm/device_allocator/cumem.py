# cumem-based pytorch pluggable allocator
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import torch

from vllm.utils import is_pin_memory_available


def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """ # noqa
    found = False
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found = True
                break
    if not found:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = line.index("/")
    path = line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), \
        f"Unexpected filename: {filename} for library {lib_name}"
    return path


cumem_available = False
try:
    from vllm.cumem_allocator import (init_module, python_create_and_map,
                                      python_unmap_and_release)
    from vllm.distributed.device_communicators.cuda_wrapper import (
        CudaRTLibrary)
    lib_name = find_loaded_library("cumem_allocator")
    libcudart = CudaRTLibrary()
    cumem_available = True
except Exception:
    # rocm platform does not support cumem allocator
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    CudaRTLibrary = None
    lib_name = None
    libcudart = None

# py_device, py_alignedSize, py_d_mem, py_p_memHandle
HandleType = Tuple[int, int, int, int]


def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[int],
                               int], python_free_func: Callable[[int, int],
                                                                None]
) -> torch.cuda.memory.CUDAPluggableAllocator:
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_name, 'my_malloc', 'my_free')
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
        python_malloc_fn: Callable[[int], int],
        python_free_func: Callable[[int, int], None]) -> None:
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
    with torch.cuda.memory.use_mem_pool(mem_pool):
        yield mem_pool


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
            # PyTorch's bug, calling torch.cuda.empty_cache() will error
            # when using pluggable allocator, see
            # https://github.com/pytorch/pytorch/issues/145168 .
            # if we have some memory allocated and then freed,
            # the memory will not be released.
            # right now it is fine, because we only use this allocator
            # during weight loading and kv cache creation, where we only
            # allocate memory.
            # TODO: we need to find a way to release the memory,
            # i.e. calling torch.cuda.empty_cache()
            self.current_mode = old_mode

    def get_current_usage(self):
        sum_bytes = 0
        for ptr, handle in self.pointer_to_handle.items():
            sum_bytes += handle[1]
        return sum_bytes
