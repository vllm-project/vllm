# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# cumem-based pytorch pluggable allocator to implement sleep mode.
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
import atexit
import gc
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from vllm.device_allocator import AllocationData, HandleType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.system_utils import find_loaded_library
from vllm.utils.torch_utils import PIN_MEMORY

logger = init_logger(__name__)


cumem_available = False
libcudart: Any = None
try:
    from vllm.cumem_allocator import (
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib_name = find_loaded_library("cumem_allocator")
    libcudart = CudaRTLibrary()
    cumem_available = True
except ModuleNotFoundError:
    # only cuda and rocm platforms support cumem allocator
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    lib_name = None


def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
) -> torch.cuda.memory.CUDAPluggableAllocator:
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_name, "my_malloc", "my_free"
    )
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
) -> Iterator[
    tuple[torch.cuda.memory.MemPool, torch.cuda.memory.CUDAPluggableAllocator]
]:
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
    with torch.cuda.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


class CuMemAllocator:
    """
    A singleton class that manages a memory pool for CUDA tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.

    Inside the `use_memory_pool(tag)` context, all tensors created will
    be allocated in the memory pool, and has the same tag as the
    tag passed to the context.

    When we call `sleep`, all tensors with the specified tag will be
    offloaded to CPU memory, and the rest of the tensors will be discarded.
    When we call `wake_up`, all tensors that are previously offloaded
    will be loaded back to GPU memory, and the rest of the tensors will
    have empty memory.

    Why it needs to be a singleton?
    When allocated tensors are garbage collected, PyTorch will call
    the free callback, which will call the `python_free_callback` method.
    The C-extension uses a global variable to store the function of an
    instance of this class. If we create multiple instances of this class,
    the global variable will be overwritten and the free callback will
    not work as expected.
    """

    instance: "CuMemAllocator | None" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        """
        CuMemAllocator is a singleton class.
        We cannot call the constructor directly.
        Call this method to get the instance.
        """
        assert cumem_available, "cumem allocator is not available"
        if CuMemAllocator.instance is None:
            CuMemAllocator.instance = CuMemAllocator()
            # Ensure MemPool/allocator wrappers are released before interpreter
            # finalization tears down PyTorch allocator internals.
            atexit.register(CuMemAllocator._shutdown_singleton)
        return CuMemAllocator.instance

    @staticmethod
    def _shutdown_singleton() -> None:
        instance = CuMemAllocator.instance
        if instance is None:
            return
        try:
            instance.release_pools()
        except Exception:
            logger.exception("CuMemAllocator singleton shutdown failed")

    def __init__(self):
        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CuMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}
        # Creating strong references to the two callbacks here to prevent
        # these ephemeral bound-method objects being garbage collected.
        # See discussions in https://github.com/vllm-project/vllm/pull/22724
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

    def release_pools(self) -> None:
        """Drop Python references to MemPool/pluggable allocators eagerly.

        A cumem ``MemPool`` outlives the ``use_memory_pool`` context (a strong
        reference is kept in ``allocator_and_pools`` to work around
        pytorch/pytorch#146431), and a captured CUDA graph can keep it alive
        longer still. ``MemPool`` only holds a non-owning pointer to the
        allocator, whose owning reference lives in the Python
        ``CUDAPluggableAllocator``. If both are instead dropped during
        interpreter shutdown, GC may finalize the allocator first; the eventual
        ``~MemPool`` -> ``emptyCache`` -> ``release_block`` then makes a virtual
        call into the freed allocator -- aborting the process with "pure virtual
        method called" (pytorch/pytorch#145168).

        Release the kept-alive pools before interpreter finalization, and keep
        the pluggable allocator wrappers alive while MemPool destructors run.
        This is safe to call more than once.
        """
        if not self.allocator_and_pools:
            return

        pool_entries = list(self.allocator_and_pools.values())
        self.allocator_and_pools.clear()

        mem_pools = [entry[0] for entry in pool_entries]
        allocators = [entry[1] for entry in pool_entries]
        pool_entries.clear()

        # Phase 1: drop MemPool refs while allocators are still strongly held.
        mem_pools.clear()
        gc.collect()

        # Phase 2: now it is safe to release allocator wrappers.
        allocators.clear()

    def close(self) -> None:
        """Compatibility alias for deterministic pool release."""
        self.release_pools()

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """
        Internal method to store the allocation data
        when memory is allocated in the memory pool."""
        py_d_mem = allocation_handle[2]
        self.pointer_to_data[py_d_mem] = AllocationData(
            allocation_handle, self.current_tag
        )
        logger.debug(
            "Allocated %s bytes for %s with address %s from cumem allocator",
            allocation_handle[1],
            self.current_tag,
            py_d_mem,
        )
        return

    def _python_free_callback(self, ptr: int) -> HandleType:
        """
        Internal method to look up the allocation data
        when memory is freed in the memory pool."""
        data = self.pointer_to_data.pop(ptr)
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        if data.is_asleep and current_platform.is_rocm():
            # On ROCm, sleep() already unmapped and released this allocation's
            # physical chunks and holds its virtual address as a placeholder
            # reservation. Return a handle with an empty chunk list so the C
            # extension skips unmap/release (avoiding a double-free) while
            # still freeing the placeholder address.
            device, size, d_mem, _ = data.handle
            return (device, size, d_mem, [])
        # Drain pending kernels before the C extension's cuMemUnmap.
        # The pluggable allocator path doesn't defer reclaim like the
        # regular caching allocator, so without this, in-flight work
        # (e.g. quant helpers' transient tensors during weight loading)
        # races the unmap and surfaces as CUDA_ERROR_ILLEGAL_ADDRESS.
        torch.cuda.synchronize(data.handle[0])
        logger.debug(
            "Freed %s bytes for %s with address %s from cumem allocator",
            data.handle[1],
            data.tag,
            ptr,
        )
        return data.handle

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.

        Args:
            offload_tags: The tags of the memory allocation that will be
                offloaded. The rest of the memory allocation will be discarded.
        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        assert isinstance(offload_tags, tuple)

        total_bytes = 0
        backup_bytes = 0

        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            total_bytes += handle[1]
            if data.tag in offload_tags:
                backup_bytes += handle[1]
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(
                    size_in_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=PIN_MEMORY,
                )
                cpu_ptr = cpu_backup_tensor.data_ptr()
                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
                data.cpu_backup_tensor = cpu_backup_tensor
            try:
                unmap_and_release(handle)
            finally:
                data.is_asleep = True

        logger.info(
            "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "
            "directly.",
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
        )

        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory.

        Args:
            tags: The tags of the memory allocation that will be loaded
                back to GPU memory. If None, all memory allocation will be loaded
                back to GPU memory.
        """
        for ptr, data in self.pointer_to_data.items():
            if tags is None or data.tag in tags:
                handle = data.handle
                create_and_map(handle)
                data.is_asleep = False
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = (
                            cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                        )
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                        data.cpu_backup_tensor = None

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """
        A context manager to use the memory pool.
        All memory allocation created inside the context will be allocated
        in the memory pool, and has the specified tag.

        Args:
            tag: The tag of the memory allocation. If None, the default tag
                will be used.
        """
        if tag is None:
            tag = CuMemAllocator.default_tag

        assert isinstance(tag, str)

        # Expandable segments are incompatible with the memory pool used for
        # sleep mode (see https://github.com/pytorch/pytorch/issues/147851).
        # If the user has enabled expandable segments via
        # PYTORCH_CUDA_ALLOC_CONF, temporarily disable them for the duration
        # of the memory pool context and restore on exit.
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        expandable_was_enabled = "expandable_segments:True" in conf
        if expandable_was_enabled:
            torch.cuda.memory._set_allocator_settings("expandable_segments:False")

        old_tag = self.current_tag
        self.current_tag = tag
        try:
            with use_memory_pool_with_allocator(
                self.python_malloc_callback, self.python_free_callback
            ) as data:
                # start to hit another PyTorch bug in PyTorch 2.6,
                # possibly because of gc-related issue w.r.t. the allocator
                # and the memory pool.
                # to avoid the issue, we keep a reference of the data.
                # see https://github.com/pytorch/pytorch/issues/146431 .
                self.allocator_and_pools[tag] = data
                yield
                # PyTorch's bug, calling torch.cuda.empty_cache() will error
                # when using pluggable allocator, see
                # https://github.com/pytorch/pytorch/issues/145168 .
                # if we have some memory allocated and then freed,
                # the memory will not be released, e.g. in online
                # quantization, where the model is created in higher
                # precision, and then quantized in lower precision.
                # Find all unused allocations and manually release them.
                # TODO: we should expose `empty_cache` method in the memory
                # pool.
                # TODO: ask for help from PyTorch team to expose this method.
                allocations = data[0].snapshot()
                for allocation in allocations:
                    if allocation["allocated_size"] == 0:
                        handle = self._python_free_callback(allocation["address"])
                        unmap_and_release(handle)
        finally:
            self.current_tag = old_tag
            if expandable_was_enabled:
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    @contextmanager
    def reuse_memory_pool(self, tag: str) -> Iterator[None]:
        """
        Re-enter an existing memory pool so that new allocations
        use the same pool and are tagged accordingly.
        """
        if tag not in self.allocator_and_pools:
            raise ValueError(f"Memory pool for tag '{tag}' does not exist.")

        mem_pool, _ = self.allocator_and_pools[tag]
        old_tag = self.current_tag
        self.current_tag = tag
        try:
            with torch.cuda.memory.use_mem_pool(mem_pool):
                yield
        finally:
            self.current_tag = old_tag

    def get_current_usage(self) -> int:
        """
        Get the total number of bytes allocated in the memory pool.
        """
        sum_bytes: int = 0
        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            sum_bytes += handle[1]
        return sum_bytes
