# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# cumem-based pytorch pluggable allocator to implement sleep mode.
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
import dataclasses
import gc
import os
from contextlib import contextmanager
from typing import Any, Callable, Optional, Union

import torch

from vllm.logger import init_logger
from vllm.utils import is_pin_memory_available

logger = init_logger(__name__)


def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """  # noqa
    found_line = None
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found_line = line
                break
    if found_line is None:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = found_line.index("/")
    path = found_line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), (
        f"Unexpected filename: {filename} for library {lib_name}"
    )
    return path


cumem_available = False
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
    # rocm platform does not support cumem allocator
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    CudaRTLibrary = None
    lib_name = None
    libcudart = None

# py_device, py_alignedSize, py_d_mem, py_p_memHandle
HandleType = tuple[int, int, int, int]


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: Optional[torch.Tensor] = None


def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[int], int], python_free_func: Callable[[int, int], None]
) -> torch.cuda.memory.CUDAPluggableAllocator:
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_name, "my_malloc", "my_free"
    )
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[int], int], python_free_func: Callable[[int, int], None]
) -> None:
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

    instance: "CuMemAllocator" = None
    default_tag: str = "default"
    graphs_tag: str = "graphs"

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
        return CuMemAllocator.instance

    def __init__(self):
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        assert "expandable_segments:True" not in conf, (
            "Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates."
        )

        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CuMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}
        # CUDA graphs metadata saved before sleep
        self._sleep_saved_cudagraphs: dict[str, Any] = {}
        # Creating strong references to the two callbacks here to prevent
        # these ephemeral bound-method objects being garbage collected.
        # See discussions in https://github.com/vllm-project/vllm/pull/22724
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

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
        logger.debug(
            "Freed %s bytes for %s with address %s from cumem allocator",
            data.handle[1],
            data.tag,
            ptr,
        )
        return data.handle

    def sleep(
            self,
            offload_tags: Optional[Union[tuple[str, ...],
                                         str]] = None,
            model_runner=None) -> None:
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.

        :param offload_tags: The tags of the memory allocation that will be
            offloaded. The rest of the memory allocation will be discarded.
        :param model_runner: Optional model runner for CUDA graph handling.
        """
        if offload_tags is None:
            # by default, allocated tensors and graphs are offloaded
            # when the allocator sleeps
            offload_tags = (CuMemAllocator.default_tag, CuMemAllocator.graphs_tag)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        assert isinstance(offload_tags, tuple)

        # Handle CUDA graphs if model_runner is provided and graphs are being
        # offloaded
        has_graphs = CuMemAllocator.graphs_tag in offload_tags
        if model_runner is not None and has_graphs:
            self._save_cuda_graphs(model_runner)

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
                    pin_memory=is_pin_memory_available(),
                )
                cpu_ptr = cpu_backup_tensor.data_ptr()
                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
                data.cpu_backup_tensor = cpu_backup_tensor
            unmap_and_release(handle)

        logger.info(
            "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "
            "directly. %s", total_bytes / 1024**3, backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
            f"Graph offloaded (tag: {CuMemAllocator.graphs_tag})" if has_graphs
            else "CUDA graphs not managed by CuMemAllocator")

        gc.collect()

    def wake_up(self, tags: Optional[list[str]] = None, model_runner=None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory.

        :param tags: The tags of the memory allocation that will be loaded
            back to GPU memory. If None, all memory allocation will be loaded
            back to GPU memory.
        :param model_runner: Optional model runner for CUDA graph handling.
        """
        for ptr, data in self.pointer_to_data.items():
            if tags is None or data.tag in tags:
                handle = data.handle
                create_and_map(handle)
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = (
                            cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                        )
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                        data.cpu_backup_tensor = None

        # Restore CUDA graphs if model_runner is provided and graphs are being restored
        if (model_runner is not None and
            (tags is None or CuMemAllocator.graphs_tag in tags)):
            self._restore_cuda_graphs(model_runner)

    @contextmanager
    def use_memory_pool(self, tag: Optional[str] = None):
        """
        A context manager to use the memory pool.
        All memory allocation created inside the context will be allocated
        in the memory pool, and has the specified tag.

        :param tag: The tag of the memory allocation. If None, the default tag
            will be used.
        """
        if tag is None:
            tag = CuMemAllocator.default_tag

        assert isinstance(tag, str)

        old_tag = self.current_tag
        self.current_tag = tag
        with use_memory_pool_with_allocator(
            self.python_malloc_callback, self.python_free_callback
        ) as data:
            # start to hit another PyTorch bug in PyTorch 2.6,
            # possibly because of gc-related issue w.r.t. the allocator and
            # the memory pool.
            # to avoid the issue, we keep a reference of the data.
            # see https://github.com/pytorch/pytorch/issues/146431 .
            self.allocator_and_pools[tag] = data
            yield
            # PyTorch's bug, calling torch.cuda.empty_cache() will error
            # when using pluggable allocator, see
            # https://github.com/pytorch/pytorch/issues/145168 .
            # if we have some memory allocated and then freed,
            # the memory will not be released, e.g. in online quantization,
            # where the model is created in higher precision, and then
            # quantized in lower precision.
            # Find all unused allocations and manually release them.
            # TODO: we should expose `empty_cache` method in the memory pool.
            # TODO: ask for help from PyTorch team to expose this method.
            allocations = data[0].snapshot()
            for allocation in allocations:
                if allocation["allocated_size"] == 0:
                    handle = self._python_free_callback(allocation["address"])
                    unmap_and_release(handle)
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

    def get_graph_pool_handle(self):
        """
        Get a graph pool handle that uses the CuMemAllocator for CUDA graphs.
        This allows CUDA graphs to be managed with the same sleep/wake
        mechanism as other tagged memory allocations.
        """
        if not hasattr(self, '_graph_pool_context'):
            logger.info("CuMemAllocator: Creating custom graph pool with tag '%s' "
                       "for sleep/wake management", CuMemAllocator.graphs_tag)

            # Create a persistent memory pool context for graphs
            self._graph_pool_context = self.use_memory_pool(
                tag=CuMemAllocator.graphs_tag)
            self._graph_pool_context.__enter__()

            # Get the actual memory pool from the context
            mem_pool, _ = self.allocator_and_pools[CuMemAllocator.graphs_tag]
            self._custom_graph_pool = mem_pool

        return self._custom_graph_pool

    def setup_graph_pool_for_sleep_mode(self) -> None:
        """
        Set up custom graph pool for sleep mode after graph capture.
        This ensures graphs captured with the native pool can be properly
        mangaged during sleep/wake cycles.
        """
        try:
            # Initialize the custom graph pool context
            graph_pool_handle = self.get_graph_pool_handle()

            logger.info("CuMemAllocator: Successfully set up custom graph "
                        "pool for sleep mode management "
                        "(handle type: %s)", type(graph_pool_handle).__name__)

        except Exception as e:
            logger.warning("CuMemAllocator: Failed to set up custom graph "
                           "pool for sleep mode: %s. Using global pool", e)

    def _save_cuda_graphs(self, model_runner) -> None:
        """Put CUDA graphs to sleep."""
        free_bytes_before = torch.cuda.mem_get_info()[0]

        # Check if model runner has cudagraph dispatcher
        if hasattr(model_runner, "cudagraph_dispatcher"):
            dispatcher = model_runner.cudagraph_dispatcher
            self._sleep_saved_cudagraphs["dispatcher"] = (
                dispatcher.get_sleep_state()
            )
            logger.info(
                "Saved cudagraph dispatcher state for %d modes",
                len(
                    self._sleep_saved_cudagraphs["dispatcher"][
                        "dispatcher_keys"
                    ]
                ),
            )

        cuda_graph_count = 0
        model = model_runner.model
        if hasattr(model, 'enter_sleep_mode'):
            sleep_state, count = model.enter_sleep_mode()
            self._sleep_saved_cudagraphs["model"] = sleep_state
            cuda_graph_count += count

        logger.info("Put %d CUDA graphs into sleep mode.", cuda_graph_count)

        # Synchronize and try to free memory
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Calculate memory freed (if any) for logging
        free_bytes_after = torch.cuda.mem_get_info()[0]
        cuda_graph_memory_freed = free_bytes_after - free_bytes_before
        if cuda_graph_memory_freed > 0:
            from vllm.utils import GiB_bytes
            logger.info(
                "CUDA graph sleep freed %.2f GiB memory.",
                cuda_graph_memory_freed / GiB_bytes,
            )


    def _restore_cuda_graphs(self, model_runner) -> None:
        """Wake up CUDA graphs using cleaner interfaces."""
        if not self._sleep_saved_cudagraphs:
            logger.info("No saved CUDA graphs to restore.")
            return

        logger.info("Waking up CUDA graphs...")
        cuda_graph_count = 0

        # Verify dispatcher state
        if (hasattr(model_runner, "cudagraph_dispatcher") and
            "dispatcher" in self._sleep_saved_cudagraphs):
            dispatcher = model_runner.cudagraph_dispatcher
            if dispatcher.verify_wake_state(self._sleep_saved_cudagraphs["dispatcher"]):
                logger.info("Cudagraph dispatcher state verified successfully.")
            else:
                logger.warning("Cudagraph dispatcher state verification failed.")

        # Restore model CUDA graphs using clean interfaces
        model = model_runner.model
        if hasattr(model, 'exit_sleep_mode') and "model" in self._sleep_saved_cudagraphs:
            cuda_graph_count += model.exit_sleep_mode(
                self._sleep_saved_cudagraphs["model"]
            )

        # Trigger CUDA to fully wake up the graph memory pool
        torch.cuda.synchronize()

        logger.info("Restored %d CUDA graphs from sleep mode.", cuda_graph_count)

        # Clear saved state
        self._sleep_saved_cudagraphs.clear()
