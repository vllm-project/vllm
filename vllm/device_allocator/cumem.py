# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# cumem-based pytorch pluggable allocator to implement sleep mode.
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
import gc
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from vllm.device_allocator import AllocationData, HandleType
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.system_utils import find_loaded_library

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
        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CuMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}
        # Dedicated stream for sleep/wake bulk copies so per-allocation
        # cudaMemcpy calls don't serialize on implicit synchronization.
        self._copy_stream: torch.cuda.Stream | None = None
        # Creating strong references to the two callbacks here to prevent
        # these ephemeral bound-method objects being garbage collected.
        # See discussions in https://github.com/vllm-project/vllm/pull/22724
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

    @staticmethod
    def _async_copy_available() -> bool:
        # Test doubles for libcudart may only provide the synchronous
        # cudaMemcpy; fall back transparently in that case.
        return hasattr(libcudart, "cudaMemcpyAsync")

    @staticmethod
    def _reuse_pinned_backup() -> bool:
        """Whether to keep pinned CPU backup buffers across sleep cycles.

        Enabled by default: page-pinning tens of GiB on every sleep costs
        seconds, and periodic sleep/wake (flash EP scaling, RL rollouts)
        pays it repeatedly. Costs the same amount of host RAM held pinned
        between cycles; disable with VLLM_CUMEM_PINNED_CACHE=0.
        """
        return os.environ.get("VLLM_CUMEM_PINNED_CACHE", "1") == "1"

    def _get_copy_stream(self) -> torch.cuda.Stream:
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        return self._copy_stream

    def _acquire_backup_buffer(self, data: AllocationData) -> torch.Tensor:
        size_in_bytes = data.handle[1]
        cached = data.cpu_backup_cache
        data.cpu_backup_cache = None
        if cached is not None and cached.numel() == size_in_bytes:
            return cached
        return torch.empty(
            size_in_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )

    def _release_backup_buffer(self, data: AllocationData) -> None:
        if self._reuse_pinned_backup():
            data.cpu_backup_cache = data.cpu_backup_tensor
        data.cpu_backup_tensor = None

    def warm_backup_cache(self, offload_tags: tuple[str, ...]) -> int:
        """Pre-allocate pinned host backup buffers for the given tags.

        Page-pinning is the dominant cost of the first sleep (seconds for
        tens of GiB); calling this off the critical path (e.g. while a
        rank drains before scaling down) makes the first sleep as fast as
        subsequent cached ones. Idempotent; returns bytes newly pinned.
        """
        warmed = 0
        for data in self.pointer_to_data.values():
            if data.tag not in offload_tags or data.cpu_backup_tensor is not None:
                continue
            size_in_bytes = data.handle[1]
            cached = data.cpu_backup_cache
            if cached is not None and cached.numel() == size_in_bytes:
                continue
            data.cpu_backup_cache = torch.empty(
                size_in_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=is_pin_memory_available(),
            )
            warmed += size_in_bytes
        return warmed

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

    def rename_tag(self, old_tag: str, new_tag: str) -> int:
        """Rename all tracked allocations that currently use ``old_tag``."""
        changed = 0
        for data in self.pointer_to_data.values():
            if data.tag == old_tag:
                data.tag = new_tag
                changed += 1
        return changed

    def retag_allocations_by_ptrs(self, ptrs: set[int], tag: str) -> int:
        """Assign ``tag`` to tracked allocations identified by device ptr."""
        changed = 0
        for ptr, data in self.pointer_to_data.items():
            if ptr in ptrs:
                data.tag = tag
                changed += 1
        return changed

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
        per_tag_total: dict[str, int] = {}
        per_tag_backup: dict[str, int] = {}
        offload_items: list[tuple[int, AllocationData]] = []

        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            # CUDA graphs that are not selected for offload must stay resident.
            # Their memory lives in the cumem pool, but discarding it (unmap
            # without a CPU backup) would leave the captured graph pointing at
            # undefined pages after wake-up -> garbage output, since graphs are
            # restored in place rather than re-captured.
            if (
                data.tag == CuMemAllocator.graphs_tag
                and data.tag not in offload_tags
            ):
                continue
            total_bytes += handle[1]
            per_tag_total[data.tag] = per_tag_total.get(data.tag, 0) + handle[1]
            if data.tag in offload_tags:
                backup_bytes += handle[1]
                per_tag_backup[data.tag] = per_tag_backup.get(data.tag, 0) + handle[1]
                offload_items.append((ptr, data))
            else:
                unmap_and_release(handle)

        if offload_items:
            if self._async_copy_available():
                # Enqueue every D2H copy on a dedicated stream and sync once,
                # instead of one implicitly-synchronizing cudaMemcpy per
                # allocation. The full device sync keeps the old null-stream
                # semantics: the copies must observe writes from all streams.
                torch.cuda.synchronize()
                copy_stream = self._get_copy_stream()
                for ptr, data in offload_items:
                    data.cpu_backup_tensor = self._acquire_backup_buffer(data)
                    libcudart.cudaMemcpyAsync(
                        data.cpu_backup_tensor.data_ptr(),
                        ptr,
                        data.handle[1],
                        copy_stream.cuda_stream,
                    )
                copy_stream.synchronize()
            else:
                for ptr, data in offload_items:
                    data.cpu_backup_tensor = self._acquire_backup_buffer(data)
                    libcudart.cudaMemcpy(
                        data.cpu_backup_tensor.data_ptr(), ptr, data.handle[1]
                    )
            # Unmap only after every copy has drained.
            for _, data in offload_items:
                unmap_and_release(data.handle)

        logger.info(
            "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "
            "directly. Per-tag MiB (total/backup): %s",
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
            {
                tag: (
                    round(per_tag_total[tag] / 1024**2, 1),
                    round(per_tag_backup.get(tag, 0) / 1024**2, 1),
                )
                for tag in per_tag_total
            },
        )

        gc.collect()
        # Flush torch's caching layer so freed blocks are returned to the
        # driver. This can raise if a pluggable mem-pool context is active (the
        # persistent CUDA-graph pool); guard so sleep still succeeds.
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("torch.cuda.empty_cache() skipped during sleep: %s", e)

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
        restore_items: list[tuple[int, AllocationData]] = []
        for ptr, data in self.pointer_to_data.items():
            if tags is None or data.tag in tags:
                create_and_map(data.handle)
                if data.cpu_backup_tensor is not None:
                    restore_items.append((ptr, data))

        if not restore_items:
            return
        if self._async_copy_available():
            copy_stream = self._get_copy_stream()
            for ptr, data in restore_items:
                libcudart.cudaMemcpyAsync(
                    ptr,
                    data.cpu_backup_tensor.data_ptr(),
                    data.handle[1],
                    copy_stream.cuda_stream,
                )
            copy_stream.synchronize()
        else:
            for ptr, data in restore_items:
                libcudart.cudaMemcpy(
                    ptr, data.cpu_backup_tensor.data_ptr(), data.handle[1]
                )
        for _, data in restore_items:
            self._release_backup_buffer(data)

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

    def get_current_usage(self) -> int:
        """
        Get the total number of bytes allocated in the memory pool.
        """
        sum_bytes: int = 0
        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            sum_bytes += handle[1]
        return sum_bytes

    def get_graph_pool_handle(self) -> tuple[int, int]:
        """Return a CUDA graph memory pool id backed by this allocator.

        CUDA graphs captured against this pool are tagged ``graphs`` and thus
        participate in the same sleep/wake offload mechanism as weights and
        the KV cache. The pool is created once and kept alive for the
        lifetime of the process, so every subsequent graph capture allocates
        into it.
        """
        if not hasattr(self, "_custom_graph_pool_id"):
            logger.info(
                "CuMemAllocator: creating CUDA graph pool with tag '%s'",
                CuMemAllocator.graphs_tag,
            )
            # MemPool backed by the cumem pluggable allocator. We do NOT enter
            # a ``use_mem_pool`` context: ``torch.cuda.graph(pool=id)`` calls
            # ``beginAllocateToPool`` for the same pool during capture, and an
            # outer ``use_mem_pool`` on the same id would fail with "already
            # recording to mempool_id".
            #
            # The ``graphs`` tag is applied by the capture caller holding
            # ``current_tag = graphs_tag`` across ``capture_model`` (see
            # ``GPUWorker._pin_sleep_mode_graph_pool``).
            new_alloc = get_pluggable_allocator(
                self.python_malloc_callback, self.python_free_callback
            )
            mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
            self.allocator_and_pools[CuMemAllocator.graphs_tag] = (
                mem_pool,
                new_alloc,
            )
            # Keep a strong ref so the pool is not destroyed.
            self._custom_graph_pool_obj = mem_pool
            self._custom_graph_pool_id = mem_pool.id
        return self._custom_graph_pool_id
