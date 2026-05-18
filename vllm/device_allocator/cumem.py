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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

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

# py_device, py_alignedSize, py_d_mem, py_p_memHandle
HandleType = tuple[int, int, int, int]

# The set of tags that vLLM's sleep/wake plumbing knows about. Selective
# sleep callers must use only these tags; anything else is rejected up
# front (see ``EngineCore.sleep``) so the engine never enters a half-
# sleep state with a bogus tag recorded.
KNOWN_SLEEP_TAGS: tuple[str, ...] = ("weights", "kv_cache")


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None
    # Whether the underlying CUDA virtual address is currently mapped to
    # physical memory. This is True at allocation time and stays True
    # across sleep() for any allocation whose tag is in `keep_tags`.
    # It is set to False when the allocation is unmapped during sleep,
    # and back to True when wake_up() remaps it.
    is_mapped: bool = True


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
        return CuMemAllocator.instance

    def __init__(self):
        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CuMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}
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
        offload_tags: tuple[str, ...] | str | None = None,
        keep_tags: tuple[str, ...] | str | None = None,
    ) -> None:
        """
        Put the allocator in sleep mode.

        Each allocation in the memory pool falls into one of three buckets,
        determined by its tag:

        1. **Offload** — allocations whose tag is in ``offload_tags`` are
           backed up to pinned CPU memory and their GPU mapping is released.
        2. **Keep**  — allocations whose tag is in ``keep_tags`` are left
           fully mapped on GPU and untouched. This is what enables
           hybrid co-location with partial rollout: e.g. offload weights to
           CPU while keeping the KV cache live on the GPU so that an
           in-flight generation can be resumed after a training step.
        3. **Discard** — allocations whose tag is in neither set have their
           GPU mapping released without a CPU backup. ``wake_up`` will
           re-map fresh empty memory at the same address.

        :param offload_tags: Tags whose allocations should be copied to CPU
            and unmapped from GPU. If ``None``, defaults to
            ``(CuMemAllocator.default_tag,)`` for backward compatibility.
            A bare string is normalized to a 1-tuple.
        :param keep_tags: Tags whose allocations should be preserved on GPU
            in their entirety. Allocations with these tags are skipped
            completely — no CPU copy, no unmap. Defaults to an empty tuple.
            A bare string is normalized to a 1-tuple. ``keep_tags`` and
            ``offload_tags`` must be disjoint.
        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        if keep_tags is None:
            keep_tags = ()
        elif isinstance(keep_tags, str):
            keep_tags = (keep_tags,)

        assert isinstance(offload_tags, tuple)
        assert isinstance(keep_tags, tuple)

        overlap = set(offload_tags) & set(keep_tags)
        assert not overlap, (
            f"offload_tags and keep_tags must be disjoint, got overlap: {overlap}"
        )

        total_bytes = 0
        backup_bytes = 0
        kept_bytes = 0

        for ptr, data in self.pointer_to_data.items():
            if not data.is_mapped:
                # Already offloaded by a prior selective sleep. Re-running
                # cudaMemcpy on the source pointer or unmap_and_release on
                # the handle would corrupt the allocator state, so we skip
                # entirely. A later wake_up will remap and restore from
                # any existing cpu_backup_tensor.
                continue
            handle = data.handle
            total_bytes += handle[1]
            if data.tag in keep_tags:
                # Allocation stays fully mapped on GPU. Do not back it up
                # to CPU and do not release the device mapping.
                kept_bytes += handle[1]
                continue
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
            data.is_mapped = False

        freed_bytes = total_bytes - kept_bytes
        logger.info(
            "CuMemAllocator: sleep freed %.2f GiB GPU memory (%.2f GiB "
            "backed up to CPU, %.2f GiB discarded), kept %.2f GiB on GPU.",
            freed_bytes / 1024**3,
            backup_bytes / 1024**3,
            (freed_bytes - backup_bytes) / 1024**3,
            kept_bytes / 1024**3,
        )

        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.

        Allocations that were unmapped during sleep are remapped on GPU.
        Those that had a CPU backup are restored from it; those that were
        discarded are left as empty memory at the same address.

        Allocations that were preserved on GPU during sleep (via
        ``keep_tags`` on :py:meth:`sleep`) are skipped — they are still
        live and must not be remapped.

        :param tags: The tags of the memory allocation that will be loaded
            back to GPU memory. If None, all memory allocation will be
            loaded back to GPU memory.
        """
        for ptr, data in self.pointer_to_data.items():
            if tags is not None and data.tag not in tags:
                continue
            if data.is_mapped:
                # Allocation was preserved on GPU during sleep (via
                # `keep_tags`). Nothing to remap or restore.
                continue
            handle = data.handle
            create_and_map(handle)
            data.is_mapped = True
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

        :param tag: The tag of the memory allocation. If None, the default tag
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
