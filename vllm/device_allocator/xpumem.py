# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from vllm.device_allocator import AllocationData, HandleType
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

MEMCPY_HOST_TO_DEVICE = 0
MEMCPY_DEVICE_TO_HOST = 1
MEMCPY_DEVICE_TO_DEVICE = 2

xpumem_available = False
xpumem_allocator: Any = None

try:
    from vllm_xpu_kernels import xpumem_allocator as _xpumem_allocator

    xpumem_allocator = _xpumem_allocator
    xpumem_available = True
except ImportError:
    xpumem_allocator = None


def _xpu_memory_module() -> Any:
    mem_mod = getattr(torch.xpu, "memory", None)
    if mem_mod is None:
        raise RuntimeError("torch.xpu.memory is not available")
    return mem_mod


def _supports_xpu_mem_pool(mem_mod: Any) -> bool:
    return hasattr(mem_mod, "MemPool") and hasattr(mem_mod, "use_mem_pool")


def _xpu_memcpy_sync(
    dst_ptr: int,
    src_ptr: int,
    n_bytes: int,
    kind: int,
    device: int,
) -> None:
    def _to_i64_ptr(ptr: int) -> int:
        # torch custom-op `int` arguments are signed int64.
        # data_ptr() may return a uint64 value above 2^63-1, so normalize it.
        return ptr if ptr < (1 << 63) else ptr - (1 << 64)

    torch.ops._C.xpu_memcpy_sync(
        _to_i64_ptr(dst_ptr),
        _to_i64_ptr(src_ptr),
        n_bytes,
        kind,
        device,
    )


def get_pluggable_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
) -> Any:
    if not xpumem_available or xpumem_allocator is None:
        raise RuntimeError("xpumem allocator extension is not available")

    xpumem_allocator.init_module(python_malloc_fn, python_free_func)
    mem_mod = _xpu_memory_module()
    alloc_cls = getattr(mem_mod, "XPUPluggableAllocator", None)
    if alloc_cls is None:
        raise RuntimeError("torch.xpu.memory.XPUPluggableAllocator is not available")

    lib_name = xpumem_allocator.__file__
    return alloc_cls(lib_name, "my_malloc", "my_free")


def create_and_allocate(allocation_handle: HandleType) -> None:
    if not xpumem_available or xpumem_allocator is None:
        raise RuntimeError("xpumem allocator extension is not available")
    xpumem_allocator.python_create_and_allocate(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    if not xpumem_available or xpumem_allocator is None:
        raise RuntimeError("xpumem allocator extension is not available")
    xpumem_allocator.python_unmap_and_release(*allocation_handle)


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
) -> Iterator[tuple[Any, Any]]:
    mem_mod = _xpu_memory_module()
    if not _supports_xpu_mem_pool(mem_mod):
        raise RuntimeError(
            "torch.xpu.memory MemPool APIs are not available "
            "(need MemPool and use_mem_pool)."
        )
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = mem_mod.MemPool(new_alloc._allocator)
    with mem_mod.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


class XpuMemAllocator:
    """A singleton pluggable allocator helper for XPU.

    Note:
    Sleep will offload selected payloads to CPU or discard and unmap XPU
    physical memory. Wake-up remaps physical memory back to the same
    reserved virtual address and restores payload.
    """

    instance: "XpuMemAllocator | None" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "XpuMemAllocator":
        assert xpumem_available, "xpumem allocator is not available"
        if XpuMemAllocator.instance is None:
            XpuMemAllocator.instance = XpuMemAllocator()
            # Ensure MemPool/allocator wrappers are released before interpreter
            # finalization tears down XPU runtime internals.
            atexit.register(XpuMemAllocator._shutdown_singleton)
        return XpuMemAllocator.instance

    @staticmethod
    def _shutdown_singleton() -> None:
        instance = XpuMemAllocator.instance
        if instance is None:
            return
        try:
            instance.release_pools()
        except Exception:
            logger.exception("XpuMemAllocator singleton shutdown failed")

    def __init__(self):
        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = XpuMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        ptr = allocation_handle[2]
        self.pointer_to_data[ptr] = AllocationData(allocation_handle, self.current_tag)
        logger.debug(
            "Allocated %s bytes for %s at %s",
            allocation_handle[1],
            self.current_tag,
            ptr,
        )

    def _python_free_callback(self, ptr: int) -> HandleType:
        data = self.pointer_to_data.pop(ptr)
        data.cpu_backup_tensor = None
        logger.debug("Freed %s bytes for %s at %s", data.handle[1], data.tag, ptr)
        return data.handle

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        if offload_tags is None:
            offload_tags = (XpuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        total_bytes = 0
        backup_bytes = 0

        for ptr, data in self.pointer_to_data.items():
            size_in_bytes = data.handle[1]
            total_bytes += size_in_bytes
            if data.tag not in offload_tags:
                unmap_and_release(data.handle)
                continue

            backup_bytes += size_in_bytes
            device, _, _, _ = data.handle
            cpu_backup = torch.empty(
                size_in_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=is_pin_memory_available(),
            )
            _xpu_memcpy_sync(
                cpu_backup.data_ptr(),
                ptr,
                size_in_bytes,
                MEMCPY_DEVICE_TO_HOST,
                device,
            )
            data.cpu_backup_tensor = cpu_backup

            unmap_and_release(data.handle)

        logger.info(
            "XpuMemAllocator: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "
            "directly.",
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
        )

        gc.collect()
        xpu_empty_cache = getattr(torch.xpu, "empty_cache", None)
        if callable(xpu_empty_cache):
            xpu_empty_cache()

    def wake_up(self, tags: list[str] | None = None) -> None:
        for ptr, data in self.pointer_to_data.items():
            if tags is not None and data.tag not in tags:
                continue
            create_and_allocate(data.handle)

            cpu_backup_tensor = data.cpu_backup_tensor
            if cpu_backup_tensor is None:
                continue

            device, size_in_bytes, _, _ = data.handle
            _xpu_memcpy_sync(
                ptr,
                cpu_backup_tensor.data_ptr(),
                size_in_bytes,
                MEMCPY_HOST_TO_DEVICE,
                device,
            )
            data.cpu_backup_tensor = None

    def release_pools(self) -> None:
        """Drop Python references to MemPool/pluggable allocators eagerly.

        This prevents pool destruction from being deferred to interpreter
        finalization, which can happen after parts of XPU runtime are already
        torn down.
        """
        if not self.allocator_and_pools:
            return

        # Note: keep allocators alive while MemPool objects are destroyed.
        # MemPool teardown may invoke allocator virtual methods (e.g. raw_delete)
        # when releasing cached blocks. If allocator wrappers are dropped first,
        # C++ can hit "pure virtual method called" during shutdown.
        pool_entries = list(self.allocator_and_pools.values())
        self.allocator_and_pools.clear()

        mem_pools = [entry[0] for entry in pool_entries]
        allocators = [entry[1] for entry in pool_entries]
        pool_entries.clear()

        xpu_sync = getattr(torch.xpu, "synchronize", None)
        if callable(xpu_sync):
            try:
                xpu_sync()
            except Exception:
                logger.debug("torch.xpu.synchronize() failed during release_pools")

        # Phase 1: drop MemPool refs while allocators are still strongly held.
        mem_pools.clear()
        gc.collect()

        # Phase 2: now it is safe to release allocator wrappers.
        allocators.clear()

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        if tag is None:
            tag = XpuMemAllocator.default_tag

        old_tag = self.current_tag
        self.current_tag = tag
        try:
            with use_memory_pool_with_allocator(
                self.python_malloc_callback,
                self.python_free_callback,
            ) as data:
                self.allocator_and_pools[tag] = data
                yield
        finally:
            self.current_tag = old_tag

    def get_current_usage(self) -> int:
        total = 0
        for data in self.pointer_to_data.values():
            total += data.handle[1]
        return total
