# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Callable, Collection, Iterable
from contextlib import AbstractContextManager
from typing import Protocol, TypeAlias

import torch

from vllm.platforms import current_platform

# py_device, py_size_or_aligned_size, py_ptr, py_handle
# py_handle has type list[int] on ROCm and int otherwise
HandleType: TypeAlias = tuple[int, int, int, list[int] | int]

# Selective wake may defer only these; any wake restores every other tag.
DEFERRABLE_TAGS: tuple[str, ...] = ("weights", "kv_cache")


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None
    is_asleep: bool = False


# Tag -> (before_unmap, after_map) of every subscriber, see register_tag_hooks.
_tag_hooks: dict[str, list[tuple[Callable[[], None], Callable[[], None]]]] = {}


def register_tag_hooks(
    tag: str, before_unmap: Callable[[], None], after_map: Callable[[], None]
) -> Callable[[], None]:
    """Have the sleep-mode allocator call `before_unmap` right before it unmaps
    allocations of `tag` (sleep, discard), and the idempotent `after_map` after
    every wake-up that leaves all of them mapped. For state on that memory that
    must live exactly as long as its mapping, such as transport registrations
    of the KV cache. Returns the function that unregisters them."""
    hooks = (before_unmap, after_map)
    _tag_hooks.setdefault(tag, []).append(hooks)
    return lambda: _tag_hooks[tag].remove(hooks)


def run_before_unmap_hooks(
    allocations: Iterable[AllocationData], tags: Collection[str] | None = None
) -> None:
    """Run the `before_unmap` hooks of each tag in `tags` (every tag if None)
    that still has mapped allocations."""
    mapped = {
        d.tag
        for d in allocations
        if not d.is_asleep and (tags is None or d.tag in tags)
    }
    for tag in mapped & _tag_hooks.keys():
        for before_unmap, _ in _tag_hooks[tag]:
            before_unmap()


def run_after_map_hooks(allocations: Iterable[AllocationData]) -> None:
    """Run the `after_map` hooks of every fully mapped tag. Being idempotent,
    they also retry one that failed on an earlier wake-up."""
    asleep = {d.tag for d in allocations if d.is_asleep}
    for tag in _tag_hooks.keys() - asleep:
        for _, after_map in _tag_hooks[tag]:
            after_map()


class MemAllocator(Protocol):
    def use_memory_pool(self, tag: str | None = None) -> AbstractContextManager: ...

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None: ...

    def discard(self, tags: tuple[str, ...] | str) -> None: ...

    def wake_up(self, tags: list[str] | None = None) -> None: ...

    def get_current_usage(self) -> int: ...


def get_mem_allocator_instance() -> MemAllocator:
    if current_platform.is_cuda_alike():
        from vllm.device_allocator.cumem import CuMemAllocator

        return CuMemAllocator.get_instance()

    if current_platform.is_xpu():
        from vllm.device_allocator.xpumem import XpuMemAllocator

        return XpuMemAllocator.get_instance()

    raise RuntimeError(
        "Sleep mode allocator is not available on platform "
        f"{type(current_platform).__name__} "
        f"(device_type={current_platform.device_type})."
    )
