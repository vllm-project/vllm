# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from contextlib import AbstractContextManager
from typing import Protocol, TypeAlias

import torch

from vllm.platforms import current_platform

# py_device, py_size_or_aligned_size, py_ptr, py_handle
# py_handle has type list[int] on ROCm and int otherwise
HandleType: TypeAlias = tuple[int, int, int, list[int] | int]


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None
    is_asleep: bool = False


def should_poison_discarded_pages() -> bool:
    """Debug-only switch for sleep/wake correctness testing (RFC #48310 SW3).

    When ``VLLM_POISON_DISCARDED_PAGES`` is set to ``1``/``true``,
    ``wake_up()`` fills the freshly remapped pages of discarded allocations
    (those without a CPU backup) with 0xFF, so stale reads of discarded
    content become deterministically detectable instead of consuming
    platform-dependent (possibly zeroed, possibly stale) memory.
    Off by default; production behavior is unchanged. ``envs`` values are
    resolved lazily on access, so tests can toggle the flag at runtime.
    """
    from vllm import envs

    return envs.VLLM_POISON_DISCARDED_PAGES


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
