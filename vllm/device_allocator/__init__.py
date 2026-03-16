# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from contextlib import AbstractContextManager
from typing import Protocol

import torch

from vllm.platforms import current_platform

# py_device, py_size_or_aligned_size, py_ptr, py_handle
HandleType = tuple[int, int, int, int]


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None


class MemAllocator(Protocol):
    def use_memory_pool(self, tag: str | None = None) -> AbstractContextManager: ...

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None: ...

    def wake_up(self, tags: list[str] | None = None) -> None: ...

    def get_current_usage(self) -> int: ...


def get_mem_allocator() -> MemAllocator:
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
