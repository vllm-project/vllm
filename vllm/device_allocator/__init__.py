# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Protocol, TypeAlias

import torch

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# py_device, py_size_or_aligned_size, py_ptr, py_handle
# py_handle has type list[int] on ROCm and int otherwise
HandleType: TypeAlias = tuple[int, int, int, list[int] | int]


def use_cudagraph_pool(
    pool: tuple[int, int] | None, vllm_config: "VllmConfig"
) -> AbstractContextManager[tuple[int, int] | None]:
    """Route CUDA graph allocations through cuMem when sleep mode is enabled."""
    if (
        vllm_config.use_v2_model_runner
        and vllm_config.model_config.enable_sleep_mode
        and vllm_config.model_config.sleep_mode_backend == "cumem"
        and current_platform.is_cuda()
    ):
        from vllm.device_allocator.cumem import CuMemAllocator

        if pool is None:
            raise ValueError("CUDA graph pool is required for cuMem capture")
        return CuMemAllocator.get_instance().use_cudagraph_pool(pool)
    return nullcontext(pool)


def release_cudagraph_pool(pool: tuple[int, int]) -> None:
    """Release a profiling pool after all its graphs have been destroyed."""
    if current_platform.is_cuda():
        from vllm.device_allocator.cumem import CuMemAllocator

        if CuMemAllocator.instance is not None:
            CuMemAllocator.instance.release_cudagraph_pool(pool)


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None
    is_asleep: bool = False


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
