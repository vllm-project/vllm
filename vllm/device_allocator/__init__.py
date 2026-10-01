# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol, TypeAlias

import torch

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# py_device, py_size_or_aligned_size, py_ptr, py_handle
# py_handle has type list[int] on ROCm and int otherwise
HandleType: TypeAlias = tuple[int, int, int, list[int] | int]

# Selective wake may defer only these; any wake restores every other tag.
DEFERRABLE_TAGS: tuple[str, ...] = ("weights", "kv_cache")


def cumem_cudagraph_pool_enabled(vllm_config: "VllmConfig") -> bool:
    """Whether CUDA graphs are captured into the cuMem pool that sleep mode
    offloads: Model Runner V2 with the cumem sleep backend on CUDA."""
    return (
        vllm_config.use_v2_model_runner
        and vllm_config.model_config is not None
        and vllm_config.model_config.enable_sleep_mode
        and vllm_config.model_config.sleep_mode_backend == "cumem"
        and current_platform.is_cuda()
    )


_plain_cudagraph_capture: ContextVar[bool] = ContextVar(
    "plain_cudagraph_capture", default=False
)


@contextmanager
def plain_cudagraph_capture() -> Iterator[None]:
    """CUDA graphs captured here stay out of cuMem. Memory profiling captures
    under it so that destroying its graphs frees their pool normally."""
    token = _plain_cudagraph_capture.set(True)
    try:
        yield
    finally:
        _plain_cudagraph_capture.reset(token)


@contextmanager
def use_cudagraph_pool(
    pool: tuple[int, int] | None, vllm_config: "VllmConfig"
) -> Iterator[tuple[int, int] | None]:
    """Yield the pool to capture into, routed through cuMem when sleep mode
    offloads graph pools, and point NCCL's graph allocator at it."""
    from vllm.distributed.device_communicators.pynccl_allocator import (
        set_graph_pool_id,
    )

    ctx: AbstractContextManager[tuple[int, int] | None] = nullcontext(pool)
    if (
        pool is not None
        and not _plain_cudagraph_capture.get()
        and cumem_cudagraph_pool_enabled(vllm_config)
    ):
        from vllm.device_allocator.cumem import CuMemAllocator

        ctx = CuMemAllocator.get_instance().use_cudagraph_pool()
    with ctx as graph_pool:
        set_graph_pool_id(graph_pool or current_platform.graph_pool_handle())
        yield graph_pool


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
