# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph pool routing for CUDA graph capture under sleep mode."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, cast

from vllm.config import VllmConfig
from vllm.device_allocator import get_mem_allocator_instance
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.device_allocator.cumem import CuMemAllocator

_outside_cumem: ContextVar[bool] = ContextVar("capture_outside_cumem", default=False)


@contextmanager
def capture_outside_cumem_pool() -> Iterator[None]:
    """CUDA graphs captured here stay out of cuMem, so that memory profiling can
    free its throwaway graphs normally."""
    token = _outside_cumem.set(True)
    try:
        yield
    finally:
        _outside_cumem.reset(token)


@contextmanager
def capture_pool(
    pool: tuple[int, int] | None, vllm_config: VllmConfig
) -> Iterator[tuple[int, int] | None]:
    """Yield the pool to capture into, the cuMem graph pool when sleep offloads
    graph memory, and point NCCL's graph allocator at it."""
    if vllm_config.use_cumem_cudagraph_pool and not _outside_cumem.get():
        allocator = cast("CuMemAllocator", get_mem_allocator_instance())
        with allocator.cudagraph_pool() as cumem_pool:
            set_graph_pool_id(cumem_pool)
            yield cumem_pool
    else:
        set_graph_pool_id(pool or current_platform.graph_pool_handle())
        yield pool
