# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph pool routing for CUDA graph capture under sleep mode."""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import TYPE_CHECKING, cast

from vllm.config import VllmConfig
from vllm.device_allocator import get_mem_allocator_instance
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.device_allocator.cumem import CuMemAllocator

_plain_capture: ContextVar[bool] = ContextVar("plain_cudagraph_capture", default=False)


def _cumem_allocator() -> "CuMemAllocator":
    # A cast, not new MemAllocator methods: only cuMem has a graph pool (XPU's
    # allocator does not), and the policy that gates every caller requires CUDA.
    return cast("CuMemAllocator", get_mem_allocator_instance())


@contextmanager
def plain_cudagraph_capture() -> Iterator[None]:
    """CUDA graphs captured here stay out of cuMem. Memory profiling captures
    under it so that destroying its graphs frees their pool normally."""
    token = _plain_capture.set(True)
    try:
        yield
    finally:
        _plain_capture.reset(token)


@contextmanager
def use_cudagraph_pool(
    pool: tuple[int, int] | None, vllm_config: VllmConfig
) -> Iterator[tuple[int, int] | None]:
    """Yield the pool to capture into, the cuMem graph pool when sleep mode
    offloads graph pools, and point NCCL's graph allocator at it."""
    ctx: AbstractContextManager[tuple[int, int] | None] = nullcontext(pool)
    if (
        pool is not None
        and not _plain_capture.get()
        and vllm_config.use_cumem_cudagraph_pool
    ):
        ctx = _cumem_allocator().use_cudagraph_pool()
    with ctx as graph_pool:
        set_graph_pool_id(graph_pool or current_platform.graph_pool_handle())
        yield graph_pool


def release_cudagraph_pool(vllm_config: VllmConfig) -> None:
    """Free the cuMem graph pool once its graphs are destroyed, so that a
    recapture starts from an empty pool."""
    if vllm_config.use_cumem_cudagraph_pool:
        _cumem_allocator().release_cudagraph_pool()
