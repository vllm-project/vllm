# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton


@triton.jit
def kv_cache_scatter_kernel(
    kv_cache_ptrs_ptr,
    source_ptr,
    token_indices_ptr,
    num_tokens_in_block,
    total_token_in_kvcache,
    num_layers,
    tokens_per_block,
    block_stride,
    head_stride,
    token_stride,
    content_stride,
    num_heads: tl.constexpr,
    content_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    layer_idx = tl.program_id(0)
    token_pos = tl.program_id(1)

    if layer_idx >= num_layers or token_pos >= num_tokens_in_block:
        return

    token_idx = tl.load(token_indices_ptr + token_pos)
    kv_cache_ptr = tl.cast(tl.load(kv_cache_ptrs_ptr + layer_idx), source_ptr.dtype)

    if token_idx >= total_token_in_kvcache:
        return

    block_idx = token_idx // tokens_per_block
    state_idx = token_idx % tokens_per_block
    hidden_size = num_heads * content_size
    for i in range(0, hidden_size, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        head_idx = offset // content_size
        content_idx = offset % content_size
        source_offset = (
            (layer_idx * num_heads + head_idx) * num_tokens_in_block + token_pos
        ) * content_size + content_idx
        target_offset = (
            block_idx * block_stride
            + head_idx * head_stride
            + state_idx * token_stride
            + content_idx * content_stride
        )
        value = tl.load(source_ptr + source_offset, mask=mask)
        tl.store(kv_cache_ptr + target_offset, value, mask=mask)


@triton.jit
def kv_cache_gather_kernel(
    kv_cache_ptrs_ptr,
    dst_ptr,
    token_indices_ptr,
    num_tokens_in_block,
    total_token_in_kvcache,
    num_layers,
    tokens_per_block,
    block_stride,
    head_stride,
    token_stride,
    content_stride,
    num_heads: tl.constexpr,
    content_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    layer_idx = tl.program_id(0)
    token_pos = tl.program_id(1)

    if layer_idx >= num_layers or token_pos >= num_tokens_in_block:
        return

    token_idx = tl.load(token_indices_ptr + token_pos)
    kv_cache_ptr = tl.cast(tl.load(kv_cache_ptrs_ptr + layer_idx), dst_ptr.dtype)

    if token_idx >= total_token_in_kvcache:
        return

    block_idx = token_idx // tokens_per_block
    state_idx = token_idx % tokens_per_block
    hidden_size = num_heads * content_size
    for i in range(0, hidden_size, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        head_idx = offset // content_size
        content_idx = offset % content_size
        kvcache_offset = (
            block_idx * block_stride
            + head_idx * head_stride
            + state_idx * token_stride
            + content_idx * content_stride
        )
        dst_offset = (
            (layer_idx * num_heads + head_idx) * num_tokens_in_block + token_pos
        ) * content_size + content_idx
        value = tl.load(kv_cache_ptr + kvcache_offset, mask=mask)
        tl.store(dst_ptr + dst_offset, value, mask=mask)


def scatter_kv_caches(
    kv_caches_ptrs: torch.Tensor,
    total_token_in_kvcache: int,
    src_tensor: torch.Tensor,
    token_indices: list[int],
    tokens_per_block: int,
    num_heads: int,
    content_size: int,
    kv_cache_strides: tuple[int, ...],
) -> None:
    """Scatter KV cache data from source tensor to KV cache storage.

    Args:
        kv_caches_ptrs: Tensor of KV cache pointers (one per layer)
        total_token_in_kvcache: Total number of tokens in KV cache
        src_tensor: Source ``[L, H, N, C]`` tensor containing data to scatter
        token_indices: List of token positions to update
        tokens_per_block: Number of stored states in each cache block
        num_heads: Size of the H axis
        content_size: Size of the C axis
        kv_cache_strides: Element strides of each ``[B, H, N, C]`` layer view
    """
    num_layers = len(kv_caches_ptrs)
    num_tokens_in_block = len(token_indices)

    assert src_tensor.shape == (
        num_layers,
        num_heads,
        num_tokens_in_block,
        content_size,
    )
    assert len(kv_cache_strides) == 4

    device = src_tensor.device
    token_indices_tensor = torch.tensor(
        token_indices, dtype=torch.int32, device="cpu"
    ).to(device, non_blocking=True)

    grid = (num_layers, num_tokens_in_block)
    BLOCK_SIZE = 128

    kv_cache_scatter_kernel[grid](
        kv_caches_ptrs,
        src_tensor,
        token_indices_tensor,
        num_tokens_in_block,
        total_token_in_kvcache,
        num_layers,
        tokens_per_block,
        *kv_cache_strides,
        num_heads=num_heads,
        content_size=content_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def gather_kv_caches(
    kv_caches_ptrs: torch.Tensor,
    total_token_in_kvcache: int,
    dst_tensor: torch.Tensor,
    token_indices: list[int],
    tokens_per_block: int,
    num_heads: int,
    content_size: int,
    kv_cache_strides: tuple[int, ...],
) -> None:
    """Gather KV cache data from KV cache storage to destination tensor.

    Args:
        kv_caches_ptrs: Tensor of KV cache pointers (one per layer)
        total_token_in_kvcache: Total number of tokens in KV cache
        dst_tensor: Destination ``[L, H, N, C]`` tensor
        token_indices: List of token positions to gather
        tokens_per_block: Number of stored states in each cache block
        num_heads: Size of the H axis
        content_size: Size of the C axis
        kv_cache_strides: Element strides of each ``[B, H, N, C]`` layer view
    """
    num_layers = kv_caches_ptrs.shape[0]
    num_tokens_in_block = len(token_indices)

    assert dst_tensor.shape == (
        num_layers,
        num_heads,
        num_tokens_in_block,
        content_size,
    )
    assert len(kv_cache_strides) == 4

    device = dst_tensor.device
    token_indices_tensor = torch.tensor(
        token_indices, dtype=torch.int32, device="cpu"
    ).to(device, non_blocking=True)

    grid = (num_layers, num_tokens_in_block)
    BLOCK_SIZE = 128

    kv_cache_gather_kernel[grid](
        kv_caches_ptrs,
        dst_tensor,
        token_indices_tensor,
        num_tokens_in_block,
        total_token_in_kvcache,
        num_layers,
        tokens_per_block,
        *kv_cache_strides,
        num_heads=num_heads,
        content_size=content_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


class CopyBufferAllocator:
    """Memory pool for tensor buffers to avoid frequent allocation/deallocation."""

    def __init__(
        self, device: torch.device, dtype: torch.dtype, shape: list, max_count: int
    ):
        self._shape = shape
        self._max_count = max_count
        self._device = device
        self._free_buffers = [
            torch.empty(shape, dtype=dtype, device=device) for _ in range(max_count)
        ]
        self._inuse_count = 0

    def alloc_buffer(self, count: int) -> list[torch.Tensor] | None:
        """Allocate buffers from the pool."""
        if count == 0:
            return []

        if self._inuse_count + count <= self._max_count:
            self._inuse_count += count
            result = self._free_buffers[-count:]
            del self._free_buffers[-count:]
            return result
        return None

    def free_buffer(self, buffers: list[torch.Tensor]) -> None:
        """Return buffers to the pool."""
        if not buffers:
            return

        if self._inuse_count >= len(buffers):
            self._inuse_count -= len(buffers)
            self._free_buffers.extend(buffers)
        else:
            raise RuntimeError("Attempted to free more buffers than allocated")


logger = init_logger(__name__)
