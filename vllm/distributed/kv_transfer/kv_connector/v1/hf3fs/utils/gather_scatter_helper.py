from typing import List, Optional

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger


@triton.jit
def kv_cache_scatter_kernel(
    kv_cache_ptrs_ptr,
    source_ptr,
    token_indices_ptr,
    num_tokens_in_block,
    hidden_size,
    total_token_in_kvcache,
    num_layers,
    is_mla,
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

    if is_mla:
        # MLA format: source [num_layers, num_tokens_in_block, hidden_size]
        # MLA format: target [total_token_in_kvcache, hidden_size] (per layer)
        source_offset = (layer_idx * num_tokens_in_block + token_pos) * hidden_size
        target_offset = token_idx * hidden_size

        for i in range(0, hidden_size, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < hidden_size
            val = tl.load(source_ptr + source_offset + offset, mask=mask)
            tl.store(kv_cache_ptr + target_offset + offset, val, mask=mask)
    else:
        # MHA format: source [num_layers, 2, num_tokens_in_block, hidden_size]
        # MHA format: target [2, total_token_in_kvcache, hidden_size]
        source_offset_k = (
            layer_idx * num_tokens_in_block * 2 + token_pos
        ) * hidden_size
        source_offset_v = (
            layer_idx * num_tokens_in_block * 2 + num_tokens_in_block + token_pos
        ) * hidden_size

        target_offset_k = token_idx * hidden_size
        target_offset_v = (total_token_in_kvcache + token_idx) * hidden_size

        for i in range(0, hidden_size, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < hidden_size

            val_k = tl.load(source_ptr + source_offset_k + offset, mask=mask)
            val_v = tl.load(source_ptr + source_offset_v + offset, mask=mask)

            tl.store(kv_cache_ptr + target_offset_k + offset, val_k, mask=mask)
            tl.store(kv_cache_ptr + target_offset_v + offset, val_v, mask=mask)


@triton.jit
def kv_cache_gather_kernel(
    kv_cache_ptrs_ptr,
    dst_ptr,
    token_indices_ptr,
    num_tokens_in_block,
    hidden_size,
    total_token_in_kvcache,
    num_layers,
    is_mla,
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

    if is_mla:
        # MLA format: source [total_token_in_kvcache, hidden_size] (per layer)
        # MLA format: dst [num_layers, num_tokens_in_block, hidden_size]
        kvcache_offset = token_idx * hidden_size
        dst_offset = (layer_idx * num_tokens_in_block + token_pos) * hidden_size

        for i in range(0, hidden_size, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < hidden_size
            val = tl.load(kv_cache_ptr + kvcache_offset + offset, mask=mask)
            tl.store(dst_ptr + dst_offset + offset, val, mask=mask)
    else:
        # MHA format: source [2, total_token_in_kvcache, hidden_size]
        # MHA format: dst [num_layers, 2, num_tokens_in_block, hidden_size]
        dst_offset_k = (layer_idx * num_tokens_in_block * 2 + token_pos) * hidden_size
        dst_offset_v = (
            layer_idx * num_tokens_in_block * 2 + num_tokens_in_block + token_pos
        ) * hidden_size

        kvcache_offset_k = token_idx * hidden_size
        kvcache_offset_v = (total_token_in_kvcache + token_idx) * hidden_size

        for i in range(0, hidden_size, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < hidden_size

            val_k = tl.load(kv_cache_ptr + kvcache_offset_k + offset, mask=mask)
            val_v = tl.load(kv_cache_ptr + kvcache_offset_v + offset, mask=mask)

            tl.store(dst_ptr + dst_offset_k + offset, val_k, mask=mask)
            tl.store(dst_ptr + dst_offset_v + offset, val_v, mask=mask)


def scatter_kv_caches(
    kv_caches_ptrs: torch.Tensor,
    total_token_in_kvcache: int,
    src_tensor: torch.Tensor,
    token_indices: List[int],
    is_mla: bool = False,
) -> None:
    """Scatter KV cache data from source tensor to KV cache storage.

    Args:
        kv_caches_ptrs: Tensor of KV cache pointers (one per layer)
        total_token_in_kvcache: Total number of tokens in KV cache
        src_tensor: Source tensor containing data to scatter
            - MHA format: [num_layers, 2, num_tokens_in_block, hidden_size]
            - MLA format: [num_layers, num_tokens_in_block, hidden_size]
        token_indices: List of token positions to update
        is_mla: Whether using MLA model format
    """
    num_layers = len(kv_caches_ptrs)
    num_tokens_in_block = len(token_indices)

    if is_mla:
        # MLA: src_tensor is [num_layers, num_tokens_in_block, hidden_size]
        assert (
            len(src_tensor.shape) == 3
        ), f"MLA src_tensor should be 3D, got {src_tensor.shape}"
        hidden_size = src_tensor.shape[2]
    else:
        # MHA: src_tensor is [num_layers, 2, num_tokens_in_block, hidden_size]
        assert (
            len(src_tensor.shape) == 4
        ), f"MHA src_tensor should be 4D, got {src_tensor.shape}"
        hidden_size = src_tensor.shape[3]

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
        hidden_size,
        total_token_in_kvcache,
        num_layers,
        is_mla,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def gather_kv_caches(
    kv_caches_ptrs: torch.Tensor,
    total_token_in_kvcache: int,
    dst_tensor: torch.Tensor,
    token_indices: List[int],
    is_mla: bool = False,
) -> None:
    """Gather KV cache data from KV cache storage to destination tensor.

    Args:
        kv_caches_ptrs: Tensor of KV cache pointers (one per layer)
        total_token_in_kvcache: Total number of tokens in KV cache
        dst_tensor: Destination tensor to store gathered data
            - MHA format: [num_layers, 2, num_tokens_in_block, hidden_size]
            - MLA format: [num_layers, num_tokens_in_block, hidden_size]
        token_indices: List of token positions to gather
        is_mla: Whether using MLA model format
    """
    num_layers = kv_caches_ptrs.shape[0]
    num_tokens_in_block = len(token_indices)

    if is_mla:
        # MLA: dst_tensor is [num_layers, num_tokens_in_block, hidden_size]
        assert (
            len(dst_tensor.shape) == 3
        ), f"MLA dst_tensor should be 3D, got {dst_tensor.shape}"
        assert (
            dst_tensor.shape[0] == num_layers
        ), f"Layer count mismatch: {dst_tensor.shape[0]} vs {num_layers}"
        assert (
            dst_tensor.shape[1] == num_tokens_in_block
        ), f"Token count mismatch: {dst_tensor.shape[1]} vs {num_tokens_in_block}"
        hidden_size = dst_tensor.shape[2]
    else:
        # MHA: dst_tensor is [num_layers, 2, num_tokens_in_block, hidden_size]
        assert (
            len(dst_tensor.shape) == 4
        ), f"MHA dst_tensor should be 4D, got {dst_tensor.shape}"
        assert (
            dst_tensor.shape[0] == num_layers
        ), f"Layer count mismatch: {dst_tensor.shape[0]} vs {num_layers}"
        assert (
            dst_tensor.shape[1] == 2
        ), f"MHA should have 2 (K,V) components, got {dst_tensor.shape[1]}"
        assert (
            dst_tensor.shape[2] == num_tokens_in_block
        ), f"Token count mismatch: {dst_tensor.shape[2]} vs {num_tokens_in_block}"
        hidden_size = dst_tensor.shape[3]

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
        hidden_size,
        total_token_in_kvcache,
        num_layers,
        is_mla,
        BLOCK_SIZE=BLOCK_SIZE,
    )


class CopyBufferAllocator:
    """Memory pool for tensor buffers to avoid frequent allocation/deallocation."""

    def __init__(
        self, device: torch.device, dtype: torch.dtype, shape: tuple, max_count: int
    ):
        self._shape = shape
        self._max_count = max_count
        self._device = device
        self._free_buffers = [
            torch.empty(shape, dtype=dtype, device=device) for _ in range(max_count)
        ]
        self._inuse_count = 0

    def alloc_buffer(self, count: int) -> Optional[List[torch.Tensor]]:
        """Allocate buffers from the pool."""
        if count == 0:
            return []

        if self._inuse_count + count <= self._max_count:
            self._inuse_count += count
            result = self._free_buffers[-count:]
            del self._free_buffers[-count:]
            return result
        return None

    def free_buffer(self, buffers: List[torch.Tensor]) -> None:
        """Return buffers to the pool."""
        if not buffers:
            return

        if self._inuse_count >= len(buffers):
            self._inuse_count -= len(buffers)
            self._free_buffers.extend(buffers)
        else:
            raise RuntimeError("Attempted to free more buffers than allocated")


logger = init_logger(__name__)
