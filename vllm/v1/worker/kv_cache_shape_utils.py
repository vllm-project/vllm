# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for restoring padded-page logical KV cache shapes."""

from collections.abc import Sequence
from math import prod

import torch

from vllm.utils.torch_utils import get_dtype_size


def scale_padded_page_size(
    padded_page_size_bytes: int,
    *,
    block_size: int,
    target_block_size: int,
) -> int:
    """Scale padded page size to a different runtime block size."""
    if block_size == target_block_size:
        return padded_page_size_bytes

    scaled_page_size = padded_page_size_bytes * target_block_size
    if scaled_page_size % block_size != 0:
        raise ValueError(
            "Padded page size scaling must be divisible by the original "
            f"block size: padded_page_size_bytes={padded_page_size_bytes}, "
            f"block_size={block_size}, target_block_size={target_block_size}"
        )
    return scaled_page_size // block_size


def get_padded_attention_kv_cache_shape(
    kv_cache_shape: Sequence[int],
    *,
    num_blocks: int,
    padded_page_size_bytes: int,
    dtype: torch.dtype,
) -> tuple[int, ...]:
    """Adjust the logical last dimension to match padded page size bytes."""
    if not kv_cache_shape:
        raise ValueError("kv_cache_shape must not be empty")
    if kv_cache_shape[0] != num_blocks:
        raise ValueError(
            "Expected kv_cache_shape to start with num_blocks: "
            f"kv_cache_shape={tuple(kv_cache_shape)}, num_blocks={num_blocks}"
        )

    dtype_size = get_dtype_size(dtype)
    if padded_page_size_bytes % dtype_size != 0:
        raise ValueError(
            "Padded page size must be divisible by dtype size: "
            f"padded_page_size_bytes={padded_page_size_bytes}, dtype={dtype}"
        )

    page_size_elements = padded_page_size_bytes // dtype_size
    elems_per_last_dim = prod(kv_cache_shape[1:-1])
    if elems_per_last_dim == 0:
        raise ValueError(
            "Invalid kv_cache_shape with zero-sized middle dimensions: "
            f"kv_cache_shape={tuple(kv_cache_shape)}"
        )
    if page_size_elements % elems_per_last_dim != 0:
        raise ValueError(
            "Padded page size does not align with kv cache shape: "
            f"kv_cache_shape={tuple(kv_cache_shape)}, "
            f"padded_page_size_bytes={padded_page_size_bytes}, dtype={dtype}"
        )

    padded_last_dim = page_size_elements // elems_per_last_dim
    if padded_last_dim < kv_cache_shape[-1]:
        raise ValueError(
            "Padded last dimension would shrink the logical kv cache shape: "
            f"kv_cache_shape={tuple(kv_cache_shape)}, "
            f"padded_last_dim={padded_last_dim}"
        )

    return (*kv_cache_shape[:-1], padded_last_dim)
