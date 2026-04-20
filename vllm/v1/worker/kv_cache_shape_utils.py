# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from math import prod

import torch

from vllm.utils.torch_utils import get_dtype_size


def maybe_adjust_kv_cache_shape_for_padded_page_size(
    kv_cache_shape: tuple[int, ...],
    *,
    num_blocks: int,
    padded_page_size_bytes: int | None,
    dtype: torch.dtype,
) -> tuple[int, ...]:
    """Adjust the KV cache shape's last dim for padded page size.

    Returns ``kv_cache_shape`` unchanged when ``padded_page_size_bytes`` is
    ``None``.
    """
    if padded_page_size_bytes is None:
        return kv_cache_shape

    dtype_size = get_dtype_size(dtype)
    if padded_page_size_bytes % dtype_size != 0:
        raise ValueError(
            "padded_page_size_bytes must be divisible by dtype size: "
            f"padded_page_size_bytes={padded_page_size_bytes}, dtype={dtype}"
        )

    total_elements = prod(kv_cache_shape)
    if total_elements % num_blocks != 0:
        raise ValueError(
            "KV cache shape total elements must be divisible by num_blocks: "
            f"shape={kv_cache_shape}, num_blocks={num_blocks}"
        )
    elements_per_block = total_elements // num_blocks

    current_last_dim = kv_cache_shape[-1]
    if elements_per_block % current_last_dim != 0:
        raise ValueError(
            "elements_per_block must be divisible by current last dimension: "
            f"elements_per_block={elements_per_block}, "
            f"current_last_dim={current_last_dim}"
        )
    elements_without_last_dim = elements_per_block // current_last_dim

    padded_elements_per_block = padded_page_size_bytes // dtype_size
    if padded_elements_per_block % elements_without_last_dim != 0:
        raise ValueError(
            "padded elements per block must map to an integer last dimension: "
            f"padded_elements_per_block={padded_elements_per_block}, "
            f"elements_without_last_dim={elements_without_last_dim}"
        )

    new_last_dim = padded_elements_per_block // elements_without_last_dim
    return (*kv_cache_shape[:-1], new_last_dim)
