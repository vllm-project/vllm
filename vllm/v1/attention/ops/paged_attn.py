# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.platforms import current_platform

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops as ops  # type: ignore[no-redef]


class PagedAttention:
    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        need_kv_flash_permute: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        if need_kv_flash_permute:
            # Data written by AITER fused kernel in flash layout:
            #   [num_blocks, block_size, num_heads, head_size]
            # Use permute (zero-cost stride change, no data copy) to
            # present the same logical shape the Triton attention kernel
            # expects while indexing into flash-layout memory.
            block_size = kv_cache.shape[2]

            key_cache = kv_cache[0].view(num_blocks, block_size, num_kv_heads, head_size // x, x)
            key_cache = key_cache.permute(0, 2, 3, 1, 4)

            value_cache = kv_cache[1].view(num_blocks, block_size, num_kv_heads, head_size)
            value_cache = value_cache.permute(0, 2, 3, 1)
        else:
            # Old logic, no changes needed
            key_cache = kv_cache[0]
            key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x, -1, x)
            value_cache = kv_cache[1]
            value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)

        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
