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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [B, H, N, 2*C] -> key [B, H, C//x, N, x], value [B, H, C, N]
        x = 16 // kv_cache.element_size()
        key_slice = kv_cache[..., :head_size]
        value_slice = kv_cache[..., head_size:]
        key_cache = (
            key_slice.permute(0, 1, 3, 2)
            .unflatten(2, (head_size // x, x))
            .transpose(3, 4)
        )
        value_cache = value_slice.permute(0, 1, 3, 2)
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
