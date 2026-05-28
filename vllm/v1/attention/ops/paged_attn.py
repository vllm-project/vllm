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
        """Return strided K/V views over the unified KV cache.

        Input ``kv_cache`` has shape ``[B, N, H, 2*C]`` with K and V
        interleaved in the trailing content dim. Returns:
          * key_cache:   ``[B, H, C//x, N, x]``
          * value_cache: ``[B, H, N, C]``

        where ``x = 16 // element_size``. V is in ``[B, H, N, C]`` (not
        the legacy ``[B, H, C, N]``) because no permutation of the
        interleaved cache yields N-innermost as a view. Triton consumers
        use explicit per-dim strides, so the layout change is invisible
        as long as callers pass strides by semantic dimension.

        TODO(RFC #42082): the ROCm C++ ``ops.paged_attention_rocm`` kernel
        still requires the legacy V layout and contiguous tensors;
        ``has_native_kv_cache_layout`` routes around it. A follow-up
        should port that HIP kernel to consume the unified layout.
        """
        x = 16 // kv_cache.element_size()
        key_slice = kv_cache[..., :head_size]
        value_slice = kv_cache[..., head_size:]
        key_cache = (
            key_slice.permute(0, 2, 1, 3)
            .unflatten(-1, (head_size // x, x))
            .transpose(2, 3)
        )
        value_cache = value_slice.permute(0, 2, 1, 3)
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
