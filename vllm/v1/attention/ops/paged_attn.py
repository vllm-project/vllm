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
        """Split a unified KV cache into legacy paged-attention key/value views.

        Input ``kv_cache`` has shape ``[B, N, H, 2*C]`` (after the caller's
        ``transpose(1, 2)`` on the logical ``[B, H, N, 2*C]`` tensor).

        Returns contiguous tensors in the legacy paged-attention format:
          * key_cache:   ``[B, H, C//x, N, x]``
          * value_cache: ``[B, H, C, N]``

        where ``x = 16 // element_size`` and ``C = head_size``.

        Because K and V are interleaved in the content dimension, the
        resulting tensors are always contiguous *copies* — callers that
        need to *write* to the cache should use stride-aware kernels
        (e.g. ``reshape_and_cache_flash``) on the raw split views instead.
        """
        x = 16 // kv_cache.element_size()

        # Slice K and V from the interleaved content dimension.
        # Result shape: [B, N, H, C]  (non-contiguous view)
        key_slice = kv_cache[..., :head_size]
        value_slice = kv_cache[..., head_size:]

        # key: [B, N, H, C] → permute → [B, H, N, C]
        #      → unflatten C → [B, H, N, C//x, x]
        #      → transpose N↔C//x → [B, H, C//x, N, x]
        #      → contiguous copy for the paged-attention kernel.
        key_cache = (
            key_slice.permute(0, 2, 1, 3)
            .unflatten(-1, (head_size // x, x))
            .transpose(2, 3)
            .contiguous()
        )

        # value: [B, N, H, C] → permute → [B, H, C, N]
        #        → contiguous copy for the paged-attention kernel.
        value_cache = value_slice.permute(0, 2, 3, 1).contiguous()

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
