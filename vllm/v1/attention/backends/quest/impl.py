# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuestSparseOffloadImpl — Phase A pure delegation to FlashAttentionImpl.

Phase B will replace the body of `forward` with real sparse logic.
"""
from __future__ import annotations

import torch

from vllm.v1.attention.backend import AttentionImpl, AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl


class QuestSparseOffloadImpl(AttentionImpl):
    """Phase A: forward calls into a held FlashAttentionImpl.

    This class exists so Phase B has a stable container to swap the forward
    body into without touching the registration / selection plumbing.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.kv_cache_dtype = kv_cache_dtype
        self._fa_impl = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
        )

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._fa_impl.forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
