#! SPDX-License-Identifier: Apache-2.0
#! SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch.nn as nn

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.hybrid_ssm_adapter import HybridSSMAdapter
from vllm.v1.attention.backends.hybrid_attn import HybridAttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec


class HybridAttentionLayer(Attention, AttentionLayerBase):
    """Attention layer that fuses sliding-window KV with an SSM history branch.

    This layer is a thin wrapper around the standard ``Attention`` module that:

    - Forces the use of ``HybridAttentionBackend`` for its attention backend.
    - Owns a ``HybridSSMAdapter`` instance representing the history branch.
    - Reuses ``Attention.get_kv_cache_spec`` so it continues to expose either
      a ``SlidingWindowSpec`` or ``FullAttentionSpec`` for its KV cache.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        *,
        ssm_state_size: int,
        ssm_conv_kernel_size: int,
        ssm_intermediate_size: int,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        # Initialize the history branch adapter first so it can participate in
        # the v1 KV cache spec discovery.
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        self.ssm_adapter = HybridSSMAdapter(
            hidden_size=num_heads * head_size,
            ssm_state_size=ssm_state_size,
            conv_kernel_size=ssm_conv_kernel_size,
            intermediate_size=ssm_intermediate_size,
            model_config=model_config,
            cache_config=cache_config or vllm_config.cache_config,
            prefix=f"{prefix}.ssm",
        )

        # Force the attention backend to be HybridAttentionBackend while reusing
        # all of Attention's internal wiring (KV cache quantization, static
        # forward context registration, etc.).
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            prefix=prefix,
            attn_backend=HybridAttentionBackend,
            **extra_impl_args,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        # Satisfy ``AttentionLayerBase`` by returning the concrete backend
        # class for this layer.
        return HybridAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Delegate KV spec computation to the parent Attention implementation.
        return super().get_kv_cache_spec(vllm_config)


