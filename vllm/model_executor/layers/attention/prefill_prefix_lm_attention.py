# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.encoder_only_attention import (
    create_encoder_only_attention_backend,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec


class PrefillPrefixLMAttention(Attention):
    """Decoder attention that runs non-causally (Prefix LM).

    This reuses the encoder-only backend wrapper, which forces
    ``causal=False`` on *every* metadata build (prefill and decode alike),
    while keeping ``attn_type=DECODER`` so a KV cache is still allocated.

    Effect by phase:
    - Prefill: query tokens attend to each other bidirectionally -- this is
      where the Prefix LM (non-causal) behavior actually takes effect.
    - Single-token decode: ``causal=False`` is a no-op. The one new query
      attends to the whole (frozen) KV cache exactly as a causal decode
      would, and cached tokens cannot attend back to it, so the output is
      identical to a causal decoder.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        cache_config: CacheConfig | None = None,
        attn_type: str | None = None,
        **kwargs,
    ):
        dtype = torch.get_default_dtype()

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
        else:
            kv_cache_dtype = "auto"

        underlying_attn_backend = get_attn_backend(
            head_size,
            dtype,
            kv_cache_dtype,
            attn_type=AttentionType.DECODER,
        )

        attn_backend = create_encoder_only_attention_backend(underlying_attn_backend)

        if attn_type is not None:
            assert attn_type == AttentionType.DECODER, (
                "PrefillPrefixLMAttention only supports AttentionType.DECODER"
            )

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=attn_backend,
            attn_type=AttentionType.DECODER,
            **kwargs,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Tag the KV cache spec as non-causal.

        The layout is identical to a regular decoder full-attention layer, so
        we reuse the base spec and only flip ``non_causal=True``. The engine
        core reads this flag (across the worker/engine process boundary, via
        the pickled spec) to disable scheduling features that assume causal
        attention -- chunked prefill and prefix caching -- which would
        otherwise corrupt the bidirectional prefill of a Prefix LM.
        """
        spec = super().get_kv_cache_spec(vllm_config)
        if isinstance(spec, FullAttentionSpec):
            return replace(spec, non_causal=True)
        return spec
