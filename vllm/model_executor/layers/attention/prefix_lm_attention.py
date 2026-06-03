# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.encoder_only_attention import (
    create_encoder_only_attention_backend,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.selector import get_attn_backend


class PrefixLMAttention(Attention):
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
                "PrefixLMAttention only supports AttentionType.DECODER"
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
