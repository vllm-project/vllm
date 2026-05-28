# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from copy import copy

import torch

from vllm.config import CacheConfig
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec


@functools.lru_cache
def create_encoder_only_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    prefix = "EncoderOnlyAttention_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class EncoderOnlyAttentionBuilder(underlying_builder):  # type: ignore
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            new_common_attn_metadata = copy(common_attn_metadata)
            new_common_attn_metadata.causal = False
            return super().build(
                common_prefix_len, new_common_attn_metadata, fast_build
            )

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=EncoderOnlyAttentionBuilder,
    )

    return attn_backend


class EncoderOnlyAttention(Attention):
    """
    Encoder attention is a special case that doesn't need a KV Cache.
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

        if attn_type is None:
            attn_type = AttentionType.ENCODER_ONLY

        underlying_attn_backend = get_attn_backend(
            head_size,
            dtype,
            kv_cache_dtype,
            attn_type=attn_type,
        )

        attn_backend = create_encoder_only_attention_backend(underlying_attn_backend)

        if attn_type != AttentionType.ENCODER_ONLY and kwargs.pop(
            "raise_on_invalid_attn_type", True
        ):
            raise ValueError(
                "EncoderOnlyAttention only supports AttentionType.ENCODER_ONLY"
            )

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=attn_backend,
            attn_type=attn_type,
            **kwargs,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Encoder-only attention does not need KV cache; other attn_types
        # (e.g. DECODER, used by recurrent PrefixLM models like HRM-Text)
        # fall through to the base Attention spec.
        if self.attn_type == AttentionType.ENCODER_ONLY:
            return None
        return super().get_kv_cache_spec(vllm_config)
