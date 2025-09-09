# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from copy import copy
from typing import Optional

import torch
from transformers import CacheConfig

from vllm import envs
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata, AttentionType)
from vllm.attention.layer import Attention
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              subclass_attention_backend)


def _get_max_encoder_len(vllm_config: VllmConfig) -> int:
    return MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(
        vllm_config.model_config)


@functools.lru_cache
def create_cross_attention_backend(
    underlying_attn_backend: AttentionBackend, ) -> type[AttentionBackend]:
    prefix = "CrossAttention_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class CrossAttentionBuilder(underlying_builder):  # type: ignore

        def build(self,
                  common_prefix_len: int,
                  common_attn_metadata: CommonAttentionMetadata,
                  fast_build: bool = False) -> AttentionMetadata:
            new_metadata = copy(common_attn_metadata)
            new_metadata.causal = False
            max_encoder_len = _get_max_encoder_len(self.vllm_config)
            new_metadata.max_seq_len = max_encoder_len

            new_metadata.seq_lens = torch.full(
                (new_metadata.num_reqs, ),
                max_encoder_len,
                dtype=torch.int32,
                device=self.device,
            )
            new_metadata.seq_lens_cpu = torch.full(
                (new_metadata.num_reqs, ),
                max_encoder_len,
                dtype=torch.int32,
                device="cpu",
            )
            return super().build(common_prefix_len, new_metadata, fast_build)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=CrossAttentionBuilder)

    return attn_backend


class CrossAttention(Attention):
    """
    Cross-attention for encoder-decoder models.
    Handles attention between decoder queries and encoder keys/values.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 cache_config: Optional[CacheConfig] = None,
                 attn_type: Optional[str] = None,
                 **kwargs):
        dtype = torch.get_default_dtype()

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        if envs.VLLM_USE_V1:
            underlying_attn_backend = get_attn_backend(head_size, dtype,
                                                       kv_cache_dtype,
                                                       block_size)

            attn_backend = create_cross_attention_backend(
                underlying_attn_backend)
        else:
            # in v0 cross attention is handled inside the backends
            attn_backend = None

        if attn_type is not None:
            assert attn_type == AttentionType.ENCODER_DECODER, (
                "CrossAttention only supports AttentionType.ENCODER_DECODER")

        super().__init__(num_heads=num_heads,
                         head_size=head_size,
                         scale=scale,
                         cache_config=cache_config,
                         attn_backend=attn_backend,
                         attn_type=AttentionType.ENCODER_DECODER,
                         **kwargs)
