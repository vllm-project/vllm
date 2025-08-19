# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from copy import copy
from typing import Optional

import torch
from transformers import CacheConfig

from vllm import envs
from vllm.attention.backends.abstract import AttentionBackend, AttentionType
from vllm.attention.layer import Attention
from vllm.attention.selector import get_attn_backend
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata, subclass_attention_backend,
    subclass_attention_metadata_builder)
from vllm.v1.core.sched.output import SchedulerOutput


@functools.lru_cache
def create_encoder_only_attention_backend(
    underlying_attn_backend: AttentionBackend, ) -> type[AttentionBackend]:
    prefix = "EncoderOnlyAttention_"

    def patch_common_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        scheduler_output: SchedulerOutput,
    ) -> CommonAttentionMetadata:
        new_metadata = copy(common_attn_metadata)
        new_metadata.causal = False
        return new_metadata

    builder_cls = subclass_attention_metadata_builder(
        name_prefix=prefix,
        builder_cls=underlying_attn_backend.get_builder_cls(),
        patch_common_attn_metadata=patch_common_attn_metadata)
    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=builder_cls)

    return attn_backend


class EncoderOnlyAttention(Attention):
    """
    Encoder attention is a special case that doesn't need a KV Cache.
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

            attn_backend = create_encoder_only_attention_backend(
                underlying_attn_backend)
        else:
            # in v0 encoder only attention is handled inside the backends
            attn_backend = None

        if attn_type is not None:
            assert attn_type == AttentionType.ENCODER_ONLY, \
                "EncoderOnlyAttention only supports AttentionType.ENCODER_ONLY"

        super().__init__(num_heads=num_heads,
                         head_size=head_size,
                         scale=scale,
                         cache_config=cache_config,
                         attn_backend=attn_backend,
                         **kwargs)
