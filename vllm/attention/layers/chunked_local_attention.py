# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import List, Optional

import torch

from vllm import envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig, QuantizationConfig
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata, make_local_attention_virtual_batches,
    subclass_attention_backend, subclass_attention_metadata_builder)

from ..layer import Attention


@functools.lru_cache
def create_chunked_local_attention_backend(
    underlying_attn_backend: AttentionBackend,
    attention_chunk_size: int,
    block_size: int,
) -> type[AttentionBackend]:
    prefix = f"ChunkedLocalAttention_{attention_chunk_size}_{block_size}_"

    def build_preprocess_fn(cm: CommonAttentionMetadata):
        return make_local_attention_virtual_batches(attention_chunk_size, cm,
                                                    block_size)

    # Dynamically create a new attention backend that wraps the
    # underlying attention backend but applies
    # `make_local_attention_virtual_batches` before calling `build(...)`
    builder_cls = subclass_attention_metadata_builder(
        name_prefix=prefix,
        builder_cls=underlying_attn_backend.get_builder_cls(),
        build_preprocess_fn=build_preprocess_fn)
    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=builder_cls)

    return attn_backend


class ChunkedLocalAttention(Attention):

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 attention_chunk_size: int,
                 num_kv_heads: Optional[int] = None,
                 alibi_slopes: Optional[List[float]] = None,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 kv_sharing_target_layer_name: Optional[str] = None,
                 prefix: str = ""):
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

            attn_backend = create_chunked_local_attention_backend(
                underlying_attn_backend, attention_chunk_size, block_size)
        else:
            # in v0 the local attention is handled inside the backends
            attn_backend = None

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            attn_backend=attn_backend)
