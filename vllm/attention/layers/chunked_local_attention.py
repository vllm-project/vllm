# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

import torch

from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata
from vllm.attention.layer import Attention
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    make_local_attention_virtual_batches,
    subclass_attention_backend,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    KVCacheSpec,
)


@functools.lru_cache
def create_chunked_local_attention_backend(
    underlying_attn_backend: AttentionBackend,
    attention_chunk_size: int,
    block_size: int,
) -> type[AttentionBackend]:
    prefix = f"ChunkedLocalAttention_{attention_chunk_size}_{block_size}_"

    underlying_builder = underlying_attn_backend.get_builder_cls()
    assert issubclass(underlying_builder, AttentionMetadataBuilder)

    class ChunkedLocalAttentionBuilder(underlying_builder):  # type: ignore
        @classmethod
        def get_cudagraph_support(
            cls: type["AttentionMetadataBuilder"],
            vllm_config: VllmConfig,
            kv_cache_spec: AttentionSpec,
        ) -> AttentionCGSupport:
            # Explicit override in case the underlying builder specialized this getter.
            # @override omitted only because of mypy limitation due to type variable.
            return AttentionCGSupport.NEVER

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            common_attn_metadata = make_local_attention_virtual_batches(
                attention_chunk_size, common_attn_metadata, block_size
            )
            return super().build(common_prefix_len, common_attn_metadata, fast_build)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=ChunkedLocalAttentionBuilder,
    )

    return attn_backend


class ChunkedLocalAttention(Attention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        attention_chunk_size: int,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        kv_sharing_target_layer_name: str | None = None,
        prefix: str = "",
    ):
        self.attention_chunk_size = attention_chunk_size
        dtype = torch.get_default_dtype()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        underlying_attn_backend = get_attn_backend(
            head_size, dtype, kv_cache_dtype, block_size
        )
        attn_backend = create_chunked_local_attention_backend(
            underlying_attn_backend, attention_chunk_size, block_size
        )

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
            attn_backend=attn_backend,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        assert self.attention_chunk_size
        return ChunkedLocalAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            attention_chunk_size=self.attention_chunk_size,
        )
