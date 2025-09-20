from dataclasses import dataclass
from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)


class DeepseekV32IndexerBackend(AttentionBackend):

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        return (0, 1, 2)

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        return DeepseekV32IndexerMetadataBuilder


@dataclass
class DeepseekV32IndexerMetadata:
    pass


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> DeepseekV32IndexerMetadata:
        return DeepseekV32IndexerMetadata()
