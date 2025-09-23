from dataclasses import dataclass
from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
import torch


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
    
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor

    num_reqs: int
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    max_seq_len: int
    
    block_table: torch.Tensor # [num_req, (max_req_len + block_size - 1) // block_size]
    slot_mapping: torch.Tensor


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> DeepseekV32IndexerMetadata:
        return DeepseekV32IndexerMetadata(
            query_start_loc = common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping)
