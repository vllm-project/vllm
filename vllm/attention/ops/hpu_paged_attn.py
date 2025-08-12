###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from vllm_hpu_extension import cache_ops, ops

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


@dataclass
class HPUPagedAttentionMetadata:
    """Metadata for PagedAttention."""
    block_list: Optional[torch.Tensor]
    block_mapping: Optional[torch.Tensor]
    block_usage: Optional[torch.Tensor]
    block_indices: Optional[torch.Tensor]
    block_offsets: Optional[torch.Tensor]
    block_scales: Optional[torch.Tensor]


class HPUPagedAttention:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 128, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(key: torch.Tensor, value: torch.Tensor,
                             key_cache: torch.Tensor,
                             value_cache: torch.Tensor,
                             slot_mapping: torch.Tensor, kv_cache_dtype: str,
                             is_prompt: bool) -> None:
        cache_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                    slot_mapping, kv_cache_dtype, is_prompt)

    @staticmethod
    def forward_decode(**kwargs) -> torch.Tensor:
        return ops.flat_pa(**kwargs)

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        subquery_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
        max_query_len: int,
        alibi_slopes: Optional[torch.Tensor],
        sliding_window: Optional[int],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "forward_prefix is not implemented for HPUPagedAttention")

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dists)
