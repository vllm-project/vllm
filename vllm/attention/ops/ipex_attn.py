from typing import Dict, List, Optional, Tuple

import intel_extension_for_pytorch.llm.modules as ipex_modules
import torch

from vllm import _custom_ops as ops


class PagedAttention:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 128, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        *args,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        *args,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, -1, head_size)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, -1, head_size)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        *args,
    ) -> None:
        ipex_modules.PagedAttention.reshape_and_cache(
            key, value, key_cache, value_cache,
            slot_mapping.flatten().int())

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: float,
        v_scale: float,
        *args,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        block_size = value_cache.shape[2]
        head_mapping = torch.arange(
            0,
            num_kv_heads,
            device="cpu",
            dtype=torch.int32,
        ).view(num_kv_heads,
               1).repeat_interleave(query.size(1) // num_kv_heads).flatten()
        ipex_modules.PagedAttention.single_query_cached_kv_attention(
            output, query.contiguous(), key_cache, value_cache, head_mapping,
            scale, block_tables, context_lens, block_size, max_context_len,
            alibi_slopes)

        return output

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_dtype: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        subquery_start_loc: torch.Tensor,
        prompt_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
        max_subquery_len: int,
        alibi_slopes: Optional[torch.Tensor],
        *args,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
        *args,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
        *args,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)
