from typing import Dict, List, Optional, Tuple

import intel_extension_for_pytorch.llm.modules as ipex_modules
import torch
from vllm import envs

from vllm import _custom_ops as ops
@torch.library.impl("myops::_single_query_cached_kv_attention", "cpu")
def _single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    is_key_cache_vnni: bool,
    value_cache: torch.Tensor,
    is_value_cache_vnni: bool,
    head_mapping: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor]):
    ipex_modules.PagedAttention.single_query_cached_kv_attention(
        output, query, key_cache, is_key_cache_vnni, value_cache, is_value_cache_vnni, head_mapping,
        scale, block_tables, context_lens, block_size, max_context_len.contiguous().item(),
        alibi_slopes)

    return output

torch.library.define(
    "myops::_single_query_cached_kv_attention",
    "(Tensor output, Tensor query, Tensor key_cache, bool is_key_cache_vnni, Tensor value_cache, bool is_value_cache_vnni, "
    + " Tensor head_mapping, float scale, Tensor block_tables, Tensor context_lens, int block_size, Tensor max_context_len, Tensor? alibi_slopes) -> (Tensor)",
)


# Note: just a workaround
v_cache_offset: int = -1
k_cache_layout: Tuple[int, ...]
v_cache_layout: Tuple[int, ...]

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
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        if envs.VLLM_CPU_VNNI_KEY_CACHE_LAYOUT or envs.VLLM_CPU_VNNI_VALUE_CACHE_LAYOUT :
            global v_cache_offset, k_cache_layout, v_cache_layout
            elem_width = torch.tensor([], dtype=dtype, device="cpu").element_size()
            if elem_width == 2:
                padded_block_size = ((block_size + 1) // 2) * 2
                padded_head_size = ((head_size + 1) // 2) * 2
            elif elem_width == 1:
                padded_block_size = ((block_size + 3) // 4) * 4
                padded_head_size = ((head_size + 3) // 4) * 4
            else:
                assert False, "unsupported dtype for VNNI KV cache layout."

            if envs.VLLM_CPU_VNNI_KEY_CACHE_LAYOUT:
                k_cache_elem_num = num_blocks * num_kv_heads * padded_head_size * block_size
                k_cache_layout = (num_blocks, num_kv_heads, padded_head_size, block_size)
            else:
                k_cache_elem_num = num_blocks * num_kv_heads * head_size * block_size
                k_cache_layout = (num_blocks, num_kv_heads, block_size, head_size)

            if envs.VLLM_CPU_VNNI_VALUE_CACHE_LAYOUT:
                v_cache_elem_num = num_blocks * num_kv_heads * padded_block_size * head_size
                v_cache_layout = (num_blocks, num_kv_heads, padded_block_size, head_size)
            else:
                v_cache_elem_num = num_blocks * num_kv_heads * block_size * head_size
                v_cache_layout = (num_blocks, num_kv_heads, block_size, head_size)

            v_cache_offset = k_cache_elem_num 
            return (k_cache_elem_num + v_cache_elem_num,)
        else:
            return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        *args,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if envs.VLLM_CPU_VNNI_KEY_CACHE_LAYOUT or envs.VLLM_CPU_VNNI_VALUE_CACHE_LAYOUT:
            key_cache = kv_cache[:v_cache_offset]
            value_cache = kv_cache[v_cache_offset:]
            key_cache = key_cache.view(*(k_cache_layout))
            value_cache = value_cache.view(*(v_cache_layout))
        else:
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
        kv_scale: float,
        *args,
    ) -> None:
        ipex_modules.PagedAttention.reshape_and_cache(
            key, value, key_cache, envs.VLLM_CPU_VNNI_KEY_CACHE_LAYOUT, value_cache, envs.VLLM_CPU_VNNI_VALUE_CACHE_LAYOUT,
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
        kv_scale: float,
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
        # ipex_modules.PagedAttention.single_query_cached_kv_attention(
        torch.ops.myops._single_query_cached_kv_attention(
            output, query.contiguous(), key_cache, envs.VLLM_CPU_VNNI_KEY_CACHE_LAYOUT, value_cache, envs.VLLM_CPU_VNNI_VALUE_CACHE_LAYOUT, head_mapping,
            scale, block_tables, context_lens, block_size, max_context_len.contiguous(),
            alibi_slopes)

        return output

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
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
        if envs.VLLM_CPU_VNNI_KEY_CACHE_LAYOUT or envs.VLLM_CPU_VNNI_VALUE_CACHE_LAYOUT:
            key_caches = [kv_cache[:v_cache_offset].view(*k_cache_layout) for kv_cache in kv_caches]
            value_caches = [kv_cache[v_cache_offset:].view(*v_cache_layout) for kv_cache in kv_caches]
        else:
            key_caches = [kv_cache[0] for kv_cache in kv_caches]
            value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)
