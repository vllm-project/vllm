# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Tuple

import aiter as rocm_aiter
import torch

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.platforms import current_platform
from vllm.utils import cdiv

FP8_DTYPE = current_platform.fp8_dtype()


def asm_V_shuffle(VC):
    # [num_blocks, num_kv_heads, head_size, block_size]
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    # [num_blocks, num_kv_heads, block_size/X, head_size, X]
    VC = VC.permute(0, 1, 3, 2, 4).contiguous()
    return VC


class AITERPagedAttention(PagedAttention):

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, asm_V_shuffle(value_cache)

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        if kv_cache_dtype not in ["int8", "fp8", "fp8_e4m3"]:
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache, slot_mapping,
                                                kv_cache_dtype, k_scale,
                                                v_scale)
        else:
            kv_cache_torch_dtype = (FP8_DTYPE
                                    if "fp8" in kv_cache_dtype else torch.int8)
            key_cache = key_cache.view(kv_cache_torch_dtype)
            value_cache = value_cache.view(kv_cache_torch_dtype)

            # reshape_and_cache_with_pertoken_quant
            # stores the key and value in the key_cache and value_cache
            # and k_scale and v_scale in ASM Layout
            # It takes input in the following format:
            # key.shape: [num_seqs, num_query_heads, head_size]
            # value.shape: [num_seqs, num_kv_heads, head_size]
            # key_cache.shape: [num_blocks, num_kv_heads, head_size // x, -1, x]
            # value_cache.shape: [num_blocks, num_kv_heads, block_size/X, head_size, X]  # noqa: E501
            # k_scale.shape: [num_blocks, num_kv_heads, block_size]
            # v_scale.shape: [num_blocks, num_kv_heads, block_size]
            rocm_aiter.reshape_and_cache_with_pertoken_quant(
                key, value, key_cache, value_cache, k_scale, v_scale,
                slot_mapping.flatten(), True)

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> torch.Tensor:
        if kv_cache_dtype not in ["int8", "fp8", "fp8_e4m3"]:
            return PagedAttention.forward_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                kv_cache_dtype=kv_cache_dtype,
                num_kv_heads=num_kv_heads,
                scale=scale,
                alibi_slopes=alibi_slopes,
                k_scale=k_scale,
                v_scale=v_scale,
                tp_rank=tp_rank,
                blocksparse_local_blocks=blocksparse_local_blocks,
                blocksparse_vert_stride=blocksparse_vert_stride,
                blocksparse_block_size=blocksparse_block_size,
                blocksparse_head_sliding_step=blocksparse_head_sliding_step)

        if "fp8" in kv_cache_dtype:
            key_cache = key_cache.view(torch.float8_e4m3fnuz)
            value_cache = value_cache.view(torch.float8_e4m3fnuz)

        if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
            # use blocksparse paged attention
            block_size = value_cache.size(-1)
            assert (blocksparse_block_size > 0 and
                    blocksparse_block_size % block_size == 0), \
                (f"{blocksparse_block_size=} needs to be a multiple of"
                 f"{block_size=} used in block_tables.")

        output = torch.empty_like(query)
        # value_cache.shape: [num_blocks, num_kv_heads, head_size, block_size]
        block_size = value_cache.shape[2] * value_cache.shape[4]
        max_num_blocks_per_seq = cdiv(max_seq_len, block_size)

        rocm_aiter.pa_fwd_asm(Q=query,
                              K=key_cache,
                              V=value_cache,
                              block_tables=block_tables,
                              context_lens=seq_lens,
                              max_num_blocks=max_num_blocks_per_seq,
                              K_QScale=k_scale,
                              V_QScale=v_scale,
                              out_=output)

        return output
