# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import aiter as rocm_aiter
import torch

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.platforms import current_platform
from vllm.utils import cdiv

FP8_DTYPE = current_platform.fp8_dtype()


class AITERPagedAttention(PagedAttention):
    is_asm_supported: bool = False

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
        if not AITERPagedAttention.is_asm_supported:
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        else:
            kv_cache_torch_dtype = FP8_DTYPE \
                        if "fp8" in kv_cache_dtype else torch.int8
            key_cache = key_cache.view(kv_cache_torch_dtype)
            value_cache = value_cache.view(kv_cache_torch_dtype)

            # rocm_aiter.reshape_and_cache_with_pertoken_quant(
            #     key, value, key_cache, value_cache, k_scale, v_scale,
            #     slot_mapping.flatten(), True)
            rocm_aiter.reshape_and_cache(key, value, key_cache, value_cache,
                                         slot_mapping.flatten(),
                                         kv_cache_dtype, k_scale, v_scale,
                                         True)

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
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty_like(query)
        block_size = value_cache.shape[3]
        if not AITERPagedAttention.is_asm_supported:
            import aiter

            max_num_partitions = (max_seq_len + 256 - 1) // 256
            assert 256 % block_size == 0
            num_seqs, num_heads, head_size = query.shape
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            return aiter.paged_attention_rocm(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
                None,
                256,
            )

        if "fp8" in kv_cache_dtype:
            kv_cache_torch_dtype = FP8_DTYPE
            # kv_cache_torch_dtype = torch.int8
            key_cache = key_cache.view(kv_cache_torch_dtype)
            value_cache = value_cache.view(kv_cache_torch_dtype)

        if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
            # use blocksparse paged attention
            block_size = value_cache.size(-1)
            assert (blocksparse_block_size > 0
                    and blocksparse_block_size % block_size == 0), (
                        f"{blocksparse_block_size=} needs to be a multiple of"
                        f"{block_size=} used in block_tables.")

        max_num_blocks_per_seq = cdiv(max_seq_len, block_size)

        rocm_aiter.pa_fwd_asm(
            query,
            key_cache,
            value_cache,
            # asm_V_shuffle(value_cache),
            block_tables,
            seq_lens,
            max_num_blocks_per_seq,
            K_QScale=k_scale,
            V_QScale=v_scale,
            out_=output,
        )
        return output
