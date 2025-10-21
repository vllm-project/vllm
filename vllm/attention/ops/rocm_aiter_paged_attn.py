# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import aiter as rocm_aiter
import torch

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()


class AITERPagedAttention(PagedAttention):
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
        is_8bit_kvcache = kv_cache_dtype in ["int8", "fp8", "fp8_e4m3"]

        if is_8bit_kvcache:
            kv_cache_torch_dtype = FP8_DTYPE if "fp8" in kv_cache_dtype else torch.int8
            key_cache = key_cache.view(kv_cache_torch_dtype)
            value_cache = value_cache.view(kv_cache_torch_dtype)

        rocm_aiter.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale=k_scale if is_8bit_kvcache else None,
            v_scale=v_scale if is_8bit_kvcache else None,
            asm_layout=True,
        )

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
        alibi_slopes: torch.Tensor | None,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty_like(query)
        is_8bit_kvcache = kv_cache_dtype in ["int8", "fp8", "fp8_e4m3"]

        if "fp8" in kv_cache_dtype:
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

            if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
                assert NotImplementedError(
                    "Blocksparse paged attention is not supported for fp8 kvcache."
                )

        rocm_aiter.pa_fwd_asm(
            Q=query,
            K=key_cache,
            V=value_cache,
            block_tables=block_tables,
            context_lens=seq_lens,
            block_tables_stride0=block_tables.stride(0),
            K_QScale=k_scale if is_8bit_kvcache else None,
            V_QScale=v_scale if is_8bit_kvcache else None,
            out_=output,
        )

        return output
