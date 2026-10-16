# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MXFP4 sparse attention indexer operations for DeepSeek V4."""

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2


def _rocm_fp4_cache_views(
    kv_cache: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    assert head_dim == 128 and kv_cache.ndim == 3
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[2] == 68
    num_blocks, block_size, _ = kv_cache.shape
    assert block_size == 64
    assert kv_cache.stride(2) == 1 and kv_cache.stride(1) == 68
    page_stride = kv_cache.stride(0)
    assert page_stride >= block_size * 68
    values = torch.as_strided(
        kv_cache,
        (num_blocks, 1, 4, block_size, 16),
        (page_stride, block_size * 64, block_size * 16, 16, 1),
    )
    scales = torch.as_strided(
        kv_cache,
        (num_blocks, 1, 4, block_size),
        (page_stride, block_size * 4, block_size, 1),
        storage_offset=kv_cache.storage_offset() + block_size * 64,
    )
    return values, scales


@triton.jit
def _rocm_fp4_fill_all_prefill_indices_kernel(
    local_ends,
    topk_indices,
    topk_stride,
    TOP_K: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, PADDED_TOP_K)
    local_end = tl.load(local_ends + row)
    tl.store(
        topk_indices + row * topk_stride + offsets,
        tl.where(offsets < local_end, offsets, -1),
        mask=offsets < TOP_K,
    )


def _rocm_fp4_sparse_attn_indexer(
    kv_cache,
    q_quant,
    q_scale,
    weights,
    head_dim,
    max_model_len,
    topk_tokens,
    topk_indices_buffer,
    metadata,
):
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
        flydsl_pa_mqa_logits_fp4,
    )
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        flydsl_pa_mqa_logits_fp4_prefill,
    )

    assert q_quant.dtype == torch.uint8 and q_quant.shape[1:] == (64, 64)
    assert q_scale is not None and q_scale.dtype == torch.uint8
    assert q_scale.shape == (q_quant.shape[0], 1, 4, 16, 4)
    values, scales = _rocm_fp4_cache_views(kv_cache, head_dim)
    block_size = kv_cache.shape[1]
    block_k = 256
    weights_bf16 = (
        weights if weights.dtype == torch.bfloat16 else weights.to(torch.bfloat16)
    )
    topk_indices_buffer[: q_quant.shape[0]].fill_(-1)

    if metadata.num_prefills:
        assert metadata.prefill is not None
        for chunk in metadata.prefill.chunks:
            if chunk.local_total_seq_lens == 0 or chunk.token_start == chunk.token_end:
                continue
            assert chunk.fp4_windows is not None
            row_to_batch, starts, ends = chunk.fp4_windows
            rows = slice(chunk.token_start, chunk.token_end)
            num_rows = chunk.token_end - chunk.token_start
            if chunk.logits_width <= topk_tokens:
                chunk_topk = topk_indices_buffer[rows, :topk_tokens]
                _rocm_fp4_fill_all_prefill_indices_kernel[(num_rows,)](
                    ends,
                    chunk_topk,
                    chunk_topk.stride(0),
                    TOP_K=topk_tokens,
                    PADDED_TOP_K=next_power_of_2(topk_tokens),
                    num_warps=8,
                )
                continue
            assert metadata.prefill.fp4_logits is not None
            assert chunk.fp4_cta_info is not None
            assert chunk.fp4_total_ctas is not None
            logits = metadata.prefill.fp4_logits[: num_rows * chunk.logits_width].view(
                num_rows, chunk.logits_width
            )
            logits = flydsl_pa_mqa_logits_fp4_prefill(
                q_quant[rows],
                q_scale[rows],
                values,
                scales,
                chunk.block_table,
                weights_bf16[rows],
                row_to_batch,
                starts,
                ends,
                min(max_model_len, chunk.logits_width),
                block_k=block_k,
                kv_block_size=block_size,
                out=logits,
                cta_info=chunk.fp4_cta_info,
                n_ctas=chunk.fp4_total_ctas,
            )
            ops.top_k_per_row_prefill(
                logits,
                starts,
                ends,
                topk_indices_buffer[rows, :topk_tokens],
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if metadata.num_decodes:
        decode = metadata.decode
        assert decode is not None and not decode.requires_padding
        assert decode.seq_lens.ndim == 2 and decode.seq_lens.shape[1] == 1
        rows = metadata.num_decode_tokens
        if rows:
            assert decode.fp4_cta_info is not None
            assert decode.fp4_total_ctas is not None
            # Capture owns this temporary in its graph pool; eager calls only
            # allocate for this batch, not the scheduler's maximum row count.
            logits = torch.empty(
                (rows, max_model_len), dtype=torch.float32, device=q_quant.device
            )
            logits = flydsl_pa_mqa_logits_fp4(
                q_quant[:rows].unsqueeze(1),
                q_scale[:rows].unsqueeze(1),
                values,
                scales,
                decode.block_table[:rows],
                weights_bf16[:rows],
                decode.seq_lens[:rows, 0],
                max_model_len,
                next_n=1,
                block_k=block_k,
                kv_block_size=block_size,
                out=logits,
                cta_info=decode.fp4_cta_info,
                total_ctas=decode.fp4_total_ctas,
            )
            ops.top_k_per_row_decode(
                logits,
                1,
                decode.seq_lens[:rows],
                topk_indices_buffer[:rows, :topk_tokens],
                rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
    return topk_indices_buffer
