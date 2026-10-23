# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4's ROCm MXFP4 sparse indexer on aiter's FlyDSL paged scorers."""

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.mla.rocm_flydsl_mxfp4_indexer import (
    FLYDSL_BLOCK_K,
    RocmFlyDslIndexerMetadata,
)


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


@eager_break_during_capture
def rocm_flydsl_mxfp4_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    head_dim: int,
    max_model_len: int,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    """Score the paged MXFP4 cache in place and write each row's top-k.

    The model supplies packed e2m1 ``q_quant`` [T, 64, 64], ``q_scale`` in
    aiter's [T, 1, 4, 16, 4] swizzle, bf16 ``weights`` with the softmax and
    head scales folded in, and ``kv_cache`` pages in the FlyDSL layout.
    """
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        # Profiling run: reserve the largest prefill logits the builder plans.
        max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(max_logits_bytes, dtype=torch.uint8, device=q_quant.device)
        return topk_indices_buffer
    metadata = attn_metadata[k_cache_prefix]
    assert isinstance(metadata, RocmFlyDslIndexerMetadata)
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
        flydsl_pa_mqa_logits_fp4,
    )
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        flydsl_pa_mqa_logits_fp4_prefill,
    )

    assert q_quant.dtype == torch.uint8 and q_quant.shape[1:] == (64, 64)
    assert q_scale.dtype == torch.uint8
    assert q_scale.shape == (q_quant.shape[0], 1, 4, 16, 4)
    values, scales = _rocm_fp4_cache_views(kv_cache, head_dim)
    block_size = kv_cache.shape[1]
    weights_bf16 = (
        weights if weights.dtype == torch.bfloat16 else weights.to(torch.bfloat16)
    )
    topk_indices_buffer[: q_quant.shape[0]].fill_(-1)

    if metadata.prefill is not None:
        for chunk, plan in zip(metadata.prefill.chunks, metadata.prefill_plans):
            if chunk.local_total_seq_lens == 0 or chunk.token_start == chunk.token_end:
                continue
            row_to_batch, starts, ends = plan.windows
            row_slice = slice(chunk.token_start, chunk.token_end)
            num_rows = chunk.token_end - chunk.token_start
            if plan.cta_info is None:
                # No wider than the top-k: every row keeps its whole window.
                chunk_topk = topk_indices_buffer[row_slice, :topk_tokens]
                _rocm_fp4_fill_all_prefill_indices_kernel[(num_rows,)](
                    ends,
                    chunk_topk,
                    chunk_topk.stride(0),
                    TOP_K=topk_tokens,
                    PADDED_TOP_K=next_power_of_2(topk_tokens),
                    num_warps=8,
                )
                continue
            assert metadata.prefill_logits is not None
            logits = metadata.prefill_logits[: num_rows * plan.width].view(
                num_rows, plan.width
            )
            logits = flydsl_pa_mqa_logits_fp4_prefill(
                q_quant[row_slice],
                q_scale[row_slice],
                values,
                scales,
                chunk.block_table,
                weights_bf16[row_slice],
                row_to_batch,
                starts,
                ends,
                min(max_model_len, plan.width),
                block_k=FLYDSL_BLOCK_K,
                kv_block_size=block_size,
                out=logits,
                cta_info=plan.cta_info,
                n_ctas=plan.total_ctas,
            )
            ops.top_k_per_row_prefill(
                logits,
                starts,
                ends,
                topk_indices_buffer[row_slice, :topk_tokens],
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    rows = metadata.num_decode_tokens
    if metadata.decode is not None and rows:
        decode = metadata.decode
        assert not decode.requires_padding
        assert metadata.decode_cta_info is not None
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
            block_k=FLYDSL_BLOCK_K,
            kv_block_size=block_size,
            out=logits,
            cta_info=metadata.decode_cta_info,
            total_ctas=metadata.decode_total_ctas,
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


class RocmFlyDslSparseAttnIndexer(SparseAttnIndexer):
    """`SparseAttnIndexer` whose HIP path scores DeepSeek V4's paged MXFP4
    cache with aiter's FlyDSL kernels. The model writes the K cache."""

    def __init__(self, k_cache, *args, **kwargs):
        super().__init__(k_cache, *args, **kwargs)
        if not (self.use_fp4_cache and self.skip_k_cache_insert):
            raise ValueError(
                "The ROCm FlyDSL indexer scores an MXFP4 K cache the model writes."
            )
        if self.head_dim != 128 or self.compress_ratio != 4:
            raise ValueError("The ROCm FlyDSL indexer requires C4 and D=128.")
        if self.candidate_blocks is not None:
            raise ValueError("The ROCm FlyDSL indexer has no candidate blocks.")

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        assert isinstance(q_quant, tuple)
        q_values, q_scale = q_quant
        return rocm_flydsl_mxfp4_sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            weights,
            self.head_dim,
            self.max_model_len,
            self.topk_tokens,
            self.topk_indices_buffer,
        )
