# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Merge AITER segmented MLA split-K partials into output plus natural-log LSE.

This mirrors AITER's own segment reduction, including its
``tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS * TILE_SIZE)`` partitioning, so
it has to move together with the ``skip_reduce=True`` call in the AITER MLA
backend. It is a rank-local split-K merge and unrelated to any collective
reduce; the natural-log LSE it returns is what the cross-rank DCP merge
consumes.
"""

import torch

from vllm.triton_utils import LOGE2, tl, triton


@triton.jit
def _merge_mla_segments_kernel(
    out_ptr,
    lse_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    num_query_heads: tl.constexpr,
    out_stride0: tl.int64,
    out_stride1: tl.int64,
    lse_stride0: tl.int64,
    TILE_SIZE: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    LOGE2: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + token_idx)
    tiles_per_segment = tl.maximum(
        1, tl.cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    )
    active_segments = tl.where(
        seq_len > 0,
        tl.cdiv(seq_len, tiles_per_segment * TILE_SIZE),
        0,
    )
    segment_offsets = tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    segment_mask = segment_offsets < active_segments

    segment_base = (
        token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + head_idx * NUM_SEGMENTS_PER_SEQ
    )
    segment_indices = segment_base + segment_offsets
    segment_max = tl.load(
        segm_max_ptr + segment_indices,
        mask=segment_mask,
        other=float("-inf"),
    )
    overall_max = tl.max(segment_max)

    segment_expsum = tl.load(
        segm_expsum_ptr + segment_indices,
        mask=segment_mask,
        other=0.0,
    )
    segment_weight = tl.where(
        segment_mask,
        tl.exp2(segment_max - overall_max),
        0.0,
    )
    overall_expsum = tl.sum(segment_expsum * segment_weight)

    output_offsets = (
        token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + head_idx * (NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + segment_offsets[:, None] * KV_LORA_RANK
        + tl.arange(0, KV_LORA_RANK)[None, :]
    )
    segment_output = tl.load(
        segm_output_ptr + output_offsets,
        mask=segment_mask[:, None],
        other=0.0,
    )
    accumulator = tl.sum(segment_output * segment_weight[:, None], axis=0)
    accumulator = tl.where(overall_expsum == 0.0, 0.0, accumulator / overall_expsum)
    lse = tl.where(
        overall_expsum == 0.0,
        float("-inf"),
        (overall_max + tl.log2(overall_expsum)) * LOGE2,
    )

    output_indices = (
        token_idx * out_stride0 + head_idx * out_stride1 + tl.arange(0, KV_LORA_RANK)
    )
    tl.store(out_ptr + output_indices, accumulator)
    tl.store(lse_ptr + token_idx * lse_stride0 + head_idx, lse)


def merge_mla_segments_triton(
    segm_output: torch.Tensor,
    segm_max: torch.Tensor,
    segm_expsum: torch.Tensor,
    seq_lens: torch.Tensor,
    tile_size: int,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge AITER base-2 segment partials into output and natural-log LSE."""
    num_tokens, num_heads, num_segments, kv_lora_rank = segm_output.shape
    output = torch.empty(
        (num_tokens, num_heads, kv_lora_rank),
        dtype=out_dtype,
        device=segm_output.device,
    )
    lse = torch.empty(
        (num_tokens, num_heads),
        dtype=torch.float32,
        device=segm_output.device,
    )
    _merge_mla_segments_kernel[(num_tokens, num_heads)](
        output,
        lse,
        segm_output,
        segm_max,
        segm_expsum,
        seq_lens,
        num_query_heads=num_heads,
        out_stride0=output.stride(0),
        out_stride1=output.stride(1),
        lse_stride0=lse.stride(0),
        TILE_SIZE=tile_size,
        KV_LORA_RANK=kv_lora_rank,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        LOGE2=LOGE2,
    )
    return output, lse
