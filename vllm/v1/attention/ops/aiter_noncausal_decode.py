# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV decode for non-causal multi-token query blocks (DSpark draft).

Adapted from AITER's ``kernel_unified_attention_3d`` / ``reduce_segments``
(``aiter/ops/triton/_triton_kernels/attention/unified_attention.py``). AITER's
kernel bounds each query row by a causal prefix. Here every query row of a
request attends to the request's whole sequence ``[0, seq_len)``, which is the
DSpark draft block semantics. All rows of a request share one KV read
per (kv head, segment) when ``BLOCK_Q`` covers the block.

Not supported: sliding window, sinks, alibi, softcap, qq bias, shuffled KV.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _cdiv(x, y):
    return (x + y - 1) // y


@triton.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _noncausal_decode_3d(
    segm_output_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS, HEAD_SIZE_PADDED]
    segm_max_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    scale,
    k_descale_ptr,
    v_descale_ptr,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,  # [num_seqs + 1]
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    tiles_per_segment = _cdiv(seq_len, NUM_SEGMENTS * TILE_SIZE)
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads
    row_mask = query_mask_0 & query_mask_1

    Q = tl.load(
        query_ptr + query_offset, mask=dim_mask[None, :] & row_mask[:, None], other=0.0
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    num_tiles = _cdiv(seq_len, TILE_SIZE)

    if k_descale_ptr is not None:
        qk_scale = qk_scale * tl.load(k_descale_ptr)
    out_factor: tl.float32 = 1.0
    if v_descale_ptr is not None:
        out_factor = tl.load(v_descale_ptr)

    for j in range(
        segm_idx * tiles_per_segment,
        tl.minimum((segm_idx + 1) * tiles_per_segment, num_tiles),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < seq_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE,
            mask=tile_mask,
            other=0,
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        K = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        ).to(Q.dtype)
        V = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        ).to(Q.dtype)

        S = qk_scale * tl.dot(Q, K)
        S = tl.where(row_mask[:, None] & tile_mask[None, :], S, float("-inf"))

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.math.exp2(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.math.exp2(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    acc = acc * out_factor

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + offs_d[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & row_mask[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS)
        + query_offset_1 * NUM_SEGMENTS
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=row_mask)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=row_mask)


@triton.jit
def _noncausal_reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    num_seqs,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    query_start_len_ptr,
    NUM_SEGMENTS: tl.constexpr,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    output_offset = (
        query_token_idx * output_stride_0 + query_head_idx * output_stride_1 + offs_d
    )

    # Rows past the last request (CUDA-graph padding) have no segments.
    num_tokens = tl.load(query_start_len_ptr + num_seqs)
    if query_token_idx >= num_tokens:
        return
    seq_idx = _find_seq_idx(query_start_len_ptr, query_token_idx, num_seqs, 1, False)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if seq_len <= 0:
        tl.store(
            output_ptr + output_offset,
            tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32).to(
                output_ptr.type.element_ty
            ),
            mask=dim_mask,
        )
        return

    tiles_per_segment = _cdiv(seq_len, NUM_SEGMENTS * TILE_SIZE)
    act_num_segments = _cdiv(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS) < act_num_segments

    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS)
        + query_head_idx * NUM_SEGMENTS
        + tl.arange(0, NUM_SEGMENTS)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.math.exp2(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS)[:, None] * HEAD_SIZE_PADDED
        + offs_d[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.math.exp2(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)
    tl.store(
        output_ptr + output_offset, acc.to(output_ptr.type.element_ty), mask=dim_mask
    )


def _block_m_q(max_seqlen_q: int, num_queries_per_kv: int) -> tuple[int, int]:
    block_m = triton.next_power_of_2(max(16, max_seqlen_q * num_queries_per_kv))
    return block_m, block_m // num_queries_per_kv


def default_num_segments(
    num_seqs: int,
    num_kv_heads: int,
    max_seqlen_q: int,
    device: torch.device,
    num_queries_per_kv: int = 4,
) -> int:
    """Segments that fill ~8 programs per CU; independent of context length."""
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    _, block_q = _block_m_q(max_seqlen_q, num_queries_per_kv)
    programs = max(1, num_seqs * num_kv_heads * triton.cdiv(max_seqlen_q, block_q))
    segments = triton.next_power_of_2(max(1, (8 * num_sms) // programs))
    return max(1, min(segments, 128))


def noncausal_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    num_segments: int | None = None,
    tile_size: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
    waves_per_eu: int = 2,
) -> torch.Tensor:
    """Every query row of request ``i`` attends to keys ``[0, seqused_k[i])``."""
    num_tokens, num_query_heads, head_size = q.shape
    _, block_size, num_kv_heads, _ = k.shape
    num_seqs = seqused_k.shape[0]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size_padded = triton.next_power_of_2(head_size)

    block_m, block_q = _block_m_q(max_seqlen_q, num_queries_per_kv)
    total_num_q_blocks = num_tokens // block_q + num_seqs
    if num_segments is None:
        num_segments = default_num_segments(
            num_seqs, num_kv_heads, max_seqlen_q, q.device, num_queries_per_kv
        )

    segm_output = torch.empty(
        num_tokens,
        num_query_heads,
        num_segments,
        head_size_padded,
        dtype=torch.float32,
        device=q.device,
    )
    segm_max = torch.empty(
        num_tokens, num_query_heads, num_segments, dtype=torch.float32, device=q.device
    )
    segm_expsum = torch.empty_like(segm_max)

    _noncausal_decode_3d[(total_num_q_blocks, num_kv_heads, num_segments)](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        k_descale_ptr=k_descale,
        v_descale_ptr=v_descale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        num_seqs=num_seqs,
        BLOCK_M=block_m,
        NUM_SEGMENTS=num_segments,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    _noncausal_reduce_segments[(num_tokens, num_query_heads)](
        output_ptr=out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=seqused_k,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        query_start_len_ptr=cu_seqlens_q,
        NUM_SEGMENTS=num_segments,
    )
    return out
