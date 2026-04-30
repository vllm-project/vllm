# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K paged decode kernel for ROCM_ATTN backend (gfx1100 / RDNA3).

The default ROCM_ATTN decode kernel ``kernel_paged_attention_2d`` launches
a 2-D grid ``(num_seqs, num_kv_heads)``: in interactive workloads
(``num_seqs`` small) and small ``num_kv_heads`` this leaves most of the
RX 7900 XTX's 96 CUs idle.  This file provides a 3-D split-K variant
that adds ``NUM_SEGMENTS`` parallel programs along the K dimension and
finishes with an online-softmax merge, matching the pattern used by the
TRITON_ATTN backend (``unified_attention``) but preserving the
ROCM_ATTN paged KV layout produced by ``PagedAttention.split_kv_cache``:

- ``key_cache``   shape ``(num_blocks, num_kv_heads, head_size // x, block_size, x)``
- ``value_cache`` shape ``(num_blocks, num_kv_heads, head_size, block_size)``

The kernel is decode-only (``query.shape[1] == num_query_heads`` with
one token per sequence, ``qlen == 1``) and assumes causal masking
(equivalent to "attend to all valid keys" when qlen == 1).  It does not
support FP8 KV cache, alibi, sliding window, sinks, or softcap; the
caller must fall back to the existing ``chunked_prefill_paged_decode``
path for those.

Cudagraph-safe: the three workspace tensors must be pre-allocated by the
metadata builder; this entry point performs no allocations.
"""
from typing import Final

import torch

from vllm.triton_utils import tl, triton

# Number of parallel split-K segments per sequence.  Matches TRITON_ATTN's
# default; chosen so that for typical Hkv >= 4 we land at >= 64 programs.
NUM_PAR_SOFTMAX_SEGMENTS: Final[int] = 16

# Below this many programs the 2-D kernel under-utilizes the GPU and the
# 3-D split-K path becomes profitable.  Used by the dispatcher (not by
# the kernel itself).
MIN_LAUNCH_GRID_SIZE_2D: Final[int] = 128


@triton.jit
def _kernel_paged_decode_split_k(
    segm_output_ptr,           # (num_seqs, num_query_heads, NUM_SEGMENTS, head_size_padded) f32
    segm_max_ptr,              # (num_seqs, num_query_heads, NUM_SEGMENTS) f32
    segm_expsum_ptr,           # (num_seqs, num_query_heads, NUM_SEGMENTS) f32
    query_ptr,                 # (num_seqs, num_query_heads, head_size)
    key_cache_ptr,             # (num_blocks, num_kv_heads, head_size//x, block_size, x)
    value_cache_ptr,           # (num_blocks, num_kv_heads, head_size, block_size)
    block_tables_ptr,          # (num_seqs, max_logical_blocks)
    seq_lens_ptr,              # (num_seqs,)
    scale,                     # float32
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    BLOCK_M: tl.constexpr,             # next_pow2(num_queries_per_kv) clamped to >= 16
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    TILE_SIZE: tl.constexpr,           # K-direction tile (e.g. 32)
    PHYSICAL_BLOCK_SIZE: tl.constexpr, # cache block_size (may be non-pow2)
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,    # next_pow2(head_size)
    x: tl.constexpr,                   # K cache vec dim (innermost)
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_k_cache_4: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    NUM_SEGMENTS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # Each segment processes a contiguous K-range of `tiles_per_segment` tiles.
    # ceil so the last segment may be partial; segments past act_num_segments
    # short-circuit (their workspace slots are masked out by the reduce kernel).
    tiles_per_segment = (seq_len + NUM_SEGMENTS * TILE_SIZE - 1) // (NUM_SEGMENTS * TILE_SIZE)
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    loop_lo = segm_idx * tiles_per_segment
    total_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE
    loop_hi_raw = (segm_idx + 1) * tiles_per_segment
    loop_hi = tl.minimum(loop_hi_raw, total_tiles)

    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    qh = kv_head_idx * num_queries_per_kv + offs_h
    head_mask = (qh < (kv_head_idx + 1) * num_queries_per_kv) & (qh < num_query_heads)
    dim_mask = offs_d < HEAD_SIZE

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    q_offset = (
        seq_idx * query_stride_0
        + qh[:, None] * query_stride_1
        + offs_d[None, :]
    )
    Q = tl.load(
        query_ptr + q_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    M = tl.full([BLOCK_M], float("-inf"), tl.float32)
    L = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], tl.float32)

    block_table_offset = seq_idx * block_table_stride

    for j in range(loop_lo, loop_hi):
        abs_tok = j * TILE_SIZE + offs_t
        tile_mask = abs_tok < seq_len

        l_block_idx = abs_tok // PHYSICAL_BLOCK_SIZE
        p_block_idx = tl.load(block_tables_ptr + block_table_offset + l_block_idx)
        internal_off = abs_tok % PHYSICAL_BLOCK_SIZE

        # K : (HEAD_SIZE_PADDED, TILE_SIZE) -- 5D address (vec dim x innermost)
        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_off[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )
        K = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        # V : (TILE_SIZE, HEAD_SIZE_PADDED) -- 4D address (slot innermost)
        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_off[:, None] * stride_v_cache_3
        )
        V = tl.load(
            value_cache_ptr + v_offset,
            mask=tile_mask[:, None] & dim_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        # S = scale * Q @ K  -> (BLOCK_M, TILE_SIZE)
        S = scale * tl.dot(Q, K)
        S = tl.where(head_mask[:, None] & tile_mask[None, :], S, float("-inf"))

        # qlen == 1: every key position 0..seq_len-1 is valid; tile_mask handles seq end.

        m_j = tl.maximum(M, tl.max(S, axis=1))
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)
        l_j = tl.sum(p, axis=1)
        alpha = tl.exp(M - m_j)
        alpha = tl.where(M == float("-inf"), 0.0, alpha)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(p.to(V.dtype), V)

    # Store per-segment partials.  The reduce kernel finalizes.
    base = (
        seq_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS)
        + qh * NUM_SEGMENTS
        + segm_idx
    )
    tl.store(segm_max_ptr + base, M, mask=head_mask)
    tl.store(segm_expsum_ptr + base, L, mask=head_mask)

    out_base = (
        seq_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS * HEAD_SIZE_PADDED)
        + qh[:, None] * (NUM_SEGMENTS * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + offs_d[None, :]
    )
    tl.store(
        segm_output_ptr + out_base,
        acc,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _kernel_reduce_segments(
    output_ptr,           # (num_seqs, num_query_heads, head_size)
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    NUM_SEGMENTS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    qhead_idx = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    tiles_per_segment = (seq_len + NUM_SEGMENTS * TILE_SIZE - 1) // (NUM_SEGMENTS * TILE_SIZE)
    # Number of segments that actually got data from the producer kernel.
    act_num_segments = (seq_len + tiles_per_segment * TILE_SIZE - 1) // (tiles_per_segment * TILE_SIZE)

    seg_offs = tl.arange(0, NUM_SEGMENTS)
    seg_mask = seg_offs < act_num_segments

    base = (
        seq_idx.to(tl.int64) * (NUM_QUERY_HEADS * NUM_SEGMENTS)
        + qhead_idx * NUM_SEGMENTS
        + seg_offs
    )
    seg_max = tl.load(segm_max_ptr + base, mask=seg_mask, other=float("-inf"))
    overall_max = tl.max(seg_max)

    seg_expsum = tl.load(segm_expsum_ptr + base, mask=seg_mask, other=0.0)
    seg_expsum = seg_expsum * tl.exp(seg_max - overall_max)
    overall_expsum = tl.sum(seg_expsum)

    seg_offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = seg_offs_d < HEAD_SIZE
    seg_out_base = (
        seq_idx.to(tl.int64) * (NUM_QUERY_HEADS * NUM_SEGMENTS * HEAD_SIZE_PADDED)
        + qhead_idx * (NUM_SEGMENTS * HEAD_SIZE_PADDED)
        + seg_offs[:, None] * HEAD_SIZE_PADDED
        + seg_offs_d[None, :]
    )
    seg_out = tl.load(
        segm_output_ptr + seg_out_base,
        mask=seg_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    seg_out = seg_out * tl.exp(seg_max - overall_max)[:, None]
    acc_sum = tl.sum(seg_out, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    out_offset = (
        seq_idx * output_stride_0
        + qhead_idx * output_stride_1
        + seg_offs_d
    )
    tl.store(output_ptr + out_offset, acc, mask=dim_mask)


def paged_decode_split_k(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    softmax_segm_output: torch.Tensor,
    softmax_segm_max: torch.Tensor,
    softmax_segm_expsum: torch.Tensor,
    num_segments: int = NUM_PAR_SOFTMAX_SEGMENTS,
    tile_size: int = 16,
) -> None:
    """3-D split-K paged decode + reduce.

    Args:
        output: (num_seqs, num_query_heads, head_size). Written.
        query: (num_seqs, num_query_heads, head_size).  qlen must be 1.
        key_cache: 5-D, layout from ``PagedAttention.split_kv_cache``.
        value_cache: 4-D, layout from ``PagedAttention.split_kv_cache``.
        block_table: (num_seqs, max_logical_blocks) int32.
        seq_lens: (num_seqs,) int32.
        scale: softmax scale (1/sqrt(head_size)).
        softmax_segm_{output,max,expsum}: workspaces, pre-allocated by the
            metadata builder.  See ``RocmAttentionMetadataBuilder``.
        num_segments: K-split degree.  Must match workspace allocation.
        tile_size: K tile size.  Must match workspace allocation.
    """
    num_seqs, num_query_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    physical_block_size = key_cache.shape[3]
    x = key_cache.shape[4]
    num_queries_per_kv = num_query_heads // num_kv_heads

    bm = max(triton.next_power_of_2(num_queries_per_kv), 16)
    head_size_padded = triton.next_power_of_2(head_size)

    # num_warps=4, num_stages=2 swept best on RX 7900 XTX (gfx1100):
    # ~594 GB/s @ Qwen3.5-shape, klen=4k, batch=1.
    grid_3d = (num_seqs, num_kv_heads, num_segments)
    _kernel_paged_decode_split_k[grid_3d](
        segm_output_ptr=softmax_segm_output,
        segm_max_ptr=softmax_segm_max,
        segm_expsum_ptr=softmax_segm_expsum,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        scale=scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        BLOCK_M=bm,
        block_table_stride=block_table.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        TILE_SIZE=tile_size,
        PHYSICAL_BLOCK_SIZE=physical_block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        x=x,
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        NUM_SEGMENTS=num_segments,
        num_warps=4,
        num_stages=2,
    )

    grid_red = (num_seqs, num_query_heads)
    _kernel_reduce_segments[grid_red](
        output_ptr=output,
        segm_output_ptr=softmax_segm_output,
        segm_max_ptr=softmax_segm_max,
        segm_expsum_ptr=softmax_segm_expsum,
        seq_lens_ptr=seq_lens,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        NUM_SEGMENTS=num_segments,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        NUM_QUERY_HEADS=num_query_heads,
        TILE_SIZE=tile_size,
    )
