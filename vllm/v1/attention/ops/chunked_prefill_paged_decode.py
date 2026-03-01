# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .prefix_prefill import context_attention_fwd

float8_info = torch.finfo(current_platform.fp8_dtype())


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def paged_attention_inner_loop(
    Q,  # (num_queries_per_kv, HEAD_SIZE_PADDED)
    M,  # (num_queries_per_kv_padded,) - running max
    L,  # (num_queries_per_kv_padded,) - running sum
    acc,  # (num_queries_per_kv_padded, HEAD_SIZE_PADDED) - accumulator
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    block_table_offset,
    seq_len,
    kv_head_idx,
    alibi_slope,  # (num_queries_per_kv_padded,) or None
    head_mask,  # (num_queries_per_kv_padded,)
    dim_mask,  # (HEAD_SIZE_PADDED,)
    scale,
    k_scale,
    v_scale,
    block_start,  # Starting block index
    block_end,  # Ending block index (exclusive)
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    x: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_k_cache_4: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
):
    """Core attention computation loop shared by 2D and 3D kernels.

    Iterates over KV cache blocks from block_start to block_end,
    computing online softmax and accumulating weighted values.

    Returns updated (M, L, acc) tensors.
    """
    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    context_len = seq_len - 1

    for j in range(block_start, block_end):
        start_n = j * BLOCK_SIZE
        # Calculate the logical location within a non-standard physical block,
        # such as 544 in Qwen/Qwen3-Next-80B-A3B-Thinking.
        # Supports non-contiguous mapping
        # from logical blocks to physical blocks
        abs_token_idx = start_n + offs_n
        l_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        # Vectorized loading of physical block IDs
        p_block_idx = tl.load(block_tables_ptr + block_table_offset + l_block_idx)
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE

        # 5D addressing logic of K
        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )

        # 4D addressing logic of V (Slot is innermost)
        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )

        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (BLOCK_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        # First calculate the dot, then apply the mask.
        qk = scale * tl.dot(Q, K)
        S = tl.where(head_mask[:, None] & seq_mask, qk, float("-inf"))

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S, -10000)

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        # m_j : (num_queries_per_kv,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # P : (num_queries_per_kv, BLOCK_SIZE,)
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)

        # l_j : (num_queries_per_kv,)
        l_j = tl.sum(p, axis=1)

        # alpha : (num_queries_per_kv, )
        alpha = tl.exp(M - m_j)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc += tl.dot(p.to(V.dtype), V)

    return M, L, acc


@triton.jit
def kernel_paged_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size_padded]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    num_queries_per_kv_padded: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    PHYSICAL_BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    x: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.int64,  # int
    stride_k_cache_4: tl.int64,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.int64,  # int
    filter_by_query_len: tl.constexpr,  # bool
    query_start_len_ptr,  # [num_seqs+1]
    USE_SINKS: tl.constexpr,  # bool
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
):
    """3D paged attention kernel with parallel softmax segments.

    Uses HND cache layout like kernel_paged_attention_2d, but adds a third
    dimension for parallel softmax computation across sequence segments.
    This improves GPU utilization for long-context decode.

    Grid: (num_seqs, num_kv_heads, num_segments)
    """
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    # Sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # Calculate tiles per segment
    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
    tiles_per_segment = cdiv_fn(num_blocks, NUM_SEGMENTS_PER_SEQ)

    # Early exit if this segment has no work
    if segm_idx * tiles_per_segment >= num_blocks:
        return

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )

    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # Q : (num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    # Initialize accumulators for this segment
    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_head_idx,
                mask=head_mask,
                other=float("-inf"),
            ).to(dtype=tl.float32)
            L = tl.where(float("-inf") < M, 1.0, 0.0)
        else:
            M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
            L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    else:
        M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
        L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)

    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    # Calculate the range of blocks for this segment
    block_start = segm_idx * tiles_per_segment
    block_end = tl.minimum((segm_idx + 1) * tiles_per_segment, num_blocks)

    # Alibi slope placeholder (will be set if USE_ALIBI_SLOPES)
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_head_idx, mask=head_mask, other=0.0
        )
    else:
        alibi_slope = None

    # Call shared inner loop for attention computation
    M, L, acc = paged_attention_inner_loop(
        Q=Q,
        M=M,
        L=L,
        acc=acc,
        key_cache_ptr=key_cache_ptr,
        value_cache_ptr=value_cache_ptr,
        block_tables_ptr=block_tables_ptr,
        block_table_offset=block_table_offset,
        seq_len=seq_len,
        kv_head_idx=kv_head_idx,
        alibi_slope=alibi_slope,
        head_mask=head_mask,
        dim_mask=dim_mask,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
        block_start=block_start,
        block_end=block_end,
        BLOCK_SIZE=BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE=PHYSICAL_BLOCK_SIZE,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        USE_ALIBI_SLOPES=USE_ALIBI_SLOPES,
        SLIDING_WINDOW=SLIDING_WINDOW,
        x=x,
        stride_k_cache_0=stride_k_cache_0,
        stride_k_cache_1=stride_k_cache_1,
        stride_k_cache_2=stride_k_cache_2,
        stride_k_cache_3=stride_k_cache_3,
        stride_k_cache_4=stride_k_cache_4,
        stride_v_cache_0=stride_v_cache_0,
        stride_v_cache_1=stride_v_cache_1,
        stride_v_cache_2=stride_v_cache_2,
        stride_v_cache_3=stride_v_cache_3,
    )

    # Store segment results (unnormalized)
    # segm_output: [num_tokens, num_query_heads, num_segments, head_size_padded]
    segm_output_offset = (
        cur_batch_in_all_start_index
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )

    # segm_max/expsum: [num_tokens, num_query_heads, num_segments]
    segm_offset = (
        cur_batch_in_all_start_index * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=head_mask)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=head_mask)


@triton.jit
def reduce_paged_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    filter_by_query_len: tl.constexpr,  # bool
    query_start_len_ptr,  # [num_seqs+1]
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    """Reduce segment outputs to final attention output.

    Grid: (num_seqs, num_query_heads)
    """
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    # Sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # Calculate actual number of active segments
    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
    tiles_per_segment = cdiv_fn(num_blocks, NUM_SEGMENTS_PER_SEQ)
    act_num_segments = cdiv_fn(num_blocks, tiles_per_segment)

    # Create masks
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # Load segment maxima
    segm_offset = (
        cur_batch_in_all_start_index * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # Load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # Load, rescale, and add segment attention outputs
    segm_output_offset = (
        cur_batch_in_all_start_index
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)

    # Safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # Write result
    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


@triton.jit
def kernel_paged_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale_inv,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    num_queries_per_kv_padded: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    PHYSICAL_BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    x: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.int64,  # int
    stride_k_cache_4: tl.int64,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.int64,  # int
    filter_by_query_len: tl.constexpr,  # bool
    query_start_len_ptr,  # [num_seqs+1]
    USE_SINKS: tl.constexpr,  # bool
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )

    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # Q : (num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
        L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_head_idx,
            mask=head_mask,
            other=float("-inf"),
        ).to(dtype=tl.float32)
        L = tl.where(float("-inf") < M, 1.0, 0.0)

    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_head_idx, mask=head_mask, other=0.0
        )
    else:
        alibi_slope = None

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # Call shared inner loop for attention computation
    M, L, acc = paged_attention_inner_loop(
        Q=Q,
        M=M,
        L=L,
        acc=acc,
        key_cache_ptr=key_cache_ptr,
        value_cache_ptr=value_cache_ptr,
        block_tables_ptr=block_tables_ptr,
        block_table_offset=block_table_offset,
        seq_len=seq_len,
        kv_head_idx=kv_head_idx,
        alibi_slope=alibi_slope,
        head_mask=head_mask,
        dim_mask=dim_mask,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
        block_start=0,
        block_end=num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE=PHYSICAL_BLOCK_SIZE,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        USE_ALIBI_SLOPES=USE_ALIBI_SLOPES,
        SLIDING_WINDOW=SLIDING_WINDOW,
        x=x,
        stride_k_cache_0=stride_k_cache_0,
        stride_k_cache_1=stride_k_cache_1,
        stride_k_cache_2=stride_k_cache_2,
        stride_k_cache_3=stride_k_cache_3,
        stride_k_cache_4=stride_k_cache_4,
        stride_v_cache_0=stride_v_cache_0,
        stride_v_cache_1=stride_v_cache_1,
        stride_v_cache_2=stride_v_cache_2,
        stride_v_cache_3=stride_v_cache_3,
    )

    # epilogue
    acc = acc / (L[:, None] + 1e-10)
    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
    )

    tl.store(
        output_ptr + output_offset[:, None] + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )


def chunked_prefill_paged_decode(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    output_scale=None,
    # Optional tensor for sinks
    sinks=None,
    is_block_table_ptr: bool = False,
    # Parameters for 3D kernel (parallel softmax)
    num_par_softmax_segments: int = 16,
    # If None, buffers will be allocated automatically when needed
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
):
    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[2] ** 0.5)

    use_alibi_slopes = alibi_slopes is not None

    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if max_query_len > 1:
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            kv_cache_dtype=kv_cache_dtype,
            k_cache=key_cache,
            v_cache=value_cache,
            b_loc=block_table,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_seq_len=max_seq_len,
            max_input_len=max_query_len,
            k_scale=k_scale,
            v_scale=v_scale,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            skip_decode=True,
            fp8_out_scale=output_scale,
            sinks=sinks,
        )

    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    # key may be None in cross-attention decode (already cached from encoder)
    num_kv_heads = key.shape[1] if key is not None else key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = query.shape[2]

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert key_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]
        assert value_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = current_platform.fp8_dtype()
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError(
                f"Unsupported FP8 kv_cache_dtype {kv_cache_dtype}: "
                f"should be one of 'fp8', 'fp8_e4m3', 'fp8_e5m2'."
            )

        key_cache = key_cache.view(target_dtype)
        value_cache = value_cache.view(target_dtype)

    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)

    from vllm.platforms.rocm import use_rocm_custom_paged_attention

    use_custom = use_rocm_custom_paged_attention(
        query.dtype,
        head_size,
        block_size,
        num_queries_per_kv,
        max_seq_len,
        sliding_window,
        kv_cache_dtype,
        alibi_slopes,
        sinks,
    )
    # Triton is only forced when encountering a non-standard block
    # like Qwen3 with a size of 544.
    # 1. Check if block_size is a power of 2 (16, 32, 64...)
    # 2. If it's a power of 2, we trust the vLLM's native use_custom decision.
    # 3. If it's not a power of 2 (such as Qwen3's 544),
    # then our Triton path is forced.
    is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)
    if not is_pow2:
        use_custom = False

    if use_custom:
        _PARTITION_SIZE_ROCM = 256
        max_num_partitions = (
            max_seq_len + _PARTITION_SIZE_ROCM - 1
        ) // _PARTITION_SIZE_ROCM
        assert _PARTITION_SIZE_ROCM % block_size == 0
        total_num_seq = block_table.shape[0]
        tmp_output = torch.empty(
            size=(total_num_seq, num_query_heads, max_num_partitions, head_size),
            dtype=query.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(total_num_seq, num_query_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

        ops.paged_attention_rocm(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale=sm_scale,
            block_tables=block_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_size=block_size,
            max_seq_len=max_seq_len,
            alibi_slopes=alibi_slopes,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            fp8_out_scale=output_scale,
        )
    else:
        real_block_size = value_cache.shape[3]
        # The standard model directly uses the original block_size.
        # Non-standard 544 uses 32 to accommodate integer division logic.
        TRITON_BLOCK_SIZE = block_size if is_pow2 else 32
        if is_block_table_ptr:
            # Using the physical base address of tensors
            kv_element_size = key_cache.element_size()
            block_byte_stride = key_cache.stride(0) * kv_element_size
            # Get the starting physical address of the KV Cache
            base_addr = key_cache.data_ptr()

            # Normalization: Directly calculate the block offset
            # of the pointer relative to the base address
            processed_block_table = ((block_table - base_addr) // block_byte_stride).to(
                torch.int32
            )
        else:
            processed_block_table = block_table.to(torch.int32)

        SEQLEN_THRESHOLD_3D = 16
        use_3d_kernel = max_query_len == 1 and max_seq_len > SEQLEN_THRESHOLD_3D

        if use_3d_kernel:
            # Auto-allocate segment buffers if not provided
            head_size_padded = triton.next_power_of_2(head_size)
            num_tokens = query.shape[0]

            if softmax_segm_output is None:
                softmax_segm_output = torch.empty(
                    (
                        num_tokens,
                        num_query_heads,
                        num_par_softmax_segments,
                        head_size_padded,
                    ),
                    dtype=torch.float32,
                    device=query.device,
                )
            if softmax_segm_max is None:
                softmax_segm_max = torch.empty(
                    (num_tokens, num_query_heads, num_par_softmax_segments),
                    dtype=torch.float32,
                    device=query.device,
                )
            if softmax_segm_expsum is None:
                softmax_segm_expsum = torch.empty(
                    (num_tokens, num_query_heads, num_par_softmax_segments),
                    dtype=torch.float32,
                    device=query.device,
                )

            # 3D kernel with parallel softmax segments for better GPU utilization
            kernel_paged_attention_3d[
                (
                    num_seqs,
                    num_kv_heads,
                    num_par_softmax_segments,
                )
            ](
                segm_output_ptr=softmax_segm_output,
                segm_max_ptr=softmax_segm_max,
                segm_expsum_ptr=softmax_segm_expsum,
                query_ptr=query,
                key_cache_ptr=key_cache,
                value_cache_ptr=value_cache,
                sink_ptr=sinks,
                block_tables_ptr=processed_block_table,
                seq_lens_ptr=seq_lens,
                alibi_slopes_ptr=alibi_slopes,
                scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                num_queries_per_kv_padded=num_queries_per_kv_padded,
                block_table_stride=processed_block_table.stride(0),
                query_stride_0=query.stride(0),
                query_stride_1=query.stride(1),
                BLOCK_SIZE=TRITON_BLOCK_SIZE,
                PHYSICAL_BLOCK_SIZE=real_block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                SLIDING_WINDOW=sliding_window,
                x=key_cache.shape[4],
                stride_k_cache_0=key_cache.stride(0),
                stride_k_cache_1=key_cache.stride(1),
                stride_k_cache_2=key_cache.stride(2),
                stride_k_cache_3=key_cache.stride(3),
                stride_k_cache_4=key_cache.stride(4),
                stride_v_cache_0=value_cache.stride(0),
                stride_v_cache_1=value_cache.stride(1),
                stride_v_cache_2=value_cache.stride(2),
                stride_v_cache_3=value_cache.stride(3),
                filter_by_query_len=True,
                query_start_len_ptr=query_start_loc,
                USE_SINKS=sinks is not None,
                NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            )
            # Reduce segment outputs to final attention output
            reduce_paged_segments[
                (
                    num_seqs,
                    num_query_heads,
                )
            ](
                output_ptr=output,
                segm_output_ptr=softmax_segm_output,
                segm_max_ptr=softmax_segm_max,
                segm_expsum_ptr=softmax_segm_expsum,
                seq_lens_ptr=seq_lens,
                num_seqs=num_seqs,
                num_query_heads=num_query_heads,
                out_scale_inv=1.0 / output_scale if output_scale is not None else 1.0,
                output_stride_0=output.stride(0),
                output_stride_1=output.stride(1),
                BLOCK_SIZE=TRITON_BLOCK_SIZE,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                filter_by_query_len=True,
                query_start_len_ptr=query_start_loc,
                NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
                USE_FP8=output_scale is not None,
            )
        else:
            # Standard 2D kernel
            kernel_paged_attention_2d[
                (
                    num_seqs,
                    num_kv_heads,
                )
            ](
                output_ptr=output,
                query_ptr=query,
                key_cache_ptr=key_cache,
                value_cache_ptr=value_cache,
                sink_ptr=sinks,
                block_tables_ptr=processed_block_table,
                seq_lens_ptr=seq_lens,
                alibi_slopes_ptr=alibi_slopes,
                scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                out_scale_inv=1.0 / output_scale if output_scale is not None else 1.0,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                num_queries_per_kv_padded=num_queries_per_kv_padded,
                block_table_stride=processed_block_table.stride(0),
                query_stride_0=query.stride(0),
                query_stride_1=query.stride(1),
                output_stride_0=output.stride(0),
                output_stride_1=output.stride(1),
                BLOCK_SIZE=TRITON_BLOCK_SIZE,
                PHYSICAL_BLOCK_SIZE=real_block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                SLIDING_WINDOW=sliding_window,
                x=key_cache.shape[4],
                stride_k_cache_0=key_cache.stride(0),
                stride_k_cache_1=key_cache.stride(1),
                stride_k_cache_2=key_cache.stride(2),
                stride_k_cache_3=key_cache.stride(3),
                stride_k_cache_4=key_cache.stride(4),
                stride_v_cache_0=value_cache.stride(0),
                stride_v_cache_1=value_cache.stride(1),
                stride_v_cache_2=value_cache.stride(2),
                stride_v_cache_3=value_cache.stride(3),
                filter_by_query_len=True,
                query_start_len_ptr=query_start_loc,
                USE_SINKS=sinks is not None,
                USE_FP8=output_scale is not None,
            )
