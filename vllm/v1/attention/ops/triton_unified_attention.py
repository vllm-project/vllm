# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
is_batch_invariant = vllm_is_batch_invariant()
float8_info = torch.finfo(current_platform.fp8_dtype())


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
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
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    q_block_offset,  # int
    decode_only: tl.constexpr,
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    q_block_global_idx = tl.program_id(0) + q_block_offset
    kv_head_idx = tl.program_id(1)

    if not decode_only:
        seq_idx = find_seq_idx(
            query_start_len_ptr,
            q_block_global_idx,
            num_seqs + q_block_offset,
            BLOCK_Q,
            True,
        )
        q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    else:
        seq_idx = q_block_global_idx
        q_block_start_idx = seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
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

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    if USE_MM_PREFIX:
        # image bidirectional attention ranges require a full range
        # including q_block padding to make sure doc mask is correct
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        # adjust for potential padding in the last q_block by considering the
        # actual sequence length
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
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

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.jit
def kernel_unified_attention_3d(
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
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    q_block_offset,  # int
    decode_only: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
):
    q_block_global_idx = tl.program_id(0) + q_block_offset
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    if not decode_only:
        seq_idx = find_seq_idx(
            query_start_len_ptr,
            q_block_global_idx,
            num_seqs + q_block_offset,
            BLOCK_Q,
            True,
        )
        q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    else:
        seq_idx = q_block_global_idx
        q_block_start_idx = seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

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

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(
        max(segm_idx * tiles_per_segment, tile_start),
        min((segm_idx + 1) * tiles_per_segment, tile_end),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
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

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    query_token_idx_offset,  # int
    decode_only: tl.constexpr,
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0) + query_token_idx_offset
    query_head_idx = tl.program_id(1)

    if not decode_only:
        seq_idx = find_seq_idx(
            query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
        )
    else:
        seq_idx = query_token_idx

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
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
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
    use_2d_kernel: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization.

    For Gemma3, use 32 for both prefill and decode to better utilize
    the larger head dimension (128/256). For other models, use
    the default vLLM behavior.
    """
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill or use_2d_kernel:
        return 32
    return 16 if element_size >= 2 else 32


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    num_prefills=None,
    num_decodes=None,
    seq_threshold_3D=None,
    split_launch=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    # Assign the following variables if they are not assigned in the attention metadata.
    # This ensures backward compatibility with callers using an earlier version of this
    # function. However, it is recommended to include these assignments in the
    # attention metadata itself, as performing them here may negatively impact
    # performance.
    if (
        seq_threshold_3D is None
        or split_launch is None
        or num_prefills is None
        or num_decodes is None
    ):
        MIN_LAUNCH_GRID_SIZE_2D = 128
        seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads
        split_launch = False
        seq_lens_q = torch.diff(cu_seqlens_q)
        num_prefills = (seq_lens_q > 1).sum().item()
        num_decodes = (seq_lens_q == 1).sum().item()

    # Assigning Q Block dimensions for prefill and decode.
    BLOCK_M_2D_PREFILL = (
        64 if num_queries_per_kv <= 64 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_M_2D_DECODE = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_M_3D_DECODE = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q_2D_PREFILL = BLOCK_M_2D_PREFILL // num_queries_per_kv
    BLOCK_Q_2D_DECODE = BLOCK_M_2D_DECODE // num_queries_per_kv
    BLOCK_Q_3D_DECODE = BLOCK_M_3D_DECODE // num_queries_per_kv

    # Tile sizes for prefill and decode. Gemma3 models use optimized values.
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    TILE_SIZE_2D_PREFILL = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=True,
        use_2d_kernel=True,
    )
    TILE_SIZE_2D_DECODE = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=False,
        use_2d_kernel=True,
    )
    TILE_SIZE_3D_DECODE = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=False,
        use_2d_kernel=False,
    )

    # Launch the 2D kernel if batch contains a prefill
    if num_prefills > 0:
        # batch contains prefills

        # Ideally we would launch with kernel with:
        # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
        # However, it is slow to realize the query_lens on cpu.
        # Instead we use upper-bound:
        # \sum_i[ceil(query_len[i] / BLOCK_Q)]
        #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
        #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
        #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
        #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
        total_num_q_blocks = (
            (q.shape[0] - num_decodes) // BLOCK_Q_2D_PREFILL + num_seqs - num_decodes
            if split_launch
            else q.shape[0] // BLOCK_Q_2D_PREFILL + num_seqs
        )

        kernel_unified_attention_2d[
            (
                total_num_q_blocks,
                num_kv_heads,
            )
        ](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_2D_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q_2D_PREFILL,
            num_seqs=num_seqs - num_decodes if split_launch else num_seqs,
            BLOCK_M=BLOCK_M_2D_PREFILL,
            q_block_offset=num_decodes if split_launch else 0,
            decode_only=False,
            USE_FP8=output_scale is not None,
        )

    if (num_decodes > 0) and ((num_prefills == 0) or split_launch):
        # Batch contains decodes that are not processed in unified fashion

        # Launch the 2D kernel if
        # 1. No intermediate tiled softmax buffers have been allocated, or
        # 2. The number of sequences exceeds the configured threshold
        # 3. Batch invariance is enabled
        if (
            seq_threshold_3D is None
            or num_par_softmax_segments is None
            or softmax_segm_output is None
            or softmax_segm_max is None
            or softmax_segm_expsum is None
            or num_decodes > seq_threshold_3D
            or is_batch_invariant
        ):
            kernel_unified_attention_2d[
                (
                    num_decodes,
                    num_kv_heads,
                )
            ](
                output_ptr=out,
                query_ptr=q,
                key_cache_ptr=k,
                value_cache_ptr=v,
                sink_ptr=sinks,
                block_tables_ptr=block_table,
                seq_lens_ptr=seqused_k,
                alibi_slopes_ptr=alibi_slopes,
                qq_bias_ptr=qq_bias,
                scale=softmax_scale,
                k_scale=k_descale,
                v_scale=v_descale,
                out_scale=1 / output_scale if output_scale is not None else 1.0,
                softcap=softcap,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                block_table_stride=block_table.stride(0),
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                output_stride_0=out.stride(0),
                output_stride_1=out.stride(1),
                qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
                BLOCK_SIZE=block_size,
                TILE_SIZE=TILE_SIZE_2D_DECODE,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                USE_ALIBI_SQRT=use_alibi_sqrt,
                USE_QQ_BIAS=use_qq_bias,
                USE_SOFTCAP=(softcap > 0),
                USE_SINKS=(sinks is not None),
                USE_MM_PREFIX=use_mm_prefix,
                MAX_MM_RANGES=max_mm_ranges,
                mm_prefix_range_ptr=mm_prefix_range,
                SLIDING_WINDOW=(1 + window_size[0]),
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                query_start_len_ptr=cu_seqlens_q,
                BLOCK_Q=BLOCK_Q_2D_DECODE,
                num_seqs=num_decodes,
                BLOCK_M=BLOCK_M_2D_DECODE,
                q_block_offset=0,
                decode_only=True,
                USE_FP8=output_scale is not None,
            )
        else:
            kernel_unified_attention_3d[
                (num_decodes, num_kv_heads, num_par_softmax_segments)
            ](
                segm_output_ptr=softmax_segm_output,
                segm_max_ptr=softmax_segm_max,
                segm_expsum_ptr=softmax_segm_expsum,
                query_ptr=q,
                key_cache_ptr=k,
                value_cache_ptr=v,
                sink_ptr=sinks,
                block_tables_ptr=block_table,
                seq_lens_ptr=seqused_k,
                alibi_slopes_ptr=alibi_slopes,
                qq_bias_ptr=qq_bias,
                scale=softmax_scale,
                k_scale=k_descale,
                v_scale=v_descale,
                softcap=softcap,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                block_table_stride=block_table.stride(0),
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
                BLOCK_SIZE=block_size,
                TILE_SIZE=TILE_SIZE_3D_DECODE,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                USE_ALIBI_SQRT=use_alibi_sqrt,
                USE_QQ_BIAS=use_qq_bias,
                USE_SOFTCAP=(softcap > 0),
                USE_SINKS=(sinks is not None),
                USE_MM_PREFIX=use_mm_prefix,
                MAX_MM_RANGES=max_mm_ranges,
                mm_prefix_range_ptr=mm_prefix_range,
                SLIDING_WINDOW=(1 + window_size[0]),
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                query_start_len_ptr=cu_seqlens_q,
                BLOCK_Q=BLOCK_Q_3D_DECODE,
                num_seqs=num_decodes,
                BLOCK_M=BLOCK_M_3D_DECODE,
                NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
                q_block_offset=0,
                decode_only=True,
            )

            reduce_segments[(num_decodes, num_query_heads)](
                output_ptr=out,
                segm_output_ptr=softmax_segm_output,
                segm_max_ptr=softmax_segm_max,
                segm_expsum_ptr=softmax_segm_expsum,
                seq_lens_ptr=seqused_k,
                num_seqs=num_decodes,
                num_query_heads=num_query_heads,
                out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
                output_stride_0=out.stride(0),
                output_stride_1=out.stride(1),
                block_table_stride=block_table.stride(0),
                TILE_SIZE=TILE_SIZE_3D_DECODE,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                query_start_len_ptr=cu_seqlens_q,
                BLOCK_Q=BLOCK_Q_3D_DECODE,
                NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
                query_token_idx_offset=0,
                decode_only=True,
                USE_FP8=output_scale is not None,
            )
