# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
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
def find_seq_idx(query_start_len_ptr, target_idx, num_seqs,
                 BLOCK_Q: tl.constexpr, use_q_block_mode: tl.constexpr):
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

# This function is borrowed from Aiter
@triton.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N,
    BLOCK_SIZE_M,
    MXFP4_QUANT_BLOCK_SIZE,
):
    """
    Converts given x (in fp32) to mxfp4 format.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32

    """
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    # Calculate scale
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127  # in fp32, we have 2&(e - 127)

    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x * quant_scale

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign, exponents and mantissa fields from FP32
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF
    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1

    # Denormal numbers
    # If exponent is less than 127, then it's a denormal number
    # See above, for denormal number mantissa is always 1 and we set bit 1 of mantissa
    adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
    m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
    # For normal numbers, bias is changed from 127 to 1, and for subnormals, we keep exponent as 0.
    # Note: E8_BIAS - E2_BIAS = 126, so for normals we subtract that.
    e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    # Combine sign, exponent, and mantissa, while saturating
    # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
    e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
    e2m1_value = ((s >> 28) | e2m1_tmp).to(tl.uint8)
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

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
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    use_native_fp4: tl.constexpr = True,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs,
                           BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr +
                                seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv
    query_offset = (query_offset_0[:, None] * query_stride_0 +
                    query_offset_1[:, None] * query_stride_1 + offs_d[None, :])

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    if use_native_fp4:
       q_fp4, scale_q = _mxfp4_quant_op(Q, HEAD_SIZE_PADDED, BLOCK_M, 32)

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
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                              mask=query_mask_1,
                              other=0.0)

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
                            )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (
        BLOCK_M - 1) // num_queries_per_kv + 1

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
    if SLIDING_WINDOW > 0:
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

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset +
                                     seq_offset // BLOCK_SIZE).to(tl.int64)

        v_offset = (physical_block_idx[:, None] * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_2 +
                    offs_d[None, :] * stride_v_cache_3 +
                    (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1)

        k_offset = (physical_block_idx[None, :] * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_2 +
                    offs_d[:, None] * stride_k_cache_3 +
                    (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1)

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(key_cache_ptr + k_offset,
                         mask=dim_mask[:, None] & tile_mask[None, :],
                         other=0.0)

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(value_cache_ptr + v_offset,
                         mask=dim_mask[None, :] & tile_mask[:, None],
                         other=0.0)

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        if use_native_fp4:
           kt = tl.trans(K)
           k_fp4, scale_k = _mxfp4_quant_op(kt, HEAD_SIZE_PADDED, TILE_SIZE, 32)

           # debug info
           pid0 = tl.program_id(axis=0)
           pid1 = tl.program_id(axis=1)
           if pid0 < -1 and pid1 < 1 and j == 0:
              tl.device_print("k_fp4:",k_fp4.shape[0], k_fp4.shape[1])

           # copied from triton def dot_scaled()
           # M, K_LHS = lhs.type.shape[-2:]
           # K_RHS, N = rhs.type.shape[-2:]
           S = scale * tl.dot_scaled(q_fp4, scale_p, "e2m1", k_fp4, scale_v, "e2m1", S, lhs_k_pack=PACK_ALONG_K,
                                   rhs_k_pack=PACK_ALONG_K, out_dtype=tl.float32)
        else:
           S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
                     S, float("-inf"))

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len + query_pos[:, None] - seq_offset)
                         < SLIDING_WINDOW, S, float("-inf"))

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

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

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        if use_native_fp4:
           p_fp4, scale_p = _mxfp4_quant_op(P, BLOCK_SIZE, BLOCK_M, 32)

           vt = tl.trans(V)
           v_fp4, scale_v = _mxfp4_quant_op(vt, BLOCK_SIZE, HEAD_SIZE_PADDED, 32)

           v_fp4_t   =  tl.trans(v_fp4)
           scale_v_t =  tl.trans(scale_v)

           # debug info
           pid0 = tl.program_id(axis=0)
           pid1 = tl.program_id(axis=1)
           if pid0 < -1 and pid1 < 1 and j == 0:
              tl.device_print("v_fp4:",v_fp4.shape[0], v_fp4.shape[1])

           # copied from triton def dot_scaled()
           # M, K_LHS = lhs.type.shape[-2:]
           # K_RHS, N = rhs.type.shape[-2:]
           acc = tl.dot_scaled(p_fp4, scale_p, "e2m1", v_fp4_t, scale_v_t, "e2m1", acc, lhs_k_pack=PACK_ALONG_K,
                                   rhs_k_pack=PACK_ALONG_K, out_dtype=tl.float32)
        else:
           acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (query_offset_0[:, None] * output_stride_0 +
                     query_offset_1[:, None] * output_stride_1 +
                     offs_d[None, :])

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.jit
def kernel_unified_attention_3d(
        segm_output_ptr,
        # [num_tokens, num_query_heads, num_segments, head_size]
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
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs,
                           BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr +
                                seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

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
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv
    query_offset = (query_offset_0[:, None] * query_stride_0 +
                    query_offset_1[:, None] * query_stride_1 + offs_d[None, :])

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
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                              mask=query_mask_1,
                              other=0.0)

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
                            )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (
        BLOCK_M - 1) // num_queries_per_kv + 1

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # iterate through tiles within current segment
    for j in range(
            segm_idx * tiles_per_segment,
            min((segm_idx + 1) * tiles_per_segment, num_tiles),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset +
                                     seq_offset // BLOCK_SIZE).to(tl.int64)

        v_offset = (physical_block_idx[:, None] * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_2 +
                    offs_d[None, :] * stride_v_cache_3 +
                    (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1)

        k_offset = (physical_block_idx[None, :] * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_2 +
                    offs_d[:, None] * stride_k_cache_3 +
                    (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1)

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(key_cache_ptr + k_offset,
                         mask=dim_mask[:, None] & tile_mask[None, :],
                         other=0.0)

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(value_cache_ptr + v_offset,
                         mask=dim_mask[None, :] & tile_mask[:, None],
                         other=0.0)

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
                     S, float("-inf"))

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len + query_pos[:, None] - seq_offset)
                         < SLIDING_WINDOW, S, float("-inf"))

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

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

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64) *
        (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        segm_idx * HEAD_SIZE_PADDED + tl.arange(0, HEAD_SIZE_PADDED)[None, :])
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (query_offset_0.to(tl.int64) *
                   (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
                   query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx)
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset,
             L,
             mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    #[num_tokens, num_query_heads, max_num_segments, head_size]
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
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(query_start_len_ptr, query_token_idx, num_seqs,
                           BLOCK_Q, False)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32)
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                        0).to(tl.int1)

    # load segment maxima
    segm_offset = (query_token_idx.to(tl.int64) *
                   (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
                   query_head_idx * NUM_SEGMENTS_PER_SEQ +
                   tl.arange(0, NUM_SEGMENTS_PER_SEQ))
    segm_max = tl.load(segm_max_ptr + segm_offset,
                       mask=segm_mask,
                       other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset,
                          mask=segm_mask,
                          other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64) *
        (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED +
        tl.arange(0, HEAD_SIZE_PADDED)[None, :])
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
    output_offset = (query_token_idx * output_stride_0 +
                     query_head_idx * output_stride_1 +
                     tl.arange(0, HEAD_SIZE_PADDED))
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


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
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
):

    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], \
        "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(
        num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    print("BLOK_M: ", num_queries_per_kv, BLOCK_M)
    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Assigning default tile sizes for prefill and decode.
    # Note: each tile size must be at least 32 for "fp8" (q.element_size() == 1)
    # and at least 16 for all other data types.
    TILE_SIZE_PREFILL = 32
    TILE_SIZE_DECODE = 16 if q.element_size() >= 2 else 32

    # if batch contains a prefill
    if max_seqlen_q > 1 or total_num_q_blocks * num_kv_heads > 128:
        kernel_unified_attention_2d[(
            total_num_q_blocks,
            num_kv_heads,
        )](
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
            TILE_SIZE=TILE_SIZE_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
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
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
        )
    else:
        # for initial version, NUM_SEGMENTS = 16 is chosen as a default
        # value that showed good performance in tests
        NUM_SEGMENTS = 16

        segm_output = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            triton.next_power_of_2(head_size),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )

        kernel_unified_attention_3d[(
            total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
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
                TILE_SIZE=TILE_SIZE_DECODE,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                USE_QQ_BIAS=use_qq_bias,
                USE_SOFTCAP=(softcap > 0),
                USE_SINKS=(sinks is not None),
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
                BLOCK_Q=BLOCK_Q,
                num_seqs=num_seqs,
                BLOCK_M=BLOCK_M,
                NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
            )
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 /
            output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
            USE_FP8=output_scale is not None,
        )
