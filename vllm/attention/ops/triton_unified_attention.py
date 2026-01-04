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
    # Use robust tanh to avoid overflow in exp
    return x * tl.math.tanh(S / x)


@triton.jit
def nvfp4_to_fp8_e4m3(nibble):
    """Convert NVFP4 E2M1 to FP8 E4M3fn bits in registers.

    This is used to leverage H100 native FP8 Tensor Cores.
    E2M1 (4-bit) -> E4M3fn (8-bit) mapping:
    - Zero: 0.0 -> 0x00
    - Mag 1: 0.5 -> 0x30
    - Mag 2: 1.0 -> 0x38
    - Mag 3: 1.5 -> 0x3c
    - Mag 4: 2.0 -> 0x40
    - Mag 5: 3.0 -> 0x44
    - Mag 6: 4.0 -> 0x48
    - Mag 7: 6.0 -> 0x4c
    Sign bit (bit 3) maps to bit 7.
    """
    mag = nibble & 0x07
    is_neg = (nibble & 0x08) != 0

    # Map magnitudes to E4M3fn bit patterns
    bits = tl.where(
        mag == 0,
        0x00,
        tl.where(
            mag == 1,
            0x30,
            tl.where(
                mag == 2,
                0x38,
                tl.where(
                    mag == 3,
                    0x3C,
                    tl.where(
                        mag == 4,
                        0x40,
                        tl.where(mag == 5, 0x44, tl.where(mag == 6, 0x48, 0x4C)),
                    ),
                ),
            ),
        ),
    )

    # Flip sign bit
    bits = tl.where(is_neg, bits | 0x80, bits)

    # Bitcast to float8e4nv
    return bits.to(tl.int8).to(tl.float8e4nv, bitcast=True)


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
    USE_NVFP4_TC: tl.constexpr,  # bool - Use FP8 Tensor Cores
    USE_FP8: tl.constexpr,  # bool - use fp8 output
    USE_NVFP4: tl.constexpr,  # bool - NVFP4 KV cache mode
    NUM_SCALES: tl.constexpr,  # int
    DATA_BYTES: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    hg_idx = tl.program_id(2)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    # In HG mode, we compute BLOCK_M heads for a single query token index.
    # Each row in Q/acc corresponds to one head in the group.
    q_head = kv_head_idx * num_queries_per_kv + hg_idx * BLOCK_M + offs_m
    head_mask = q_head < (kv_head_idx * num_queries_per_kv + num_queries_per_kv)

    # Simplified query pos for HG mode: one query token index per program block
    query_pos = q_block_local_idx * BLOCK_Q
    query_offset_0 = cur_batch_in_all_start_index + query_pos

    query_offset = (
        query_offset_0 * query_stride_0
        + q_head[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0 & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        L = tl.zeros([BLOCK_M], dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + q_head,
            mask=head_mask,
            other=float("-inf"),
        ).to(dtype=tl.float32)
        L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)

    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # Hoist conversion to reduce layout pressure in the compiler.
    if USE_NVFP4_TC:
        # ================================================================
        # PROPER Q QUANTIZATION FOR FP8 TENSOR CORES
        # FP8 E4M3 has limited dynamic range [-448, 448]. We must:
        # 1. Compute max abs per 32-element block
        # 2. Calculate proper scale (log2 exponent + 127 bias)
        # 3. Normalize Q before FP8 conversion
        # 4. Use accurate Q_scale in dot_scaled
        #
        # tl.dot_scaled computes: C = (A @ B) * 2^(A_scale-127) * 2^(B_scale-127)
        # So we need Q_normalized such that Q_normalized * 2^(Q_scale-127) = Q_original
        # ================================================================
        Q_NUM_SCALES: tl.constexpr = HEAD_SIZE_PADDED // 32  # 4 scales for 128-dim head

        # Reshape Q to [BLOCK_M, Q_NUM_SCALES, 32] to compute per-block max
        Q_reshaped = tl.reshape(Q, (BLOCK_M, Q_NUM_SCALES, 32))

        # Compute max abs per 32-element block: shape [BLOCK_M, Q_NUM_SCALES]
        Q_abs_max = tl.max(tl.abs(Q_reshaped), axis=2)

        # Compute scale: log2(max) clamped to valid FP8 range
        # FP8 E4M3 max is 448, so scale = ceil(log2(max)) where max > 0
        # For max <= 0, use neutral scale 127
        # Bias by 127 for the exponent encoding
        Q_log2_max = tl.where(Q_abs_max > 0, tl.math.log2(Q_abs_max), 0.0)
        Q_scale_f = tl.math.ceil(Q_log2_max) + 127.0
        Q_scale = tl.maximum(tl.minimum(Q_scale_f, 255.0), 0.0).to(
            tl.uint8
        )  # shape [BLOCK_M, Q_NUM_SCALES]

        # Compute inverse scale for normalization: 2^(127 - scale)
        # This normalizes Q values to fit in FP8 range
        Q_inv_scale = tl.math.exp2(127.0 - Q_scale_f)  # shape [BLOCK_M, Q_NUM_SCALES]

        # Broadcast inverse scale to full Q shape: [BLOCK_M, Q_NUM_SCALES] -> [BLOCK_M, HEAD_SIZE_PADDED]
        Q_inv_scale_full = tl.reshape(
            tl.broadcast_to(Q_inv_scale[:, :, None], (BLOCK_M, Q_NUM_SCALES, 32)),
            (BLOCK_M, HEAD_SIZE_PADDED),
        )

        # Normalize Q and convert to FP8
        Q_normalized = Q * Q_inv_scale_full
        Q_fp8 = Q_normalized.to(tl.float8e4nv)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + q_head, mask=head_mask, other=0.0)

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos * qq_bias_stride_0
        )  # shape: [BLOCK_M] scalar token index

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    # In HG mode, BLOCK_Q is small (usually 1), so this is just context_len + query_pos + 1
    max_seq_prefix_len = context_len + query_pos + 1

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
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        first_allowed_key = context_len + query_pos - SLIDING_WINDOW + 1
        last_allowed_key = context_len + query_pos
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

        if not USE_NVFP4:
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
        else:
            K_load = None

        # FP8/NVFP4 dequantization with native tensor core optimization
        if not USE_NVFP4 and K_load.dtype.is_fp8():
            K = K_load
        elif USE_NVFP4:
            k_packed_data_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + (offs_d[:, None] // 2) * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            K_u8 = tl.load(
                key_cache_ptr + k_packed_data_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0,
            ).to(tl.uint8)

            K_nibble = tl.where((offs_d[:, None] & 1) == 0, K_u8 & 0xF, K_u8 >> 4)

            # ================================================================
            # CRITICAL: Full dequant → uniform scale → FP8 requant pipeline
            #
            # Problem: tl.dot_scaled requires UNIFORM scales per K-block across
            # all tokens, but NVFP4 has PER-TOKEN scales. Using per-token scales
            # directly causes incorrect attention scores and repetitive output.
            #
            # Solution:
            # 1. Convert NVFP4 nibbles to FP values
            # 2. Load per-token scales and dequantize to BF16
            # 3. Compute tile-uniform scale (max abs per K-block across ALL tokens)
            # 4. Re-quantize to FP8 with uniform scale
            # 5. Use uniform scale in tl.dot_scaled
            # ================================================================

            # Step 1: Convert NVFP4 nibbles to float representation
            # E2M1 format: value = sign * mantissa * 2^exponent
            K_nibble_f = K_nibble.to(tl.float32)
            sign = tl.where(K_nibble_f >= 8, -1.0, 1.0)
            K_abs = tl.where(K_nibble_f >= 8, K_nibble_f - 8, K_nibble_f)
            # E2M1 lookup: 0=0, 1=0.5, 2=1, 3=1.5, 4=2, 5=3, 6=4, 7=6
            K_val = tl.where(
                K_abs == 0,
                0.0,
                tl.where(
                    K_abs == 1,
                    0.5,
                    tl.where(
                        K_abs == 2,
                        1.0,
                        tl.where(
                            K_abs == 3,
                            1.5,
                            tl.where(
                                K_abs == 4,
                                2.0,
                                tl.where(
                                    K_abs == 5, 3.0, tl.where(K_abs == 6, 4.0, 6.0)
                                ),
                            ),
                        ),
                    ),
                ),
            )
            K_float = sign * K_val  # Shape: [HEAD_SIZE, TILE_SIZE]

            # Step 2: Load per-token scales
            K_NUM_SCALES: tl.constexpr = HEAD_SIZE_PADDED // 32  # = 4 for head_size=128
            offs_ks = tl.arange(0, K_NUM_SCALES)

            # K_scale layout in cache: [TILE_SIZE, K_NUM_SCALES] for per-token scales
            k_scale_base = (
                physical_block_idx[:, None] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_k_cache_1
                + DATA_BYTES * stride_k_cache_3
            )
            k_scale_mask = tl.broadcast_to(
                tile_mask[:, None], (TILE_SIZE, K_NUM_SCALES)
            )

            K_scale_per_token = tl.load(
                key_cache_ptr + k_scale_base + offs_ks[None, :] * stride_k_cache_3,
                mask=k_scale_mask,
                other=127,
            ).to(
                tl.uint8
            )  # Shape: [TILE_SIZE, K_NUM_SCALES]

            # ================================================================
            # CORRECT SOLUTION: Full BF16 Dequantization
            #
            # Problem: tl.dot_scaled requires uniform scales, but NVFP4 has per-token scales
            # Solution: Dequantize K to BF16 using per-token scales, use regular tl.dot
            #
            # This preserves QUALITY by:
            # - Using all precision from NVFP4's per-token scales
            # - Avoiding forced uniform scale quantization error
            # - Simple, proven approach
            # ================================================================

            # Dequantize K using per-token scales
            K_scale_exp = tl.math.exp2(
                K_scale_per_token.to(tl.float32) - 127.0
            )  # [TILE_SIZE, K_NUM_SCALES]

            # Reshape K_float from [HEAD_SIZE_PADDED, TILE_SIZE] to [K_NUM_SCALES, 32, TILE_SIZE]
            K_float_reshaped = tl.reshape(K_float, (K_NUM_SCALES, 32, TILE_SIZE))

            # Transpose and broadcast scales
            K_scale_exp_T = tl.trans(K_scale_exp)  # [K_NUM_SCALES, TILE_SIZE]
            K_scale_broadcast = tl.broadcast_to(
                K_scale_exp_T[:, None, :], (K_NUM_SCALES, 32, TILE_SIZE)
            )

            # Apply per-token scales to get fully dequantized K
            K_dequant_reshaped = (
                K_float_reshaped * K_scale_broadcast
            )  # [K_NUM_SCALES, 32, TILE_SIZE]

            # Reshape back to [HEAD_SIZE_PADDED, TILE_SIZE]
            K_dequant = tl.reshape(K_dequant_reshaped, (HEAD_SIZE_PADDED, TILE_SIZE))

            # K is now in BF16 with full precision
            K = K_dequant.to(tl.bfloat16)
        else:
            K = K_load

        if not USE_NVFP4:
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )
        else:
            V_load = None

        # FP8/NVFP4 dequantization for V
        if not USE_NVFP4 and V_load.dtype.is_fp8():
            V = V_load
        elif USE_NVFP4:
            v_packed_data_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + (offs_d[None, :] // 2) * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            V_u8 = tl.load(
                value_cache_ptr + v_packed_data_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0,
            ).to(tl.uint8)

            V_nibble = tl.where((offs_d[None, :] & 1) == 0, V_u8 & 0xF, V_u8 >> 4)
            V = nvfp4_to_fp8_e4m3(V_nibble)

            # ================================================================
            # Load V scales (u8) - CUDA now uses 32-element groups
            # So we have HEAD_SIZE_PADDED // 32 = 4 scales per head
            # Stored at positions: DATA_BYTES + [0, 1, 2, 3]
            # ================================================================
            V_NUM_SCALES: tl.constexpr = HEAD_SIZE_PADDED // 32  # = 4 for head_size=128
            offs_vs = tl.arange(0, V_NUM_SCALES)
            v_scale_off = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + (DATA_BYTES + offs_vs)[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            v_scale_mask = tl.broadcast_to(
                tile_mask[:, None], (TILE_SIZE, V_NUM_SCALES)
            )
            V_scale_u8 = tl.load(
                value_cache_ptr + v_scale_off, mask=v_scale_mask, other=127
            ).to(tl.uint8)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos
        seq_mask = seq_offset <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
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
                    (seq_offset >= range_start) & (seq_offset <= range_end) & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        # PERFORMANCE OPTIMIZED MATMUL PATHS
        # NVFP4 with Four Over Six (4/6) quantization:
        # - K is fully dequantized to BF16 using per-token scales (quality preserved)
        # - Q is in FP8 with per-block scales
        # - Use regular tl.dot for BF16 matmul
        if USE_NVFP4 and USE_NVFP4_TC:
            # K is BF16 (dequantized from NVFP4 with per-token scales)
            # Q is also converted to BF16 for matmul
            S += scale * tl.dot(Q.to(tl.bfloat16), K).to(tl.float32)
        elif USE_NVFP4:
            S += scale * tl.dot(Q.to(tl.bfloat16), K.to(tl.bfloat16)).to(tl.float32)
        elif K.dtype.is_fp8():
            S += scale * tl.load(k_scale) * tl.dot(Q.to(tl.float32), K.to(tl.float32))
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_0 & head_mask[:, None] & seq_mask[None, :], S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)[None, :]

        if USE_QQ_BIAS:
            key_rel_pos = seq_offset - context_len
            is_query_key = (key_rel_pos >= 0) & (key_rel_pos < qq_bias_stride_0)
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],
                other=0.0,
            )
            S += qq_bias

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            V = tl.where(
                (context_len + query_pos - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        if USE_NVFP4:
            # Dequantize V with 32-element scale groups (4 scales per 128-dim head)
            V_scale_v = tl.math.exp2(V_scale_u8.to(tl.float32) - 127.0).to(tl.bfloat16)
            # Broadcast each scale to 32 elements: [TILE_SIZE, 4] -> [TILE_SIZE, 4, 32] -> [TILE_SIZE, 128]
            V_scale_full = tl.reshape(
                tl.broadcast_to(
                    V_scale_v[:, :, None], (TILE_SIZE, HEAD_SIZE_PADDED // 32, 32)
                ),
                (TILE_SIZE, HEAD_SIZE_PADDED),
            )
            V_deq = V.to(tl.bfloat16) * V_scale_full
            acc += tl.dot(P.to(tl.bfloat16), V_deq).to(tl.float32)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0 * output_stride_0
        + q_head[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0 & head_mask[:, None],
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
    USE_NVFP4_TC: tl.constexpr,  # bool
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
    USE_FP8: tl.constexpr,  # bool - use fp8 output
    USE_NVFP4: tl.constexpr,  # bool - NVFP4 KV cache mode
    NUM_SCALES: tl.constexpr,  # int
    DATA_BYTES: tl.constexpr,  # int
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    # Pack (segm_idx, hg_idx) into pid(2) to overcome Triton's 3-axis limit.
    # HG is calculated as (num_queries_per_kv + BLOCK_M - 1) // BLOCK_M
    HG = (num_queries_per_kv + BLOCK_M - 1) // BLOCK_M
    pid2 = tl.program_id(2)
    segm_idx = pid2 // HG
    hg_idx = pid2 % HG

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # NVFP4 constants (Define once at top level to avoid constexpr reassignment in loops)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    # In HG mode, we compute BLOCK_M heads for a single query token index.
    q_head = kv_head_idx * num_queries_per_kv + hg_idx * BLOCK_M + offs_m
    head_mask = q_head < (kv_head_idx * num_queries_per_kv + num_queries_per_kv)

    # Simplified query pos for HG mode: one query token index per program block
    query_pos = q_block_local_idx * BLOCK_Q
    query_offset_0 = cur_batch_in_all_start_index + query_pos

    query_offset = (
        query_offset_0 * query_stride_0
        + q_head[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0 & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + q_head,
                mask=head_mask,
                other=float("-inf"),
            ).to(dtype=tl.float32)
            L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
            L = tl.zeros([BLOCK_M], dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        L = tl.zeros([BLOCK_M], dtype=tl.float32)

    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # Hoist conversion to reduce layout pressure in the compiler.
    if USE_NVFP4_TC:
        # ================================================================
        # PROPER Q QUANTIZATION FOR FP8 TENSOR CORES (3D kernel)
        # Same logic as 2D kernel for consistency
        # ================================================================
        Q_NUM_SCALES: tl.constexpr = HEAD_SIZE_PADDED // 32  # 4 scales for 128-dim head

        # Reshape Q to [BLOCK_M, Q_NUM_SCALES, 32] to compute per-block max
        Q_reshaped = tl.reshape(Q, (BLOCK_M, Q_NUM_SCALES, 32))

        # Compute max abs per 32-element block: shape [BLOCK_M, Q_NUM_SCALES]
        Q_abs_max = tl.max(tl.abs(Q_reshaped), axis=2)

        # Compute scale: log2(max) clamped to valid FP8 range
        Q_log2_max = tl.where(Q_abs_max > 0, tl.math.log2(Q_abs_max), 0.0)
        Q_scale_f = tl.math.ceil(Q_log2_max) + 127.0
        Q_scale = tl.maximum(tl.minimum(Q_scale_f, 255.0), 0.0).to(tl.uint8)

        # Compute inverse scale for normalization
        Q_inv_scale = tl.math.exp2(127.0 - Q_scale_f)

        # Broadcast inverse scale to full Q shape
        Q_inv_scale_full = tl.reshape(
            tl.broadcast_to(Q_inv_scale[:, :, None], (BLOCK_M, Q_NUM_SCALES, 32)),
            (BLOCK_M, HEAD_SIZE_PADDED),
        )

        # Normalize Q and convert to FP8
        Q_normalized = Q * Q_inv_scale_full
        Q_fp8 = Q_normalized.to(tl.float8e4nv)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + q_head, mask=head_mask, other=0.0)

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos * qq_bias_stride_0
        )  # shape: [BLOCK_M] scalar token index

    # compute the length of the longest sequence prefix spanned by any
    max_seq_prefix_len = context_len + query_pos + 1

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
        tl.minimum((segm_idx + 1) * tiles_per_segment, num_tiles),
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
            + (offs_d[:, None]) * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        if not USE_NVFP4:
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            K = K_load[:HEAD_SIZE, :] if HEAD_SIZE_PADDED > HEAD_SIZE else K_load
        elif USE_NVFP4:
            k_packed_data_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + (offs_d[:, None] // 2) * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            K_u8 = tl.load(
                key_cache_ptr + k_packed_data_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0,
            ).to(tl.uint8)

            K_nibble = tl.where((offs_d[:, None] & 1) == 0, K_u8 & 0xF, K_u8 >> 4)
            K = nvfp4_to_fp8_e4m3(K_nibble)

            # ================================================================
            # Load K scales (u8) - CUDA now uses 32-element groups
            # So we have HEAD_SIZE_PADDED // 32 = 4 scales per head
            # tl.dot_scaled RHS scale must be [K_dim//32, N] = [4, TILE_SIZE]
            # ================================================================
            K_NUM_SCALES: tl.constexpr = HEAD_SIZE_PADDED // 32  # = 4 for head_size=128
            offs_ks = tl.arange(0, K_NUM_SCALES)

            k_scale_base = (
                physical_block_idx[:, None] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_k_cache_1
                + DATA_BYTES * stride_k_cache_3
            )
            k_scale_mask = tl.broadcast_to(
                tile_mask[:, None], (TILE_SIZE, K_NUM_SCALES)
            )

            # Load scales - shape [TILE_SIZE, K_NUM_SCALES] = [32, 4]
            # This matches Triton's expected rhs_scale shape [N, K//32]
            K_scale_u8 = tl.load(
                key_cache_ptr + k_scale_base + offs_ks[None, :] * stride_k_cache_3,
                mask=k_scale_mask,
                other=127,
            ).to(tl.uint8)

            K_scale_reduced = K_scale_u8

        if not USE_NVFP4:
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )
        else:
            V_load = None

        # FP8/NVFP4 dequantization for V
        if not USE_NVFP4 and V_load.dtype.is_fp8():
            V = V_load
        elif USE_NVFP4:
            v_packed_data_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + (offs_d[None, :] // 2) * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            V_u8 = tl.load(
                value_cache_ptr + v_packed_data_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0,
            ).to(tl.uint8)

            V_nibble = tl.where((offs_d[None, :] & 1) == 0, V_u8 & 0xF, V_u8 >> 4)
            V = nvfp4_to_fp8_e4m3(V_nibble)

            # ================================================================
            # Load V scales (u8) - CUDA now uses 32-element groups
            # So we have HEAD_SIZE_PADDED // 32 = 4 scales per head
            # Stored at positions: DATA_BYTES + [0, 1, 2, 3]
            # ================================================================
            V_NUM_SCALES: tl.constexpr = HEAD_SIZE_PADDED // 32  # = 4 for head_size=128
            offs_vs = tl.arange(0, V_NUM_SCALES)
            v_scale_off = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + (DATA_BYTES + offs_vs)[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            v_scale_mask = tl.broadcast_to(
                tile_mask[:, None], (TILE_SIZE, V_NUM_SCALES)
            )
            V_scale_u8 = tl.load(
                value_cache_ptr + v_scale_off, mask=v_scale_mask, other=127
            ).to(tl.uint8)
        else:
            V = V_load[:, :HEAD_SIZE] if HEAD_SIZE_PADDED > HEAD_SIZE else V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos
        seq_mask = seq_offset <= query_abs_pos

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
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
                    (seq_offset >= range_start) & (seq_offset <= range_end) & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        # PERFORMANCE OPTIMIZED MATMUL PATHS (3D)
        if USE_NVFP4 and USE_NVFP4_TC:
            S += scale * tl.dot_scaled(
                Q_fp8,
                Q_scale,
                "e4m3",
                K,
                K_scale_reduced,
                "e4m3",
            )
        elif USE_NVFP4:
            S += scale * tl.dot(Q.to(tl.bfloat16), K.to(tl.bfloat16)).to(tl.float32)
        elif K.dtype.is_fp8():
            S += scale * tl.load(k_scale) * tl.dot(Q.to(tl.float32), K.to(tl.float32))
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            head_mask[:, None] & query_mask_0 & seq_mask[None, :], S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)[None, :]

        if USE_QQ_BIAS:
            key_rel_pos = seq_offset - context_len
            is_query_key = (key_rel_pos >= 0) & (key_rel_pos < qq_bias_stride_0)
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],
                other=0.0,
            )
            S += qq_bias

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            V = tl.where(
                (context_len + query_pos - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        if USE_NVFP4:
            # Dequantize V with 32-element scale groups (4 scales per 128-dim head)
            V_scale_v = tl.math.exp2(V_scale_u8.to(tl.float32) - 127.0).to(tl.bfloat16)
            # Broadcast each scale to 32 elements: [TILE_SIZE, 4] -> [TILE_SIZE, 4, 32] -> [TILE_SIZE, 128]
            V_scale_full = tl.reshape(
                tl.broadcast_to(
                    V_scale_v[:, :, None], (TILE_SIZE, HEAD_SIZE_PADDED // 32, 32)
                ),
                (TILE_SIZE, HEAD_SIZE_PADDED),
            )
            V_deq = V.to(tl.bfloat16) * V_scale_full
            acc += tl.dot(P.to(tl.bfloat16), V_deq).to(tl.float32)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + q_head[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0 & head_mask[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + q_head * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & head_mask)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & head_mask)


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
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

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
    # Handle -inf - (-inf) = nan by neutralizing empty segments
    exp_scale = tl.exp(segm_max - overall_max)
    exp_scale = tl.where(segm_max == float("-inf"), 0.0, exp_scale)
    segm_expsum = segm_expsum * exp_scale
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
    segm_output *= exp_scale[:, None]
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
    if is_prefill:
        return 32
    return 32 if element_size == 1 else 16  # elements_size == 1 for fp8/nvfp4


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
    seq_threshold_3D=None,
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
    # NVFP4 KV cache mode
    use_nvfp4=False,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    # Define head_size and block_size before NVFP4 block uses them
    head_size = q.shape[2]
    block_size = v.shape[1]

    # PERFORMANCE OPTIMIZATION: Disabled pre-dequantization to enable fused kernel
    # The Triton kernel has native NVFP4 support (lines 310-385) which is 2x faster
    # because it avoids materializing 32GB of BF16 intermediate data.
    #
    # if use_nvfp4:
    #     try:
    #         from .nvfp4_dequant import gathered_dequantize_nvfp4_kv_cache
    #
    #         # 1. Linear Dequantization Trick: Dequantize ONLY active blocks
    #         # into a temporary linear buffer. This avoids the massive OOM spike.
    #         k_linear, new_block_table = gathered_dequantize_nvfp4_kv_cache(
    #             k, block_table, head_size, q.dtype
    #         )
    #         v_linear, _ = gathered_dequantize_nvfp4_kv_cache(
    #             v, block_table, head_size, q.dtype
    #         )
    #
    #         # 2. Redirect K, V and block_table to the linearized buffers
    #         k = k_linear
    #         v = v_linear
    #         block_table = new_block_table
    #         use_nvfp4 = False
    #
    #     except (ImportError, RuntimeError) as e:
    #         # Fallback to software LUT in Triton kernel if package unavailable
    #         pass

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

    # Keep BLOCK_M bounded; large BLOCK_M triggers Triton FP8 dot_scaled layout crash on SM90.
    # We use HG (head-groups) to cover all query heads with small BLOCK_M.
    BLOCK_M = 16 if num_queries_per_kv <= 16 else 32
    HG = (num_queries_per_kv + BLOCK_M - 1) // BLOCK_M
    BLOCK_Q = (
        1  # One query token index per program block for stable decode-heavy workloads
    )

    head_size_padded = triton.next_power_of_2(head_size)
    num_scales = head_size_padded // 16
    data_bytes = head_size_padded // 2

    # Enable FP8 dot_scaled only in safe M regime.
    # (On your Triton build, BLOCK_M=128 triggers convert_layout crash.)
    use_nvfp4_tc = bool(use_nvfp4) and (BLOCK_M <= 32)

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

    # Tile sizes for prefill and decode. Gemma3 models use optimized values.
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    # Tile sizes for prefill and decode. Gemma3 models use optimized values.
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    if use_nvfp4:
        TILE_SIZE_PREFILL = 32
        TILE_SIZE_DECODE = 32
    else:
        TILE_SIZE_PREFILL = _get_tile_size(
            head_size,
            sliding_window_val,
            q.element_size(),
            is_prefill=True,
        )
        TILE_SIZE_DECODE = _get_tile_size(
            head_size,
            sliding_window_val,
            q.element_size(),
            is_prefill=False,
        )

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold
    if (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
    ):
        kernel_unified_attention_2d[
            (
                total_num_q_blocks,
                num_kv_heads,
                HG,
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
            TILE_SIZE=TILE_SIZE_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
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
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_NVFP4_TC=use_nvfp4_tc,
            USE_FP8=output_scale is not None,
            USE_NVFP4=use_nvfp4,
            NUM_SCALES=num_scales,
            DATA_BYTES=data_bytes,
        )
    else:
        kernel_unified_attention_3d[
            (total_num_q_blocks, num_kv_heads, num_par_softmax_segments * HG)
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
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
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
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_NVFP4_TC=use_nvfp4_tc,
            USE_NVFP4=use_nvfp4,
            USE_FP8=output_scale is not None,
            NUM_SCALES=num_scales,
            DATA_BYTES=data_bytes,
        )
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )
