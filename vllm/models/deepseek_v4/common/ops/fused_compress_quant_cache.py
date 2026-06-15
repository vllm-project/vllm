# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused compressor + FP8/MXFP4 UE8M0 quantization + KV cache insert kernels.

Three specialized kernels:
  - _fused_kv_compress_norm_rope_insert_sparse_attn:
        head=512, nope=448 FP8 + rope=64 bf16
  - _fused_kv_compress_norm_rope_insert_indexer_attn:
        head=128, all FP8, 1 block/token
  - _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn:
        head=128, MXFP4 (block=32), 4 ue8m0 bytes

RoPE is register-based via tl.reshape -> tl.split -> tl.interleave (or the
even/odd halves are consumed directly for MXFP4, no interleave needed).
FP8 UE8M0 quant uses tl.reshape to tile [N_QUANT_BLOCKS, QUANT_BLOCK] for
per-block absmax entirely in registers. MXFP4 does the same tiling on the
even/odd halves, producing (N_QUANT_BLOCKS, MXFP4_BLOCK/2) packed nibbles
and N_QUANT_BLOCKS ue8m0 bytes.
"""

from typing import Any

import torch

from vllm.triton_utils import tl, triton

from .fused_indexer_q import _fp32x2_to_fp4x2


def compress_norm_rope_store_triton(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """Shared triton launcher for the fused compress+norm+RoPE+insert path.

    Picks one of the three kernels in this module based on ``head_dim`` and
    ``use_fp4_cache``. Identical launch signature for all three.
    """
    if head_dim == 512:
        kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
        num_warps = 4
    elif use_fp4_cache:
        kernel = _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
        num_warps = 1
    else:
        kernel = _fused_kv_compress_norm_rope_insert_indexer_attn
        num_warps = 1

    kernel[(num_actual,)](
        # state cache
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        # metadata
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        # RMSNorm
        rms_norm_weight,
        rms_norm_eps,
        # RoPE
        cos_sin_cache,
        cos_sin_cache.stride(0),
        # KV cache
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],  # paged KV cache block size (tokens per block)
        # constexprs
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=448.0,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=num_warps,
        **pdl_kwargs,
    )


# =============================================================================
# DeepseekV4 PCP compressor path
# =============================================================================
@triton.jit
def _dsv4_pcp_compressor_partial_stats_kernel(
    # ── state cache (compressor internal state) ──
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    # ── metadata ──
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # ── partial softmax stats output ──
    partial_m_ptr,
    partial_s_ptr,
    partial_v_ptr,
    partial_stride0,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    out_offsets = token_idx * partial_stride0 + block

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        tl.store(partial_m_ptr + out_offsets, 0.0, mask=mask)
        tl.store(partial_s_ptr + out_offsets, 0.0, mask=mask)
        tl.store(partial_v_ptr + out_offsets, 0.0, mask=mask)
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        tl.store(partial_m_ptr + out_offsets, 0.0, mask=mask)
        tl.store(partial_s_ptr + out_offsets, 0.0, mask=mask)
        tl.store(partial_v_ptr + out_offsets, 0.0, mask=mask)
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    block_indices = pos // virtual_block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    virtual_block_offsets = pos - block_indices * virtual_block_size
    is_local = (
        virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
    ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
    local_block_offsets = (
        virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
    ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
        virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
    )
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    row_base = (
        state_cache_ptr
        + block_numbers.to(tl.int64) * state_cache_stride0
        + local_block_offsets * state_cache_stride1
        + head_offset
    )
    combined_mask = mask_pos[:, None] & is_local[:, None] & mask[None, :]

    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    local_m = tl.max(score, axis=0)
    has_local = local_m != -float("inf")
    shifted = tl.where(has_local[None, :], score - local_m[None, :], -float("inf"))
    weight = tl.exp(shifted)
    local_s = tl.sum(weight, axis=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )
    local_v = tl.sum(kv * weight, axis=0)

    tl.store(partial_m_ptr + out_offsets, local_m, mask=mask)
    tl.store(partial_s_ptr + out_offsets, local_s, mask=mask)
    tl.store(partial_v_ptr + out_offsets, local_v, mask=mask)


@triton.jit
def _dsv4_pcp_rescale_partial_stats_kernel(
    partial_m_ptr,
    partial_s_ptr,
    partial_v_ptr,
    global_m_ptr,
    global_s_ptr,
    global_v_ptr,
    partial_stride0,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    offsets = token_idx * partial_stride0 + block

    partial_m = tl.load(partial_m_ptr + offsets, mask=mask, other=-float("inf"))
    global_m = tl.load(global_m_ptr + offsets, mask=mask, other=-float("inf"))
    partial_s = tl.load(partial_s_ptr + offsets, mask=mask, other=0.0)
    partial_v = tl.load(partial_v_ptr + offsets, mask=mask, other=0.0)

    valid = (
        (partial_m == partial_m)
        & (partial_m != float("inf"))
        & (partial_m != -float("inf"))
        & (global_m == global_m)
        & (global_m != float("inf"))
        & (global_m != -float("inf"))
    )
    scale = tl.where(valid, tl.exp(partial_m - global_m), 0.0)
    tl.store(global_s_ptr + offsets, partial_s * scale, mask=mask)
    tl.store(global_v_ptr + offsets, partial_v * scale, mask=mask)


@triton.jit
def _dsv4_pcp_finalize_sparse_attn_kernel(
    # ── merged compressor stats ──
    global_s_ptr,
    global_v_ptr,
    global_stride0,
    # ── metadata ──
    positions_ptr,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    stat_offsets = token_idx * global_stride0 + block
    global_s = tl.load(global_s_ptr + stat_offsets, mask=mask, other=1.0)
    global_v = tl.load(global_v_ptr + stat_offsets, mask=mask, other=0.0)
    compressed_kv = global_v / global_s

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)
    block_absmax = tl.maximum(block_absmax, 1e-4)

    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_scaled = quant_2d * inv_scales_col
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))

    nope_mask = block < NOPE_HEAD_DIM
    tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)

    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = exponents + 127.0
    encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
    tl.store(
        scale_ptr + scale_idx,
        encoded.to(tl.uint8),
        mask=scale_idx < N_NOPE_BLOCKS,
    )
    tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)

    bf16_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)


@triton.jit
def _dsv4_pcp_finalize_indexer_attn_kernel(
    # ── merged compressor stats ──
    global_s_ptr,
    global_v_ptr,
    global_stride0,
    # ── metadata ──
    positions_ptr,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    stat_offsets = token_idx * global_stride0 + block
    global_s = tl.load(global_s_ptr + stat_offsets, mask=mask, other=1.0)
    global_v = tl.load(global_v_ptr + stat_offsets, mask=mask, other=0.0)
    compressed_kv = global_v / global_s

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(normed_2d)

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)

    tl.static_assert(
        TRITON_BLOCK_SIZE == QUANT_BLOCK,
        "Indexer expects one quant block (QUANT_BLOCK == TRITON_BLOCK_SIZE)",
    )
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    result_bf16 = result.to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(result_bf16), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    raw_scale = absmax * INV_FP8_MAX
    exponent = tl.ceil(tl.log2(raw_scale))
    inv_scale = tl.exp2(-exponent)

    x_scaled = result_bf16 * inv_scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)

    tl.store(fp8_ptr + block, x_uint8, mask=mask)

    scale_val = tl.exp2(exponent)
    tl.store(scale_ptr.to(tl.pointer_type(tl.float32)), scale_val)


@triton.jit
def _dsv4_pcp_finalize_indexer_mxfp4_attn_kernel(
    # ── merged compressor stats ──
    global_s_ptr,
    global_v_ptr,
    global_stride0,
    # ── metadata ──
    positions_ptr,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    stat_offsets = token_idx * global_stride0 + block
    global_s = tl.load(global_s_ptr + stat_offsets, mask=mask, other=1.0)
    global_v = tl.load(global_v_ptr + stat_offsets, mask=mask, other=0.0)
    compressed_kv = global_v / global_s

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(normed_2d)

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v

    new_even = new_even.to(tl.bfloat16).to(tl.float32)
    new_odd = new_odd.to(tl.bfloat16).to(tl.float32)

    N_QUANT_BLOCKS: tl.constexpr = HEAD_SIZE // QUANT_BLOCK
    HALF_BLOCK: tl.constexpr = QUANT_BLOCK // 2
    tl.static_assert(TRITON_BLOCK_SIZE == HEAD_SIZE)
    tl.static_assert(HEAD_SIZE % QUANT_BLOCK == 0)
    tl.static_assert(TOKEN_STRIDE == HEAD_SIZE // 2)
    tl.static_assert(SCALE_DIM == N_QUANT_BLOCKS)

    even_2d = tl.reshape(new_even, (N_QUANT_BLOCKS, HALF_BLOCK))
    odd_2d = tl.reshape(new_odd, (N_QUANT_BLOCKS, HALF_BLOCK))

    amax = tl.maximum(
        tl.max(tl.abs(even_2d), axis=1),
        tl.max(tl.abs(odd_2d), axis=1),
    )
    amax = tl.maximum(amax, 6.0 * (2**-126))

    log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.exp2(-log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)

    inv_scale_col = tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1))
    packed = _fp32x2_to_fp4x2(even_2d * inv_scale_col, odd_2d * inv_scale_col)
    packed_flat = tl.reshape(packed, (TOKEN_STRIDE,))

    tl.store(val_ptr + tl.arange(0, TOKEN_STRIDE), packed_flat)
    tl.store(scale_ptr + tl.arange(0, SCALE_DIM), ue8m0)


# =============================================================================
# DeepseekV4 Attention path (head=512, nope=448 FP8 + rope=64 bf16)
# =============================================================================
@triton.jit
def _fused_kv_compress_norm_rope_insert_sparse_attn(
    # ── state cache (compressor internal state) ──
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    # ── metadata ──
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,  # 448.0
    QUANT_BLOCK: tl.constexpr,  # 64 for DeepseekV4
    TOKEN_STRIDE: tl.constexpr,  # 576 for DeepseekV4
    SCALE_DIM: tl.constexpr,  # 8 for DeepseekV4 (7 real + 1 pad)
    KV_BLOCK_STRIDE: tl.constexpr,
):
    """Fused compress → RMSNorm → FP8 quant (nope) → RoPE → bf16 store (rope).

    One program per token; early-exits for non-boundary positions.

    Cache block layout (``block_size`` tokens):
      [0, bs*576):       token data (448 fp8 + 128 bf16 each)
      [bs*576, +bs*8):   uint8 UE8M0 scales (7 real + 1 pad each)
    """
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    # ── Gather state cache entries ────────────────────────────────────
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)

    # Precomputed row base shared by score and kv loads
    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )

    combined_mask = mask_pos[:, None] & mask[None, :]

    # ── Softmax + weighted sum ───────────────────────────────────────
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )

    compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32

    # ── RMSNorm (fp32 throughout) ──────────────────────────────────────
    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    # ── KV cache pointers ────────────────────────────────────────────
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM  # 448
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2  # 32

    # FP8 UE8M0 quant: cast fp32 → bf16 → fp32 before quant to match reference.
    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK  # 7
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)  # [N_QUANT_BLOCKS] fp32
    block_absmax = tl.maximum(block_absmax, 1e-4)

    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_scaled = quant_2d * inv_scales_col
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))

    nope_mask = block < NOPE_HEAD_DIM
    tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)

    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = exponents + 127.0
    encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
    tl.store(
        scale_ptr + scale_idx,
        encoded.to(tl.uint8),
        mask=scale_idx < N_NOPE_BLOCKS,
    )
    tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    # Register-based GPT-J RoPE in fp32.
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)  # each [NUM_PAIRS] fp32

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)  # [TRITON_BLOCK_SIZE] fp32

    # Store rotated rope portion as bf16 into the cache's bf16 area.
    bf16_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)


# =============================================================================
# Indexer path (head=128, all FP8, single quant block)
# =============================================================================
@triton.jit
def _fused_kv_compress_norm_rope_insert_indexer_attn(
    # ── state cache (compressor internal state) ──
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    # ── metadata ──
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,  # 448.0
    QUANT_BLOCK: tl.constexpr,  # 128 for indexer
    TOKEN_STRIDE: tl.constexpr,  # 128 for indexer
    SCALE_DIM: tl.constexpr,  # 4 for indexer (1 float32)
    KV_BLOCK_STRIDE: tl.constexpr,
):
    """Fused compress → RMSNorm → RoPE → FP8 quant → store.

    One program per token; early-exits for non-boundary positions.

    Cache block layout:
      [0, bs*128):       FP8 data (128 bytes/token)
      [bs*128, +bs*4):   float32 scales (4 bytes/token)

    For head_dim=128 we have exactly one quant block, so we skip the
    [N_QUANT_BLOCKS, QUANT_BLOCK] reshape entirely and use a flat
    ``tl.max`` reduction.
    """
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    # ── Gather state cache entries ────────────────────────────────────
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)

    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )

    combined_mask = mask_pos[:, None] & mask[None, :]

    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )

    compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32

    # ── RMSNorm (fp32 throughout) ──────────────────────────────────────
    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    # ── KV cache pointers ────────────────────────────────────────────
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    # ── Register-based GPT-J forward RoPE in fp32 ─────────────────────
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(normed_2d)  # each [NUM_PAIRS] fp32

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)  # fp32

    # ── FP8 UE8M0 quant: single block, flat reduction ────────────────
    tl.static_assert(
        TRITON_BLOCK_SIZE == QUANT_BLOCK,
        "Indexer expects one quant block (QUANT_BLOCK == TRITON_BLOCK_SIZE)",
    )
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    result_bf16 = result.to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(result_bf16), axis=0)  # scalar
    absmax = tl.maximum(absmax, 1e-4)
    raw_scale = absmax * INV_FP8_MAX
    exponent = tl.ceil(tl.log2(raw_scale))
    inv_scale = tl.exp2(-exponent)

    x_scaled = result_bf16 * inv_scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)

    tl.store(fp8_ptr + block, x_uint8, mask=mask)

    # Single float32 scale
    scale_val = tl.exp2(exponent)
    tl.store(scale_ptr.to(tl.pointer_type(tl.float32)), scale_val)


# =============================================================================
# Indexer path (head=128, MXFP4: 2 nibbles/byte + ue8m0 per 32-elem block)
# =============================================================================
@triton.jit
def _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
    # ── state cache (compressor internal state) ──
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    # ── metadata ──
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,  # unused for MXFP4 (kept for signature parity)
    QUANT_BLOCK: tl.constexpr,  # 32 for MXFP4
    TOKEN_STRIDE: tl.constexpr,  # HEAD_SIZE // 2 = 64 packed bytes/token
    SCALE_DIM: tl.constexpr,  # HEAD_SIZE // QUANT_BLOCK = 4 ue8m0 bytes/token
    KV_BLOCK_STRIDE: tl.constexpr,
):
    """Fused compress → RMSNorm → RoPE → MXFP4 quant → store.

    One program per token; early-exits for non-boundary positions.

    Cache block layout (``block_size`` tokens per cache block):
      [0, bs*TOKEN_STRIDE):        packed MXFP4 nibbles (2 values/byte)
      [bs*TOKEN_STRIDE, +bs*SCALE_DIM): ue8m0 scale bytes (one per 32-elem block)

    MXFP4 format:
      - E2M1 4-bit values packed two per byte (low nibble first, then high).
      - Per-32-element block scale = 2^ceil(log2(amax / 6.0)), stored ue8m0
        (byte = exponent + 127).
      - Max representable magnitude = 6.0.
    """
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    # ── Gather state cache entries ────────────────────────────────────
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)

    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )

    combined_mask = mask_pos[:, None] & mask[None, :]

    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )

    compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32

    # ── RMSNorm (fp32 throughout) ──────────────────────────────────────
    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    # ── KV cache pointers (segregated: values first, then scales) ────
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    # ── Register-based GPT-J forward RoPE in fp32 ─────────────────────
    # We keep the even/odd halves (no tl.interleave afterwards) because the
    # MXFP4 per-block absmax / pack naturally operates on (even, odd) pairs.
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(normed_2d)  # each [NUM_PAIRS] fp32

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v

    # bf16 roundtrip for parity with reference / Q-side kernel numerics.
    new_even = new_even.to(tl.bfloat16).to(tl.float32)
    new_odd = new_odd.to(tl.bfloat16).to(tl.float32)

    # ── MXFP4 quant: tile even/odd halves into (N_BLOCKS, HALF_BLOCK) ──
    # Each MXFP4 block of QUANT_BLOCK elements = HALF_BLOCK consecutive pairs,
    # so (N_BLOCKS, HALF_BLOCK) rows of even/odd each land exactly one block.
    N_QUANT_BLOCKS: tl.constexpr = HEAD_SIZE // QUANT_BLOCK
    HALF_BLOCK: tl.constexpr = QUANT_BLOCK // 2
    tl.static_assert(TRITON_BLOCK_SIZE == HEAD_SIZE)
    tl.static_assert(HEAD_SIZE % QUANT_BLOCK == 0)
    tl.static_assert(TOKEN_STRIDE == HEAD_SIZE // 2)
    tl.static_assert(SCALE_DIM == N_QUANT_BLOCKS)

    even_2d = tl.reshape(new_even, (N_QUANT_BLOCKS, HALF_BLOCK))
    odd_2d = tl.reshape(new_odd, (N_QUANT_BLOCKS, HALF_BLOCK))

    amax = tl.maximum(
        tl.max(tl.abs(even_2d), axis=1),
        tl.max(tl.abs(odd_2d), axis=1),
    )
    amax = tl.maximum(amax, 6.0 * (2**-126))

    # ue8m0 block scale: 2^ceil(log2(amax / 6.0)), stored as (exp + 127) byte.
    log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.exp2(-log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)  # [N_QUANT_BLOCKS]

    inv_scale_col = tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1))
    packed = _fp32x2_to_fp4x2(
        even_2d * inv_scale_col, odd_2d * inv_scale_col
    )  # (N_BLOCKS, HALF_BLOCK) uint8
    packed_flat = tl.reshape(packed, (TOKEN_STRIDE,))

    tl.store(val_ptr + tl.arange(0, TOKEN_STRIDE), packed_flat)
    tl.store(scale_ptr + tl.arange(0, SCALE_DIM), ue8m0)
