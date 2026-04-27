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

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .fused_indexer_q import _e2m1_nibble


# =============================================================================
# DeepseekV4 Attention path (head=512, nope=448 FP8 + rope=64 bf16)
# =============================================================================
def _gather_compressor_state(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    valid_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    head_size: int,
    state_width: int,
    span: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised gather of (kv, score) rows from the paged compressor
    state cache. Returns (kv_rows, score_rows) of shape (N, span, head_size)
    with -inf score and zero kv for out-of-range positions.

    State cache last-dim layout:  [kv_curr | kv_overlap | score_curr | score_overlap]
    each of width head_size. For C4A (overlap=True), span=8 with first 4
    tokens reading the [_curr] regions and last 4 reading the [_overlap]
    regions; for C128A, span=128 and head_offset is uniformly 0.
    """
    device = state_cache.device
    pos_offsets = torch.arange(span, device=device, dtype=torch.int64) - span + 1
    abs_pos = positions[valid_indices].to(torch.int64).unsqueeze(-1) + pos_offsets
    mask_pos = abs_pos >= 0
    pos_safe = abs_pos.clamp_min(0)
    block_indices = pos_safe // block_size
    block_offsets = pos_safe % block_size
    req_idx = token_to_req_indices[valid_indices].to(torch.int64)
    block_numbers = block_table[req_idx.unsqueeze(-1), block_indices]  # (N, span)

    # Single batched gather: (N, span, 2 * state_width).
    gathered = state_cache[block_numbers, block_offsets].to(torch.float32)

    # Split kv/score halves: state_cache[..., :state_width] is kv, the rest is score.
    kv_half = gathered[..., :state_width]
    score_half = gathered[..., state_width : 2 * state_width]

    if span == compress_ratio:
        kv_rows = kv_half[..., :head_size]
        score_rows = score_half[..., :head_size]
    else:
        # span = 2 * compress_ratio (C4A): first half from offset 0,
        # second half from offset head_size.
        kv_rows = torch.cat(
            [
                kv_half[:, :compress_ratio, :head_size],
                kv_half[:, compress_ratio:, head_size : 2 * head_size],
            ],
            dim=1,
        )
        score_rows = torch.cat(
            [
                score_half[:, :compress_ratio, :head_size],
                score_half[:, compress_ratio:, head_size : 2 * head_size],
            ],
            dim=1,
        )

    invalid = ~mask_pos.unsqueeze(-1)
    kv_rows = torch.where(invalid, torch.zeros_like(kv_rows), kv_rows)
    score_rows = torch.where(
        invalid,
        torch.full_like(score_rows, float("-inf")),
        score_rows,
    )
    return kv_rows, score_rows


def _scatter_packed_kv_cache(
    k_cache: torch.Tensor,
    fp8_bytes: torch.Tensor,
    bf16_bytes: torch.Tensor | None,
    scale_bytes: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    valid_indices: torch.Tensor,
    kv_cache_block_size: int,
    token_stride: int,
    scale_dim: int,
    fp8_dim: int,
) -> None:
    """Vectorised scatter of (FP8 nope | bf16 rope | UE8M0 scales) bytes
    into the paged uint8 KV cache."""
    k_cache_u8 = k_cache.view(torch.uint8) if k_cache.dtype != torch.uint8 else k_cache
    n_blocks = k_cache_u8.shape[0]
    block_stride = k_cache_u8[0].numel()  # bytes per block
    flat = k_cache_u8.view(n_blocks, block_stride)

    slots = kv_slot_mapping[valid_indices].to(torch.int64)
    block_idx = slots // kv_cache_block_size
    pos_in_block = slots % kv_cache_block_size

    token_offsets = pos_in_block * token_stride
    scale_offsets = kv_cache_block_size * token_stride + pos_in_block * scale_dim

    # FP8 nope byte scatter: flat[block_idx, token_offsets+0..fp8_dim] = fp8_bytes
    fp8_idx = token_offsets.unsqueeze(-1) + torch.arange(
        fp8_dim, device=k_cache_u8.device, dtype=torch.int64
    )
    flat[block_idx.unsqueeze(-1), fp8_idx] = fp8_bytes

    # Optional bf16 rope byte scatter
    if bf16_bytes is not None:
        bf16_n = bf16_bytes.shape[-1]
        bf16_off = (
            token_offsets.unsqueeze(-1)
            + fp8_dim
            + torch.arange(bf16_n, device=k_cache_u8.device, dtype=torch.int64)
        )
        flat[block_idx.unsqueeze(-1), bf16_off] = bf16_bytes

    # Scales scatter
    s_idx = scale_offsets.unsqueeze(-1) + torch.arange(
        scale_dim, device=k_cache_u8.device, dtype=torch.int64
    )
    flat[block_idx.unsqueeze(-1), s_idx] = scale_bytes


def _fused_kv_compress_norm_rope_insert_sparse_attn_torch(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    head_size: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
    rope_head_dim: int,
    fp8_max: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """SM80 PyTorch fallback for the V4 K-cache compressor (head_dim=512).
    Vectorised: single batched gather + matmul softmax + RMSNorm + RoPE +
    UE8M0 FP8 quant + scattered cache write. No per-token Python loop or
    `.item()` calls."""
    device = state_cache.device
    nope_head_dim = head_size - rope_head_dim
    n_nope_blocks = nope_head_dim // quant_block
    half_rope = rope_head_dim // 2
    span = (1 + int(overlap)) * compress_ratio

    # cudagraph dummy runs may pass kv_slot_mapping with different padding
    # than slot_mapping; align to the smaller (state-side) length.
    n_match = min(slot_mapping.shape[0], kv_slot_mapping.shape[0])
    slot_mapping = slot_mapping[:n_match]
    kv_slot_mapping = kv_slot_mapping[:n_match]
    positions = positions[:n_match]
    token_to_req_indices = token_to_req_indices[:n_match]
    valid_mask = (
        (slot_mapping >= 0)
        & ((positions + 1) % compress_ratio == 0)
        & (kv_slot_mapping >= 0)
    )
    if not valid_mask.any():
        return
    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)

    kv_rows, score_rows = _gather_compressor_state(
        state_cache,
        token_to_req_indices,
        positions,
        valid_indices,
        block_table,
        block_size,
        head_size,
        state_width,
        span,
        compress_ratio,
    )

    # Softmax over the span axis per (token, feature), then weighted sum.
    weights = torch.softmax(score_rows, dim=1)
    compressed_kv = (kv_rows * weights).sum(dim=1)  # (N, head_size)

    # RMSNorm
    variance = (compressed_kv * compressed_kv).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_norm_weight.to(torch.float32)

    # FP8 quant on NoPE region (block-wise UE8M0).
    nope = normed[:, :nope_head_dim].view(-1, n_nope_blocks, quant_block)
    nope_bf16 = nope.to(torch.bfloat16).to(torch.float32)
    block_absmax = nope_bf16.abs().amax(dim=-1).clamp_min(1e-4)
    raw_scale = block_absmax / fp8_max
    exponents = torch.ceil(torch.log2(raw_scale))
    inv_scale = torch.pow(2.0, -exponents).unsqueeze(-1)
    quantized = (nope_bf16 * inv_scale).clamp(-fp8_max, fp8_max)
    fp8_nope = quantized.to(torch.float8_e4m3fn).view(-1, nope_head_dim)
    fp8_nope_bytes = fp8_nope.view(torch.uint8)

    encoded_scales = (exponents + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    pad_col = torch.zeros(
        encoded_scales.shape[0],
        scale_dim - n_nope_blocks,
        device=device,
        dtype=torch.uint8,
    )
    scales_full = torch.cat([encoded_scales, pad_col], dim=-1)

    # Forward GPT-J RoPE on the RoPE region.
    rope_part = normed[:, nope_head_dim:]
    even = rope_part[:, 0::2]
    odd = rope_part[:, 1::2]
    valid_positions = positions[valid_indices].to(torch.long)
    compressed_pos = (valid_positions // compress_ratio) * compress_ratio
    cs = cos_sin_cache.index_select(0, compressed_pos)
    cos = cs[:, :half_rope].to(torch.float32)
    sin = cs[:, half_rope : 2 * half_rope].to(torch.float32)
    new_even = even * cos - odd * sin
    new_odd = odd * cos + even * sin
    rope_out = torch.empty_like(rope_part)
    rope_out[:, 0::2] = new_even
    rope_out[:, 1::2] = new_odd
    rope_bytes = rope_out.to(torch.bfloat16).view(torch.uint8)

    _scatter_packed_kv_cache(
        k_cache,
        fp8_nope_bytes,
        rope_bytes,
        scales_full,
        kv_slot_mapping,
        valid_indices,
        kv_cache_block_size,
        token_stride,
        scale_dim,
        fp8_dim=nope_head_dim,
    )


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


def _fused_kv_compress_norm_rope_insert_indexer_attn_torch(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    head_size: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
    rope_head_dim: int,
    fp8_max: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """SM80 PyTorch fallback for the indexer K-cache FP8 compressor
    (head_dim=128). Vectorised — no per-token Python loop."""
    nope_head_dim = head_size - rope_head_dim
    half_rope = rope_head_dim // 2
    span = (1 + int(overlap)) * compress_ratio
    assert head_size == quant_block, (
        "Indexer compressor expects QUANT_BLOCK == HEAD_SIZE"
    )

    # cudagraph dummy runs may pass kv_slot_mapping with different padding
    # than slot_mapping; align to the smaller (state-side) length.
    n_match = min(slot_mapping.shape[0], kv_slot_mapping.shape[0])
    slot_mapping = slot_mapping[:n_match]
    kv_slot_mapping = kv_slot_mapping[:n_match]
    positions = positions[:n_match]
    token_to_req_indices = token_to_req_indices[:n_match]
    valid_mask = (
        (slot_mapping >= 0)
        & ((positions + 1) % compress_ratio == 0)
        & (kv_slot_mapping >= 0)
    )
    if not valid_mask.any():
        return
    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)

    kv_rows, score_rows = _gather_compressor_state(
        state_cache,
        token_to_req_indices,
        positions,
        valid_indices,
        block_table,
        block_size,
        head_size,
        state_width,
        span,
        compress_ratio,
    )

    weights = torch.softmax(score_rows, dim=1)
    compressed_kv = (kv_rows * weights).sum(dim=1)  # (N, head_size)
    variance = (compressed_kv * compressed_kv).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_norm_weight.to(torch.float32)

    # Forward GPT-J RoPE on the last rope_head_dim elements.
    nope = normed[:, :nope_head_dim]
    rope_part = normed[:, nope_head_dim:]
    even = rope_part[:, 0::2]
    odd = rope_part[:, 1::2]
    valid_positions = positions[valid_indices].to(torch.long)
    compressed_pos = (valid_positions // compress_ratio) * compress_ratio
    cs = cos_sin_cache.index_select(0, compressed_pos)
    cos = cs[:, :half_rope].to(torch.float32)
    sin = cs[:, half_rope : 2 * half_rope].to(torch.float32)
    new_even = even * cos - odd * sin
    new_odd = odd * cos + even * sin
    rope_out = torch.empty_like(rope_part)
    rope_out[:, 0::2] = new_even
    rope_out[:, 1::2] = new_odd

    full = torch.cat([nope, rope_out], dim=-1)
    full_bf16 = full.to(torch.bfloat16).to(torch.float32)
    absmax = full_bf16.abs().amax(dim=-1).clamp_min(1e-4)
    raw_scale = absmax / fp8_max
    exponents = torch.ceil(torch.log2(raw_scale))
    inv_scale = torch.pow(2.0, -exponents).unsqueeze(-1)
    quantized = (full_bf16 * inv_scale).clamp(-fp8_max, fp8_max)
    fp8_full = quantized.to(torch.float8_e4m3fn)
    fp8_bytes = fp8_full.view(torch.uint8)

    scale_vals = torch.pow(2.0, exponents).to(torch.float32)
    scale_bytes = scale_vals.view(-1, 1).view(torch.uint8)  # (N, 4)

    _scatter_packed_kv_cache(
        k_cache,
        fp8_bytes,
        bf16_bytes=None,
        scale_bytes=scale_bytes,
        kv_slot_mapping=kv_slot_mapping,
        valid_indices=valid_indices,
        kv_cache_block_size=kv_cache_block_size,
        token_stride=token_stride,
        scale_dim=scale_dim,
        fp8_dim=head_size,
    )


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
    amax = tl.maximum(amax, 1e-4)

    # ue8m0 block scale: 2^ceil(log2(amax / 6.0)), stored as (exp + 127) byte.
    log2_ratio = tl.ceil(tl.log2(amax / 6.0))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.exp2(-log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)  # [N_QUANT_BLOCKS]

    inv_scale_col = tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1))
    lo_nib = _e2m1_nibble(even_2d * inv_scale_col)  # (N_BLOCKS, HALF_BLOCK) uint8
    hi_nib = _e2m1_nibble(odd_2d * inv_scale_col)
    packed = lo_nib | (hi_nib << 4)
    packed_flat = tl.reshape(packed, (TOKEN_STRIDE,))

    tl.store(val_ptr + tl.arange(0, TOKEN_STRIDE), packed_flat)
    tl.store(scale_ptr + tl.arange(0, SCALE_DIM), ue8m0)


# =============================================================================
# Custom-op wrappers for cudagraph splitting on SM80 reference paths.
#
# The compressor PyTorch fallbacks use .any() / .nonzero() / .item() — all of
# which are illegal during cudagraph capture (data-dependent control flow).
# Registering them as vllm:: custom ops + listing them in CompilationConfig.
# _attention_ops makes vllm split the cudagraph at the call site so the
# fallback body runs eager outside the captured replay.
# =============================================================================


def _compressor_sparse_attn_torch_op(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    block_size: int,
    rms_norm_eps: float,
    kv_cache_block_size: int,
    head_size: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
    rope_head_dim: int,
    fp8_max: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    _fused_kv_compress_norm_rope_insert_sparse_attn_torch(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        block_size=block_size,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        cos_sin_cache=cos_sin_cache,
        k_cache=k_cache,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_cache_block_size,
        head_size=head_size,
        state_width=state_width,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rope_head_dim=rope_head_dim,
        fp8_max=fp8_max,
        quant_block=quant_block,
        token_stride=token_stride,
        scale_dim=scale_dim,
    )


def _compressor_sparse_attn_torch_op_fake(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    block_size: int,
    rms_norm_eps: float,
    kv_cache_block_size: int,
    head_size: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
    rope_head_dim: int,
    fp8_max: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    return None


def _compressor_indexer_attn_torch_op(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    block_size: int,
    rms_norm_eps: float,
    kv_cache_block_size: int,
    head_size: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
    rope_head_dim: int,
    fp8_max: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    _fused_kv_compress_norm_rope_insert_indexer_attn_torch(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        block_size=block_size,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        cos_sin_cache=cos_sin_cache,
        k_cache=k_cache,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_cache_block_size,
        head_size=head_size,
        state_width=state_width,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rope_head_dim=rope_head_dim,
        fp8_max=fp8_max,
        quant_block=quant_block,
        token_stride=token_stride,
        scale_dim=scale_dim,
    )


direct_register_custom_op(
    op_name="deepseek_v4_compressor_sparse_sm80",
    op_func=_compressor_sparse_attn_torch_op,
    mutates_args=["k_cache", "state_cache"],
    fake_impl=_compressor_sparse_attn_torch_op_fake,
)
direct_register_custom_op(
    op_name="deepseek_v4_compressor_indexer_sm80",
    op_func=_compressor_indexer_attn_torch_op,
    mutates_args=["k_cache", "state_cache"],
    fake_impl=_compressor_sparse_attn_torch_op_fake,
)
