# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN decode path — Stage 5.a.

Two small Triton kernels + a thin Python driver that calls FlashAttention.

  - ``_kvarn_dequant_blocks_kernel``: dequantises one (block, kv_head) tile
    from the int4 cache into rotated fp16 at a caller-supplied destination
    in a packed varlen ``[total_kv_tokens, Hk, D]`` buffer.
  - ``_kvarn_pool_gather_packed_kernel``: copies sink / trailing-tail tokens
    from the (already rotated) fp16 tail pool into the same packed buffer.
  - ``kvarn_decode_attention``: orchestrates Q rotation → dequant + gather →
    ``flash_attn_varlen_func`` → output un-rotation.

Cache layout (per block, head) — 17920 bytes, see KVarNConfig.

The task plan for both kernels (which cache blocks to dequant, which pool
blocks to gather, where they land in the packed buffer) is built once per
batch in ``KVarNMetadataBuilder.build`` and reused across all 28+ attention
layer forwards in a step.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

# Number of KV-sequence splits for the split-K flash-decoding kernel. More
# splits = better load-balancing of ragged burst seqlens across SMs, at the cost
# of a larger fp32 partial-output scratch + more stage-2 combine work.
KVARN_NUM_KV_SPLITS = 8


# ──────────────────────────────────────────────────────────────────────────────
# Dequant kernel: int4 quantised tile → rotated fp16 packed buffer.
# SUPERSEDED by `_kvarn_build_packed_kv_kernel` (the unified block_table-driven
# kernel). Kept only for the standalone unit test scripts/test_dequant_kernel.py.
# Not called by the decode driver.
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _kvarn_dequant_blocks_kernel(
    KV_cache_ptr,       # [num_blocks, num_kv_heads, TILE_BYTES]   uint8
    Block_ids_ptr,      # [num_tasks]                              int32
    Dst_offsets_ptr,    # [num_tasks]                              int32
    K_out_ptr,          # [max_total_tokens, num_kv_heads, D]      fp16
    V_out_ptr,          # [max_total_tokens, num_kv_heads, D]      fp16
    # strides
    stride_kv_b, stride_kv_h,
    stride_out_t, stride_out_h,
    # constexprs
    D: tl.constexpr,
    GROUP: tl.constexpr,
    K_PACKED_OFFSET: tl.constexpr,
    K_S_COL_OFFSET: tl.constexpr,
    K_ZP_OFFSET: tl.constexpr,
    K_S_ROW_OFFSET: tl.constexpr,
    V_PACKED_OFFSET: tl.constexpr,
    V_S_COL_OFFSET: tl.constexpr,
    V_S_ROW_OFFSET: tl.constexpr,
    V_ZP_OFFSET: tl.constexpr,
):
    """Dequantise one (block, kv_head) tile per program.

    Grid: ``(num_tasks, num_kv_heads)``.

    For task ``t``:
      - reads cache block ``Block_ids_ptr[t]`` at head ``hk``
      - dequantises both K and V tiles
      - writes the resulting fp16 K_rot[g, d] and V_rot[g, d] into
        K_out_ptr / V_out_ptr starting at token ``Dst_offsets_ptr[t]``,
        head ``hk``.

    Dequant identities (matching ``kvarn_decode.py``):
      K_rot[d, g] = (q_K[d, g] * s_col_K[d] + zp_K[d]) * s_row_K[g]
      V_rot[g, d] = (q_V[g, d] * s_row_V[g] + zp_V[g]) * s_col_V[d]
    """
    task_id = tl.program_id(0)
    hk = tl.program_id(1)

    block_id = tl.load(Block_ids_ptr + task_id).to(tl.int64)
    dst = tl.load(Dst_offsets_ptr + task_id).to(tl.int64)
    tile_base = block_id * stride_kv_b + hk * stride_kv_h

    d_offs = tl.arange(0, D)
    g_offs = tl.arange(0, GROUP)
    g_byte_k = g_offs // 2
    g_shift_k = (g_offs % 2) * 4
    PACK_K: tl.constexpr = 8 // K_BITS
    PACK_V: tl.constexpr = 8 // V_BITS
    MASK_K: tl.constexpr = (1 << K_BITS) - 1
    MASK_V: tl.constexpr = (1 << V_BITS) - 1
    d_byte_v = d_offs // PACK_V
    d_shift_v = (d_offs % PACK_V) * V_BITS

    # K scales
    sk_lo = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2).to(tl.uint16)
    sk_hi = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2 + 1).to(tl.uint16)
    s_col_K = ((sk_lo | (sk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

    zk_lo = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2).to(tl.uint16)
    zk_hi = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2 + 1).to(tl.uint16)
    zp_K = ((zk_lo | (zk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

    srk_lo = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + g_offs * 2).to(tl.uint16)
    srk_hi = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + g_offs * 2 + 1).to(tl.uint16)
    s_row_K = ((srk_lo | (srk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

    # K dequant: cache stores [D, group/2] packed; produce [GROUP, D] for output.
    k_addrs = (
        tile_base + K_PACKED_OFFSET
        + d_offs[:, None] * (GROUP // 2)
        + g_byte_k[None, :]
    )
    k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)
    q_K = ((k_bytes >> g_shift_k[None, :]) & 0xF).to(tl.float32)
    K_rot = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]   # [D, GROUP]
    K_rot_out = tl.trans(K_rot)                                           # [GROUP, D]

    # V scales
    scv_lo = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2).to(tl.uint16)
    scv_hi = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2 + 1).to(tl.uint16)
    s_col_V = ((scv_lo | (scv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

    srv_lo = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + g_offs * 2).to(tl.uint16)
    srv_hi = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + g_offs * 2 + 1).to(tl.uint16)
    s_row_V = ((srv_lo | (srv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

    zpv_lo = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + g_offs * 2).to(tl.uint16)
    zpv_hi = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + g_offs * 2 + 1).to(tl.uint16)
    zp_V = ((zpv_lo | (zpv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

    # V dequant: cache stores [group, D/2] packed; result is [GROUP, D].
    v_addrs = (
        tile_base + V_PACKED_OFFSET
        + g_offs[:, None] * (D // 2)
        + d_byte_v[None, :]
    )
    v_bytes = tl.load(KV_cache_ptr + v_addrs).to(tl.int32)
    q_V = ((v_bytes >> d_shift_v[None, :]) & 0xF).to(tl.float32)
    V_rot = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]   # [GROUP, D]

    # Write to scratch ([token, kv_head, dim] layout).
    dst_token_offs = dst + g_offs
    out_addrs = (
        dst_token_offs[:, None] * stride_out_t
        + hk * stride_out_h
        + d_offs[None, :]
    )
    tl.store(K_out_ptr + out_addrs, K_rot_out.to(tl.float16))
    tl.store(V_out_ptr + out_addrs, V_rot.to(tl.float16))


# ──────────────────────────────────────────────────────────────────────────────
# Stage α-2 scatter store: writes (already-rotated) k, v into the tail pool at
# positions derived from slot_mapping. Replaces the Python for-loop in
# do_kv_cache_update so the whole store path is capturable.
#
# Pool layout: [POOL_SIZE, group, Hk, D] fp16. The pool slot is found via the
# sparse Block_to_slot lookup (block_id → slot, -1 if the block lives in the
# int4 cache). pos = slot_mapping % group. The lookup table is mutated only by
# the metadata builder (outside any captured region).
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _kvarn_scatter_store_kernel(
    K_in_ptr,           # [N, Hk, D]                    fp16 (already rotated)
    V_in_ptr,           # [N, Hk, D]                    fp16
    Slot_mapping_ptr,   # [N]                           int32   (slot < 0 ⇒ pad)
    Block_to_slot_ptr,  # [num_blocks_lookup]           int32   (-1 = no slot)
    Pool_K_ptr,         # [POOL_SIZE, group, Hk, D]     fp16
    Pool_V_ptr,         # [POOL_SIZE, group, Hk, D]     fp16
    # strides
    stride_in_n, stride_in_h,
    stride_pool_b, stride_pool_t, stride_pool_h,
    # constexprs
    GROUP: tl.constexpr,
    D: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    """Scatter one (token, kv_head) row from k, v into pool[slot, pos, hk, :].
    slot = Block_to_slot_ptr[slot_mapping[i] // GROUP],
    pos  = slot_mapping[i] % GROUP.
    Skips i where slot_mapping < 0 (padding) or block_to_slot < 0 (no slot
    yet — shouldn't happen if the metadata builder pre-allocated correctly).
    Grid: (N, Hk).
    """
    i = tl.program_id(0)
    hk = tl.program_id(1)

    sm = tl.load(Slot_mapping_ptr + i)
    if sm < 0:
        return

    block_id = sm // GROUP
    pos = (sm % GROUP).to(tl.int64)
    in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
    if not in_range:
        return
    pool_slot = tl.load(Block_to_slot_ptr + block_id)
    if pool_slot < 0:
        return

    d = tl.arange(0, D)
    src_offs = i * stride_in_n + hk * stride_in_h + d
    k_row = tl.load(K_in_ptr + src_offs)
    v_row = tl.load(V_in_ptr + src_offs)

    dst_offs = (pool_slot.to(tl.int64) * stride_pool_b
                + pos * stride_pool_t
                + hk * stride_pool_h
                + d)
    tl.store(Pool_K_ptr + dst_offs, k_row)
    tl.store(Pool_V_ptr + dst_offs, v_row)


# ──────────────────────────────────────────────────────────────────────────────
# Stage α-2 capture-correct: ONE block_table-driven build-packed-KV kernel.
# Reads vLLM's persistent block_table + seq_lens directly (so a captured CUDA
# graph sees fresh data each replay), and writes the packed varlen fp16 K/V
# that flash_attn_varlen consumes. Fixed grid (B * MAX_BLOCKS_PER_REQ, Hk) so
# the launch dims are constant per captured batch size. Per (block, head):
#   - pool_slot >= 0  → fp16 already-rotated tokens copied from the tail pool
#                       (sink at k==0; in-progress tail at k==n_full).
#   - pool_slot <  0  → block lives in the int4 cache; dequant it in-place.
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _kvarn_build_packed_kv_kernel(
    Block_table_ptr,    # [B, max_blocks]                          int32
    Seq_lens_ptr,       # [B]                                      int32
    Cu_seqlens_ptr,     # [B+1]                                    int32 (prefix sum of seq_lens)
    Block_to_slot_ptr,  # [num_blocks_lookup]                      int32 (-1 = in int4 cache)
    KV_cache_ptr,       # [num_blocks, num_kv_heads, TILE_BYTES]   uint8
    Tail_K_pool_ptr,    # [POOL_SIZE, group, Hk, D]                fp16 (rotated)
    Tail_V_pool_ptr,    # [POOL_SIZE, group, Hk, D]                fp16
    K_out_ptr,          # [max_total_tokens, Hk, D]                fp16 (packed, rotated)
    V_out_ptr,          # [max_total_tokens, Hk, D]                fp16
    # strides
    stride_bt_b,
    stride_kv_b, stride_kv_h,
    stride_pool_b, stride_pool_t, stride_pool_h,
    stride_out_t, stride_out_h,
    # constexprs
    MAX_BLOCKS_PER_REQ: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
    K_PACKED_OFFSET: tl.constexpr,
    K_S_COL_OFFSET: tl.constexpr,
    K_ZP_OFFSET: tl.constexpr,
    K_S_ROW_OFFSET: tl.constexpr,
    V_PACKED_OFFSET: tl.constexpr,
    V_S_COL_OFFSET: tl.constexpr,
    V_S_ROW_OFFSET: tl.constexpr,
    V_ZP_OFFSET: tl.constexpr,
):
    """Grid: (B * MAX_BLOCKS_PER_REQ, Hk). One (request-block, head) per program.
    b is always < B by construction (grid dim 0 == B*MAX_BLOCKS_PER_REQ), so no
    runtime-B guard is needed — avoiding it keeps the kernel free of a
    non-constexpr early-return that Triton's type inference mishandles."""
    bk = tl.program_id(0)
    hk = tl.program_id(1)
    b = bk // MAX_BLOCKS_PER_REQ
    k = bk % MAX_BLOCKS_PER_REQ

    seq_len = tl.load(Seq_lens_ptr + b)
    # tokens this block contributes = clamp(seq_len - k*GROUP, 0, GROUP).
    # n_tok <= 0 ⇒ padding program (block beyond this request's length).
    rem = seq_len - k * GROUP
    n_tok = tl.minimum(tl.maximum(rem, 0), GROUP)
    if n_tok <= 0:
        return

    block_id = tl.load(Block_table_ptr + b * stride_bt_b + k)
    dst_base = tl.load(Cu_seqlens_ptr + b).to(tl.int64) + k.to(tl.int64) * GROUP

    d_offs = tl.arange(0, D)
    g_offs = tl.arange(0, GROUP)
    g_mask = g_offs < n_tok

    # Always-tensor slot lookup (masked load → -1 when block_id out of range).
    # Avoids mixing a Python int with a tensor across the branch below, which
    # Triton's type inference rejects in the general-batch compilation.
    in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
    safe_bid = tl.where(in_range, block_id, 0)
    pool_slot = tl.load(Block_to_slot_ptr + safe_bid, mask=in_range, other=-1)

    out_addrs = (
        (dst_base + g_offs)[:, None] * stride_out_t
        + hk * stride_out_h
        + d_offs[None, :]
    )

    if pool_slot >= 0:
        # ── fp16 tokens already rotated in the pool (sink / in-progress tail) ──
        pool_base = pool_slot.to(tl.int64) * stride_pool_b + hk * stride_pool_h
        src_addrs = pool_base + g_offs[:, None] * stride_pool_t + d_offs[None, :]
        K_chunk = tl.load(Tail_K_pool_ptr + src_addrs, mask=g_mask[:, None], other=0.0)
        V_chunk = tl.load(Tail_V_pool_ptr + src_addrs, mask=g_mask[:, None], other=0.0)
        tl.store(K_out_ptr + out_addrs, K_chunk, mask=g_mask[:, None])
        tl.store(V_out_ptr + out_addrs, V_chunk, mask=g_mask[:, None])
    else:
        # ── int4 quantised block: dequant in-place to rotated fp16 ─────────────
        tile_base = block_id.to(tl.int64) * stride_kv_b + hk * stride_kv_h
        g_byte_k = g_offs // 2
        g_shift_k = (g_offs % 2) * 4
        d_byte_v = d_offs // 2
        d_shift_v = (d_offs % 2) * 4

        sk_lo = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2).to(tl.uint16)
        sk_hi = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2 + 1).to(tl.uint16)
        s_col_K = ((sk_lo | (sk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        zk_lo = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2).to(tl.uint16)
        zk_hi = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2 + 1).to(tl.uint16)
        zp_K = ((zk_lo | (zk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        srk_lo = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + g_offs * 2).to(tl.uint16)
        srk_hi = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + g_offs * 2 + 1).to(tl.uint16)
        s_row_K = ((srk_lo | (srk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

        k_addrs = (tile_base + K_PACKED_OFFSET
                   + d_offs[:, None] * (GROUP // 2) + g_byte_k[None, :])
        k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)
        q_K = ((k_bytes >> g_shift_k[None, :]) & 0xF).to(tl.float32)
        K_rot = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]   # [D, GROUP]
        K_rot_out = tl.trans(K_rot)                                          # [GROUP, D]

        scv_lo = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2).to(tl.uint16)
        scv_hi = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2 + 1).to(tl.uint16)
        s_col_V = ((scv_lo | (scv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        srv_lo = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + g_offs * 2).to(tl.uint16)
        srv_hi = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + g_offs * 2 + 1).to(tl.uint16)
        s_row_V = ((srv_lo | (srv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        zpv_lo = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + g_offs * 2).to(tl.uint16)
        zpv_hi = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + g_offs * 2 + 1).to(tl.uint16)
        zp_V = ((zpv_lo | (zpv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

        v_addrs = (tile_base + V_PACKED_OFFSET
                   + g_offs[:, None] * (D // 2) + d_byte_v[None, :])
        v_bytes = tl.load(KV_cache_ptr + v_addrs).to(tl.int32)
        q_V = ((v_bytes >> d_shift_v[None, :]) & 0xF).to(tl.float32)
        V_rot = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]   # [GROUP, D]

        tl.store(K_out_ptr + out_addrs, K_rot_out.to(tl.float16), mask=g_mask[:, None])
        tl.store(V_out_ptr + out_addrs, V_rot.to(tl.float16), mask=g_mask[:, None])


# ──────────────────────────────────────────────────────────────────────────────
# Pool→packed gather: copy already-rotated fp16 tail-pool tokens (sink + tail)
# into the same packed buffer that flash_attn_varlen consumes.
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _kvarn_pool_gather_packed_kernel(  # noqa: SUPERSEDED by _kvarn_build_packed_kv_kernel; not called.
    Tail_K_pool_ptr,        # [POOL_SIZE, group, Hk, D] fp16 (rotated)
    Tail_V_pool_ptr,        # [POOL_SIZE, group, Hk, D] fp16
    Block_to_slot_ptr,      # [num_blocks_lookup] int32  block_id → pool slot
    Block_ids_ptr,          # [num_tasks] int32
    Src_starts_ptr,         # [num_tasks] int32
    Dst_offsets_ptr,        # [num_tasks] int32
    Lengths_ptr,            # [num_tasks] int32
    K_out_ptr,              # [max_total_tokens, Hk, D] fp16 (packed)
    V_out_ptr,              # [max_total_tokens, Hk, D] fp16
    # strides
    stride_pool_b, stride_pool_t, stride_pool_h,
    stride_out_t, stride_out_h,
    # constexprs
    NUM_BLOCKS_LOOKUP: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Copy a (block_id, [src_start:src_start+length]) chunk from the rotated
    tail pool into the packed varlen output buffer.

    Stage α-2: sparse slot indirection via Block_to_slot_ptr.
    """
    task_id = tl.program_id(0)
    hk = tl.program_id(1)

    block_id = tl.load(Block_ids_ptr + task_id)
    in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
    if in_range:
        pool_slot = tl.load(Block_to_slot_ptr + block_id)
        if pool_slot >= 0:
            src_start = tl.load(Src_starts_ptr + task_id).to(tl.int64)
            dst_off = tl.load(Dst_offsets_ptr + task_id).to(tl.int64)
            length = tl.load(Lengths_ptr + task_id)

            g_offs = tl.arange(0, GROUP)
            d_offs = tl.arange(0, D)
            mask_g = g_offs < length

            pool_base = pool_slot.to(tl.int64) * stride_pool_b + hk * stride_pool_h
            src_addrs = pool_base + (src_start + g_offs)[:, None] * stride_pool_t + d_offs[None, :]
            K_chunk = tl.load(Tail_K_pool_ptr + src_addrs, mask=mask_g[:, None], other=0.0)
            V_chunk = tl.load(Tail_V_pool_ptr + src_addrs, mask=mask_g[:, None], other=0.0)

            dst_token_offs = dst_off + g_offs
            dst_addrs = dst_token_offs[:, None] * stride_out_t + hk * stride_out_h + d_offs[None, :]
            tl.store(K_out_ptr + dst_addrs, K_chunk, mask=mask_g[:, None])
            tl.store(V_out_ptr + dst_addrs, V_chunk, mask=mask_g[:, None])


# ──────────────────────────────────────────────────────────────────────────────
# FUSED decode: dequant int4 in registers + flash-decoding online softmax.
# Reads the int4 cache (history) and the fp16 tail pool (sink / partial tail)
# directly — NEVER materializes a full fp16 K/V buffer to HBM. This is the path
# that can beat FP16/TurboQuant: per step it moves ~int4 (0.25x FP16) KV traffic
# for the bulk history instead of the materialize path's ~2.25x.
#
# One-stage flash-decode (correctness-first): grid (B, Hq). Each program handles
# one (request, query-head), loops that request's KV blocks with online softmax.
# Split-K parallelism over KV is a later optimization for long context.
#
# AUTO-TUNED across BLOCK_N ∈ {16, 32, 64}. The autotune key (D, GROUP, Q_PER_KV,
# K_BITS, V_BITS) makes Triton re-tune per model architecture × quant config:
# - Q_PER_KV=2 (Qwen3-0.6B) / =4 (Qwen3-4B) / =8 (Qwen3-30B-A3B, 32B) typically
#   favour different BLOCK_N; the autotuner picks once on first call and caches.
# - num_warps=8 was empirically slower at Q_PER_KV=8 (tl.dot with small M
#   under-utilises threads), so we only include 4 warps; if that turns out wrong
#   for some model, extending the config list is cheap.
# - num_stages=3 vs 2 showed no difference in the micro-bench; we keep =2.
# ──────────────────────────────────────────────────────────────────────────────


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
    ],
    key=["D", "GROUP", "Q_PER_KV", "K_BITS", "V_BITS"],
)
@triton.jit
def _kvarn_fused_decode_kernel(
    Q_ptr,              # [B, Hq, D]                               fp16 (rotated)
    Block_table_ptr,    # [B, max_blocks]                          int32
    Seq_lens_ptr,       # [B]                                      int32
    Block_to_slot_ptr,  # [num_blocks_lookup]                      int32 (-1 = int4)
    KV_cache_ptr,       # [num_blocks, num_kv_heads, TILE_BYTES]   uint8
    Tail_K_pool_ptr,    # [POOL_SIZE, group, Hk, D]                fp16 (rotated)
    Tail_V_pool_ptr,    # [POOL_SIZE, group, Hk, D]                fp16
    Out_ptr,            # [B, Hq, D]                               fp16 (rotated out)
    scale,
    # strides
    stride_q_b, stride_q_h,
    stride_bt_b,
    stride_kv_b, stride_kv_h,
    stride_pool_b, stride_pool_t, stride_pool_h,
    stride_o_b, stride_o_h,
    # constexprs
    MAX_BLOCKS_PER_REQ: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
    K_PACKED_OFFSET: tl.constexpr,
    K_S_COL_OFFSET: tl.constexpr,
    K_ZP_OFFSET: tl.constexpr,
    K_S_ROW_OFFSET: tl.constexpr,
    V_PACKED_OFFSET: tl.constexpr,
    V_S_COL_OFFSET: tl.constexpr,
    V_S_ROW_OFFSET: tl.constexpr,
    V_ZP_OFFSET: tl.constexpr,
):
    # GQA head-grouping: ONE program per (request, KV head) serves all Q_PER_KV
    # query heads that share this KV head, so each int4 K/V tile is dequantized
    # ONCE (not Q_PER_KV times). The redundant-dequant penalty of the per-Q-head
    # version scales with Q_PER_KV (4× on Qwen3-4B) and was the dominant cost.
    b = tl.program_id(0)
    hk = tl.program_id(1)
    qh = tl.arange(0, Q_PER_KV)                            # query heads in this group
    hq0 = hk * Q_PER_KV

    seq_len = tl.load(Seq_lens_ptr + b)

    d_offs = tl.arange(0, D)
    PACK_K: tl.constexpr = 8 // K_BITS
    PACK_V: tl.constexpr = 8 // V_BITS
    MASK_K: tl.constexpr = (1 << K_BITS) - 1
    MASK_V: tl.constexpr = (1 << V_BITS) - 1
    d_byte_v = d_offs // PACK_V
    d_shift_v = (d_offs % PACK_V) * V_BITS

    # q: [Q_PER_KV, D]
    q = tl.load(Q_ptr + b * stride_q_b + (hq0 + qh)[:, None] * stride_q_h
                + d_offs[None, :]).to(tl.float32)

    m_i = tl.full([Q_PER_KV], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_PER_KV], dtype=tl.float32)
    acc = tl.zeros([Q_PER_KV, D], dtype=tl.float32)

    n_blocks = (seq_len + GROUP - 1) // GROUP
    for k in range(0, n_blocks):
        rem = seq_len - k * GROUP
        n_tok = tl.minimum(tl.maximum(rem, 0), GROUP)

        block_id = tl.load(Block_table_ptr + b * stride_bt_b + k)
        in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
        safe_bid = tl.where(in_range, block_id, 0)
        pool_slot = tl.load(Block_to_slot_ptr + safe_bid, mask=in_range, other=-1)

        tile_base = block_id.to(tl.int64) * stride_kv_b + hk * stride_kv_h
        safe_slot = tl.where(pool_slot >= 0, pool_slot, 0)
        pool_base = safe_slot.to(tl.int64) * stride_pool_b + hk * stride_pool_h

        # Per-channel (per-d) K/V scales — constant across the 128 tokens; load
        # once. (Garbage but unused for pool blocks.)
        sk_lo = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2).to(tl.uint16)
        sk_hi = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2 + 1).to(tl.uint16)
        s_col_K = ((sk_lo | (sk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        zk_lo = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2).to(tl.uint16)
        zk_hi = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2 + 1).to(tl.uint16)
        zp_K = ((zk_lo | (zk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        scv_lo = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2).to(tl.uint16)
        scv_hi = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2 + 1).to(tl.uint16)
        s_col_V = ((scv_lo | (scv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

        for c0 in range(0, GROUP, BLOCK_N):
            cols = c0 + tl.arange(0, BLOCK_N)              # [BN] token indices in tile
            cmask = cols < n_tok

            if pool_slot >= 0:
                # fp16 already-rotated tokens in the pool (sink / partial tail).
                src = pool_base + cols[:, None] * stride_pool_t + d_offs[None, :]
                Kc = tl.load(Tail_K_pool_ptr + src, mask=cmask[:, None], other=0.0).to(tl.float32)  # [BN, D]
                Vc = tl.load(Tail_V_pool_ptr + src, mask=cmask[:, None], other=0.0).to(tl.float32)  # [BN, D]
                K_dg = tl.trans(Kc)                                       # [D, BN]
            else:
                # int4 dequant for this chunk of tokens (ONCE, shared by all q heads).
                cb_k = cols // PACK_K
                cs_k = (cols % PACK_K) * K_BITS
                srk_lo = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + cols * 2).to(tl.uint16)
                srk_hi = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + cols * 2 + 1).to(tl.uint16)
                s_row_K = ((srk_lo | (srk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)  # [BN]
                k_addrs = (tile_base + K_PACKED_OFFSET
                           + d_offs[:, None] * (GROUP // PACK_K) + cb_k[None, :])
                k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)                  # [D, BN]
                q_K = ((k_bytes >> cs_k[None, :]) & MASK_K).to(tl.float32)
                K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]      # [D, BN]

                srv_lo = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + cols * 2).to(tl.uint16)
                srv_hi = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + cols * 2 + 1).to(tl.uint16)
                s_row_V = ((srv_lo | (srv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)  # [BN]
                zpv_lo = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + cols * 2).to(tl.uint16)
                zpv_hi = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + cols * 2 + 1).to(tl.uint16)
                zp_V = ((zpv_lo | (zpv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)     # [BN]
                v_addrs = (tile_base + V_PACKED_OFFSET
                           + cols[:, None] * (D // PACK_V) + d_byte_v[None, :])
                v_bytes = tl.load(KV_cache_ptr + v_addrs).to(tl.int32)                  # [BN, D]
                q_V = ((v_bytes >> d_shift_v[None, :]) & MASK_V).to(tl.float32)
                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]        # [BN, D]

            # scores[h,c] = q·Kᵀ via tensor cores (q [Q_PER_KV,D] · K_dg [D,BN]).
            scores = tl.dot(q, K_dg)                                                   # [Q_PER_KV, BN]
            scores = tl.where(cmask[None, :], scores * scale, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))                            # [Q_PER_KV]
            p = tl.exp(scores - m_new[:, None])                                        # [Q_PER_KV, BN]
            alpha = tl.exp(m_i - m_new)                                                # [Q_PER_KV]
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, Vc)                                 # [Q_PER_KV, D]
            m_i = m_new

    out = (acc / l_i[:, None]).to(tl.float16)                                          # [Q_PER_KV, D]
    tl.store(Out_ptr + b * stride_o_b + (hq0 + qh)[:, None] * stride_o_h + d_offs[None, :], out)


# ──────────────────────────────────────────────────────────────────────────────
# SPLIT-K (flash-decoding) variant: stage 1 computes partial attention over a
# contiguous slice of each request's KV blocks; the extra grid dim parallelizes
# the KV sequence so ragged burst seqlens are load-balanced across SMs. Stage 2
# combines the per-split partials via the log-sum-exp trick. This is what makes
# the decode kernel competitive with TurboQuant's _tq_decode_stage1 at burst.
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _kvarn_fused_decode_stage1(
    Q_ptr, Block_table_ptr, Seq_lens_ptr, Block_to_slot_ptr, KV_cache_ptr,
    Tail_K_pool_ptr, Tail_V_pool_ptr,
    MidO_ptr,           # [N, NUM_KV_SPLITS, D]            fp32 (O_s = acc/l per split)
    MidLse_ptr,         # [N, NUM_KV_SPLITS]               fp32 (lse_s = m + log l)
    scale,
    stride_q_b, stride_q_h,
    stride_bt_b,
    stride_kv_b, stride_kv_h,
    stride_pool_b, stride_pool_t, stride_pool_h,
    stride_mo_n, stride_mo_s,          # MidO: row (N), split
    stride_ml_n,                       # MidLse: row (N)
    MAX_BLOCKS_PER_REQ: tl.constexpr, D: tl.constexpr, GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr, Q_PER_KV: tl.constexpr, NUM_KV_SPLITS: tl.constexpr,
    HQ: tl.constexpr, K_BITS: tl.constexpr, V_BITS: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
    K_PACKED_OFFSET: tl.constexpr, K_S_COL_OFFSET: tl.constexpr,
    K_ZP_OFFSET: tl.constexpr, K_S_ROW_OFFSET: tl.constexpr,
    V_PACKED_OFFSET: tl.constexpr, V_S_COL_OFFSET: tl.constexpr,
    V_S_ROW_OFFSET: tl.constexpr, V_ZP_OFFSET: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    split = tl.program_id(2)
    qh = tl.arange(0, Q_PER_KV)
    hq0 = hk * Q_PER_KV

    seq_len = tl.load(Seq_lens_ptr + b)
    n_blocks = (seq_len + GROUP - 1) // GROUP
    blocks_per_split = (n_blocks + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS
    blk_lo = split * blocks_per_split
    blk_hi = tl.minimum(blk_lo + blocks_per_split, n_blocks)

    d_offs = tl.arange(0, D)
    PACK_K: tl.constexpr = 8 // K_BITS
    PACK_V: tl.constexpr = 8 // V_BITS
    MASK_K: tl.constexpr = (1 << K_BITS) - 1
    MASK_V: tl.constexpr = (1 << V_BITS) - 1
    d_byte_v = d_offs // PACK_V
    d_shift_v = (d_offs % PACK_V) * V_BITS
    q = tl.load(Q_ptr + b * stride_q_b + (hq0 + qh)[:, None] * stride_q_h
                + d_offs[None, :]).to(tl.float32)

    m_i = tl.full([Q_PER_KV], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_PER_KV], dtype=tl.float32)
    acc = tl.zeros([Q_PER_KV, D], dtype=tl.float32)

    for k in range(blk_lo, blk_hi):
        rem = seq_len - k * GROUP
        n_tok = tl.minimum(tl.maximum(rem, 0), GROUP)
        block_id = tl.load(Block_table_ptr + b * stride_bt_b + k)
        in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
        safe_bid = tl.where(in_range, block_id, 0)
        pool_slot = tl.load(Block_to_slot_ptr + safe_bid, mask=in_range, other=-1)
        tile_base = block_id.to(tl.int64) * stride_kv_b + hk * stride_kv_h
        safe_slot = tl.where(pool_slot >= 0, pool_slot, 0)
        pool_base = safe_slot.to(tl.int64) * stride_pool_b + hk * stride_pool_h

        sk_lo = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2).to(tl.uint16)
        sk_hi = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2 + 1).to(tl.uint16)
        s_col_K = ((sk_lo | (sk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        zk_lo = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2).to(tl.uint16)
        zk_hi = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2 + 1).to(tl.uint16)
        zp_K = ((zk_lo | (zk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        scv_lo = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2).to(tl.uint16)
        scv_hi = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2 + 1).to(tl.uint16)
        s_col_V = ((scv_lo | (scv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

        for c0 in range(0, GROUP, BLOCK_N):
            cols = c0 + tl.arange(0, BLOCK_N)
            cmask = cols < n_tok
            if pool_slot >= 0:
                src = pool_base + cols[:, None] * stride_pool_t + d_offs[None, :]
                Kc = tl.load(Tail_K_pool_ptr + src, mask=cmask[:, None], other=0.0).to(tl.float32)
                Vc = tl.load(Tail_V_pool_ptr + src, mask=cmask[:, None], other=0.0).to(tl.float32)
                K_dg = tl.trans(Kc)
            else:
                cb_k = cols // PACK_K
                cs_k = (cols % PACK_K) * K_BITS
                srk_lo = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + cols * 2).to(tl.uint16)
                srk_hi = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + cols * 2 + 1).to(tl.uint16)
                s_row_K = ((srk_lo | (srk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
                k_addrs = (tile_base + K_PACKED_OFFSET + d_offs[:, None] * (GROUP // 2) + cb_k[None, :])
                k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)
                q_K = ((k_bytes >> cs_k[None, :]) & MASK_K).to(tl.float32)
                K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                srv_lo = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + cols * 2).to(tl.uint16)
                srv_hi = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + cols * 2 + 1).to(tl.uint16)
                s_row_V = ((srv_lo | (srv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
                zpv_lo = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + cols * 2).to(tl.uint16)
                zpv_hi = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + cols * 2 + 1).to(tl.uint16)
                zp_V = ((zpv_lo | (zpv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
                v_addrs = (tile_base + V_PACKED_OFFSET + cols[:, None] * (D // 2) + d_byte_v[None, :])
                v_bytes = tl.load(KV_cache_ptr + v_addrs).to(tl.int32)
                q_V = ((v_bytes >> d_shift_v[None, :]) & MASK_V).to(tl.float32)
                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]

            scores = tl.dot(q, K_dg)
            scores = tl.where(cmask[None, :], scores * scale, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, Vc)
            m_i = m_new

    # Partial output O_s = acc / l_i and lse_s = m_i + log(l_i); empty split → 0 / -inf.
    nonempty = l_i > 0
    O_s = acc / tl.where(nonempty, l_i, 1.0)[:, None]                  # [Q_PER_KV, D]
    lse_s = tl.where(nonempty, m_i + tl.log(tl.where(nonempty, l_i, 1.0)), -float("inf"))
    rows = b * HQ + hq0 + qh                                          # [Q_PER_KV] (= N rows)
    tl.store(MidO_ptr + rows[:, None] * stride_mo_n + split * stride_mo_s + d_offs[None, :], O_s)
    tl.store(MidLse_ptr + rows * stride_ml_n + split, lse_s)


@triton.jit
def _kvarn_fused_decode_stage2(
    MidO_ptr,           # [N, NUM_KV_SPLITS, D] fp32
    MidLse_ptr,         # [N, NUM_KV_SPLITS]    fp32
    Out_ptr,            # [N, D] fp16
    stride_mo_n, stride_mo_s,
    stride_ml_n,
    stride_o_n,
    D: tl.constexpr, NUM_KV_SPLITS: tl.constexpr,
):
    # One program per output row (= one (request, query-head)). Combine splits.
    n = tl.program_id(0)
    d_offs = tl.arange(0, D)
    s_offs = tl.arange(0, NUM_KV_SPLITS)
    lse = tl.load(MidLse_ptr + n * stride_ml_n + s_offs)             # [SPLITS]
    g = tl.max(lse, axis=0)                                          # global max lse
    w = tl.exp(lse - g)                                             # [SPLITS] weights
    denom = tl.sum(w, axis=0)
    # weighted sum of partial outputs
    O = tl.load(MidO_ptr + n * stride_mo_n + s_offs[:, None] * stride_mo_s + d_offs[None, :])  # [SPLITS, D]
    out = tl.sum(w[:, None] * O, axis=0) / denom                    # [D]
    tl.store(Out_ptr + n * stride_o_n + d_offs, out.to(tl.float16))


# ──────────────────────────────────────────────────────────────────────────────
# Python driver
# ──────────────────────────────────────────────────────────────────────────────


def kvarn_decode_attention(
    query: torch.Tensor,                # [B, Hq, D]   fp16/bf16
    kv_cache: torch.Tensor,             # [num_blocks, num_kv_heads, TILE_BYTES] uint8
    hadamard: torch.Tensor,             # [D, D]       fp32
    scale: float,
    cfg,
    impl,                               # KVarNAttentionImpl (has pool + scratch buffers)
    md,                                 # KVarNMetadata (carries the precomputed task plan)
) -> torch.Tensor:
    """Stage 5.a decode driver — dequant + FlashAttention.

    For each layer call:

        1. Rotate the query: q_rot = q · H.
        2. Build a packed varlen ``[total_kv_tokens, Hk, D]`` fp16 K, V by
           dequantising the quantised blocks (``_kvarn_dequant_blocks_kernel``)
           and copying sink + trailing-tail tokens from the rotated fp16 tail
           pool (``_kvarn_pool_gather_packed_kernel``).
        3. Call ``flash_attn_varlen_func`` with the assembled K, V.
        4. Un-rotate the output: out = out_rot · H.

    The task plan (dequant_block_ids, pool_block_ids, dst_offsets, lengths,
    cu_seqlens) is precomputed once per batch in ``KVarNMetadataBuilder.build``
    and passed in via ``md`` — no per-layer host→GPU allocations.

    Output: ``[B, Hq, D]`` in ``query``'s dtype, in the un-rotated frame.
    """
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    B, Hq, D = query.shape
    Hk = kv_cache.shape[1]
    device = query.device
    out_dtype = query.dtype
    group = cfg.group
    N = B * Hq  # rows for the 2D Q rotation matmul

    # 1. Q rotation — single fp16 tensor-core matmul into the persistent buffer.
    #    Use the SAME fp16 Hadamard the K/V store used (_H_fp16) so QKᵀ stays
    #    invariant; the old fp32 path added two [N,D] copies + a slower fp32 GEMM
    #    per layer for no accuracy benefit (H is orthonormal, well-conditioned).
    H16 = impl._H_fp16 if impl._H_fp16 is not None else hadamard.to(torch.float16)
    q_rot_fp16 = impl._q_rot_fp16_buf[:N]
    with torch.profiler.record_function("kvarn_q_rotation"):
        torch.mm(query.reshape(N, D), H16, out=q_rot_fp16)

    # 2+3. Attention. Two paths (KVARN_FUSED_DECODE, default fused):
    #   FUSED      — one kernel reads int4/pool directly, dequants in registers,
    #                online-softmax; never materializes fp16 K/V to HBM. Moves
    #                ~0.25x FP16 KV traffic for the bulk history → the only path
    #                that can beat FP16/TurboQuant on latency.
    #   MATERIALIZE — build packed fp16 K/V then stock FlashAttention (≥2.25x
    #                FP16 KV traffic; kept for A/B and as a fallback).
    max_blocks_per_req = md.fa_max_blocks_per_req
    use_fused = os.environ.get("KVARN_FUSED_DECODE", "1") == "1"
    # Single-stage kernel is @triton.autotune'd over BLOCK_N (keyed on
    # D/GROUP/Q_PER_KV/K_BITS/V_BITS) — no BLOCK_N/num_warps/num_stages here.
    # Split-K path (stage1) keeps explicit knobs for the rare low-batch regime.
    _bn = int(os.environ.get("KVARN_BLOCK_N", "16"))
    _nw = int(os.environ.get("KVARN_NUM_WARPS", "4"))
    _ns = int(os.environ.get("KVARN_NUM_STAGES", "2"))
    common = dict(
        MAX_BLOCKS_PER_REQ=max_blocks_per_req, D=D, GROUP=group,
        Q_PER_KV=Hq // Hk, K_BITS=cfg.key_bits, V_BITS=cfg.value_bits,
        NUM_BLOCKS_LOOKUP=impl._block_lookup_size,
        K_PACKED_OFFSET=cfg.k_packed_offset, K_S_COL_OFFSET=cfg.k_s_col_offset,
        K_ZP_OFFSET=cfg.k_zp_offset, K_S_ROW_OFFSET=cfg.k_s_row_offset,
        V_PACKED_OFFSET=cfg.v_packed_offset, V_S_COL_OFFSET=cfg.v_s_col_offset,
        V_S_ROW_OFFSET=cfg.v_s_row_offset, V_ZP_OFFSET=cfg.v_zp_offset,
    )
    # SPLIT-K (KVARN_SPLIT_K=1): two-stage flash-decoding — only a win in the
    # LOW-batch / long-context regime (few programs ⇒ the KV-split dim adds the
    # missing parallelism). At BURST (high batch) the single-stage (B,Hk) grid
    # already saturates the GPU, so split-K's mid-buffer round-trip + stage-2 +
    # empty-split waste roughly HALVE throughput. Default: single-stage.
    use_fused = use_fused and True
    split_k = use_fused and os.environ.get("KVARN_SPLIT_K", "0") == "1"
    if use_fused and not split_k:
        fused_out = impl._fused_out_buf[:N]               # [N, D] fp16
        with torch.profiler.record_function("kvarn_fused_decode"):
            _kvarn_fused_decode_kernel[(B, Hk)](
                q_rot_fp16, md.block_table, md.seq_lens, impl._block_to_slot_t,
                kv_cache, impl._tail_K_pool, impl._tail_V_pool, fused_out, scale,
                Hq * D, D, md.block_table.stride(0),
                kv_cache.stride(0), kv_cache.stride(1),
                impl._tail_K_pool.stride(0), impl._tail_K_pool.stride(1), impl._tail_K_pool.stride(2),
                Hq * D, D, **common,                       # autotune fills BLOCK_N + warps + stages
            )
        output_rot = fused_out
    elif split_k:
        SPLITS = KVARN_NUM_KV_SPLITS
        mid_o = impl._mid_o_buf
        mid_lse = impl._mid_lse_buf
        fused_out = impl._fused_out_buf[:N]
        with torch.profiler.record_function("kvarn_fused_decode_s1"):
            _kvarn_fused_decode_stage1[(B, Hk, SPLITS)](
                q_rot_fp16, md.block_table, md.seq_lens, impl._block_to_slot_t,
                kv_cache, impl._tail_K_pool, impl._tail_V_pool, mid_o, mid_lse, scale,
                Hq * D, D, md.block_table.stride(0),
                kv_cache.stride(0), kv_cache.stride(1),
                impl._tail_K_pool.stride(0), impl._tail_K_pool.stride(1), impl._tail_K_pool.stride(2),
                mid_o.stride(0), mid_o.stride(1), mid_lse.stride(0),
                BLOCK_N=_bn, NUM_KV_SPLITS=SPLITS, HQ=Hq,
                num_warps=_nw, num_stages=_ns, **common,
            )
        with torch.profiler.record_function("kvarn_fused_decode_s2"):
            _kvarn_fused_decode_stage2[(N,)](
                mid_o, mid_lse, fused_out,
                mid_o.stride(0), mid_o.stride(1), mid_lse.stride(0), fused_out.stride(0),
                D=D, NUM_KV_SPLITS=SPLITS, num_warps=2,
            )
        output_rot = fused_out
    else:
        K_packed = impl._fa_K_buf
        V_packed = impl._fa_V_buf
        with torch.profiler.record_function("kvarn_build_packed_kv"):
            _kvarn_build_packed_kv_kernel[(B * max_blocks_per_req, Hk)](
                md.block_table, md.seq_lens, md.fa_cu_seqlens_k,
                impl._block_to_slot_t,
                kv_cache, impl._tail_K_pool, impl._tail_V_pool,
                K_packed, V_packed,
                md.block_table.stride(0),
                kv_cache.stride(0), kv_cache.stride(1),
                impl._tail_K_pool.stride(0), impl._tail_K_pool.stride(1), impl._tail_K_pool.stride(2),
                K_packed.stride(0), K_packed.stride(1),
                MAX_BLOCKS_PER_REQ=max_blocks_per_req,
                D=D, GROUP=group,
                NUM_BLOCKS_LOOKUP=impl._block_lookup_size,
                K_PACKED_OFFSET=cfg.k_packed_offset, K_S_COL_OFFSET=cfg.k_s_col_offset,
                K_ZP_OFFSET=cfg.k_zp_offset, K_S_ROW_OFFSET=cfg.k_s_row_offset,
                V_PACKED_OFFSET=cfg.v_packed_offset, V_S_COL_OFFSET=cfg.v_s_col_offset,
                V_S_ROW_OFFSET=cfg.v_s_row_offset, V_ZP_OFFSET=cfg.v_zp_offset,
                num_warps=4, num_stages=2,
            )
        with torch.profiler.record_function("kvarn_flash_attn"):
            output_rot = flash_attn_varlen_func(
                q=q_rot_fp16.view(B, Hq, D),
                k=K_packed, v=V_packed,
                cu_seqlens_q=md.fa_cu_seqlens_q,
                cu_seqlens_k=md.fa_cu_seqlens_k,
                max_seqlen_q=1,
                max_seqlen_k=md.fa_max_seqlen_k_fixed,
                softmax_scale=scale,
                causal=False,
            )

    # 4. Un-rotate output — single fp16 tensor-core matmul (V was rotated, so
    #    the attention output lives in the rotated frame). out = out_rot · H.
    with torch.profiler.record_function("kvarn_output_unrotation"):
        out_unrot = torch.mm(output_rot.reshape(N, D), H16)   # fresh fp16
        return out_unrot.view(B, Hq, D).to(out_dtype)
