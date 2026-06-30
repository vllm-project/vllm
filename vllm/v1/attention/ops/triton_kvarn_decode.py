# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN decode and speculative-verify Triton kernels.

Store/flush kernels pack filled KV tiles into the int4 cache; the decode and
verify paths attend over that cache (dequantizing selected tiles and reading the
fp16 tail pool for sink / unflushed blocks). Python entry points:
``kvarn_decode_attention`` and ``kvarn_verify_attention``.

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
KVARN_NUM_KV_SPLITS = int(os.environ.get("KVARN_NUM_KV_SPLITS", "16"))
KVARN_MAX_KV_SPLITS = 64  # cap of the context-adaptive schedule below

# Shared autotune space for the decode kernels (single-token, split-K stage1,
# and spec-verify). ncu on the burst single-stage kernel showed it pinned at
# ~25% occupancy (register-limited to 3 blocks/SM) and bottlenecked on L1/TEX
# transaction rate, not DRAM bandwidth. So beyond BLOCK_N x num_warps we let the
# autotuner trade pipelining for occupancy: num_stages=1 (no pipeline buffers,
# fewer registers) and a couple of maxnreg caps (more resident blocks to hide
# the L1 latency). The autotuner keeps whichever is fastest per shape, so this
# is pure upside; online-softmax / split-K make the output reduction-order
# invariant (fp noise only), independent of the config chosen.
_DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bn in (16, 32, 64)
    for nw in (2, 4)
    for ns in (1, 2)
] + [
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2, maxnreg=mr)
    for mr in (64, 96)
]


def adaptive_num_kv_splits(max_blocks_per_req: int) -> int:
    """Context-adaptive split-K count (single source of truth for the decode
    driver AND the partial-buffer sizing, so they can never diverge).

    Depends only on the deployment's max_model_len (via max_blocks_per_req =
    ceil(max_model_len/group)), so it is CONSTANT per deployment -> CUDA-graph
    safe and changes nothing for short-context deployments. 16 split under-
    parallelized the stage-1 (B, Hk, SPLITS) grid at low batch: the single-token
    decode microbench (Qwen3-4B, ctx 4.6K) measured 37us at 16 vs ~27us at 32,
    a ~28% stage-1 win, growing at longer ctx (16K: 82->49us). Split-K is
    log-sum-exp-combined, so the count never changes the OUTPUT, only occupancy;
    32 is the floor up to 256 blocks. KVARN_NUM_KV_SPLITS overrides."""
    env = os.environ.get("KVARN_NUM_KV_SPLITS")
    if env is not None:
        return int(env)
    if max_blocks_per_req <= 256:
        return 32
    return KVARN_MAX_KV_SPLITS


# ──────────────────────────────────────────────────────────────────────────────


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
    K_in_ptr,  # [N, Hk, D]                    fp16 (already rotated)
    V_in_ptr,  # [N, Hk, D]                    fp16
    Slot_mapping_ptr,  # [N]                           int32   (slot < 0 ⇒ pad)
    Block_to_slot_ptr,  # [num_blocks_lookup]           int32   (-1 = no slot)
    Pool_K_ptr,  # [POOL_SIZE, group, Hk, D]     fp16
    Pool_V_ptr,  # [POOL_SIZE, group, Hk, D]     fp16
    # strides
    stride_in_n,
    stride_in_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
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

    dst_offs = (
        pool_slot.to(tl.int64) * stride_pool_b
        + pos * stride_pool_t
        + hk * stride_pool_h
        + d
    )
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
    Block_table_ptr,  # [B, max_blocks]                          int32
    Seq_lens_ptr,  # [B]                                      int32
    Cu_seqlens_ptr,  # [B+1] int32 (prefix sum of seq_lens)
    Block_to_slot_ptr,  # [num_blocks_lookup] int32 (-1 = in int4 cache)
    KV_cache_ptr,  # [num_blocks, num_kv_heads, TILE_BYTES]   uint8
    Tail_K_pool_ptr,  # [POOL_SIZE, group, Hk, D]                fp16 (rotated)
    Tail_V_pool_ptr,  # [POOL_SIZE, group, Hk, D]                fp16
    K_out_ptr,  # [max_total_tokens, Hk, D]                fp16 (packed, rotated)
    V_out_ptr,  # [max_total_tokens, Hk, D]                fp16
    # strides
    stride_bt_b,
    stride_kv_b,
    stride_kv_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
    stride_out_t,
    stride_out_h,
    # constexprs
    MAX_BLOCKS_PER_REQ: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
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
        # ── quantised block: dequant in-place to rotated fp16. Bit-width-aware:
        # the K/V packing strides follow K_BITS / V_BITS exactly as
        # in the fused decode kernels. The old hardcoded 4-bit V layout (stride
        # D//2, shift (d%2)*4, mask 0xF) read past the 2-bit-V packed region of
        # the default k4v2 preset into the V scales -> garbage V on this path.
        PACK_K: tl.constexpr = 8 // K_BITS
        PACK_V: tl.constexpr = 8 // V_BITS
        MASK_K: tl.constexpr = (1 << K_BITS) - 1
        MASK_V: tl.constexpr = (1 << V_BITS) - 1
        tile_base = block_id.to(tl.int64) * stride_kv_b + hk * stride_kv_h
        g_byte_k = g_offs // PACK_K
        g_shift_k = (g_offs % PACK_K) * K_BITS
        d_byte_v = d_offs // PACK_V
        d_shift_v = (d_offs % PACK_V) * V_BITS

        sk_lo = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        sk_hi = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        s_col_K = ((sk_lo | (sk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        zk_lo = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        zk_hi = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        zp_K = ((zk_lo | (zk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        srk_lo = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + g_offs * 2).to(
            tl.uint16
        )
        srk_hi = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + g_offs * 2 + 1).to(
            tl.uint16
        )
        s_row_K = ((srk_lo | (srk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

        k_addrs = (
            tile_base
            + K_PACKED_OFFSET
            + d_offs[:, None] * (GROUP // PACK_K)
            + g_byte_k[None, :]
        )
        k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)
        q_K = ((k_bytes >> g_shift_k[None, :]) & MASK_K).to(tl.float32)
        K_rot = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[
            None, :
        ]  # [D, GROUP]
        K_rot_out = tl.trans(K_rot)  # [GROUP, D]

        scv_lo = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        scv_hi = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        s_col_V = ((scv_lo | (scv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        srv_lo = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + g_offs * 2).to(
            tl.uint16
        )
        srv_hi = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + g_offs * 2 + 1).to(
            tl.uint16
        )
        s_row_V = ((srv_lo | (srv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        zpv_lo = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + g_offs * 2).to(
            tl.uint16
        )
        zpv_hi = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + g_offs * 2 + 1).to(
            tl.uint16
        )
        zp_V = ((zpv_lo | (zpv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

        v_addrs = (
            tile_base
            + V_PACKED_OFFSET
            + g_offs[:, None] * (D // PACK_V)
            + d_byte_v[None, :]
        )
        v_bytes = tl.load(KV_cache_ptr + v_addrs).to(tl.int32)
        q_V = ((v_bytes >> d_shift_v[None, :]) & MASK_V).to(tl.float32)
        V_rot = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[
            None, :
        ]  # [GROUP, D]

        tl.store(K_out_ptr + out_addrs, K_rot_out.to(tl.float16), mask=g_mask[:, None])
        tl.store(V_out_ptr + out_addrs, V_rot.to(tl.float16), mask=g_mask[:, None])


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
    configs=_DECODE_AUTOTUNE_CONFIGS,
    key=["D", "GROUP", "Q_PER_KV", "K_BITS", "V_BITS"],
)
@triton.jit
def _kvarn_fused_decode_kernel(
    Q_ptr,  # [B, Hq, D]                               fp16 (rotated)
    Req_row_ptr,  # [B] int32 — block-table row per program row (VQ_INDIRECT)
    Block_table_ptr,  # [B, max_blocks]                          int32
    Seq_lens_ptr,  # [B]                                      int32
    Block_to_slot_ptr,  # [num_blocks_lookup]                      int32 (-1 = int4)
    KV_cache_ptr,  # [num_blocks, num_kv_heads, TILE_BYTES]   uint8
    Tail_K_pool_ptr,  # [POOL_SIZE, group, Hk, D]                fp16 (rotated)
    Tail_V_pool_ptr,  # [POOL_SIZE, group, Hk, D]                fp16
    Out_ptr,  # [B, Hq, D]                               fp16 (rotated out)
    scale,
    # strides
    stride_q_b,
    stride_q_h,
    stride_bt_b,
    stride_kv_b,
    stride_kv_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
    stride_o_b,
    stride_o_h,
    # constexprs
    MAX_BLOCKS_PER_REQ: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    Q_PER_KV_PAD: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
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
    VQ_INDIRECT: tl.constexpr,
):
    # GQA head-grouping: ONE program per (request, KV head) serves all Q_PER_KV
    # query heads that share this KV head, so each int4 K/V tile is dequantized
    # ONCE (not Q_PER_KV times). The redundant-dequant penalty of the per-Q-head
    # version scales with Q_PER_KV (4× on Qwen3-4B) and was the dominant cost.
    # Q_PER_KV is padded to a power of 2 (Q_PER_KV_PAD) because tl.arange / tl.dot
    # require pow2 dims; padded query heads are masked off (e.g. Qwen3.5 GQA 24/4
    # = ratio 6 -> pad to 8).
    #
    # VQ_INDIRECT (multi-query verify): program row b is a QUERY TOKEN, not a
    # request — Req_row_ptr[b] gives its block-table row and Seq_lens_ptr[b] its
    # bottom-right causal length (cached_len + token_idx + 1). Q/Out stay
    # token-major, so everything else is unchanged: the speculative-decode
    # verify step runs the same dual-source (int4 + pool) online-softmax read
    # instead of materializing the whole context to fp16 scratch (O(context)
    # per step — the long-context MTP slowdown).
    b = tl.program_id(0)
    hk = tl.program_id(1)
    qh = tl.arange(0, Q_PER_KV_PAD)  # padded query-head lane
    qmask = qh < Q_PER_KV  # real heads in this group
    hq0 = hk * Q_PER_KV

    bt_row = b
    if VQ_INDIRECT:
        bt_row = tl.load(Req_row_ptr + b)
    seq_len = tl.load(Seq_lens_ptr + b)
    # Padded rows (uniform-batch graph capture/replay pads the token count)
    # carry seq_len <= 0: nothing to attend, the output row is never read.
    if seq_len <= 0:
        return

    d_offs = tl.arange(0, D)
    PACK_K: tl.constexpr = 8 // K_BITS
    PACK_V: tl.constexpr = 8 // V_BITS
    MASK_K: tl.constexpr = (1 << K_BITS) - 1
    MASK_V: tl.constexpr = (1 << V_BITS) - 1
    d_byte_v = d_offs // PACK_V
    d_shift_v = (d_offs % PACK_V) * V_BITS

    # q: [Q_PER_KV_PAD, D] — padded lanes masked to 0 (no OOB read past Hq).
    q = tl.load(
        Q_ptr + b * stride_q_b + (hq0 + qh)[:, None] * stride_q_h + d_offs[None, :],
        mask=qmask[:, None],
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full([Q_PER_KV_PAD], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_PER_KV_PAD], dtype=tl.float32)
    acc = tl.zeros([Q_PER_KV_PAD, D], dtype=tl.float32)

    n_blocks = (seq_len + GROUP - 1) // GROUP
    # Sliding-window: this query (last token) only attends to the last
    # SLIDING_WINDOW keys, so start the block loop at the first block that
    # overlaps the window (massive saving: ~window/GROUP blocks instead of all).
    win_start = 0
    blk_lo = 0
    if SLIDING_WINDOW > 0:
        win_start = tl.maximum(seq_len - SLIDING_WINDOW, 0)
        blk_lo = win_start // GROUP
    for k in range(blk_lo, n_blocks):
        rem = seq_len - k * GROUP
        n_tok = tl.minimum(tl.maximum(rem, 0), GROUP)

        block_id = tl.load(Block_table_ptr + bt_row * stride_bt_b + k)
        in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
        safe_bid = tl.where(in_range, block_id, 0)
        pool_slot = tl.load(Block_to_slot_ptr + safe_bid, mask=in_range, other=-1)

        tile_base = block_id.to(tl.int64) * stride_kv_b + hk * stride_kv_h
        safe_slot = tl.where(pool_slot >= 0, pool_slot, 0)
        pool_base = safe_slot.to(tl.int64) * stride_pool_b + hk * stride_pool_h

        # Per-channel (per-d) K/V scales — constant across the 128 tokens; load
        # once. fp16 fields live at even byte offsets in the uint8 tile, so load
        # them as a single uint16 (half the L1 transactions of the lo/hi byte
        # pair). (Garbage but unused for pool blocks.)
        ku16 = (KV_cache_ptr + tile_base).to(tl.pointer_type(tl.uint16))
        s_col_K = (
            tl.load(ku16 + (K_S_COL_OFFSET // 2) + d_offs)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        zp_K = (
            tl.load(ku16 + (K_ZP_OFFSET // 2) + d_offs)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        s_col_V = (
            tl.load(ku16 + (V_S_COL_OFFSET // 2) + d_offs)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )

        for c0 in range(0, GROUP, BLOCK_N):
            cols = c0 + tl.arange(0, BLOCK_N)  # [BN] token indices in tile
            cmask = cols < n_tok
            if SLIDING_WINDOW > 0:  # mask keys before the window boundary
                cmask = cmask & ((k * GROUP + cols) >= win_start)

            if pool_slot >= 0:
                # fp16 already-rotated tokens in the pool (sink / partial tail).
                src = pool_base + cols[:, None] * stride_pool_t + d_offs[None, :]
                Kc = tl.load(Tail_K_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )  # [BN, D]
                Vc = tl.load(Tail_V_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )  # [BN, D]
                K_dg = tl.trans(Kc)  # [D, BN]
            else:
                # int4 dequant for this chunk of tokens (ONCE, shared by all q heads).
                cb_k = cols // PACK_K
                cs_k = (cols % PACK_K) * K_BITS
                s_row_K = (
                    tl.load(ku16 + (K_S_ROW_OFFSET // 2) + cols)
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )  # [BN]
                k_addrs = (
                    tile_base
                    + K_PACKED_OFFSET
                    + d_offs[:, None] * (GROUP // PACK_K)
                    + cb_k[None, :]
                )
                k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)  # [D, BN]
                q_K = ((k_bytes >> cs_k[None, :]) & MASK_K).to(tl.float32)
                K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[
                    None, :
                ]  # [D, BN]

                s_row_V = (
                    tl.load(ku16 + (V_S_ROW_OFFSET // 2) + cols)
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )  # [BN]
                zp_V = (
                    tl.load(ku16 + (V_ZP_OFFSET // 2) + cols)
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )  # [BN]
                v_addrs = (
                    tile_base
                    + V_PACKED_OFFSET
                    + cols[:, None] * (D // PACK_V)
                    + d_byte_v[None, :]
                )
                v_bytes = tl.load(KV_cache_ptr + v_addrs).to(tl.int32)  # [BN, D]
                q_V = ((v_bytes >> d_shift_v[None, :]) & MASK_V).to(tl.float32)
                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[
                    None, :
                ]  # [BN, D]

            # scores[h,c] = q·Kᵀ via tensor cores (q [Q_PER_KV,D] · K_dg [D,BN]).
            scores = tl.dot(q, K_dg)  # [Q_PER_KV, BN]
            scores = tl.where(cmask[None, :], scores * scale, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))  # [Q_PER_KV]
            p = tl.exp(scores - m_new[:, None])  # [Q_PER_KV, BN]
            alpha = tl.exp(m_i - m_new)  # [Q_PER_KV]
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, Vc)  # [Q_PER_KV, D]
            m_i = m_new

    out = (acc / l_i[:, None]).to(tl.float16)  # [Q_PER_KV_PAD, D]
    tl.store(
        Out_ptr + b * stride_o_b + (hq0 + qh)[:, None] * stride_o_h + d_offs[None, :],
        out,
        mask=qmask[:, None],
    )


# ──────────────────────────────────────────────────────────────────────────────
# SPLIT-K (flash-decoding) variant: stage 1 computes partial attention over a
# contiguous slice of each request's KV blocks; the extra grid dim parallelizes
# the KV sequence so ragged burst seqlens are load-balanced across SMs. Stage 2
# combines the per-split partials via the log-sum-exp trick. This is what makes
# the decode kernel competitive with TurboQuant's _tq_decode_stage1 at burst.
# ──────────────────────────────────────────────────────────────────────────────


# AUTO-TUNED across BLOCK_N x num_warps x num_stages (keyed per model arch ×
# quant config, like the single-stage kernel). Stage1 was previously launched
# with a hardcoded BLOCK_N=16 / num_warps=4; the microbench (Qwen3-4B) showed
# BLOCK_N=32 / num_warps=2 is ~25-40% faster across 4.6K-32K ctx. Split-K is
# LSE-combined so BLOCK_N never affects the output. Warmed in
# _warm_decode_kernels (pre-CUDA-graph-capture), so autotune never triggers
# mid-capture.
@triton.autotune(
    configs=_DECODE_AUTOTUNE_CONFIGS,
    key=["D", "GROUP", "Q_PER_KV", "K_BITS", "V_BITS"],
)
@triton.jit
def _kvarn_fused_decode_stage1(
    Q_ptr,
    Req_row_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Block_to_slot_ptr,
    KV_cache_ptr,
    Tail_K_pool_ptr,
    Tail_V_pool_ptr,
    MidO_ptr,  # [N, NUM_KV_SPLITS, D]            fp32 (O_s = acc/l per split)
    MidLse_ptr,  # [N, NUM_KV_SPLITS]               fp32 (lse_s = m + log l)
    scale,
    stride_q_b,
    stride_q_h,
    stride_bt_b,
    stride_kv_b,
    stride_kv_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
    stride_mo_n,
    stride_mo_s,  # MidO: row (N), split
    stride_ml_n,  # MidLse: row (N)
    MAX_BLOCKS_PER_REQ: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    HQ: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    Q_PER_KV_PAD: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
    K_PACKED_OFFSET: tl.constexpr,
    K_S_COL_OFFSET: tl.constexpr,
    K_ZP_OFFSET: tl.constexpr,
    K_S_ROW_OFFSET: tl.constexpr,
    V_PACKED_OFFSET: tl.constexpr,
    V_S_COL_OFFSET: tl.constexpr,
    V_S_ROW_OFFSET: tl.constexpr,
    V_ZP_OFFSET: tl.constexpr,
    VQ_INDIRECT: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    split = tl.program_id(2)
    qh = tl.arange(0, Q_PER_KV_PAD)  # padded to pow2; mask below
    qmask = qh < Q_PER_KV
    hq0 = hk * Q_PER_KV

    # VQ_INDIRECT: row b is a query TOKEN (verify step); see the single-stage
    # kernel's note. Block table via Req_row_ptr, causal length via Seq_lens.
    bt_row = b
    if VQ_INDIRECT:
        bt_row = tl.load(Req_row_ptr + b)
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
    q = tl.load(
        Q_ptr + b * stride_q_b + (hq0 + qh)[:, None] * stride_q_h + d_offs[None, :],
        mask=qmask[:, None],
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full([Q_PER_KV_PAD], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_PER_KV_PAD], dtype=tl.float32)
    acc = tl.zeros([Q_PER_KV_PAD, D], dtype=tl.float32)

    for k in range(blk_lo, blk_hi):
        rem = seq_len - k * GROUP
        n_tok = tl.minimum(tl.maximum(rem, 0), GROUP)
        block_id = tl.load(Block_table_ptr + bt_row * stride_bt_b + k)
        in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
        safe_bid = tl.where(in_range, block_id, 0)
        pool_slot = tl.load(Block_to_slot_ptr + safe_bid, mask=in_range, other=-1)
        tile_base = block_id.to(tl.int64) * stride_kv_b + hk * stride_kv_h
        safe_slot = tl.where(pool_slot >= 0, pool_slot, 0)
        pool_base = safe_slot.to(tl.int64) * stride_pool_b + hk * stride_pool_h

        # Single uint16 load per fp16 scale (half the L1 transactions of the
        # lo/hi byte pair); fp16 fields are at even byte offsets in the tile.
        ku16 = (KV_cache_ptr + tile_base).to(tl.pointer_type(tl.uint16))
        s_col_K = (
            tl.load(ku16 + (K_S_COL_OFFSET // 2) + d_offs)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        zp_K = (
            tl.load(ku16 + (K_ZP_OFFSET // 2) + d_offs)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        s_col_V = (
            tl.load(ku16 + (V_S_COL_OFFSET // 2) + d_offs)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )

        for c0 in range(0, GROUP, BLOCK_N):
            cols = c0 + tl.arange(0, BLOCK_N)
            cmask = cols < n_tok
            if SLIDING_WINDOW > 0:
                cmask = cmask & (
                    (k * GROUP + cols) >= tl.maximum(seq_len - SLIDING_WINDOW, 0)
                )
            if pool_slot >= 0:
                src = pool_base + cols[:, None] * stride_pool_t + d_offs[None, :]
                Kc = tl.load(Tail_K_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )
                Vc = tl.load(Tail_V_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )
                K_dg = tl.trans(Kc)
            else:
                cb_k = cols // PACK_K
                cs_k = (cols % PACK_K) * K_BITS
                s_row_K = (
                    tl.load(ku16 + (K_S_ROW_OFFSET // 2) + cols)
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
                k_addrs = (
                    tile_base
                    + K_PACKED_OFFSET
                    + d_offs[:, None] * (GROUP // PACK_K)
                    + cb_k[None, :]
                )
                k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)
                q_K = ((k_bytes >> cs_k[None, :]) & MASK_K).to(tl.float32)
                K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                s_row_V = (
                    tl.load(ku16 + (V_S_ROW_OFFSET // 2) + cols)
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
                zp_V = (
                    tl.load(ku16 + (V_ZP_OFFSET // 2) + cols)
                    .to(tl.float16, bitcast=True)
                    .to(tl.float32)
                )
                # FIX: V packed-row stride is D/PACK_V bytes (PACK_V = 8/V_BITS).
                # Was hardcoded `D // 2` (correct only for 4-bit V); with the shipped
                # k4v2 preset (V_BITS=2 -> PACK_V=4) it strode 2x too far -> read
                # garbage V + indexed past the tile (OOB illegal-access at long ctx).
                # The single-stage kernel already used (D // PACK_V); this matches it.
                v_addrs = (
                    tile_base
                    + V_PACKED_OFFSET
                    + cols[:, None] * (D // PACK_V)
                    + d_byte_v[None, :]
                )
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
    O_s = acc / tl.where(nonempty, l_i, 1.0)[:, None]  # [Q_PER_KV, D]
    lse_s = tl.where(
        nonempty, m_i + tl.log(tl.where(nonempty, l_i, 1.0)), -float("inf")
    )
    rows = b * HQ + hq0 + qh  # [Q_PER_KV_PAD] (= N rows)
    tl.store(
        MidO_ptr + rows[:, None] * stride_mo_n + split * stride_mo_s + d_offs[None, :],
        O_s,
        mask=qmask[:, None],
    )
    tl.store(MidLse_ptr + rows * stride_ml_n + split, lse_s, mask=qmask)


@triton.jit
def _kvarn_fused_decode_stage2(
    MidO_ptr,  # [N, NUM_KV_SPLITS, D] fp32
    MidLse_ptr,  # [N, NUM_KV_SPLITS]    fp32
    Out_ptr,  # [N, D] fp16
    stride_mo_n,
    stride_mo_s,
    stride_ml_n,
    stride_o_n,
    D: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
):
    # One program per output row (= one (request, query-head)). Combine splits.
    n = tl.program_id(0)
    d_offs = tl.arange(0, D)
    s_offs = tl.arange(0, NUM_KV_SPLITS)
    lse = tl.load(MidLse_ptr + n * stride_ml_n + s_offs)  # [SPLITS]
    g = tl.max(lse, axis=0)  # global max lse
    w = tl.exp(lse - g)  # [SPLITS] weights
    denom = tl.sum(w, axis=0)
    # weighted sum of partial outputs
    o_parts = tl.load(
        MidO_ptr + n * stride_mo_n + s_offs[:, None] * stride_mo_s + d_offs[None, :]
    )  # [SPLITS, D]
    out = tl.sum(w[:, None] * o_parts, axis=0) / denom  # [D]
    tl.store(Out_ptr + n * stride_o_n + d_offs, out.to(tl.float16))


# ──────────────────────────────────────────────────────────────────────────────
# Python driver
# ──────────────────────────────────────────────────────────────────────────────


def kvarn_decode_attention(
    query: torch.Tensor,  # [B, Hq, D]   fp16/bf16
    kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, TILE_BYTES] uint8
    hadamard: torch.Tensor,  # [D, D]       fp32
    scale: float,
    cfg,
    impl,  # KVarNAttentionImpl (has pool + scratch buffers)
    md,  # KVarNMetadata (carries the precomputed task plan)
) -> torch.Tensor:
    """Decode driver — dequant + FlashAttention.

    For each layer call:

        1. Rotate the query: q_rot = q · H.
        2. Build a packed varlen ``[total_kv_tokens, Hk, D]`` fp16 K, V with
           ``_kvarn_build_packed_kv_kernel``: dequantise the int4 blocks and
           copy sink + trailing-tail tokens from the rotated fp16 tail pool.
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
    # Both the single-stage kernel and stage1 are @triton.autotune'd over
    # BLOCK_N/num_warps (keyed on D/GROUP/Q_PER_KV/K_BITS/V_BITS) — no
    # BLOCK_N/num_warps/num_stages passed at the launch sites.
    _qpk = Hq // Hk
    # Pad Q_PER_KV to a power of 2 for tl.arange / tl.dot (e.g. Qwen3.5 GQA
    # 24q/4kv = ratio 6 -> 8); padded query heads are masked off in-kernel.
    _qpk_pad = 1 << (_qpk - 1).bit_length() if _qpk > 1 else 1
    common = dict(
        MAX_BLOCKS_PER_REQ=max_blocks_per_req,
        D=D,
        GROUP=group,
        Q_PER_KV=_qpk,
        Q_PER_KV_PAD=_qpk_pad,
        SLIDING_WINDOW=int(getattr(impl, "sliding_window", 0) or 0),
        K_BITS=cfg.key_bits,
        V_BITS=cfg.value_bits,
        NUM_BLOCKS_LOOKUP=impl._block_lookup_size,
        K_PACKED_OFFSET=cfg.k_packed_offset,
        K_S_COL_OFFSET=cfg.k_s_col_offset,
        K_ZP_OFFSET=cfg.k_zp_offset,
        K_S_ROW_OFFSET=cfg.k_s_row_offset,
        V_PACKED_OFFSET=cfg.v_packed_offset,
        V_S_COL_OFFSET=cfg.v_s_col_offset,
        V_S_ROW_OFFSET=cfg.v_s_row_offset,
        V_ZP_OFFSET=cfg.v_zp_offset,
        VQ_INDIRECT=False,
    )
    # SPLIT-K (KVARN_SPLIT_K=1): two-stage flash-decoding — only a win in the
    # LOW-batch / long-context regime (few programs ⇒ the KV-split dim adds the
    # missing parallelism). At BURST (high batch) the single-stage (B,Hk) grid
    # already saturates the GPU, so split-K's mid-buffer round-trip + stage-2 +
    # empty-split waste roughly HALVE throughput. Default: single-stage.
    use_fused = use_fused and True
    # Split-K decision. Single-stage grid is (B, Hk) programs, each serially
    # walking the WHOLE context; at long context that serial loop dominates and
    # leaves the GPU under-occupied -> split-K parallelizes the KV dim for a big
    # win (Qwen3.5-27B head_dim256 16K: 0.59x -> 0.96x same-batch, and lets KVarN
    # out-throughput FP16's max feasible batch). But at short context / high
    # occupancy the mid-buffer round-trip + stage-2 + empty-split waste roughly
    # HALVE throughput. So auto-enable only in the long-context, under-occupied
    # regime; KVARN_SPLIT_K env (0/1) is an explicit override.
    # The split-K mid buffers are sized for the pure-decode regime
    # (max_num_seqs * Hq rows); never split if this batch would overflow them
    # (defensive — real decode batches always fit, but a padded dummy run can
    # be wider). The single-stage kernel handles any batch size.
    _mid_fits = impl._mid_o_buf is not None and impl._mid_o_buf.shape[0] >= N
    _sk_env = os.environ.get("KVARN_SPLIT_K")
    if _sk_env is not None:
        split_k = use_fused and _sk_env == "1" and _mid_fits
    else:
        sm_count = (
            getattr(impl, "_sm_count", 0)
            or torch.cuda.get_device_properties(device).multi_processor_count
        )
        # long context (>= ~16 blocks of group tokens) AND single-stage grid does
        # not already fill the SMs.
        # Sliding-window layers read only ~window/GROUP blocks (single-stage is
        # plenty + the windowed loop is in the single-stage kernel), so never split.
        _sw = int(getattr(impl, "sliding_window", 0) or 0)
        split_k = (
            use_fused
            and (_sw <= 0)
            and (max_blocks_per_req >= 16)
            and (B * Hk <= sm_count)
            and _mid_fits
        )
    if use_fused and not split_k:
        fused_out = impl._fused_out_buf[:N]  # [N, D] fp16
        with torch.profiler.record_function("kvarn_fused_decode"):
            _kvarn_fused_decode_kernel[(B, Hk)](
                q_rot_fp16,
                md.seq_lens,
                md.block_table,
                md.seq_lens,
                impl._block_to_slot_t,
                kv_cache,
                impl._tail_K_pool,
                impl._tail_V_pool,
                fused_out,
                scale,
                Hq * D,
                D,
                md.block_table.stride(0),
                kv_cache.stride(0),
                kv_cache.stride(1),
                impl._tail_K_pool.stride(0),
                impl._tail_K_pool.stride(1),
                impl._tail_K_pool.stride(2),
                Hq * D,
                D,
                **common,  # autotune fills BLOCK_N + warps + stages
            )
        output_rot = fused_out
    elif split_k:
        SPLITS = adaptive_num_kv_splits(max_blocks_per_req)
        mid_o = impl._mid_o_buf
        mid_lse = impl._mid_lse_buf
        fused_out = impl._fused_out_buf[:N]
        with torch.profiler.record_function("kvarn_fused_decode_s1"):
            _kvarn_fused_decode_stage1[(B, Hk, SPLITS)](
                q_rot_fp16,
                md.seq_lens,
                md.block_table,
                md.seq_lens,
                impl._block_to_slot_t,
                kv_cache,
                impl._tail_K_pool,
                impl._tail_V_pool,
                mid_o,
                mid_lse,
                scale,
                Hq * D,
                D,
                md.block_table.stride(0),
                kv_cache.stride(0),
                kv_cache.stride(1),
                impl._tail_K_pool.stride(0),
                impl._tail_K_pool.stride(1),
                impl._tail_K_pool.stride(2),
                mid_o.stride(0),
                mid_o.stride(1),
                mid_lse.stride(0),
                NUM_KV_SPLITS=SPLITS,
                HQ=Hq,
                **common,  # BLOCK_N/warps autotuned
            )
        with torch.profiler.record_function("kvarn_fused_decode_s2"):
            _kvarn_fused_decode_stage2[(N,)](
                mid_o,
                mid_lse,
                fused_out,
                mid_o.stride(0),
                mid_o.stride(1),
                mid_lse.stride(0),
                fused_out.stride(0),
                D=D,
                NUM_KV_SPLITS=SPLITS,
                num_warps=2,
            )
        output_rot = fused_out
    else:
        K_packed = impl._fa_K_buf
        V_packed = impl._fa_V_buf
        with torch.profiler.record_function("kvarn_build_packed_kv"):
            _kvarn_build_packed_kv_kernel[(B * max_blocks_per_req, Hk)](
                md.block_table,
                md.seq_lens,
                md.fa_cu_seqlens_k,
                impl._block_to_slot_t,
                kv_cache,
                impl._tail_K_pool,
                impl._tail_V_pool,
                K_packed,
                V_packed,
                md.block_table.stride(0),
                kv_cache.stride(0),
                kv_cache.stride(1),
                impl._tail_K_pool.stride(0),
                impl._tail_K_pool.stride(1),
                impl._tail_K_pool.stride(2),
                K_packed.stride(0),
                K_packed.stride(1),
                MAX_BLOCKS_PER_REQ=max_blocks_per_req,
                D=D,
                GROUP=group,
                K_BITS=cfg.key_bits,
                V_BITS=cfg.value_bits,
                NUM_BLOCKS_LOOKUP=impl._block_lookup_size,
                K_PACKED_OFFSET=cfg.k_packed_offset,
                K_S_COL_OFFSET=cfg.k_s_col_offset,
                K_ZP_OFFSET=cfg.k_zp_offset,
                K_S_ROW_OFFSET=cfg.k_s_row_offset,
                V_PACKED_OFFSET=cfg.v_packed_offset,
                V_S_COL_OFFSET=cfg.v_s_col_offset,
                V_S_ROW_OFFSET=cfg.v_s_row_offset,
                V_ZP_OFFSET=cfg.v_zp_offset,
                num_warps=4,
                num_stages=2,
            )
        with torch.profiler.record_function("kvarn_flash_attn"):
            output_rot = flash_attn_varlen_func(
                q=q_rot_fp16.view(B, Hq, D),
                k=K_packed,
                v=V_packed,
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
        out_unrot = torch.mm(output_rot.reshape(N, D), H16)  # fresh fp16
        return out_unrot.view(B, Hq, D).to(out_dtype)


def kvarn_verify_attention(
    query: torch.Tensor,  # [NQ, Hq, D]  fp16/bf16 (token-major)
    kv_cache: torch.Tensor,  # [num_blocks, Hk, TILE_BYTES] uint8
    block_table: torch.Tensor,  # [B, max_blocks] int32
    scale: float,
    cfg,
    impl,  # KVarNAttentionImpl
    vq_req: torch.Tensor,  # [NQ] int32 — block-table row per token
    vq_seqlen: torch.Tensor,  # [NQ] int32 — causal len: cached+i+1
    max_ctx_blocks: int,  # ceil(max context / group) upper bound
    qlen: int = 0,  # uniform query length (>= 2), else 0
    seq_lens: torch.Tensor | None = None,  # [B] int32 (uniform path)
) -> torch.Tensor:
    """Fused multi-query verify (speculative decode), reading int4 tiles +
    the fp16 tail pool directly — no fp16 materialization of the context
    (whose O(context)-per-step cost dominated MTP decode).

    Two modes:
    - UNIFORM (qlen >= 2, the captured/common case): one program per
      (request, kv head, split) — the request's QLEN tokens SHARE each
      block's dequant, so KV bytes and dequant ALU match single-token decode.
    - per-token fallback (qlen == 0, non-uniform eager batches): one program
      per (query token, kv head) via the vq plan; QLEN-x redundant dequant.

    Output: ``[NQ, Hq, D]`` in ``query``'s dtype, un-rotated frame.
    """
    NQ, Hq, D = query.shape
    Hk = kv_cache.shape[1]
    device = query.device
    out_dtype = query.dtype
    group = cfg.group
    Nrows = NQ * Hq

    H16 = (
        impl._H_fp16
        if impl._H_fp16 is not None
        else impl._hadamard(device).to(torch.float16)
    )
    q_rot = torch.mm(query.reshape(Nrows, D).to(torch.float16), H16)

    _qpk = Hq // Hk
    _qpk_pad = 1 << (_qpk - 1).bit_length() if _qpk > 1 else 1
    common = dict(
        MAX_BLOCKS_PER_REQ=max_ctx_blocks,
        D=D,
        GROUP=group,
        Q_PER_KV=_qpk,
        Q_PER_KV_PAD=_qpk_pad,
        SLIDING_WINDOW=int(getattr(impl, "sliding_window", 0) or 0),
        K_BITS=cfg.key_bits,
        V_BITS=cfg.value_bits,
        NUM_BLOCKS_LOOKUP=impl._block_lookup_size,
        K_PACKED_OFFSET=cfg.k_packed_offset,
        K_S_COL_OFFSET=cfg.k_s_col_offset,
        K_ZP_OFFSET=cfg.k_zp_offset,
        K_S_ROW_OFFSET=cfg.k_s_row_offset,
        V_PACKED_OFFSET=cfg.v_packed_offset,
        V_S_COL_OFFSET=cfg.v_s_col_offset,
        V_S_ROW_OFFSET=cfg.v_s_row_offset,
        V_ZP_OFFSET=cfg.v_zp_offset,
    )

    out_rot = torch.empty(NQ, Hq, D, dtype=torch.float16, device=device)

    _m = qlen * (1 << ((Hq // Hk) - 1).bit_length() if Hq // Hk > 1 else 1)
    if (
        qlen >= 2
        and seq_lens is not None
        and NQ % qlen == 0
        and (_m & (_m - 1)) == 0  # Q-tile rows must be a power of 2
        # DEFAULT OFF: numerically validated in isolation (matches the
        # per-token kernel within fp32 reduction noise on live inputs,
        # incl. on the failing trajectory), but serving with it corrupts
        # the MTP drafter's proposals (invalid [-1,...] spec tokens,
        # embedding index asserts at temperature>0, degenerate greedy
        # output) through a mechanism not yet isolated — suspicion is an
        # interaction with async scheduling / drafter metadata rather
        # than kernel math. Re-enable for debugging only.
        and os.environ.get("KVARN_SHARED_VERIFY", "0") == "1"
    ):
        # SHARED-DEQUANT uniform path: split-K shaped (SPLITS=1 degenerates
        # cleanly); stage2 combines into the flat [NQ*Hq, D] output.
        B = NQ // qlen
        SPLITS = (
            adaptive_num_kv_splits(max_ctx_blocks)
            if max_ctx_blocks >= 16
            and B * Hk
            <= (
                getattr(impl, "_sm_count", 0)
                or torch.cuda.get_device_properties(device).multi_processor_count
            )
            else 1
        )
        mid_o = torch.empty(Nrows, SPLITS, D, dtype=torch.float32, device=device)
        mid_lse = torch.empty(Nrows, SPLITS, dtype=torch.float32, device=device)
        _kvarn_fused_verify_stage1[(B, Hk, SPLITS)](
            q_rot,
            block_table,
            vq_seqlen,
            impl._block_to_slot_t,
            kv_cache,
            impl._tail_K_pool,
            impl._tail_V_pool,
            mid_o,
            mid_lse,
            scale,
            Hq * D,
            D,
            block_table.stride(0),
            kv_cache.stride(0),
            kv_cache.stride(1),
            impl._tail_K_pool.stride(0),
            impl._tail_K_pool.stride(1),
            impl._tail_K_pool.stride(2),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_lse.stride(0),
            QLEN=qlen,
            HQ=Hq,
            NUM_KV_SPLITS=SPLITS,
            **common,
        )
        out_flat = out_rot.view(Nrows, D)
        _kvarn_fused_decode_stage2[(Nrows,)](
            mid_o,
            mid_lse,
            out_flat,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_lse.stride(0),
            out_flat.stride(0),
            D=D,
            NUM_KV_SPLITS=SPLITS,
            num_warps=2,
        )
        out_unrot = torch.mm(out_rot.reshape(Nrows, D), H16)
        return out_unrot.view(NQ, Hq, D).to(out_dtype)

    common["VQ_INDIRECT"] = True

    # Split-K mirrors the decode driver's heuristic: long context with too few
    # programs to fill the SMs. Verify batches are tiny (NQ <= maxq * B), so
    # long-context verify nearly always wants the split.
    sm_count = (
        getattr(impl, "_sm_count", 0)
        or torch.cuda.get_device_properties(device).multi_processor_count
    )
    _sw = int(getattr(impl, "sliding_window", 0) or 0)
    _sk_env = os.environ.get("KVARN_SPLIT_K")
    if _sk_env is not None:
        split_k = _sk_env == "1"
    else:
        split_k = (_sw <= 0) and (max_ctx_blocks >= 16) and (NQ * Hk <= sm_count)

    if not split_k:
        _kvarn_fused_decode_kernel[(NQ, Hk)](
            q_rot,
            vq_req,
            block_table,
            vq_seqlen,
            impl._block_to_slot_t,
            kv_cache,
            impl._tail_K_pool,
            impl._tail_V_pool,
            out_rot,
            scale,
            Hq * D,
            D,
            block_table.stride(0),
            kv_cache.stride(0),
            kv_cache.stride(1),
            impl._tail_K_pool.stride(0),
            impl._tail_K_pool.stride(1),
            impl._tail_K_pool.stride(2),
            Hq * D,
            D,
            **common,
        )
    else:
        SPLITS = adaptive_num_kv_splits(max_ctx_blocks)
        mid_o = torch.empty(Nrows, SPLITS, D, dtype=torch.float32, device=device)
        mid_lse = torch.empty(Nrows, SPLITS, dtype=torch.float32, device=device)
        _kvarn_fused_decode_stage1[(NQ, Hk, SPLITS)](
            q_rot,
            vq_req,
            block_table,
            vq_seqlen,
            impl._block_to_slot_t,
            kv_cache,
            impl._tail_K_pool,
            impl._tail_V_pool,
            mid_o,
            mid_lse,
            scale,
            Hq * D,
            D,
            block_table.stride(0),
            kv_cache.stride(0),
            kv_cache.stride(1),
            impl._tail_K_pool.stride(0),
            impl._tail_K_pool.stride(1),
            impl._tail_K_pool.stride(2),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_lse.stride(0),
            NUM_KV_SPLITS=SPLITS,
            HQ=Hq,
            **common,  # BLOCK_N/warps autotuned
        )
        out_flat = out_rot.view(Nrows, D)
        _kvarn_fused_decode_stage2[(Nrows,)](
            mid_o,
            mid_lse,
            out_flat,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_lse.stride(0),
            out_flat.stride(0),
            D=D,
            NUM_KV_SPLITS=SPLITS,
            num_warps=2,
        )

    out_unrot = torch.mm(out_rot.reshape(Nrows, D), H16)
    return out_unrot.view(NQ, Hq, D).to(out_dtype)


# ──────────────────────────────────────────────────────────────────────────────
# SHARED-DEQUANT verify kernel: one program per (REQUEST, kv-head, split) — all
# QLEN verify tokens of a request share each block's dequant (the per-token
# VQ_INDIRECT path above re-walks the context once per token, i.e. QLEN
# redundant dequants). Q tile is [QLEN * Q_PER_KV_PAD, D] with a per-row
# bottom-right causal limit: row (token j, lane h) attends kv positions
# < seq_len - QLEN + j + 1. Uniform QLEN is a constexpr (uniform-batch graph
# capture guarantees it); non-uniform eager batches fall back to the per-token
# kernel. Scale vectors are loaded via fp16 pointer casts (the tile offsets are
# 2-byte aligned) instead of the byte-pair loads of the older kernels.
# ──────────────────────────────────────────────────────────────────────────────


@triton.autotune(
    configs=_DECODE_AUTOTUNE_CONFIGS,
    key=["D", "GROUP", "Q_PER_KV", "QLEN", "K_BITS", "V_BITS"],
)
@triton.jit
def _kvarn_fused_verify_stage1(
    Q_ptr,  # [NQ = B*QLEN, Hq, D] fp16 (rotated, token-major)
    Block_table_ptr,  # [B, max_blocks] int32
    Seq_lens_ptr,  # [NQ] int32 — the vq plan (per-token causal lengths);
    # the request's FULL length is its LAST token's entry.
    # Built CPU-side in the builder: under async spec
    # decode the device seq_lens tensor can disagree with
    # the builder's CPU view, and the CPU view is the one
    # the (validated) per-token path uses.
    Block_to_slot_ptr,
    KV_cache_ptr,
    Tail_K_pool_ptr,
    Tail_V_pool_ptr,
    MidO_ptr,  # [NQ*Hq, NUM_KV_SPLITS, D] fp32
    MidLse_ptr,  # [NQ*Hq, NUM_KV_SPLITS]    fp32
    scale,
    stride_q_t,
    stride_q_h,
    stride_bt_b,
    stride_kv_b,
    stride_kv_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
    stride_mo_n,
    stride_mo_s,
    stride_ml_n,
    MAX_BLOCKS_PER_REQ: tl.constexpr,  # unused; kept for launch-dict parity
    D: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QLEN: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    Q_PER_KV_PAD: tl.constexpr,
    HQ: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
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
    b = tl.program_id(0)
    hk = tl.program_id(1)
    split = tl.program_id(2)

    seq_len = tl.load(Seq_lens_ptr + b * QLEN + (QLEN - 1))
    # Padded rows (uniform-batch capture/replay) carry seq_len <= 0.
    if seq_len <= 0:
        return

    M: tl.constexpr = QLEN * Q_PER_KV_PAD
    r = tl.arange(0, M)
    j = r // Q_PER_KV_PAD  # token idx in request
    lane = r % Q_PER_KV_PAD  # query-head lane
    rmask = lane < Q_PER_KV
    limit = seq_len - QLEN + j + 1  # [M] causal kv limit
    hq0 = hk * Q_PER_KV
    d_offs = tl.arange(0, D)

    PACK_K: tl.constexpr = 8 // K_BITS
    PACK_V: tl.constexpr = 8 // V_BITS
    MASK_K: tl.constexpr = (1 << K_BITS) - 1
    MASK_V: tl.constexpr = (1 << V_BITS) - 1
    d_byte_v = d_offs // PACK_V
    d_shift_v = (d_offs % PACK_V) * V_BITS

    tok_row = b * QLEN + j  # [M] token-major Q row
    q = tl.load(
        Q_ptr
        + tok_row[:, None] * stride_q_t
        + (hq0 + lane)[:, None] * stride_q_h
        + d_offs[None, :],
        mask=rmask[:, None],
        other=0.0,
    ).to(tl.float32)  # [M, D]

    m_i = tl.full([M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)
    acc = tl.zeros([M, D], dtype=tl.float32)

    n_blocks = (seq_len + GROUP - 1) // GROUP
    blocks_per_split = (n_blocks + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS
    blk_lo = split * blocks_per_split
    blk_hi = tl.minimum(blk_lo + blocks_per_split, n_blocks)

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

        # Per-channel scales — direct fp16 loads (2-byte-aligned offsets).
        s_col_K = tl.load(
            (KV_cache_ptr + tile_base + K_S_COL_OFFSET).to(tl.pointer_type(tl.float16))
            + d_offs
        ).to(tl.float32)
        zp_K = tl.load(
            (KV_cache_ptr + tile_base + K_ZP_OFFSET).to(tl.pointer_type(tl.float16))
            + d_offs
        ).to(tl.float32)
        s_col_V = tl.load(
            (KV_cache_ptr + tile_base + V_S_COL_OFFSET).to(tl.pointer_type(tl.float16))
            + d_offs
        ).to(tl.float32)

        for c0 in range(0, GROUP, BLOCK_N):
            cols = c0 + tl.arange(0, BLOCK_N)
            cmask = cols < n_tok
            kvpos = k * GROUP + cols  # [BN]

            if pool_slot >= 0:
                src = pool_base + cols[:, None] * stride_pool_t + d_offs[None, :]
                Kc = tl.load(Tail_K_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )  # [BN, D]
                Vc = tl.load(Tail_V_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )  # [BN, D]
                K_dg = tl.trans(Kc)  # [D, BN]
            else:
                cb_k = cols // PACK_K
                cs_k = (cols % PACK_K) * K_BITS
                s_row_K = tl.load(
                    (KV_cache_ptr + tile_base + K_S_ROW_OFFSET).to(
                        tl.pointer_type(tl.float16)
                    )
                    + cols
                ).to(tl.float32)
                k_addrs = (
                    tile_base
                    + K_PACKED_OFFSET
                    + d_offs[:, None] * (GROUP // PACK_K)
                    + cb_k[None, :]
                )
                k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)
                q_K = ((k_bytes >> cs_k[None, :]) & MASK_K).to(tl.float32)
                K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                s_row_V = tl.load(
                    (KV_cache_ptr + tile_base + V_S_ROW_OFFSET).to(
                        tl.pointer_type(tl.float16)
                    )
                    + cols
                ).to(tl.float32)
                zp_V = tl.load(
                    (KV_cache_ptr + tile_base + V_ZP_OFFSET).to(
                        tl.pointer_type(tl.float16)
                    )
                    + cols
                ).to(tl.float32)
                v_addrs = (
                    tile_base
                    + V_PACKED_OFFSET
                    + cols[:, None] * (D // PACK_V)
                    + d_byte_v[None, :]
                )
                v_bytes = tl.load(KV_cache_ptr + v_addrs).to(tl.int32)
                q_V = ((v_bytes >> d_shift_v[None, :]) & MASK_V).to(tl.float32)
                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]

            scores = tl.dot(q, K_dg)  # [M, BN]
            smask = cmask[None, :] & (kvpos[None, :] < limit[:, None])
            if SLIDING_WINDOW > 0:
                smask = smask & (
                    kvpos[None, :] >= tl.maximum(limit[:, None] - SLIDING_WINDOW, 0)
                )
            scores = tl.where(smask, scores * scale, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, Vc)
            m_i = m_new

    nonempty = l_i > 0
    O_s = acc / tl.where(nonempty, l_i, 1.0)[:, None]
    lse_s = tl.where(
        nonempty, m_i + tl.log(tl.where(nonempty, l_i, 1.0)), -float("inf")
    )
    rows = tok_row * HQ + hq0 + lane  # [M] N-row index
    tl.store(
        MidO_ptr + rows[:, None] * stride_mo_n + split * stride_mo_s + d_offs[None, :],
        O_s,
        mask=rmask[:, None],
    )
    tl.store(MidLse_ptr + rows * stride_ml_n + split, lse_s, mask=rmask)
