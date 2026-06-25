# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU-only HP -> int2 flush for the unified mixed KV pool.

Decode-time flush pipeline for ``UnifiedInt2HPKVPool``. Three device kernels
run per flush step; between flushes the pipeline is skipped entirely (the
caller gates on ``pool._flush_step_counter % pool.flush_interval``).

    plan:  per-request, emit up to ``flush_interval`` (flush_pos, src_hp_slot,
           dst_quant_slot) triples identifying HP-recent slots to demote.
    quant: one fused triton launch over a ``(num_flush_tokens, num_heads,
           num_layers)`` grid that quantizes HP K and HP V for every
           (flush_token, head, layer) tile. This collapses the previous
           per-layer Python loop + ``index_select`` + ``_pretransformed``
           calls (~4 launches per layer) into a single launch per flush
           step. The ``num_layers`` axis is on the grid (not a
           ``tl.static_range`` unroll) so the kernel's TTIR stays small
           and Triton JIT time is proportional to one layer of work
           rather than ``num_layers``.
    remap: for each flushed (req, flush_pos), overwrite ``req_to_token`` with
           the quant slot id so subsequent attention steps see the token as
           int2.

No CPU-side iteration, no ``torch.unique``, no eviction retry. The caller
passes ``returned_slot_ids`` straight to ``allocator.free`` in one call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@dataclass
class FlushPlan:
    """Output of ``gpu_flush_int2_plan``; consumed by ``gpu_flush_int2_apply``.

    Holds only the per-step tensors and scalars produced/needed across the
    plan -> apply boundary. Pool-side metadata (kv pointers, strides, scalar
    config) is re-supplied to ``apply`` by the caller.
    """

    returned_slot_ids: torch.Tensor  # int64 [bs * flush_interval]
    src_hp_slot: torch.Tensor  # int64 [bs * flush_interval]
    flush_pos: torch.Tensor  # int32 [bs * flush_interval]
    valid_mask: torch.Tensor  # int8  [bs * flush_interval]
    dst_quant_slots: torch.Tensor  # int64 [bs * flush_interval] (carried through)
    bs: int
    flush_interval: int


# ---------------------------------------------------------------------------
# Plan kernel
# ---------------------------------------------------------------------------


@triton.jit
def _flush_plan_kernel(
    seq_lens_ptr,  # int32 [bs]
    prefix_lens_ptr,  # int32 [bs]
    req_pool_indices_ptr,  # int64 [bs]
    dst_quant_slot_ptr,  # int64 [bs * FLUSH_INTERVAL]  pre-allocated
    req_to_token_ptr,  # int32 [num_req_slots, max_ctx]
    flush_mask_ptr,  # int8  [bs] -- 1 if request flushes this step
    src_hp_slot_out_ptr,  # int64 [bs, FLUSH_INTERVAL]   src hp slot or -1
    returned_slot_ids_ptr,  # int64 [bs, FLUSH_INTERVAL]   hp slot or dst_quant_slot
    flush_pos_out_ptr,  # int32 [bs, FLUSH_INTERVAL]   flush_pos or -1
    valid_mask_out_ptr,  # int8  [bs, FLUSH_INTERVAL]   1 if flushed else 0
    max_ctx,
    rtt_stride_row,
    HP_PREFIX_TOKENS: tl.constexpr,
    HP_RECENT_TOKENS: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    FLUSH_INTERVAL: tl.constexpr,
):
    """Per request, decide which FLUSH_INTERVAL HP-recent slots to demote.

    Per-request gating: if ``flush_mask[i] == 0`` this request does not flush
    this step. We then write ``valid=0`` and ``returned_slot_ids = dst_q`` for
    all FLUSH_INTERVAL entries so the caller's bulk free returns the entire
    quant page back to the allocator (whole-page free, no aliasing live
    slots).

    For flushing requests (``flush_mask[i] == 1``): by construction (counter
    init in ``_alloc_for_extend_mixed``) the request now has exactly
    ``hp_recent + FLUSH_INTERVAL - 1`` HP-recent positions all of which are
    valid HP slots, and the K oldest are at positions
    ``[seq_len - hp_recent - (K-1) .. seq_len - hp_recent]``. We demote all K.
    The legacy ``loc >= HP_OFFSET`` per-position check is kept as a defensive
    invariant guard — in normal operation it is always true on flushing
    requests.
    """
    i = tl.program_id(0)
    do_flush = tl.load(flush_mask_ptr + i).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + i).to(tl.int32)
    prefix_len = tl.load(prefix_lens_ptr + i).to(tl.int32)
    req_pool_idx = tl.load(req_pool_indices_ptr + i).to(tl.int64)

    for j in tl.static_range(FLUSH_INTERVAL):
        out_idx = i * FLUSH_INTERVAL + j
        dst_q = tl.load(dst_quant_slot_ptr + out_idx).to(tl.int64)

        valid = 0
        src_hp = tl.full((), -1, tl.int64)
        flush_pos = tl.full((), -1, tl.int32)

        if do_flush == 1 and HP_RECENT_TOKENS > 0:
            fp = seq_len - HP_RECENT_TOKENS - (FLUSH_INTERVAL - 1) + j
            if fp >= prefix_len and fp >= 0:
                loc = tl.load(
                    req_to_token_ptr + req_pool_idx * rtt_stride_row + fp.to(tl.int64)
                ).to(tl.int64)
                if loc >= HP_OFFSET:
                    src_hp = loc - HP_OFFSET
                    valid = 1
                    flush_pos = fp

        tl.store(valid_mask_out_ptr + out_idx, tl.full((), valid, tl.int8))
        tl.store(flush_pos_out_ptr + out_idx, flush_pos)

        if valid == 1:
            # The freed slot is the HP slot (as a global id) — the HP tier has
            # a non-zero offset so the allocator.free path can decode the tier.
            tl.store(returned_slot_ids_ptr + out_idx, src_hp + HP_OFFSET)
            tl.store(src_hp_slot_out_ptr + out_idx, src_hp)
        else:
            # Hand the unused quant slot back so the caller can free it in the
            # same bulk call. For non-flushing requests this returns the whole
            # quant page (all FLUSH_INTERVAL slots) → page is freed cleanly.
            tl.store(returned_slot_ids_ptr + out_idx, dst_q)
            tl.store(src_hp_slot_out_ptr + out_idx, -1)


# ---------------------------------------------------------------------------
# Fused quant kernel: one launch flushes every (flush_tokens_tile, head, layer)
# triple, both K and V sides. Multi-row tiled with BLOCK_TOK = flush_interval
# so per-tile early bail aligns with per-request flush gating, and 128-bit /
# thread vectorized loads on the head_dim axis (8 bf16 / 16 fp8 elements per
# thread). Inline clip via K_CLIP_INDEX / V_CLIP_INDEX constexpr (-1 disables)
# replaces the previous separate threshold-precompute kernel.
# ---------------------------------------------------------------------------


@triton.jit
def _fused_flush_quant_body(
    hp_base,  # pointer to hp_dtype arena for one (layer, K|V)
    quant_base,  # pointer to uint8 arena (int2 packed view)
    sz_base,  # pointer to scale_dtype arena
    src_hp_slot,  # int64 [BLOCK_TOK]
    dst_quant_slot,  # int64 [BLOCK_TOK]
    active,  # int1  [BLOCK_TOK]  per-row valid mask
    head_idx,  # int64 scalar
    HP_STRIDE_LOC: tl.constexpr,
    HP_STRIDE_HEAD: tl.constexpr,
    HP_STRIDE_DIM: tl.constexpr,
    Q_STRIDE_LOC: tl.constexpr,
    Q_STRIDE_HEAD: tl.constexpr,
    Q_STRIDE_DIM: tl.constexpr,
    SZ_STRIDE_LOC: tl.constexpr,
    SZ_STRIDE_HEAD: tl.constexpr,
    SZ_STRIDE_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_QUARTER: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    CLIP_INDEX: tl.constexpr,
    BSEARCH_ITERS: tl.constexpr,
):
    """Quantize ``BLOCK_TOK`` (src_hp_slot, head) HP rows into int2 at the
    matching ``dst_quant_slot``s.

    Mirrors the clip-aware single-call kernel
    (``_pretransformed_int2_set_kv_clip_grouped_kernel``):

      1. Load full ``[BLOCK_TOK, HEAD_DIM]`` row tile in one shot.
      2. If ``CLIP_INDEX >= 0``, compute per-row clip threshold via
         ``tl.sort`` + masked sum, then clamp.
      3. Group-wise reshape / min / max / scale / zero.
      4. Quantize on the 3-D grouped tile (so each ``(t, g, k)`` sees its
         own ``scale[t, g] / zero[t, g]`` without a separate gather).
      5. Quartered split via ``reshape + permute + split×2`` for the int2
         pack.

    ``HEAD_DIM`` is required to be a power of two by the launcher so the
    grouped reshape and the quartered split (``BLOCK_QUARTER == HEAD_DIM/4``
    exactly) land cleanly.
    """
    full_offs = tl.arange(0, HEAD_DIM)
    base = (
        src_hp_slot[:, None] * HP_STRIDE_LOC
        + head_idx * HP_STRIDE_HEAD
        + full_offs[None, :] * HP_STRIDE_DIM
    )
    acc = tl.load(
        hp_base + base,
        mask=active[:, None],
        other=0.0,
    ).to(tl.float32)  # [BLOCK_TOK, HEAD_DIM]

    if CLIP_INDEX >= 0:
        abs_acc = tl.abs(acc)
        if BSEARCH_ITERS > 0:
            target_above = HEAD_DIM - CLIP_INDEX
            thr_lo = tl.zeros([BLOCK_TOK], dtype=tl.float32)
            thr_hi = tl.max(abs_acc, axis=1)
            for _ in tl.static_range(BSEARCH_ITERS):
                thr_mid = (thr_lo + thr_hi) * 0.5
                cnt_above = tl.sum((abs_acc > thr_mid[:, None]).to(tl.int32), axis=1)
                too_many = cnt_above > target_above
                thr_lo = tl.where(too_many, thr_mid, thr_lo)
                thr_hi = tl.where(too_many, thr_hi, thr_mid)
            thr = thr_hi
        else:
            sorted_acc = tl.sort(abs_acc)
            pick = (full_offs == CLIP_INDEX)[None, :]
            thr = tl.sum(tl.where(pick, sorted_acc, 0.0), axis=1)  # [BLOCK_TOK]
        acc = tl.minimum(
            tl.maximum(acc, -thr[:, None]),
            thr[:, None],
        )

    grouped = tl.reshape(acc, (BLOCK_TOK, NUM_GROUPS, GROUP_SIZE))
    val_min = tl.min(grouped, axis=2)  # [BLOCK_TOK, NUM_GROUPS]
    val_max = tl.max(grouped, axis=2)
    scale = tl.maximum(val_max - val_min, 1e-8) / 3.0
    zero = tl.math.div_rn(-val_min, scale)

    # Quartered split via reshape + permute + split×2. Matches the layout
    # produced by ``_pretransformed_grouped_int2_set_kv_kernel`` (the
    # non-fused reference): byte ``i`` packs values at positions
    # ``i, BQ+i, 2*BQ+i, 3*BQ+i``.
    #
    # We split fp32 ``acc`` (not the post-quant uint8 tile): empirically the
    # uint8-permute-reshape-split chain produced wrong q2 lanes on this
    # Triton/sm90 build, while the fp32 chain matches the bit-level
    # reference. Per-position scale/zero come from broadcasting the per-
    # group ``scale``/``zero`` to ``HEAD_DIM`` and feeding them through
    # the same reshape+permute+split pipeline.
    acc_r = tl.reshape(acc, (BLOCK_TOK, 4, BLOCK_QUARTER))
    acc_p = tl.permute(acc_r, (0, 2, 1))
    acc_s = tl.reshape(acc_p, (BLOCK_TOK, BLOCK_QUARTER, 2, 2))
    a_even, a_odd = tl.split(acc_s)
    vals0, vals2 = tl.split(a_even)
    vals1, vals3 = tl.split(a_odd)

    scale_3d = tl.broadcast_to(scale[:, :, None], (BLOCK_TOK, NUM_GROUPS, GROUP_SIZE))
    zero_3d = tl.broadcast_to(zero[:, :, None], (BLOCK_TOK, NUM_GROUPS, GROUP_SIZE))
    scale_flat = tl.reshape(scale_3d, (BLOCK_TOK, HEAD_DIM))
    zero_flat = tl.reshape(zero_3d, (BLOCK_TOK, HEAD_DIM))

    sr = tl.reshape(scale_flat, (BLOCK_TOK, 4, BLOCK_QUARTER))
    sp = tl.permute(sr, (0, 2, 1))
    ss = tl.reshape(sp, (BLOCK_TOK, BLOCK_QUARTER, 2, 2))
    s_even, s_odd = tl.split(ss)
    s0, s2 = tl.split(s_even)
    s1, s3 = tl.split(s_odd)

    zr = tl.reshape(zero_flat, (BLOCK_TOK, 4, BLOCK_QUARTER))
    zp = tl.permute(zr, (0, 2, 1))
    zs = tl.reshape(zp, (BLOCK_TOK, BLOCK_QUARTER, 2, 2))
    z_even, z_odd = tl.split(zs)
    z0, z2 = tl.split(z_even)
    z1, z3 = tl.split(z_odd)

    q0 = (tl.math.div_rn(vals0, s0) + z0 + 0.5).to(tl.uint8)
    q1 = (tl.math.div_rn(vals1, s1) + z1 + 0.5).to(tl.uint8)
    q2 = (tl.math.div_rn(vals2, s2) + z2 + 0.5).to(tl.uint8)
    q3 = (tl.math.div_rn(vals3, s3) + z3 + 0.5).to(tl.uint8)
    packed = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)

    dim_offs_q = tl.arange(0, BLOCK_QUARTER)
    cache_offset = (
        dst_quant_slot[:, None] * Q_STRIDE_LOC
        + head_idx * Q_STRIDE_HEAD
        + dim_offs_q[None, :] * Q_STRIDE_DIM
    )
    tl.store(quant_base + cache_offset, packed, mask=active[:, None])

    group_ids = tl.arange(0, NUM_GROUPS)
    sz_offset_base = dst_quant_slot[:, None] * SZ_STRIDE_LOC + head_idx * SZ_STRIDE_HEAD
    tl.store(
        sz_base + sz_offset_base + (group_ids[None, :] * 2) * SZ_STRIDE_DIM,
        scale,
        mask=active[:, None],
    )
    tl.store(
        sz_base + sz_offset_base + (group_ids[None, :] * 2 + 1) * SZ_STRIDE_DIM,
        zero,
        mask=active[:, None],
    )


@triton.jit
def _fused_flush_quant_kernel(
    # Per-layer base pointers (int64 addresses, one per layer).
    hp_k_ptrs_ptr,
    hp_v_ptrs_ptr,
    quant_k_ptrs_ptr,
    quant_v_ptrs_ptr,
    k_sz_ptrs_ptr,
    v_sz_ptrs_ptr,
    # Dtype-carrying sample tensors. Triton infers the element type of each
    # pointer class from these; the actual kernel body uses the runtime
    # addresses loaded from the ``*_ptrs_ptr`` arrays above.
    hp_k_sample_ptr,
    hp_v_sample_ptr,
    quant_k_sample_ptr,
    quant_v_sample_ptr,
    k_sz_sample_ptr,
    v_sz_sample_ptr,
    # Flush plan (flat view of [bs, FLUSH_INTERVAL])
    src_hp_slot_ptr,  # int64 [num_flush_tokens]  clamped >= 0
    dst_quant_slot_ptr,  # int64 [num_flush_tokens]
    valid_mask_ptr,  # int8  [num_flush_tokens]
    num_flush_tokens,
    num_heads,
    num_layers,
    # Strides -- uniform across layers; enforced at pool construction.
    HP_K_STRIDE_LOC: tl.constexpr,
    HP_K_STRIDE_HEAD: tl.constexpr,
    HP_K_STRIDE_DIM: tl.constexpr,
    HP_V_STRIDE_LOC: tl.constexpr,
    HP_V_STRIDE_HEAD: tl.constexpr,
    HP_V_STRIDE_DIM: tl.constexpr,
    Q_K_STRIDE_LOC: tl.constexpr,
    Q_K_STRIDE_HEAD: tl.constexpr,
    Q_K_STRIDE_DIM: tl.constexpr,
    Q_V_STRIDE_LOC: tl.constexpr,
    Q_V_STRIDE_HEAD: tl.constexpr,
    Q_V_STRIDE_DIM: tl.constexpr,
    K_SZ_STRIDE_LOC: tl.constexpr,
    K_SZ_STRIDE_HEAD: tl.constexpr,
    K_SZ_STRIDE_DIM: tl.constexpr,
    V_SZ_STRIDE_LOC: tl.constexpr,
    V_SZ_STRIDE_HEAD: tl.constexpr,
    V_SZ_STRIDE_DIM: tl.constexpr,
    K_HEAD_DIM: tl.constexpr,
    K_BLOCK_QUARTER: tl.constexpr,
    K_NUM_GROUPS: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    V_BLOCK_QUARTER: tl.constexpr,
    V_NUM_GROUPS: tl.constexpr,
    V_GROUP_SIZE: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    K_CLIP_INDEX: tl.constexpr,
    V_CLIP_INDEX: tl.constexpr,
    K_BSEARCH_ITERS: tl.constexpr,
    V_BSEARCH_ITERS: tl.constexpr,
):
    """Grid: ``(cdiv(num_flush_tokens, BLOCK_TOK), num_heads, num_layers)``.

    Each program processes ``BLOCK_TOK`` consecutive flush_tokens for the
    same ``(head, layer)`` and packs both K and V. The launcher picks
    ``BLOCK_TOK`` and ``num_warps`` so each thread reads exactly 128 bits
    along the contiguous head_dim axis (8 bf16 / 16 fp8 elements).

    Per-tile early bail: ``BLOCK_TOK`` is set to the pool's
    ``flush_interval`` so all rows in a tile come from the same request and
    therefore share the same ``valid`` status. ``tl.max(valid) == 0``
    skips the entire tile on the K-1 of K decode steps when no request in
    that request-tile flushes.

    ``K_CLIP_INDEX`` / ``V_CLIP_INDEX`` of ``-1`` disable in-kernel clip;
    a non-negative value selects the per-row threshold from
    ``sort(abs(row))[CLIP_INDEX]`` (oscar path). The legacy two-pass
    ``compute_flush_clip_thresholds_triton`` + threshold-tensor flow is
    replaced by this inline computation.
    """
    pid_tok = tl.program_id(0)
    head = tl.program_id(1)
    layer = tl.program_id(2)
    if head >= num_heads or layer >= num_layers:
        return

    tok_offs = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tok_mask = tok_offs < num_flush_tokens

    valid = tl.load(valid_mask_ptr + tok_offs, mask=tok_mask, other=0).to(tl.int32)
    # Bail the whole tile when no token in it is valid. Under per-request
    # flush gating + ``BLOCK_TOK == flush_interval`` this branch is uniform
    # across the tile, so it skips the bulk of work on non-flushing decode
    # steps without dragging dead lanes through the load + sort.
    if tl.max(valid, axis=0) == 0:
        return

    active = tok_mask & (valid != 0)
    src = tl.load(src_hp_slot_ptr + tok_offs, mask=tok_mask, other=0).to(tl.int64)
    dst = tl.load(dst_quant_slot_ptr + tok_offs, mask=tok_mask, other=0).to(tl.int64)
    head64 = head.to(tl.int64)

    # K side
    hp_k_base = tl.load(hp_k_ptrs_ptr + layer).to(
        tl.pointer_type(hp_k_sample_ptr.dtype.element_ty)
    )
    q_k_base = tl.load(quant_k_ptrs_ptr + layer).to(
        tl.pointer_type(quant_k_sample_ptr.dtype.element_ty)
    )
    sz_k_base = tl.load(k_sz_ptrs_ptr + layer).to(
        tl.pointer_type(k_sz_sample_ptr.dtype.element_ty)
    )
    _fused_flush_quant_body(
        hp_k_base,
        q_k_base,
        sz_k_base,
        src,
        dst,
        active,
        head64,
        HP_K_STRIDE_LOC,
        HP_K_STRIDE_HEAD,
        HP_K_STRIDE_DIM,
        Q_K_STRIDE_LOC,
        Q_K_STRIDE_HEAD,
        Q_K_STRIDE_DIM,
        K_SZ_STRIDE_LOC,
        K_SZ_STRIDE_HEAD,
        K_SZ_STRIDE_DIM,
        K_HEAD_DIM,
        K_BLOCK_QUARTER,
        K_NUM_GROUPS,
        K_GROUP_SIZE,
        BLOCK_TOK,
        K_CLIP_INDEX,
        K_BSEARCH_ITERS,
    )
    # V side
    hp_v_base = tl.load(hp_v_ptrs_ptr + layer).to(
        tl.pointer_type(hp_v_sample_ptr.dtype.element_ty)
    )
    q_v_base = tl.load(quant_v_ptrs_ptr + layer).to(
        tl.pointer_type(quant_v_sample_ptr.dtype.element_ty)
    )
    sz_v_base = tl.load(v_sz_ptrs_ptr + layer).to(
        tl.pointer_type(v_sz_sample_ptr.dtype.element_ty)
    )
    _fused_flush_quant_body(
        hp_v_base,
        q_v_base,
        sz_v_base,
        src,
        dst,
        active,
        head64,
        HP_V_STRIDE_LOC,
        HP_V_STRIDE_HEAD,
        HP_V_STRIDE_DIM,
        Q_V_STRIDE_LOC,
        Q_V_STRIDE_HEAD,
        Q_V_STRIDE_DIM,
        V_SZ_STRIDE_LOC,
        V_SZ_STRIDE_HEAD,
        V_SZ_STRIDE_DIM,
        V_HEAD_DIM,
        V_BLOCK_QUARTER,
        V_NUM_GROUPS,
        V_GROUP_SIZE,
        BLOCK_TOK,
        V_CLIP_INDEX,
        V_BSEARCH_ITERS,
    )


# ---------------------------------------------------------------------------
# Remap kernel: update req_to_token for each flushed (req, flush_pos) entry.
# ---------------------------------------------------------------------------


@triton.jit
def _flush_remap_kernel(
    req_pool_indices_ptr,  # int64 [bs]
    flush_pos_ptr,  # int32 [bs * FLUSH_INTERVAL]
    dst_quant_slot_ptr,  # int64 [bs * FLUSH_INTERVAL]
    valid_mask_ptr,  # int8  [bs * FLUSH_INTERVAL]
    req_to_token_ptr,  # int32 [num_req_slots, max_ctx]
    rtt_stride_row,
    FLUSH_INTERVAL: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    out_idx = i * FLUSH_INTERVAL + j
    valid = tl.load(valid_mask_ptr + out_idx).to(tl.int32)
    if valid == 0:
        return
    req = tl.load(req_pool_indices_ptr + i).to(tl.int64)
    fp = tl.load(flush_pos_ptr + out_idx).to(tl.int64)
    dst = tl.load(dst_quant_slot_ptr + out_idx).to(tl.int32)
    tl.store(req_to_token_ptr + req * rtt_stride_row + fp, dst)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_kv_quant_config(
    head_dim: int, num_scale_groups: int
) -> tuple[int, int, int]:
    """Return (BLOCK_QUARTER, NUM_GROUPS, GROUP_SIZE) for the quant kernel."""
    if head_dim % num_scale_groups != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by num_scale_groups "
            f"({num_scale_groups})"
        )
    if (head_dim & (head_dim - 1)) != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be a power of two for the multi-row "
            f"flush quant kernel"
        )
    block_quarter = head_dim // 4
    group_size = head_dim // num_scale_groups
    return block_quarter, num_scale_groups, group_size


def _flush_clip_index(clip_ratio: float, head_dim: int) -> int:
    """``-1`` disables in-kernel clip; otherwise floor(ratio*head_dim) into
    ``[0, head_dim - 1]``. Mirrors the host-side logic in the per-row clip
    kernels.
    """
    if clip_ratio <= 0.0:
        return -1
    idx = int(clip_ratio * head_dim)
    if idx >= head_dim:
        idx = head_dim - 1
    if idx < 0:
        idx = 0
    return idx


def _flush_elements_per_thread(dtype: torch.dtype) -> int:
    """Elements per thread for a 128-bit vectorized load; bf16 -> 8, fp8 -> 16.
    Other dtypes raise — the flush kernel is only used in the oscar / hadamard
    paths where the HP buffer is bf16 (or fp8 on MLA configs).
    """
    if dtype == torch.bfloat16:
        return 8
    if dtype.is_floating_point and dtype.itemsize == 1:
        return 16
    raise AssertionError(
        f"flush quant kernel requires bf16 or fp8 HP dtype, got {dtype}"
    )


def _flush_block_tok_and_num_warps(
    flush_interval: int, head_dim: int, elements_per_thread: int
) -> tuple[int, int]:
    """Pick the smallest pow2 ``BLOCK_TOK`` that (a) gives one full warp of
    128-bit/thread vectorized loads along head_dim, and (b) is ``≤
    flush_interval`` so every row in a tile still comes from one request
    (clean per-tile early bail on ``valid==0``).

    Smaller ``BLOCK_TOK`` is preferred: it keeps per-program register
    pressure low and produces more grid programs to expose to the GPU,
    which boosted SM throughput from 44%→78% on Qwen3-8B flush traces.
    Larger only when one warp's worth of vectorized loads doesn't fit at
    the smaller size (very small head_dim).
    """
    fi_pow2 = triton.next_power_of_2(max(1, int(flush_interval)))
    block_tok = 2
    while block_tok * head_dim < 32 * elements_per_thread:
        block_tok *= 2
    if block_tok > fi_pow2:
        block_tok = fi_pow2
    total_elems = block_tok * head_dim
    assert total_elems % (32 * elements_per_thread) == 0, (
        f"BLOCK_TOK={block_tok} head_dim={head_dim} "
        f"epp={elements_per_thread}: tile doesn't divide cleanly into "
        "128-bit/thread loads"
    )
    num_warps = total_elems // (32 * elements_per_thread)
    return block_tok, num_warps


def gpu_flush_int2_plan(
    *,
    seq_lens: torch.Tensor,  # int32 [bs]
    prefix_lens: torch.Tensor,  # int32 [bs]
    req_pool_indices: torch.Tensor,  # int64 [bs]
    dst_quant_slots: torch.Tensor,  # int64 [bs * flush_interval]
    req_to_token: torch.Tensor,  # int32 [num_req_slots, max_ctx]
    flush_mask: torch.Tensor,  # bool [bs] -- per-request gate
    hp_prefix_tokens: int,
    hp_recent_tokens: int,
    hp_global_offset: int,
    flush_interval: int,
):
    """Plan-only phase: emit ``returned_slot_ids`` etc. without touching KV.

    This is the part of ``gpu_flush_int2`` that has no race with a concurrent
    forward batch — it only reads ``req_to_token`` and ``flush_mask`` and
    writes freshly-allocated metadata tensors. Splitting it out lets the
    caller free ``returned_slot_ids`` (which contains a sync-bearing
    ``torch.unique``) before issuing the schedule-stream wait on the previous
    forward; see ``_alloc_for_decode_mixed`` in ``mem_cache/common.py``.

    Returns ``None`` if ``bs == 0`` or ``flush_interval <= 0`` (no work).
    """
    bs = int(seq_lens.shape[0])
    if bs == 0 or flush_interval <= 0:
        return None

    assert req_to_token.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert prefix_lens.dtype == torch.int32
    assert req_pool_indices.dtype == torch.int64
    assert dst_quant_slots.dtype == torch.int64
    assert dst_quant_slots.numel() == bs * flush_interval
    assert flush_mask.shape == (bs,), (
        f"flush_mask shape {tuple(flush_mask.shape)} != ({bs},)"
    )
    flush_mask_i8 = flush_mask.to(torch.int8)

    device = seq_lens.device
    total_flush_slots = bs * flush_interval
    returned_slot_ids = torch.empty(
        (total_flush_slots,), dtype=torch.int64, device=device
    )
    src_hp_slot = torch.empty((total_flush_slots,), dtype=torch.int64, device=device)
    flush_pos = torch.empty((total_flush_slots,), dtype=torch.int32, device=device)
    valid_mask = torch.empty((total_flush_slots,), dtype=torch.int8, device=device)

    rtt_stride_row = int(req_to_token.stride(0))

    _flush_plan_kernel[(bs,)](
        seq_lens,
        prefix_lens,
        req_pool_indices,
        dst_quant_slots,
        req_to_token,
        flush_mask_i8,
        src_hp_slot,
        returned_slot_ids,
        flush_pos,
        valid_mask,
        int(req_to_token.shape[1]),
        rtt_stride_row,
        HP_PREFIX_TOKENS=int(hp_prefix_tokens),
        HP_RECENT_TOKENS=int(hp_recent_tokens),
        HP_OFFSET=int(hp_global_offset),
        FLUSH_INTERVAL=int(flush_interval),
        num_warps=1,
        num_stages=1,
    )

    return FlushPlan(
        returned_slot_ids=returned_slot_ids,
        src_hp_slot=src_hp_slot,
        flush_pos=flush_pos,
        valid_mask=valid_mask,
        dst_quant_slots=dst_quant_slots,
        bs=bs,
        flush_interval=int(flush_interval),
    )


def gpu_flush_int2_apply(
    plan: FlushPlan,
    *,
    req_pool_indices: torch.Tensor,  # int64 [bs]
    req_to_token: torch.Tensor,  # int32 [num_req_slots, max_ctx]
    # Pool-held metadata (built once at pool construction):
    hp_k_ptrs: torch.Tensor,
    hp_v_ptrs: torch.Tensor,
    quant_k_ptrs: torch.Tensor,
    quant_v_ptrs: torch.Tensor,
    k_sz_ptrs: torch.Tensor,
    v_sz_ptrs: torch.Tensor,
    hp_k_sample: torch.Tensor,
    hp_v_sample: torch.Tensor,
    quant_k_sample: torch.Tensor,
    quant_v_sample: torch.Tensor,
    k_sz_sample: torch.Tensor,
    v_sz_sample: torch.Tensor,
    hp_k_strides: tuple[int, int, int],
    hp_v_strides: tuple[int, int, int],
    quant_k_strides: tuple[int, int, int],
    quant_v_strides: tuple[int, int, int],
    k_sz_strides: tuple[int, int, int],
    v_sz_strides: tuple[int, int, int],
    # Scalar config:
    num_heads: int,
    head_dim: int,
    v_head_dim: int,
    k_num_scale_groups: int,
    v_num_scale_groups: int,
    num_layers: int,
    k_clip_ratio: float = 0.0,
    v_clip_ratio: float = 0.0,
) -> None:
    """Apply phase: run fused quant + remap kernels using a prepared plan.

    Must be issued *after* the schedule-stream wait on the previous forward —
    the remap kernel writes ``req_to_token`` at positions the previous
    forward's attention is concurrently reading.
    """
    bs = plan.bs
    flush_interval = plan.flush_interval
    total_flush_slots = bs * flush_interval

    # src_hp_slot.clamp(min=0) routes invalid rows to HP slot 0 (the reserved
    # padding row). Their quant outputs are garbage but are freed by the
    # caller in the same bulk free as the valid flushes.
    safe_src_hp_slot = plan.src_hp_slot.clamp(min=0)

    k_block_quarter, k_num_groups, k_group_size = _resolve_kv_quant_config(
        head_dim, k_num_scale_groups
    )
    v_block_quarter, v_num_groups, v_group_size = _resolve_kv_quant_config(
        v_head_dim, v_num_scale_groups
    )

    k_clip_index = _flush_clip_index(k_clip_ratio, head_dim)
    v_clip_index = _flush_clip_index(v_clip_ratio, v_head_dim)

    elements_per_thread = _flush_elements_per_thread(hp_k_sample.dtype)
    block_tok, num_warps = _flush_block_tok_and_num_warps(
        flush_interval, head_dim, elements_per_thread
    )
    grid = (
        triton.cdiv(total_flush_slots, block_tok),
        num_heads,
        int(num_layers),
    )
    _fused_flush_quant_kernel[grid](
        hp_k_ptrs,
        hp_v_ptrs,
        quant_k_ptrs,
        quant_v_ptrs,
        k_sz_ptrs,
        v_sz_ptrs,
        hp_k_sample,
        hp_v_sample,
        quant_k_sample,
        quant_v_sample,
        k_sz_sample,
        v_sz_sample,
        safe_src_hp_slot,
        plan.dst_quant_slots,
        plan.valid_mask,
        total_flush_slots,
        num_heads,
        int(num_layers),
        HP_K_STRIDE_LOC=hp_k_strides[0],
        HP_K_STRIDE_HEAD=hp_k_strides[1],
        HP_K_STRIDE_DIM=hp_k_strides[2],
        HP_V_STRIDE_LOC=hp_v_strides[0],
        HP_V_STRIDE_HEAD=hp_v_strides[1],
        HP_V_STRIDE_DIM=hp_v_strides[2],
        Q_K_STRIDE_LOC=quant_k_strides[0],
        Q_K_STRIDE_HEAD=quant_k_strides[1],
        Q_K_STRIDE_DIM=quant_k_strides[2],
        Q_V_STRIDE_LOC=quant_v_strides[0],
        Q_V_STRIDE_HEAD=quant_v_strides[1],
        Q_V_STRIDE_DIM=quant_v_strides[2],
        K_SZ_STRIDE_LOC=k_sz_strides[0],
        K_SZ_STRIDE_HEAD=k_sz_strides[1],
        K_SZ_STRIDE_DIM=k_sz_strides[2],
        V_SZ_STRIDE_LOC=v_sz_strides[0],
        V_SZ_STRIDE_HEAD=v_sz_strides[1],
        V_SZ_STRIDE_DIM=v_sz_strides[2],
        K_HEAD_DIM=int(head_dim),
        K_BLOCK_QUARTER=k_block_quarter,
        K_NUM_GROUPS=k_num_groups,
        K_GROUP_SIZE=k_group_size,
        V_HEAD_DIM=int(v_head_dim),
        V_BLOCK_QUARTER=v_block_quarter,
        V_NUM_GROUPS=v_num_groups,
        V_GROUP_SIZE=v_group_size,
        BLOCK_TOK=block_tok,
        K_CLIP_INDEX=k_clip_index,
        V_CLIP_INDEX=v_clip_index,
        K_BSEARCH_ITERS=(int(head_dim).bit_length() - 1) if head_dim >= 64 else 0,
        V_BSEARCH_ITERS=(int(v_head_dim).bit_length() - 1) if v_head_dim >= 64 else 0,
        num_warps=num_warps,
        num_stages=1,
    )

    rtt_stride_row = int(req_to_token.stride(0))
    _flush_remap_kernel[(bs, flush_interval)](
        req_pool_indices,
        plan.flush_pos,
        plan.dst_quant_slots,
        plan.valid_mask,
        req_to_token,
        rtt_stride_row,
        FLUSH_INTERVAL=int(flush_interval),
        num_warps=1,
        num_stages=1,
    )


def gpu_flush_int2(
    *,
    seq_lens: torch.Tensor,  # int32 [bs]
    prefix_lens: torch.Tensor,  # int32 [bs]
    req_pool_indices: torch.Tensor,  # int64 [bs]
    dst_quant_slots: torch.Tensor,  # int64 [bs * flush_interval]
    req_to_token: torch.Tensor,  # int32 [num_req_slots, max_ctx]
    flush_mask: torch.Tensor,  # bool [bs] -- per-request gate
    # Pool-held metadata (built once at pool construction):
    hp_k_ptrs: torch.Tensor,  # int64 [num_layers]
    hp_v_ptrs: torch.Tensor,  # int64 [num_layers]
    quant_k_ptrs: torch.Tensor,  # int64 [num_layers]
    quant_v_ptrs: torch.Tensor,  # int64 [num_layers]
    k_sz_ptrs: torch.Tensor,  # int64 [num_layers]
    v_sz_ptrs: torch.Tensor,  # int64 [num_layers]
    hp_k_sample: torch.Tensor,
    hp_v_sample: torch.Tensor,
    quant_k_sample: torch.Tensor,
    quant_v_sample: torch.Tensor,
    k_sz_sample: torch.Tensor,
    v_sz_sample: torch.Tensor,
    hp_k_strides: tuple[int, int, int],
    hp_v_strides: tuple[int, int, int],
    quant_k_strides: tuple[int, int, int],
    quant_v_strides: tuple[int, int, int],
    k_sz_strides: tuple[int, int, int],
    v_sz_strides: tuple[int, int, int],
    # Scalar config:
    hp_prefix_tokens: int,
    hp_recent_tokens: int,
    hp_global_offset: int,
    num_heads: int,
    head_dim: int,
    v_head_dim: int,
    k_num_scale_groups: int,
    v_num_scale_groups: int,
    num_layers: int,
    flush_interval: int,
    k_clip_ratio: float = 0.0,
    v_clip_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run plan -> fused quant -> remap for one decode flush step.

    Thin wrapper over :func:`gpu_flush_int2_plan` and
    :func:`gpu_flush_int2_apply` that preserves the original monolithic
    signature and return contract for callers that don't need to interleave
    other work between the two phases.

    Returns:
        returned_slot_ids : int64 [bs * flush_interval]
            For each entry, either the HP slot that was flushed (global id) or
            the unused pre-allocated quant slot.
        valid_mask : int8 [bs * flush_interval]
            1 where a flush occurred; 0 otherwise.
    """
    plan = gpu_flush_int2_plan(
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        req_pool_indices=req_pool_indices,
        dst_quant_slots=dst_quant_slots,
        req_to_token=req_to_token,
        flush_mask=flush_mask,
        hp_prefix_tokens=hp_prefix_tokens,
        hp_recent_tokens=hp_recent_tokens,
        hp_global_offset=hp_global_offset,
        flush_interval=flush_interval,
    )
    if plan is None:
        device = seq_lens.device
        empty = torch.empty((0,), dtype=torch.int64, device=device)
        mask = torch.empty((0,), dtype=torch.int8, device=device)
        return empty, mask

    gpu_flush_int2_apply(
        plan,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        hp_k_ptrs=hp_k_ptrs,
        hp_v_ptrs=hp_v_ptrs,
        quant_k_ptrs=quant_k_ptrs,
        quant_v_ptrs=quant_v_ptrs,
        k_sz_ptrs=k_sz_ptrs,
        v_sz_ptrs=v_sz_ptrs,
        hp_k_sample=hp_k_sample,
        hp_v_sample=hp_v_sample,
        quant_k_sample=quant_k_sample,
        quant_v_sample=quant_v_sample,
        k_sz_sample=k_sz_sample,
        v_sz_sample=v_sz_sample,
        hp_k_strides=hp_k_strides,
        hp_v_strides=hp_v_strides,
        quant_k_strides=quant_k_strides,
        quant_v_strides=quant_v_strides,
        k_sz_strides=k_sz_strides,
        v_sz_strides=v_sz_strides,
        num_heads=num_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        k_num_scale_groups=k_num_scale_groups,
        v_num_scale_groups=v_num_scale_groups,
        num_layers=num_layers,
        k_clip_ratio=k_clip_ratio,
        v_clip_ratio=v_clip_ratio,
    )

    return plan.returned_slot_ids, plan.valid_mask
