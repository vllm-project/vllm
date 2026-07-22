# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Compressor boundary kernel for V4 attention (CSA / sparse_attn path).

Replaces the Python pool → RMSNorm → RoPE → (quant) → kv_cache scatter
chain in `Compressor.forward` (see `atom/models/deepseek_v4.py`).

SGLang plan-style batched dispatch (vs. the earlier per-seq launcher):
  Each compression boundary across the entire fwd is one row in
  `compress_plan_gpu` — a packed `[num_compress, 4] int32` tensor where each
  row is `[ragged_id, batch_id, position, window_len]`. The kernel grid is
  the caller-supplied slice length (decode CG: `_decode_compress_cap[ratio]`
  / eager prefill: `n_compress`); inactive plan rows are sentinel-marked
  (`position == -1`) and bail at the top of the kernel. Each program does
  ONE 4×i32 load to get all the metadata it needs:

    ragged_id  → row index in the ragged kv_in / score_in stream
    batch_id   → seq index → state_slot_mapping[batch_id], block_table[batch_id]
    position   → absolute token position (drives RoPE + paged scatter); -1 = skip
    window_len → number of leading K-loop iterations that read state cache
                 (instead of the ragged input). K = STATE_SIZE.

State cache vs. input vs. padding dispatch (replaces the old `s >= start_pos`
test):

    s = position - K + 1 + k_static
    is_padding = s < 0
    is_input   = k_static >= window_len
    is_state   = (~is_input) & (~is_padding)

Correctness invariant (caller-side):
  This kernel reads state cache as-of-the-end-of-the-PREVIOUS-fwd. Therefore
  the caller MUST invoke this kernel BEFORE `update_compressor_states` runs
  (which would overwrite the historic positions this kernel needs).

Quant modes (constexpr-selected by the `quant` arg of the Python wrapper):
  - quant=False (CSA Main / HCA Main): writes BF16 rows into the paged BF16
    `kv_cache` at compressed slot `position // ratio`.
  - quant=True (CSA Indexer-inner): per-row amax → ue8m0 (or raw) scale →
    fp8 cast → preshuffled (MFMA 16x16 tile) write into the FP8 `kv_cache`,
    plus fp32 scale into the per-block scale region (`cache_scale` — a
    strided view of the same allocation built by the V4 builder). Bit-exact
    match for `indexer_k_quant_and_cache` /
    `cp_gather_indexer_k_quant_cache` (cache_kernels.cu:1145+).

Output: side-effecting only — cache scatter IS the only output. The earlier
caller-visible `[num_compress, head_dim]` BF16 return tensor was vestigial
(paged_decode/paged_prefill read the scattered compress entries directly
from `unified_kv` (Main) or the FP8 indexer pool, not from the kernel
return).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from vllm.models.deepseek_v4.amd.atom.model_ops.v4_kernels.compress_plan import CompressPlan
from vllm.models.deepseek_v4.amd.atom.utils import envs

# Optional flydsl path (aiter ROCm kernels). Falls back to Triton when
# unavailable. HCA = compress + norm_rope_scatter 2-kernel split for
# D=512 ratio=128 overlap=False.
try:
    from aiter.ops.flydsl.kernels.fused_compress_attn import flydsl_fused_compress_attn
    from aiter.ops.flydsl.kernels.fused_compress_attn_hca import (
        flydsl_hca_compress_attn,
    )
except Exception:
    flydsl_fused_compress_attn = None
    flydsl_hca_compress_attn = None

# Supported (head_dim, rope_head_dim, ratio, overlap) tuples for the flydsl
# kernel — matches V4-Pro Main (D=512) and Indexer-inner (D=128) compressors.
# Extend as more configs are validated.
_FLYDSL_SUPPORTED = {
    (512, 64, 4, True),  # V4-Pro CSA Main BF16   (ratio=4, OVERLAP)
    (128, 64, 4, True),  # V4-Pro CSA Indexer FP8 (ratio=4, OVERLAP)
    (512, 64, 128, False),  # V4-Pro HCA Main BF16  (ratio=128, no overlap)
}


@triton.jit
def _fused_compress_attn_kernel(
    # ── source: INPUT (this fwd's projection) ───────────────────────────
    kv_in_ptr,  # [num_q_tokens, dim_full] fp32 (strided allowed)
    kv_in_row_stride,  # row stride; ≥ dim_full when fused upstream split
    score_in_ptr,  # [num_q_tokens, dim_full] fp32 (raw, no ape; strided allowed)
    score_in_row_stride,
    dim_full,  # = 2*head_dim if OVERLAP else head_dim
    # ── plan: per-boundary packed metadata ──────────────────────────────
    plan_ptr,  # [num_compress, 4] int32 (ragged_id, batch_id, position, window_len)
    # ── source: state cache (previous fwd's writes; score has ape) ──────
    kv_state_ptr,
    kv_state_slot_stride,
    kv_state_pos_stride,
    score_state_ptr,
    score_state_slot_stride,
    score_state_pos_stride,
    state_slot_mapping_ptr,  # [bs] int32 — per-seq state cache slot
    # ── ape (for INPUT-source rows only) ────────────────────────────────
    ape_ptr,  # [RATIO, dim_full] fp32
    # ── RMSNorm ─────────────────────────────────────────────────────────
    rms_weight_ptr,  # [head_dim] fp32
    rms_eps,
    # ── RoPE (separate cos / sin caches) ────────────────────────────────
    cos_cache_ptr,  # [max_seq, rope_head_dim/2] bf16 (after .squeeze)
    sin_cache_ptr,
    cos_sin_pos_stride,  # = rope_head_dim // 2
    # ── KV cache scatter (paged) ────────────────────────────────────────
    kv_cache_ptr,  # bf16: [NB, k_per_block, head_dim] / fp8: [NB, k_per_block, head_dim]
    kv_cache_block_stride,
    kv_cache_token_stride,
    cache_scale_ptr,  # fp32 [NB, k_per_block] (QUANT path only; dummy ptr otherwise)
    cache_scale_block_stride,  # fp32 elements per block (= k_per_block * cache_stride / 4)
    block_table_ptr,  # [bs, max_blocks_per_seq] int32
    block_table_seq_stride,  # row stride
    k_per_block,
    head_dim,
    rope_head_dim,
    # ── constexpr ───────────────────────────────────────────────────────
    BLOCK_D: tl.constexpr,  # = next_pow2(head_dim)
    HALF_ROPE: tl.constexpr,  # = rope_head_dim // 2
    OVERLAP: tl.constexpr,
    RATIO: tl.constexpr,
    STATE_SIZE: tl.constexpr,  # ring buffer modulo = kv_state.shape[1] (≥ K_pool;
    #   spec decode: K_pool + max_spec_steps so R's rejected writes fall outside
    #   R+1's K_pool-wide read window; non-spec decode: K_pool exactly — no
    #   rejection ever happens and causal writes preclude any read-before-overwrite)
    K: tl.constexpr,  # pool-window reduce dim (= 2*RATIO if OVERLAP else RATIO);
    #   ≤ STATE_SIZE; used for `s = position - K + 1 + k_static` loop bound
    HAS_BLOCK_TABLE: tl.constexpr,
    QUANT: tl.constexpr,  # 0 = raw BF16 (CSA/HCA Main), 1 = FP8 e4m3 + ue8m0 scale (Indexer)
    USE_UE8M0: tl.constexpr,  # round scale to power-of-2 (only when QUANT == 1)
    PRESHUFFLE: tl.constexpr,  # MFMA 16x16 preshuffled FP8 layout (only when QUANT == 1)
    # E4M3 max (=448 for E4M3FN, =240 for E4M3FNUZ). Constexpr so the clamp
    # bounds and reciprocal fold to compile-time constants.  Ignored when
    # QUANT == 0 (caller passes 1.0 as a placeholder).
    FP8_MAX: tl.constexpr = 1.0,
):
    """One program per boundary in the plan. Grid = caller-supplied slice
    length (decode CG: `_decode_compress_cap[ratio]` for capture/replay
    address stability; eager prefill: tight `n_compress`). Inactive rows
    are sentinel-marked (position == -1) and bail before any load /
    store / scatter."""
    pid = tl.program_id(0)
    plan_base = plan_ptr + pid * 4
    ragged_id = tl.load(plan_base + 0)
    batch_id = tl.load(plan_base + 1)
    position = tl.load(plan_base + 2)
    window_len = tl.load(plan_base + 3)

    if position < 0:
        return

    slot = tl.load(state_slot_mapping_ptr + batch_id)

    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim

    # ── 1. Per-source-position load + online softmax-pool ──────────────
    # Two-phase split (vs. the older single masked loop): for each program
    # `is_input = k_static >= window_len` partitions the K iterations into
    # a leading state-cache-only run and a trailing input-only run. Issuing
    # only the live side's loads (rather than masked-off both) cuts HBM
    # bandwidth ~40% on AMD CDNA where masked tl.load still issues the LD
    # instruction (predicate only suppresses register write-back).
    #
    # Padding invariant: padding (`s < 0` ⟺ `k_static < K-1-position`) lies
    # entirely within `k_static < window_len` because
    # `window_len = K - min(j_in_seq+1, K) ≥ K - 1 - j_in_seq ≥ K - 1 - position`
    # (`position = prefix_len + j_in_seq`, `prefix_len ≥ 0`). The input phase
    # therefore needs no padding mask.
    NEG_INF: tl.constexpr = float("-inf")
    m_acc = tl.full([BLOCK_D], NEG_INF, tl.float32)
    kv_acc = tl.zeros([BLOCK_D], tl.float32)
    w_acc = tl.zeros([BLOCK_D], tl.float32)

    # ── Phase 1: state cache (k_static ∈ [0, window_len)) ──
    # Dynamic bound (window_len is per-program, not constexpr) — Triton
    # cannot static-unroll this. Loop body issues only state-side loads.
    for k_static in tl.range(0, window_len):
        s = position - K + 1 + k_static
        is_padding = s < 0
        # B-side (k >= RATIO): cols [head_dim:]; A-side (k < RATIO): cols [:head_dim].
        # HCA (no overlap, K=RATIO): col_off=0 always.
        col_off = (k_static >= RATIO) * head_dim if OVERLAP else 0

        s_safe = tl.maximum(s, 0)
        ring = s_safe % STATE_SIZE
        state_row_off = (
            slot * kv_state_slot_stride + ring * kv_state_pos_stride + col_off
        )
        kv_b = tl.load(
            kv_state_ptr + state_row_off + d,
            mask=(~is_padding) & d_mask,
            other=0.0,
        )
        score_b = tl.load(
            score_state_ptr
            + slot * score_state_slot_stride
            + ring * score_state_pos_stride
            + col_off
            + d,
            mask=(~is_padding) & d_mask,
            other=NEG_INF,
        )

        m_new = tl.maximum(m_acc, score_b)
        scale = tl.where(m_acc == NEG_INF, 0.0, tl.exp(m_acc - m_new))
        # Padding lanes have score_b = NEG_INF → w_k = 0, contributes nothing.
        w_k = tl.where(score_b == NEG_INF, 0.0, tl.exp(score_b - m_new))
        kv_acc = kv_acc * scale + w_k * kv_b
        w_acc = w_acc * scale + w_k
        m_acc = m_new

    # ── Phase 2: ragged input (k_static ∈ [window_len, K)) ──
    # No padding here (per invariant above). All loads unconditional in
    # the position dimension; only the head_dim mask remains.
    for k_static in tl.range(window_len, K):
        col_off = (k_static >= RATIO) * head_dim if OVERLAP else 0
        ape_row = k_static % RATIO
        # k_static = K-1 → s = position (the boundary token itself,
        # = ragged_id row). Earlier k_static map to earlier ragged rows.
        in_row = ragged_id - (K - 1 - k_static)

        # kv_in / score_in: single-use per program → evict_first to keep
        # state-cache lines (small, possibly shared across programs) hot.
        kv_a = tl.load(
            kv_in_ptr + in_row * kv_in_row_stride + col_off + d,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        score_a = tl.load(
            score_in_ptr + in_row * score_in_row_stride + col_off + d,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        ape_v = tl.load(
            ape_ptr + ape_row * dim_full + col_off + d,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        score_k = score_a + ape_v

        m_new = tl.maximum(m_acc, score_k)
        scale = tl.where(m_acc == NEG_INF, 0.0, tl.exp(m_acc - m_new))
        # score_k always finite in input phase → no NEG_INF guard needed.
        w_k = tl.exp(score_k - m_new)
        kv_acc = kv_acc * scale + w_k * kv_a
        w_acc = w_acc * scale + w_k
        m_acc = m_new

    compressed = kv_acc / w_acc  # [BLOCK_D] fp32

    # ── 2. RMSNorm (fp32) ──────────────────────────────────────────────
    rms_w = tl.load(rms_weight_ptr + d, mask=d_mask, other=0.0)
    compressed_masked = tl.where(d_mask, compressed, 0.0)
    var = tl.sum(compressed_masked * compressed_masked, axis=0) / head_dim
    rrms = tl.rsqrt(var + rms_eps)
    normed = compressed_masked * rrms * rms_w  # [BLOCK_D] fp32

    # ── 3. RoPE on rope_head_dim segment (GPT-J interleaved, fp32) ────
    comp_pos = (position // RATIO) * RATIO
    NUM_PAIRS: tl.constexpr = BLOCK_D // 2
    NOPE_PAIRS = (head_dim - rope_head_dim) // 2

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even_v, odd_v = tl.split(pair_2d)  # each [NUM_PAIRS]

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    cos_per_pair = tl.load(
        cos_cache_ptr + comp_pos * cos_sin_pos_stride + cs_idx,
        mask=is_rope_pair,
        other=1.0,
    ).to(tl.float32)
    sin_per_pair = tl.load(
        sin_cache_ptr + comp_pos * cos_sin_pos_stride + cs_idx,
        mask=is_rope_pair,
        other=0.0,
    ).to(tl.float32)

    new_even = even_v * cos_per_pair - odd_v * sin_per_pair
    new_odd = odd_v * cos_per_pair + even_v * sin_per_pair
    rotated = tl.interleave(new_even, new_odd)  # [BLOCK_D] fp32

    # ── 4. Cache scatter (paged) ───────────────────────────────────────
    # The Compressor's BF16 return value was historically consumed by sparse
    # attention but is now vestigial — paged_decode/paged_prefill read the
    # scattered compress entries directly from `unified_kv` (Main) or the FP8
    # indexer pool. So no caller-visible `out` write; the cache scatter IS
    # the only output.
    if HAS_BLOCK_TABLE:
        # block_table[batch_id, ci // k_per_block] → physical_block.
        # slot_in_block = ci % k_per_block. Same address resolution as
        # `indexer_k_quant_and_cache` (which used a pre-flattened slot_mapping
        # = physical_block * k_per_block + slot_in_block).
        ci = position // RATIO
        block_in_seq = ci // k_per_block
        slot_in_block = ci % k_per_block
        physical_block = tl.load(
            block_table_ptr + batch_id * block_table_seq_stride + block_in_seq
        ).to(tl.int64)

        if QUANT:
            # FP8 e4m3 quantization: per-row amax → ue8m0 (or raw) scale →
            # clamp+cast to fp8 → write preshuffled (MFMA 16x16 tile) or
            # linear layout into the FP8 region; write the fp32 scale into
            # the per-block scale region (separate `cache_scale_ptr` view).
            # Bit-exact match for `indexer_k_quant_and_cache` /
            # `cp_gather_indexer_k_quant_cache` (cache_kernels.cu:1145+).
            #
            # Quant pattern follows aiter's `_fp8_quant_op` /
            # `_fused_rms_gated_fp8_group_quant_kernel` — the explicit
            # `fp_downcast_rounding="rtne"` is intentionally NOT used here:
            # on AMD it forces a slow software-RTNE path and bypasses the
            # `v_cvt_pk_fp8_f32` HW intrinsic (which already rounds RTNE).
            # The pre-cast `tl.clamp` saturates intermediate overflow and
            # mirrors aiter's hand-tuned HIP `aiter::scaled_cast`.
            rotated_for_amax = tl.where(d_mask, tl.abs(rotated), 0.0)
            amax = tl.max(rotated_for_amax, axis=0)  # scalar (per row)
            scale = tl.maximum(amax, 1e-4) * (1.0 / FP8_MAX)
            if USE_UE8M0:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))
            inv_scale = 1.0 / scale
            scaled = tl.clamp(rotated * inv_scale, -FP8_MAX, FP8_MAX)
            fp8_val = scaled.to(kv_cache_ptr.dtype.element_ty)
            if PRESHUFFLE:
                TILE: tl.constexpr = 16
                token_tile_id = slot_in_block // TILE
                token_in_tile = slot_in_block % TILE
                col_tile_id = d // TILE
                col_in_tile = d % TILE
                fp8_offset = (
                    physical_block * kv_cache_block_stride
                    + token_tile_id * (TILE * head_dim)
                    + col_tile_id * (TILE * TILE)
                    + token_in_tile * TILE
                    + col_in_tile
                )
            else:
                fp8_offset = (
                    physical_block * kv_cache_block_stride
                    + slot_in_block * head_dim
                    + d
                )
            # Streaming write — these slots aren't reread inside this kernel.
            tl.store(
                kv_cache_ptr + fp8_offset, fp8_val, mask=d_mask, cache_modifier=".cs"
            )
            # Scale: one fp32 per row, packed at end of block.
            scale_offset = physical_block * cache_scale_block_stride + slot_in_block
            tl.store(cache_scale_ptr + scale_offset, scale, cache_modifier=".cs")
        else:
            cache_addr = (
                physical_block * kv_cache_block_stride
                + slot_in_block * kv_cache_token_stride
                + d
            )
            tl.store(kv_cache_ptr + cache_addr, rotated.to(tl.bfloat16), mask=d_mask)


def fused_compress_attn(
    *,
    # Source tensors (ragged across all seqs in batch)
    kv_in: torch.Tensor,  # [num_q_tokens, dim_full] fp32
    score_in: torch.Tensor,  # [num_q_tokens, dim_full] fp32 (raw, no ape)
    kv_state: torch.Tensor,  # [num_slots, STATE_SIZE, dim_full] fp32
    score_state: torch.Tensor,  # same shape, score has ape pre-added
    # Plan + per-seq metadata
    plan: CompressPlan,
    state_slot_mapping: torch.Tensor,  # [bs] int32 — per-seq state cache slot
    # Compressor params
    ape: torch.Tensor,  # [ratio, dim_full] fp32
    rms_weight: torch.Tensor,  # [head_dim] fp32
    rms_eps: float,
    cos_cache: torch.Tensor,  # [max_seq, ..., rope_head_dim/2] bf16/fp16
    sin_cache: torch.Tensor,  # same shape
    # KV cache scatter
    kv_cache: Optional[
        torch.Tensor
    ],  # bf16: [NB, k_per_block, head_dim] / fp8: same shape, fp8
    block_tables: Optional[torch.Tensor],  # [bs, max_blocks_per_seq] int32
    k_per_block: int,
    # Geometry
    overlap: bool,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    # FP8 quant fusion (Indexer-inner Compressor path)
    quant: bool = False,
    cache_scale: Optional[
        torch.Tensor
    ] = None,  # fp32 [NB, k_per_block]; required when quant=True
    use_ue8m0: bool = True,  # round scale to power-of-2 (UE8M0); only when quant=True
    preshuffle: bool = True,  # MFMA 16x16 preshuffled FP8 layout; only when quant=True
    fp8_max: Optional[float] = None,  # E4M3 max; required when quant=True
) -> None:
    """Batched fused per-source-position pool + RMSNorm + RoPE + cache scatter,
    dispatched via SGLang-style packed plan.

    Two scatter modes (constexpr-selected by `quant`):

      - quant=False (CSA Main / HCA Main): writes BF16 rows into the paged
        BF16 kv_cache (compressed slot = `position // ratio`).

      - quant=True (CSA Indexer-inner): per-row amax → ue8m0 scale → fp8 cast
        → preshuffled (MFMA 16x16 tile) write into the FP8 kv_cache, plus
        fp32 scale into `cache_scale` (the per-block scale region of the
        same allocation). Bit-exact match for `indexer_k_quant_and_cache` /
        `cp_gather_indexer_k_quant_cache` (cache_kernels.cu:1145+).

    Side-effecting: cache scatter IS the only output (Main path's BF16
    return tensor was vestigial — paged_decode/paged_prefill read directly
    from `unified_kv` and the indexer FP8 pool, not from the kernel return).
    Grid is always `plan_capacity` (CUDAGraph-safe); inactive plan rows are
    sentinel-skipped (`position == -1`) inside the kernel.

    Caller MUST invoke BEFORE `update_compressor_states` (state cache reads
    must see previous-fwd data).
    """
    plan_capacity = plan.compress_plan_gpu.shape[0]
    num_compress = plan.num_compress
    if plan_capacity == 0:
        return  # nothing to do — no plan rows ever populated.

    # ------------------------------------------------------------------
    # flydsl dispatch. Pure-GPU time on V4-Pro beats Triton 0.9x→2.9x
    # across the relevant N_compress range; the small-N gap is bridged
    # by the kernel doing both BF16 and FP8 paths through a single
    # launcher (less per-call Python overhead at the boundary).
    # ------------------------------------------------------------------
    # "auto": use flydsl on supported shapes (V4-Pro), else fall back to Triton.
    _flydsl_mode = "auto"
    _shape_key = (head_dim, rope_head_dim, ratio, overlap)
    _flydsl_shape_ok = _shape_key in _FLYDSL_SUPPORTED
    _flydsl_use = (
        flydsl_fused_compress_attn is not None
        and _flydsl_mode in ("auto", "always")
        and _flydsl_shape_ok
    )
    if _flydsl_mode == "always" and not _flydsl_shape_ok:
        raise RuntimeError(
            f"ATOM_FUSED_COMPRESS_USE_FLYDSL=always but shape "
            f"{_shape_key} is not in supported set {_FLYDSL_SUPPORTED}"
        )
    # HCA 2-kernel-split: BF16-only on V4-Pro HCA Main shape
    # (D=512 ratio=128 overlap=False). HCA wins single-kernel at all N
    # (1.06-3.7×) post slice_size + VEC=8 refactor.
    _hca_use = (
        _flydsl_use
        and flydsl_hca_compress_attn is not None
        and not quant
        and _shape_key == (512, 64, 128, False)
    )
    if _hca_use:
        flydsl_hca_compress_attn(
            kv_in=kv_in,
            score_in=score_in,
            kv_state=kv_state,
            score_state=score_state,
            state_slot_mapping=state_slot_mapping,
            plan_gpu=plan.compress_plan_gpu,
            ape=ape,
            rms_weight=rms_weight,
            rms_eps=rms_eps,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            kv_cache=kv_cache,
            block_tables=block_tables,
            k_per_block=k_per_block,
            ratio=ratio,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
        )
        return
    if _flydsl_use:
        flydsl_fused_compress_attn(
            kv_in=kv_in,
            score_in=score_in,
            kv_state=kv_state,
            score_state=score_state,
            plan_gpu=plan.compress_plan_gpu,
            state_slot_mapping=state_slot_mapping,
            ape=ape,
            rms_weight=rms_weight,
            rms_eps=rms_eps,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            kv_cache=kv_cache,
            block_tables=block_tables,
            k_per_block=k_per_block,
            overlap=overlap,
            ratio=ratio,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            quant=quant,
            cache_scale=cache_scale,
            use_ue8m0=use_ue8m0,
            preshuffle=preshuffle,
        )
        return

    # Validate shapes
    dim_full = (2 if overlap else 1) * head_dim
    K_pool = (2 if overlap else 1) * ratio  # pool window size (algorithm-defined)
    state_size = kv_state.shape[
        1
    ]  # ring buffer modulo (≥ K_pool; V4-Pro: K_pool + max_spec_steps spec / K_pool non-spec)
    assert (
        kv_in.dim() == 2 and kv_in.shape[1] == dim_full
    ), f"kv_in {kv_in.shape}, expected [*, {dim_full}]"
    assert score_in.shape == kv_in.shape
    assert (
        state_size >= K_pool and kv_state.shape[2] == dim_full
    ), f"kv_state {kv_state.shape}, expected [*, ≥{K_pool}, {dim_full}]"
    assert score_state.shape == kv_state.shape
    assert ape.shape == (ratio, dim_full)
    assert rms_weight.shape == (head_dim,)
    assert plan.compress_plan_gpu.shape == (
        plan_capacity,
        4,
    ), f"plan {plan.compress_plan_gpu.shape}, expected ({plan_capacity}, 4)"
    assert plan.compress_plan_gpu.dtype == torch.int32
    assert num_compress <= plan_capacity, (
        f"plan.num_compress ({num_compress}) > capacity ({plan_capacity}); "
        f"caller must size the plan buffer to the worst-case num_compress."
    )
    assert state_slot_mapping.dim() == 1 and state_slot_mapping.dtype == torch.int32
    assert cos_cache.shape[-1] == rope_head_dim // 2
    assert sin_cache.shape[-1] == rope_head_dim // 2
    assert (
        cos_cache.stride(0) == rope_head_dim // 2
    ), f"cos_cache outer stride {cos_cache.stride(0)} != rope_head_dim/2"
    # kv_in / score_in row-strided allowed (e.g. zero-copy split halves of the
    # fused wkv_gate output). Inner column stride must be 1 — kernel uses
    # `+ d` for the BLOCK_D offset.
    assert kv_in.stride(-1) == 1 and score_in.stride(-1) == 1
    assert kv_state.is_contiguous() and score_state.is_contiguous()
    assert ape.is_contiguous() and rms_weight.is_contiguous()
    has_bt = block_tables is not None
    if has_bt:
        assert kv_cache is not None and kv_cache.dim() == 3
        assert block_tables.dim() == 2 and block_tables.is_contiguous()
        bt_seq_stride = block_tables.stride(0)
    else:
        bt_seq_stride = 0

    # Quant validation. quant=True is FP8 cache write (Indexer-inner path);
    # requires a valid block_tables (slot resolution) AND a paired fp32 scale
    # view of the same allocation (see Compressor.forward for how to slice it).
    if quant:
        assert has_bt, "quant=True requires block_tables for slot resolution"
        assert (
            kv_cache.dtype != torch.bfloat16
        ), f"quant=True expects an FP8/uint8 kv_cache; got {kv_cache.dtype}"
        assert (
            cache_scale is not None and cache_scale.dtype == torch.float32
        ), "quant=True requires `cache_scale` (fp32 [NB, k_per_block])"
        assert cache_scale.dim() == 2 and cache_scale.shape[0] == kv_cache.shape[0]
        assert fp8_max is not None and fp8_max > 0
        if preshuffle:
            assert (
                head_dim % 16 == 0
            ), f"preshuffle requires head_dim%16==0, got {head_dim}"
            assert (
                k_per_block % 16 == 0
            ), f"preshuffle requires k_per_block%16==0, got {k_per_block}"

    BLOCK_D = triton.next_power_of_2(head_dim)
    HALF_ROPE = rope_head_dim // 2
    K = K_pool  # pool window reduce-dim (constexpr; not equal to ring modulo)

    # Cache-scale args (only consumed by the quant path; pass placeholders
    # otherwise so the constexpr branch is never taken).
    if quant:
        cache_scale_arg = cache_scale
        cache_scale_block_stride_arg = cache_scale.stride(0)
        fp8_max_arg = float(fp8_max)
    else:
        cache_scale_arg = state_slot_mapping  # placeholder int32 ptr (unused)
        cache_scale_block_stride_arg = 0
        fp8_max_arg = 1.0  # placeholder; FP8_MAX is constexpr, must be > 0

    # Fixed grid for CUDAGraph compat: launch one program per plan row;
    # sentinel rows (position=-1) skip inside the kernel.
    grid = (plan_capacity,)
    _fused_compress_attn_kernel[grid](
        kv_in,
        kv_in.stride(0),
        score_in,
        score_in.stride(0),
        dim_full,
        plan.compress_plan_gpu,
        kv_state,
        kv_state.stride(0),
        kv_state.stride(1),
        score_state,
        score_state.stride(0),
        score_state.stride(1),
        state_slot_mapping,
        ape,
        rms_weight,
        rms_eps,
        cos_cache,
        sin_cache,
        cos_cache.stride(0),
        kv_cache if has_bt else cos_cache,  # placeholder when no scatter
        kv_cache.stride(0) if has_bt else 0,
        kv_cache.stride(1) if has_bt else 0,
        cache_scale_arg,
        cache_scale_block_stride_arg,
        block_tables if has_bt else state_slot_mapping,  # placeholder
        bt_seq_stride,
        k_per_block,
        head_dim,
        rope_head_dim,
        BLOCK_D=BLOCK_D,
        HALF_ROPE=HALF_ROPE,
        OVERLAP=int(overlap),
        RATIO=ratio,
        STATE_SIZE=state_size,
        K=K,
        HAS_BLOCK_TABLE=int(has_bt),
        QUANT=int(quant),
        FP8_MAX=fp8_max_arg,
        USE_UE8M0=int(use_ue8m0),
        PRESHUFFLE=int(preshuffle),
    )


def fused_compress_attn_reference(
    *,
    kv_in: torch.Tensor,
    score_in: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    plan: CompressPlan,
    state_slot_mapping: torch.Tensor,
    ape: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    block_tables: Optional[torch.Tensor],
    k_per_block: int,
    overlap: bool,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Optional[torch.Tensor]:
    """Pure-PyTorch reference equivalent of `fused_compress_attn` (plan path).

    Returns `[num_compress, head_dim]` BF16 in plan order. None if num_compress=0.
    """
    if plan.num_compress == 0:
        return None
    device = kv_in.device
    K = (2 if overlap else 1) * ratio  # pool window
    state_size = kv_state.shape[1]  # ring buffer modulo (≥ K)
    plan_cpu = plan.compress_plan_gpu.detach().cpu()
    slot_map_cpu = state_slot_mapping.detach().cpu()
    if block_tables is not None:
        bt_cpu = block_tables.detach().cpu()
    else:
        bt_cpu = None

    out = torch.empty(plan.num_compress, head_dim, dtype=out_dtype, device=device)

    for pid in range(plan.num_compress):
        ragged_id, batch_id, position, window_len = plan_cpu[pid].tolist()
        slot = int(slot_map_cpu[batch_id].item())

        kv_rows = []
        score_rows = []
        for k in range(K):
            s = position - K + 1 + k
            if overlap:
                col_off = head_dim if k >= ratio else 0
            else:
                col_off = 0
            ape_row = k % ratio
            d_slice = slice(col_off, col_off + head_dim)
            is_padding = s < 0
            is_input = k >= window_len

            if is_padding:
                kv_rows.append(
                    torch.zeros(head_dim, dtype=torch.float32, device=device)
                )
                score_rows.append(
                    torch.full(
                        (head_dim,), float("-inf"), dtype=torch.float32, device=device
                    )
                )
            elif is_input:
                in_row = ragged_id - (K - 1 - k)
                kv_rows.append(kv_in[in_row, d_slice].float())
                score_rows.append(
                    score_in[in_row, d_slice].float() + ape[ape_row, d_slice].float()
                )
            else:
                ring = s % state_size
                kv_rows.append(kv_state[slot, ring, d_slice].float())
                score_rows.append(score_state[slot, ring, d_slice].float())

        kv_stack = torch.stack(kv_rows, dim=0)  # [K, head_dim]
        sc_stack = torch.stack(score_rows, dim=0)  # [K, head_dim]
        weights = torch.softmax(sc_stack, dim=0)
        compressed = (weights * kv_stack).sum(dim=0)  # [head_dim] fp32

        var = (compressed * compressed).mean()
        normed = compressed * torch.rsqrt(var + rms_eps) * rms_weight.float()

        comp_pos = (position // ratio) * ratio
        rope_seg = normed[-rope_head_dim:].clone()
        cos_v = cos_cache[comp_pos].view(-1).float()
        sin_v = sin_cache[comp_pos].view(-1).float()
        even = rope_seg[0::2]
        odd = rope_seg[1::2]
        new_even = even * cos_v - odd * sin_v
        new_odd = odd * cos_v + even * sin_v
        rotated_seg = torch.stack([new_even, new_odd], dim=-1).flatten()
        normed[-rope_head_dim:] = rotated_seg

        out_bf16 = normed.to(out_dtype)
        out[pid] = out_bf16

        if bt_cpu is not None and kv_cache is not None:
            ci = position // ratio
            block_in_seq = ci // k_per_block
            slot_in_block = ci % k_per_block
            physical = int(bt_cpu[batch_id, block_in_seq].item())
            kv_cache[physical, slot_in_block] = out_bf16

    return out
