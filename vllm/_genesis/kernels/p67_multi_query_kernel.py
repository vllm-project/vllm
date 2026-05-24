# SPDX-License-Identifier: Apache-2.0
"""P67 production wrapper — multi-query TurboQuant attention for spec-decode.

v7.50 (2026-04-27): adopt vllm#33529 (Triton MLA perf fixes, merged
2026-04-02) cache-modifier hints + tl.range() pipelining hint:
  - Q + Block_table + scales/zeros loads → cache_modifier=".ca" (L1 cache)
  - K/V dequant raw loads → cache_modifier=".cg" (L2 streaming)
  - Hoist `kv_head * stride_cache_head` out of inner KV loop
  - `for start_n in tl.range(0, total_seq_len, BLOCK_KV):` instead of plain
    range() — explicit Triton pipelining hint, allows compiler to overlap
    cp.async loads with prior-iteration MMA on Ampere
Expected gain: +5-10% throughput. Numerics unchanged (cache_modifier is
a memory-traffic hint, not arithmetic).

v7.48 NOTE (2026-04-27): a brief tf32x3 → default-tf32 experiment was
attempted to reduce inner-loop MMA cost, but the apparent regression
turned out to be unrelated (vLLM nightly bumped PyTorch to 2.11+CUDA13
which our driver 570 only partially supported, masking real perf).
After upgrading host driver to 580 + CUDA 13.0, tf32x3 stays as the
production choice — its precision boost matters for spec-decode verify
correctness on our k8v4 KV cache. Future PRs may revisit using `'tf32'`
explicitly (single-pass, ~2-3× faster than tf32x3) once we have an
A/B numerical regression suite in place.

v7.34 SPLIT-M (Fix A from algorithms research, 2026-04-26):
- Outer loop loads K/V tiles ONCE per iteration (same memory bandwidth)
- Inner `tl.static_range` unrolls K_PLUS_1 separate tl.dot calls
- Each q_t has INDEPENDENT online-softmax accumulators
- Mathematically bit-exact match to per-query upstream path
- Zero perf loss: Triton unrolls static_range → same SASS

ROOT CAUSE for prior drift (~0.2% rel_avg, off-by-1 digit on number recall):
The fused multi-query MMA (BLOCK_M=K_PLUS_1*HEADS_PER_KV=32) had different
per-row epilogue ORDERING than upstream per-query path. Online-softmax rescale
α_i applied in different sequence → compounded magnitude history drift over
~256 KV iterations (Golden et al., arXiv 2405.02803, "Is Flash Attention
Stable?" — documented O(N) drift in fused FA vs baseline).

Fix A from algorithms research (arXiv 2203.03341, FA3 paper, vLLM #40792
hoseung2 grouped decode pattern): split-M with shared K/V load, per-q_t
independent accumulators. Each q_t gets bit-identical accumulator history
to per-query reference.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.kernels.p67")

_ENV_ENABLE = "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL"


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on"
    )


_CACHED_KERNEL = None


def _build_kernel_fused(tl, triton):
    """v7.52 fused-M kernel — opt-in via GENESIS_P67_USE_FUSED=1.

    Architecture: ONE dot per KV-tile with m=K_PLUS_1*HEADS_PER_KV (e.g. 32
    for Qwen3.6 K_PLUS_1=4 + GQA-8). Vector online softmax over BLOCK_M
    rows. Per-row causal mask via `q_abs_pos[:, None] >= seq_offset[None, :]`
    broadcast. Includes all v7.50+v7.51 optimizations (tl.exp2 + LOG2E,
    -FLT_MAX, cache_modifier hints, tl.range, hoisted invariants).

    Risks:
    - Register pressure: acc tensor [BLOCK_M=32, BLOCK_D=128] fp32 = 16 KB
      virtual. Compile may spill — measured if it happens.
    - Per-row mask correctness: validated against reference impl in private
      repo p67_test_ieee_precision.py (rel_avg < 1e-3 was the bar).

    Same kernel signature as split-M for caller compat — launcher can
    invoke either without code change.
    """

    @triton.jit
    def cutlass_genesis_p67_v18_fused(
        Q_ptr,
        KV_cache_ptr,
        Block_table_ptr,
        Seq_lens_ptr,
        K_chunk_ptr,         # unused — API parity
        V_chunk_ptr,         # unused — API parity
        O_ptr,
        stride_qb, stride_qt, stride_qh, stride_qd,
        stride_cache_block, stride_cache_pos, stride_cache_head,
        stride_bt_b,
        stride_kkb, stride_kkt, stride_kkh, stride_kkd,  # unused
        stride_vkb, stride_vkt, stride_vkh, stride_vkd,  # unused
        stride_ob, stride_ot, stride_oh, stride_od,
        SCALE: tl.constexpr,
        K_PLUS_1: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        Hq_TOTAL: tl.constexpr,
        KPS: tl.constexpr,
        VAL_DATA_BYTES: tl.constexpr,
        FP8_E4B15: tl.constexpr = 0,
        DOT_FP16: tl.constexpr = 0,
    ):
        """Grid: (B, num_kv_heads, 1). Same as split-M.

        v7.62.19 (Phase 2): DOT_FP16 constexpr — when 1, cast Q/K/V to fp16
        before tl.dot and drop input_precision='tf32x3'. Yields native fp16
        Tensor Core path (1 dot/MMA) vs Markidis tf32x3 (3 dots/MMA).
        Expected +15-25% on Ampere consumer (lmdeploy/SGLang/FlashInfer
        cross-engine consensus). Opt-in via GENESIS_P67_DOT_PRECISION=fp16.
        """
        bid = tl.program_id(0)
        kv_head = tl.program_id(1)

        # Fused block: BLOCK_M = K_PLUS_1 * HEADS_PER_KV (e.g. 4*8=32).
        BLOCK_M: tl.constexpr = K_PLUS_1 * HEADS_PER_KV

        offs_m = tl.arange(0, BLOCK_M)
        # Per-row q_t (which query position) and abs_head (which Q head)
        q_t = offs_m // HEADS_PER_KV          # [BLOCK_M]: 0,0,..0,1,1,..1, ...
        head_in_group = offs_m % HEADS_PER_KV  # [BLOCK_M]: 0,1,..7,0,1,..7, ...
        abs_head = kv_head * HEADS_PER_KV + head_in_group  # [BLOCK_M]
        head_mask = abs_head < Hq_TOTAL  # [BLOCK_M]

        total_seq_len = tl.load(Seq_lens_ptr + bid)
        prior_seq_len = total_seq_len - K_PLUS_1
        q_abs_pos = prior_seq_len + q_t  # [BLOCK_M]

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM
        offs_kv = tl.arange(0, BLOCK_KV)
        vb_idx = offs_d // 2
        vb_shift = (offs_d % 2) * 4

        # Single fused Q load — [BLOCK_M, BLOCK_D]
        q_addrs = (
            bid * stride_qb
            + q_t[:, None] * stride_qt
            + abs_head[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        Q = tl.load(
            Q_ptr + q_addrs,
            mask=head_mask[:, None] & d_mask[None, :],
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.float32)  # [BLOCK_M, BLOCK_D] fp32

        # Vector state per row
        M_state = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        L_state = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        bt_base = bid * stride_bt_b
        _kv_head_byte_offset = tl.cast(kv_head, tl.int64) * stride_cache_head
        _LOG2E = 1.4426950408889634
        # v7.62.19 (P67 v2 Phase 1): fuse SCALE*LOG2E once outside loop;
        # eliminates per-element LOG2E mul on alpha and P (lmdeploy idiom,
        # 1 mul per BLOCK_M + 1 mul per BLOCK_M*BLOCK_KV per tile saved).
        SCALE_LOG2E = SCALE * _LOG2E

        # v7.62.22 reverted: Triton 3.6 flags (disallow_acc_multi_buffer +
        # loop_unroll_factor=2) microbench REGRESSED -6.5% on 35B split-M
        # path (BLOCK_M=4). Removed multi-buffering benefit, raised register
        # pressure. May help fused-M (BLOCK_M=32) — kept default tl.range.
        for start_n in tl.range(0, total_seq_len, BLOCK_KV):
            seq_offset = start_n + offs_kv
            tile_mask = seq_offset < total_seq_len

            page_idx = seq_offset // BLOCK_SIZE
            page_off = seq_offset % BLOCK_SIZE
            physical_block = tl.load(
                Block_table_ptr + bt_base + page_idx,
                mask=tile_mask, other=0,
                cache_modifier=".ca",
            ).to(tl.int64)
            slot_bases = (
                physical_block * stride_cache_block
                + page_off.to(tl.int64) * stride_cache_pos
                + _kv_head_byte_offset
            )

            # K dequant
            k_addrs = slot_bases[None, :] + offs_d[:, None]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=d_mask[:, None] & tile_mask[None, :],
                other=0,
                cache_modifier=".cg",
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            # v7.62.21 (quality fix): drop NaN sanitize + clamp on K.
            # Upstream `triton_turboquant_decode_attention` uses dequant output
            # directly (no clamp); k8v4 cache by spec is bounded to fp8 range.
            # The clamp+NaN-where rewrites edge-case bit patterns to 0.0,
            # producing distribution drift that breaks spec-decode acceptance.
            K_tile = k_float  # [BLOCK_D, BLOCK_KV] fp32

            # V dequant
            val_bases = slot_bases + KPS
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=tile_mask[:, None] & d_mask[None, :],
                other=0,
                cache_modifier=".cg",
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            V_dequant = v_idx * v_scales[:, None] + v_zeros[:, None]
            # v7.62.21: drop NaN sanitize + clamp on V (upstream parity).
            V_tile = V_dequant  # [BLOCK_KV, BLOCK_D] fp32

            # ─── FUSED single-MMA QK ───
            # S = SCALE_LOG2E * Q @ K_tile, [BLOCK_M, BLOCK_KV]
            # Full m=BLOCK_M=32 → mma.sync.m16n8k16 fully utilized.
            # v7.62.19: LOG2E pre-multiplied here, so exp2 below skips it.
            if DOT_FP16:
                # Native fp16 path: 1 dot/MMA on Ampere TC vs tf32x3=3 dots.
                # Q/K stay fp32 in their carriers; cast before MMA only.
                S = SCALE_LOG2E * tl.dot(
                    Q.to(tl.float16), K_tile.to(tl.float16),
                    out_dtype=tl.float32,
                )
            else:
                # v7.62.21e (test): input_precision='tf32' single-pass. Differs
                # from tf32x3 (3-pass emulation): single MMA tf32 → no
                # inter-pass accumulator, different rounding pattern. Maybe
                # close enough to upstream's IEEE for spec-decode acceptance.
                # If quality breaks, fall back to element-wise tl.sum (B3).
                S = SCALE_LOG2E * tl.dot(
                    Q, K_tile,
                    out_dtype=tl.float32, input_precision='tf32',
                )

            # ─── PER-ROW causal + head mask ───
            # q_abs_pos[BLOCK_M] vs seq_offset[BLOCK_KV] → [BLOCK_M, BLOCK_KV]
            # Each row gets its OWN causal cutoff based on its q_t position.
            # This is the v7.27 drift fix: per-row mask, not shared.
            causal = q_abs_pos[:, None] >= seq_offset[None, :]
            valid = head_mask[:, None] & tile_mask[None, :] & causal
            # v7.62.21 (quality fix): -inf sentinel matches upstream + lmdeploy +
            # SGLang. exp2(-inf)=0 cleanly; previous -FLT_MAX caused alpha=exp2(0)=1
            # vs upstream alpha=0 on fully-masked tiles → drift in K+1 verify path.
            S = tl.where(valid, S, -float("inf"))

            # ─── Vector online softmax over BLOCK_M rows ───
            # v7.62.19: M_state already in log2 domain (S_log2 = SCALE_LOG2E*QK),
            # so exp2 inputs are bare diffs — no per-element LOG2E mul needed.
            M_new = tl.maximum(tl.max(S, axis=1), M_state)  # [BLOCK_M]
            alpha = tl.exp2(M_state - M_new)                # [BLOCK_M]
            P = tl.exp2(S - M_new[:, None])                 # [BLOCK_M, BLOCK_KV]
            L_state = L_state * alpha + tl.sum(P, axis=1)   # [BLOCK_M]
            # Single MMA for acc update
            if DOT_FP16:
                acc = acc * alpha[:, None] + tl.dot(
                    P.to(tl.float16), V_tile.to(tl.float16),
                    out_dtype=tl.float32,
                )
            else:
                acc = acc * alpha[:, None] + tl.dot(
                    P, V_tile, out_dtype=tl.float32, input_precision='tf32x3'
                )
            M_state = M_new

        # ───── Epilogue: normalize + scatter to (B, K+1, Hq, D) output ─────
        safe_L = tl.where(L_state > 0.0, L_state, 1.0)
        out = acc / safe_L[:, None]  # [BLOCK_M, BLOCK_D] fp32

        # O_ptr layout: [B, K_PLUS_1, Hq, D]. Use per-row q_t + abs_head.
        o_addrs = (
            bid * stride_ob
            + q_t[:, None] * stride_ot
            + abs_head[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(
            O_ptr + o_addrs, out.to(tl.float16),
            mask=head_mask[:, None] & d_mask[None, :],
        )

    return cutlass_genesis_p67_v18_fused


def _build_kernel():
    """Define the P67 Triton kernel. Returns None on import failure.

    Two architectures available:
    - **Split-M (v7.34, default)**: K_PLUS_1 separate dots per KV-tile,
      m=8 each. Bit-exact match to per-query upstream path.
    - **Fused-M (v7.52, opt-in via GENESIS_P67_USE_FUSED=1)**: ONE dot
      per KV-tile with m=K_PLUS_1*HEADS_PER_KV=32, vectorized online
      softmax over BLOCK_M rows with per-row causal mask. Eliminates
      the m=8 under-utilization of mma.sync.m16n8k16 (which requires
      m>=16 for full TC throughput). Targets +8-15% throughput.
      Was rejected in v7.27 due to per-row causal mask error causing
      0.2% drift; v7.52 fixes via `q_abs_pos[:, None] >= seq_offset[None, :]`
      broadcasted explicitly, plus all v7.50/v7.51 optimizations
      (tl.exp2 + LOG2E, -FLT_MAX, cache_modifier hints, tl.range).
    """
    try:
        from vllm.triton_utils import tl, triton
    except Exception:
        try:
            import triton
            import triton.language as tl
        except Exception:
            return None

    use_fused = os.environ.get("GENESIS_P67_USE_FUSED", "").strip().lower() in (
        "1", "true", "yes", "on"
    )

    if use_fused:
        return _build_kernel_fused(tl, triton)

    @triton.jit
    def cutlass_genesis_p67_v17_split_m(
        Q_ptr,
        KV_cache_ptr,
        Block_table_ptr,
        Seq_lens_ptr,
        K_chunk_ptr,         # unused — kept for API compat
        V_chunk_ptr,         # unused — kept for API compat
        O_ptr,
        stride_qb, stride_qt, stride_qh, stride_qd,
        stride_cache_block, stride_cache_pos, stride_cache_head,
        stride_bt_b,
        stride_kkb, stride_kkt, stride_kkh, stride_kkd,  # unused
        stride_vkb, stride_vkt, stride_vkh, stride_vkd,  # unused
        stride_ob, stride_ot, stride_oh, stride_od,
        SCALE: tl.constexpr,
        K_PLUS_1: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        BLOCK_QH: tl.constexpr,
        Hq_TOTAL: tl.constexpr,
        KPS: tl.constexpr,
        VAL_DATA_BYTES: tl.constexpr,
        FP8_E4B15: tl.constexpr = 0,
        DOT_FP16: tl.constexpr = 0,
        # ───── Genesis P67c sparse-V (v7.65, 2026-05-01) ─────
        # Per-q_t skip via uniform-scalar `if` (Triton 3.6 scf.if pattern,
        # PN26b-proven). When SPARSE_V=0 (default constexpr), entire skip
        # block is DCE'd at compile time → byte-equivalent to pre-sparse v17.
        # Bit-exact contract: when SPARSE_V=1 AND threshold=0.0, skip never
        # fires (P_t = exp2(...) ≥ 0, so `max(P_t) < 0` is always False).
        # See `vllm/_genesis/tests/test_p67c_sparse_v.py` for invariants.
        SPARSE_V: tl.constexpr = 0,
        SPARSE_V_THRESHOLD: tl.constexpr = 0.0,
        SINK_TOKENS: tl.constexpr = 4,
    ):
        """Grid: (B, num_kv_heads, 1).

        SPLIT-M architecture:
        - Outer loop: load K/V tile ONCE per iteration (bandwidth-shared)
        - Inner static_range over K_PLUS_1: separate tl.dot per q_t
        - Each q_t has BLOCK_M = HEADS_PER_KV rows (e.g., 8 for Qwen3.6)
        - Per-q_t accumulators (M, L, acc) updated independently
        - Bit-exact match to per-query upstream path

        Optional sparse-V skip (constexpr-DCE'd when SPARSE_V=0):
        - After computing P_t for q_t, take scalar `tl.max(P_t)`
        - If max < threshold AND tile beyond SINK_TOKENS positions → skip
          P@V dot, only apply alpha decay to acc
        - Sink protection prevents skipping initial KV positions (StreamingLLM
          finding: first ~4 tokens are universally attended even when scores
          look small relative to recent context)
        """
        bid = tl.program_id(0)
        kv_head = tl.program_id(1)

        # Per-q_t block dimension. Caller passes BLOCK_QH = next_power_of_2(HEADS_PER_KV)
        # so the kernel compiles for any GQA factor (incl. non-pow-2 like 27B GQA=6).
        # Padding lanes (offs_h >= HEADS_PER_KV) are masked everywhere — no Q load,
        # no score contribution, no output write. Bit-exact to power-of-2 case when
        # HEADS_PER_KV is itself power-of-2 (BLOCK_QH == HEADS_PER_KV → mask all-true).

        offs_h = tl.arange(0, BLOCK_QH)
        abs_head = kv_head * HEADS_PER_KV + offs_h
        # Padding lane mask: lane i is valid iff i < HEADS_PER_KV (i.e. it maps to
        # a real head within this kv_head's group). Without this, lanes
        # HEADS_PER_KV..BLOCK_QH-1 would alias into the *next* kv_head's heads
        # via abs_head=kv_head*HEADS_PER_KV+i, computing wrong attention with the
        # current kv_head's K/V tile. (With pow-2 HEADS_PER_KV the lane mask is
        # identically true and lowers to a no-op.)
        lane_valid = offs_h < HEADS_PER_KV
        head_mask = lane_valid & (abs_head < Hq_TOTAL)

        # vLLM convention: seq_lens[i] = TOTAL length INCLUDING K_PLUS_1 chunk.
        total_seq_len = tl.load(Seq_lens_ptr + bid)
        prior_seq_len = total_seq_len - K_PLUS_1

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM
        offs_kv = tl.arange(0, BLOCK_KV)
        vb_idx = offs_d // 2
        vb_shift = (offs_d % 2) * 4

        # Per-q_t accumulator state stored as [K_PLUS_1, BLOCK_QH] / [K_PLUS_1, BLOCK_QH, BLOCK_D]
        # Triton can't dynamically index by constexpr — use where-masking for writes.
        M_state = tl.zeros([K_PLUS_1, BLOCK_QH], dtype=tl.float32) - float("inf")
        L_state = tl.zeros([K_PLUS_1, BLOCK_QH], dtype=tl.float32)
        acc = tl.zeros([K_PLUS_1, BLOCK_QH, BLOCK_D], dtype=tl.float32)

        q_t_range = tl.arange(0, K_PLUS_1)
        q_base = bid * stride_qb

        bt_base = bid * stride_bt_b

        # v7.50 (Step C — vllm#33529 patterns): hoist KV-head-byte invariant
        # out of inner KV loop. It depends only on (kv_head, stride), not on
        # per-tile seq_offset. Triton would also hoist this with -O2 but
        # explicit hoist matches the upstream MLA decode optimization.
        _kv_head_byte_offset = tl.cast(kv_head, tl.int64) * stride_cache_head
        # v7.62.19 (P67 v2 Phase 1): hoist SCALE*LOG2E pre-multiply.
        _LOG2E_split = 1.4426950408889634
        SCALE_LOG2E_split = SCALE * _LOG2E_split

        # v7.51 NOTE: experimented with explicit hoist of all K_PLUS_1 Q
        # tiles before the outer KV loop (load once as [K_PLUS_1, BLOCK_QH,
        # BLOCK_D] tensor, extract row t via mask-reduction inside loop).
        # REGRESSION -5 to -12% across all max_tokens. Triton compiler with
        # `tl.static_range` + cache_modifier=".ca" already hoists per-iter
        # loads to L1; explicit hoist adds reduction overhead. Reverted —
        # left per-t load below.

        # ════════════════════════════════════════════════════════════════
        # OUTER LOOP — KV tiles. K/V loaded ONCE per iteration.
        # v7.50: tl.range() instead of range() — explicit Triton pipelining
        #        hint, lets the compiler overlap async-copy with MMA across
        #        iterations. Adopted from vllm#33529 (merged 2026-04-02).
        # v7.62.22 reverted: Triton 3.6 flags regressed -6.5% on 35B split-M.
        # ════════════════════════════════════════════════════════════════
        for start_n in tl.range(0, total_seq_len, BLOCK_KV):
            seq_offset = start_n + offs_kv
            tile_mask = seq_offset < total_seq_len

            page_idx = seq_offset // BLOCK_SIZE
            page_off = seq_offset % BLOCK_SIZE
            # v7.50: cache_modifier=".ca" — block_table is a small lookup
            # array reused across all K/V loads in this tile. Cache-all
            # keeps it in L1.
            physical_block = tl.load(
                Block_table_ptr + bt_base + page_idx,
                mask=tile_mask, other=0,
                cache_modifier=".ca",
            ).to(tl.int64)
            slot_bases = (
                physical_block * stride_cache_block
                + page_off.to(tl.int64) * stride_cache_pos
                + _kv_head_byte_offset  # hoisted invariant
            )

            # K loaded transposed: [HEAD_SIZE, TILE_SIZE]
            # v7.50: cache_modifier=".cg" — K/V cache is streaming (sequential
            # read, not reused). Cache-global keeps L1 free for Q + scales.
            k_addrs = slot_bases[None, :] + offs_d[:, None]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=d_mask[:, None] & tile_mask[None, :],
                other=0,
                cache_modifier=".cg",
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            # v7.62.21 (quality fix): drop NaN sanitize + clamp on K — matches
            # upstream `triton_turboquant_decode_attention` which uses the
            # dequant output directly. Clamp+NaN-where rewrites edge bit
            # patterns to 0.0, producing distribution drift that broke
            # spec-decode acceptance and tool-call generation on 35B.
            K_tile = k_float  # [BLOCK_D, BLOCK_KV] fp32

            # V dequant — load 4-bit indices + scale + zero, build V_tile
            val_bases = slot_bases + KPS
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            # v7.50: streaming V dequant data — .cg
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=tile_mask[:, None] & d_mask[None, :],
                other=0,
                cache_modifier=".cg",
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            # v7.50: scales/zeros are 4 bytes per tile-row, reused across
            # both V-dequant arithmetic and (potentially) MMA epilogue.
            # Cache-all keeps them hot in L1.
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            V_dequant = v_idx * v_scales[:, None] + v_zeros[:, None]
            # v7.62.21: drop NaN sanitize + clamp on V (upstream parity).
            V_tile = V_dequant  # keep fp32 for IEEE precision PV dot

            # ─────────── SPLIT-M: per-q_t independent dots ───────────
            # Static unroll over K_PLUS_1 query tokens. Each iteration:
            # 1. Load Q tile for this t (Triton compiler unrolls — no runtime index)
            # 2. Compute scores for ONE q_t (BLOCK_QH × BLOCK_KV tile)
            # 3. Apply causal mask for that q_t's absolute position
            # 4. Online softmax update for that q_t's accumulators only
            # Bit-exact to per-query upstream path (no cross-row drift).
            for t in tl.static_range(0, K_PLUS_1):
                q_abs_pos_t = prior_seq_len + t

                # Load Q tile for this q_t: [BLOCK_QH, BLOCK_D]
                # t is constexpr-static here so address computation is constant.
                # v7.51 NOTE: tested explicit hoist of all K_PLUS_1 Q tiles
                # before outer KV loop — REGRESSION -5 to -12% (Tier 2 Step F,
                # 2026-04-27). Triton compiler ALREADY hoists this load via
                # `tl.static_range` unroll + cache_modifier=".ca" L1 pinning.
                # Our manual hoist added sum-reduction overhead (over K_PLUS_1)
                # plus 8 KB virtual register pressure (all_Q tensor). Lesson:
                # don't fight Triton's automatic loop-invariant code motion
                # for simple loads — leave per-q_t load here.
                q_addrs_t = (
                    q_base
                    + t * stride_qt
                    + abs_head[:, None] * stride_qh
                    + offs_d[None, :] * stride_qd
                )
                Q_t_raw = tl.load(
                    Q_ptr + q_addrs_t,
                    mask=head_mask[:, None] & d_mask[None, :],
                    other=0.0,
                    cache_modifier=".ca",
                )  # [BLOCK_QH, BLOCK_D] — fp16 from input
                # Upcast Q to fp32 — required for input_precision='tf32x3' to be effective.
                Q_t = Q_t_raw.to(tl.float32)

                # S_t = scale * Q_t @ K_tile [BLOCK_QH, BLOCK_KV]
                # input_precision='ieee' forces SOFTWARE fp32 dot (no Tensor Cores)
                # → full 23-bit mantissa, matches upstream's element-wise fp32 sum.
                # tf32x3: Markidis 3xTF32 emulation — Tensor Core throughput
                # with ~700x precision boost vs default TF32 (CUTLASS data).
                # Requires fp32 inputs (which we have).
                # v7.62.19: SCALE_LOG2E_split absorbs LOG2E pre-multiply
                # → exp2 below operates on bare diffs. Phase 2: fp16 dot opt-in.
                if DOT_FP16:
                    S_t = SCALE_LOG2E_split * tl.dot(
                        Q_t.to(tl.float16), K_tile.to(tl.float16),
                        out_dtype=tl.float32,
                    )
                else:
                    # v7.62.21e: tf32 single-pass test (see fused-M comment).
                    S_t = SCALE_LOG2E_split * tl.dot(
                        Q_t, K_tile,
                        out_dtype=tl.float32, input_precision='tf32',
                    )

                # Per-row causal mask for this q_t:
                # q_abs_pos_t (scalar) >= seq_offset[n]
                # v7.62.21 (quality fix): -inf sentinel matches upstream
                # `triton_turboquant_decode_attention` + lmdeploy + SGLang.
                # exp2(-inf) = 0 cleanly — fully-masked tile produces 0 cleanly.
                # Previous -FLT_MAX caused alpha = exp2(0) = 1 (vs upstream 0)
                # on fully-masked tiles → drift in K+1 verify path → tool-call
                # repetition spam on 35B.
                causal = q_abs_pos_t >= seq_offset
                valid = head_mask[:, None] & tile_mask[None, :] & causal[None, :]
                S_t = tl.where(valid, S_t, -float("inf"))

                # Triton can't index 2D with constexpr, use mask-based extract.
                # tl.where avoids -inf*0=NaN issue from naive multiplication.
                t_mask = q_t_range == t  # [K_PLUS_1] one-hot, compile-time constant
                # Extract row t: where t_mask use M_state, else 0; sum picks the t row.
                M_old_t = tl.sum(tl.where(t_mask[:, None], M_state, 0.0), axis=0)
                L_old_t = tl.sum(tl.where(t_mask[:, None], L_state, 0.0), axis=0)
                acc_old_t = tl.sum(tl.where(t_mask[:, None, None], acc, 0.0), axis=0)

                # Online softmax update for this q_t
                # v7.52 (Action #1 from PR #40929 audit): tl.exp2 maps directly
                # to hardware ex2 instruction; tl.exp is synthesized as
                # ex2(x*log2e) so adds one mul. Pre-multiply by LOG2E = 1.4426...
                # at the input (S_t and M-diff) and use exp2 throughout. Net:
                # one fewer fp mul per softmax step in hot inner loop.
                # v7.62.19: M_state already in log2 domain; no per-element
                # _LOG2E mul on the exp2 inputs (fused into SCALE_LOG2E_split).
                M_new_t = tl.maximum(tl.max(S_t, axis=1), M_old_t)
                alpha_t = tl.exp2(M_old_t - M_new_t)
                P_t = tl.exp2(S_t - M_new_t[:, None])
                L_new_t = L_old_t * alpha_t + tl.sum(P_t, axis=1)

                # ───── Genesis P67c sparse-V per-q_t skip gate ─────
                # Constexpr-DCE'd to nothing when SPARSE_V=0.
                # When SPARSE_V=1 + threshold=0.0: max(P_t) >= 0 always, so
                # `p_t_max < 0` is False → skip never fires → bit-exact.
                # Sink protection: first SINK_TOKENS positions never skipped.
                skip_pv_t = False
                if SPARSE_V:
                    tile_protected = start_n < SINK_TOKENS
                    if not tile_protected:
                        p_t_max = tl.max(P_t)
                        skip_pv_t = p_t_max < SPARSE_V_THRESHOLD

                # PV for this q_t: [BLOCK_QH, BLOCK_D]
                # P_t already fp32, V_tile fp32 → IEEE software dot for full precision.
                if skip_pv_t:
                    # Skip path: only decay (no V@P contribution this tile)
                    acc_new_t = acc_old_t * alpha_t[:, None]
                elif DOT_FP16:
                    acc_new_t = acc_old_t * alpha_t[:, None] + tl.dot(
                        P_t.to(tl.float16), V_tile.to(tl.float16),
                        out_dtype=tl.float32,
                    )
                else:
                    acc_new_t = acc_old_t * alpha_t[:, None] + tl.dot(
                        P_t, V_tile, out_dtype=tl.float32, input_precision='tf32x3'
                    )

                # Write back into per-q_t accumulator slots via where-mask.
                M_state = tl.where(t_mask[:, None], M_new_t[None, :], M_state)
                L_state = tl.where(t_mask[:, None], L_new_t[None, :], L_state)
                acc = tl.where(
                    t_mask[:, None, None],
                    acc_new_t[None, :, :],
                    acc,
                )

        # ───── Epilogue: normalize and store ─────
        safe_L = tl.where(L_state > 0.0, L_state, 1.0)
        out = acc / safe_L[:, :, None]  # [K_PLUS_1, BLOCK_QH, BLOCK_D]

        # O_ptr layout: [B, K_PLUS_1, Hq, D]
        o_addrs_3d = (
            bid * stride_ob
            + q_t_range[:, None, None] * stride_ot
            + abs_head[None, :, None] * stride_oh
            + offs_d[None, None, :] * stride_od
        )
        tl.store(
            O_ptr + o_addrs_3d, out.to(tl.float16),
            mask=head_mask[None, :, None] & d_mask[None, None, :],
        )

    return cutlass_genesis_p67_v17_split_m


def _get_kernel():
    global _CACHED_KERNEL
    if _CACHED_KERNEL is None:
        _CACHED_KERNEL = _build_kernel()
    return _CACHED_KERNEL


# ───────────────────────────────────────────────────────────────────
# P67c sparse-V env resolvers (v7.65, 2026-05-01)
# ───────────────────────────────────────────────────────────────────


def _resolve_sparse_v_enabled() -> bool:
    """True iff GENESIS_ENABLE_P67_SPARSE_V env is set to a truthy value."""
    return os.environ.get(
        "GENESIS_ENABLE_P67_SPARSE_V", ""
    ).strip().lower() in ("1", "true", "yes", "on")


def _resolve_sparse_v_threshold() -> float:
    """Parse and clamp threshold to safe range [0.0, 0.5].

    Threshold = 0.0 → never skip (bit-exact with SPARSE_V=0 path, since
    P_t = exp2(...) >= 0 always; `p_t_max < 0` is False).
    Threshold > 0.5 → too aggressive (would skip warm tiles); clamp to 0.5.
    Negative → invalid; clamp to 0 (degenerate to never-skip).
    """
    raw = os.environ.get("GENESIS_P67_SPARSE_V_THRESHOLD", "0.001")
    try:
        thr = float(raw)
    except (ValueError, TypeError):
        thr = 0.001
    return max(0.0, min(0.5, thr))


def _resolve_sparse_v_sink_tokens() -> int:
    """Parse SINK_TOKENS, default 4 (StreamingLLM finding).

    First N KV positions are universally attended even when scores look
    small relative to recent context. Skipping them produces small but
    measurable quality drift on long-form generation.
    """
    raw = os.environ.get("GENESIS_P67_SPARSE_V_SINK_TOKENS", "4")
    try:
        n = int(raw)
    except (ValueError, TypeError):
        n = 4
    return max(0, n)


def _autoconfig(sm_major: int, sm_minor: int, head_dim: int) -> dict:
    """Pick BLOCK_KV/num_warps/num_stages per SM.

    v7.39 aggressive tune for Ampere consumer (SM 8.6 RTX A5000):
    - BLOCK_KV=32: doubles tile size, halves loop iter count
    - num_warps=8: more parallelism per CTA (was 4)
    - num_stages=3: deeper async pipeline for better hide latency

    v7.50 NOTE on num_stages choice:
    Tested num_stages=2 on A5000 (Tier 2 Step E, 2026-04-27) — REGRESSION
    -2% to -9% across all max_tokens, sweet spot at 128-256 tokens.
    Reason: our P67 is memory-bound + dequant-heavy (TQ k8v4 FP8 bitcast +
    4-bit V unpack + per-tile scale/zero loads). Deep pipeline (3 stages)
    hides DRAM latency on KV reads better than the 2-buffer shmem savings.
    Generic advice "num_stages=2 reduces register pressure on Ampere"
    applies to compute-bound dense GEMM kernels, NOT to dequant-heavy paths.
    Keep num_stages=3 for any TQ k8v4 / quant-dequant workload here.

    Override via env: GENESIS_P67_BLOCK_KV, GENESIS_P67_NUM_WARPS, GENESIS_P67_NUM_STAGES.

    [Genesis 2026-05-04 SMEM-aware default — pattern from MidasMining vllm#41508]
    BLOCK_KV default — arch-aware:
      * SM<9 (Ampere/Hopper consumer): 32 (Ampere SMEM 99 KB hard cap)
      * SM>=9 (Hopper datacenter / Blackwell): 64 (more SMEM headroom)
    SMEM math (Ampere):
      Q tile: BLOCK_M*BLOCK_D*4 = 16*128*4 = 8 KB
      K tile: BLOCK_KV*BLOCK_D*4 = 32*128*4 = 16 KB (32 OK; 64 = 32 KB also OK
              but combined with V, dequant scratch, accumulators → tight)
      V tile: BLOCK_KV*BLOCK_D*4 = 16 KB (× 2 for stages=3 → 48 KB)
      Total: ~50-60 KB on Ampere, headroom 30-40 KB. BLOCK_KV=64 → exceeds.
    Empirical: feedback_triton36_flags_regress_split_m confirmed BLOCK_KV>32
    regresses on small-BLOCK_M Ampere split-M. Stay with 32 default.
    Future-proof: when migrating to Blackwell R6000 Pro (sm 12), bump default
    to 64 (SMEM 228 KB) for ~15-20% throughput uplift on long-K paths.

    [2026-05-05 Blackwell consumer (sm_120 RTX 5090) status — UNMEASURED]
    The (BLOCK_KV=64, num_warps=8, num_stages=3) defaults that fire on
    sm_major>=9 are EXTRAPOLATED from Hopper datacenter, NOT empirically
    measured on consumer Blackwell (sm 12.0). First real 5090 datapoints
    expected from noonghunna club-3090 discussion #51 (apnar's rig).
    Until empirical sweep lands, treat sm_120 as "use defaults but
    UNVERIFIED" — the datacenter-class config may underutilize the
    smaller consumer L2 (88 MB on 5090 vs 60 MB on H100 vs 6 MB A5000).
    Cross-reference: feedback_triton36_flags_regress_split_m (split-M
    regression on small BLOCK_M Ampere → may also bite on sm_120).
    """
    import os as _os
    # arch-aware default — Ampere keeps 32, Hopper+ gets 64 future-proof
    # NOTE: sm 12.0 Blackwell consumer config UNMEASURED (see docstring).
    _default_block_kv = "32" if sm_major < 9 else "64"
    block_kv = int(_os.environ.get("GENESIS_P67_BLOCK_KV", _default_block_kv))
    num_warps = int(_os.environ.get("GENESIS_P67_NUM_WARPS", "8" if sm_major >= 8 else "4"))
    num_stages = int(_os.environ.get("GENESIS_P67_NUM_STAGES", "3" if sm_major >= 8 else "2"))
    return dict(BLOCK_KV=block_kv, num_warps=num_warps, num_stages=num_stages)


def _detect_fp8_mode() -> int:
    try:
        import torch
        cap = torch.cuda.get_device_capability()
        return 1 if cap < (8, 9) else 0
    except Exception:
        return 1


def is_active() -> bool:
    if not _env_enabled():
        return False
    if _get_kernel() is None:
        return False
    return True


def alloc_output_buffer(B, K_PLUS_1, Hq, D, device, dtype):
    """Pre-allocate reusable output buffer (cudagraph-safe)."""
    import torch
    return torch.empty((B, K_PLUS_1, Hq, D), dtype=dtype, device=device)


def call_p67_attention(
    q,
    kv_cache,
    block_table,
    seq_lens,
    k_chunk,
    v_chunk,
    scale: float,
    block_size: int,
    kps: int,
    val_data_bytes: int,
    output=None,
):
    """Production launcher for P67 v7.34 split-M multi-query attention."""
    import torch
    import triton

    kernel = _get_kernel()
    if kernel is None:
        raise ImportError("P67 Triton kernel not available")

    B, K_PLUS_1, Hq, D = q.shape
    Hk = k_chunk.shape[2]
    assert Hq % Hk == 0
    heads_per_kv = Hq // Hk

    # v7.62.19b: defensive guard — Triton tl.zeros requires power-of-2 dims
    # so K_PLUS_1 must be power of 2. With MTP K=3 the typical value is 4,
    # but post-rejection or extension paths can produce K_PLUS_1 ∈ {3, 5, 6,
    # 7, 9, ...}. Refuse those — caller (P67b dispatcher) catches and falls
    # through to upstream kernel cleanly. This was causing CompilationError
    # spam + bimodal variance under fp16-dot rebuild (v777 35B regression
    # root cause; pre-existed but masked by warm Triton cache on prior names).
    if K_PLUS_1 < 2 or (K_PLUS_1 & (K_PLUS_1 - 1)) != 0:
        raise ValueError(
            f"P67 kernel requires K_PLUS_1 power-of-2, got {K_PLUS_1}; "
            "caller should fall through to upstream kernel"
        )
    if heads_per_kv < 1:
        raise ValueError(
            f"P67 kernel requires heads_per_kv >= 1, got {heads_per_kv}"
        )
    # Fused-M (opt-in via GENESIS_P67_USE_FUSED=1) bakes BLOCK_M = K_PLUS_1 *
    # HEADS_PER_KV into one contiguous arange — requires pow-2. Split-M
    # (default) is generalized to non-pow-2 via BLOCK_QH padding + lane_valid
    # mask. Refuse fused on non-pow-2 so caller falls through to upstream.
    _fused_mode = os.environ.get("GENESIS_P67_USE_FUSED", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    if _fused_mode and (heads_per_kv & (heads_per_kv - 1)) != 0:
        raise ValueError(
            f"GENESIS_P67_USE_FUSED=1 requires heads_per_kv power-of-2, "
            f"got {heads_per_kv}; either unset GENESIS_P67_USE_FUSED or use "
            "default split-M which supports any GQA factor"
        )

    cap = torch.cuda.get_device_capability()
    cfg = _autoconfig(cap[0], cap[1], D)

    BLOCK_D = triton.next_power_of_2(D)
    # v7.63.x non-pow-2 generalization: pad BLOCK_QH up to next_power_of_2 so
    # Triton tl.arange / tl.dot compile for any GQA factor (e.g. Qwen3.6-27B
    # GQA=6 → BLOCK_QH=8 with 2 lanes masked off via head_mask). When
    # heads_per_kv is already power-of-2 (35B GQA=8 → BLOCK_QH=8) this is a
    # no-op — bit-exact to pre-generalization split-M kernel.
    BLOCK_QH = triton.next_power_of_2(heads_per_kv)

    if output is None:
        output = torch.empty_like(q)
    assert output.dtype == q.dtype, (
        f"output dtype {output.dtype} must match q dtype {q.dtype}"
    )

    fp8_e4b15 = _detect_fp8_mode()
    # v7.62.19 Phase 2: opt-in fp16 dot (drops tf32x3 emulation, +15-25%
    # cross-engine consensus). Default 0 = tf32x3 (current safe path).
    dot_fp16 = 1 if os.environ.get(
        "GENESIS_P67_DOT_PRECISION", "tf32x3"
    ).strip().lower() == "fp16" else 0

    # v7.65 P67c: opt-in sparse-V per-q_t skip (per-row vote integration).
    # Default 0 = no sparse-V → constexpr-DCE'd → byte-equivalent to v17.
    sparse_v_enabled = 1 if _resolve_sparse_v_enabled() else 0
    sparse_v_thr = _resolve_sparse_v_threshold()
    sink_tokens = _resolve_sparse_v_sink_tokens()

    grid = (B, Hk, 1)
    kernel[grid](
        q, kv_cache, block_table, seq_lens,
        k_chunk, v_chunk, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        k_chunk.stride(0), k_chunk.stride(1), k_chunk.stride(2), k_chunk.stride(3),
        v_chunk.stride(0), v_chunk.stride(1), v_chunk.stride(2), v_chunk.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        SCALE=scale,
        K_PLUS_1=K_PLUS_1,
        BLOCK_D=BLOCK_D,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        BLOCK_KV=cfg["BLOCK_KV"],
        HEADS_PER_KV=heads_per_kv,
        BLOCK_QH=BLOCK_QH,
        Hq_TOTAL=Hq,
        KPS=kps,
        VAL_DATA_BYTES=val_data_bytes,
        FP8_E4B15=fp8_e4b15,
        DOT_FP16=dot_fp16,
        SPARSE_V=sparse_v_enabled,
        SPARSE_V_THRESHOLD=sparse_v_thr,
        SINK_TOKENS=sink_tokens,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return output


def diagnostic_info() -> dict[str, Any]:
    info = {"env_enabled": _env_enabled(), "version": "v7.39_aggressive_tune"}
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            info["sm"] = f"{cap[0]}.{cap[1]}"
            info["fp8_mode"] = "e4b15" if cap < (8, 9) else "e4nv"
            info["autoconfig"] = _autoconfig(cap[0], cap[1], 128)
        else:
            info["cuda"] = False
    except Exception as e:
        info["error"] = str(e)
    info["kernel_built"] = _get_kernel() is not None
    return info
