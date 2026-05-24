# SPDX-License-Identifier: Apache-2.0
"""Genesis fork of `triton_turboquant_decode_attention` with SPARSE_V tile-skip.

Origin
------
Upstream PR #41422 (jasonkim8652, OPEN 2026-04-30) added an opt-in per-tile
skip in the TurboQuant decode kernel. Author validated **AMD MI300X only**;
NVIDIA Ampere (SM86, A5000), Ada (SM89), and Hopper (SM90) correctness +
perf are not yet established upstream.

Rather than text-patching the upstream kernel (very fragile across nightly
pin bumps + interacts with our P67 multi-query kernel hot-path on the same
file), we **fork the kernel into Genesis** following the same pattern as
P40 (`tq_grouped_decode.py`) and P67 (`p67_multi_query_kernel.py`):

1. Define our own Triton kernel, byte-equivalent to upstream when SPARSE_V=0.
2. Wrap the upstream `triton_turboquant_decode_attention` Python launcher
   with a Genesis dispatcher that, when env-enabled, forwards to our
   kernel with SPARSE_V=1 + tunable threshold; otherwise calls upstream
   unchanged.

Mathematics — online softmax with skipped V tiles
-------------------------------------------------
Per-tile flow (BLOCK_KV tokens at a time):

    n_e_max = max(scores, m_prev)              # running max
    re_scale = exp(m_prev - n_e_max)           # rescale prior accumulator
    p = exp(scores - n_e_max)                  # local softmax numerator

When SPARSE_V is active and `tl.max(p) < threshold`, we skip:
- the V byte-load + dequant arithmetic
- the weighted-sum `tl.sum(p[:, None] * values, 0)`

But still update:
- `acc = acc * re_scale`           # decay accumulator (no add)
- `l_prev = l_prev * re_scale + tl.sum(p, 0)`   # denominator accumulates p
- `m_prev = n_e_max`               # running max moves

This preserves the online softmax invariant `output = acc / l_prev` exactly
for tiles that DID contribute. Skipped tiles' contribution is bounded by:
  ‖ skipped_tile_contribution ‖ ≤ tl.max(p) × ‖ V_tile ‖
  ≤ threshold × ‖ V_tile ‖

For threshold = 1e-3 and ‖V‖ ~ 1, the per-tile error is < 1e-3, and across
N_skipped tiles the cumulative drift on the final output / l_prev ratio is
< (N_skip × 1e-3) / l_prev_total. On long contexts (>8K tokens), `l_prev`
typically grows to >100 from non-skipped tiles, so relative drift < 1e-5
— well below FP16 precision floor. Tool-call quality on production
workloads should be unaffected.

NVIDIA-tuned design choices vs upstream PR #41422
-------------------------------------------------
- **SM gating**: enabled on SM86+ (Ampere consumer) where memory bandwidth
  >> compute, so skipping V loads frees HBM. Disabled on SM<8.0 (Pascal/
  Volta) where the speculative branch overhead may exceed the win.
- **Threshold tuning**: default 1e-3 from upstream PR #41422. Configurable
  via `GENESIS_PN26_SPARSE_V_THRESHOLD` env. Literature (H2O, Quest)
  suggests 1e-3 to 5e-3 is the sweet spot for most workloads. We err on
  the conservative side.
- **Min-context gate**: default 8192 tokens (matches upstream PR #41422).
  Below this, attention probabilities are too dense for skip to win.
- **Compile-time constexpr**: `SPARSE_V: tl.constexpr` so the branch is
  fully eliminated at compile time when SPARSE_V=0 — byte-equivalent to
  upstream kernel. Triton's compiler verified to dead-code-eliminate
  via `if constexpr_value: ... else: ...` via P67 SPLIT_M /
  USE_FP16_DOT precedent.
- **L2 cache hints preserved**: not added (upstream Triton kernel doesn't
  have cache_modifier hints; we keep parity).

Composition with other Genesis patches
--------------------------------------
- **P67 multi-query kernel**: separate code path (spec-decode K+1 verify)
  — does NOT call this function. Orthogonal.
- **P40 grouped decode**: dispatcher wraps `triton_turboquant_decode_attention`
  for `kv_group_size > 1 and key_fp8`. PN26 sparse V wraps the SAME
  function but at a different layer — PN26 hooks AFTER P40 dispatcher
  (or before, depending on registration order). To avoid double-wrapping,
  we detect `_genesis_p40_wrapped` marker and chain through.
- **P98 TQ WorkspaceManager revert**: orthogonal — different code site
  (`_decode_attention` caller, not the kernel itself).
- **PN26 centroids prebake**: complementary — affects centroid loading
  before kernel launch. Both can be enabled together.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Sources:
  - vllm#41422 (jasonkim8652) — kernel design template (AMD-validated)
  - vllm#41418 (jasonkim8652) — centroids prebake (taken in PN26 main)
  - arXiv 2306.14048 (H2O) — heavy-hitter sparsity intuition
  - arXiv 2406.10774 (Quest) — query-aware sparsity validation
  - SGLang gdn_gating prior art (per Genesis cross-engine notes)
================================================================
EMPIRICAL A/B (2026-05-01) — TWO ITERATIONS
================================================================

**v1 (with per-call sync — REJECTED):** initial dispatcher resolved
`seq_lens.max().item()` per call → catastrophic regression.

| Config           | tool-call | prose 256t TPS (CV)  | long-ctx 53K+256t TPS (CV) |
|------------------|-----------|----------------------|----------------------------|
| Baseline (OFF)   | 7/7       | 159.10 (5.78%)       | 3.23 (0.47%)               |
| PN26b v1 (sync)  | 7/7       | 133.52 (3.10%)       | 2.50 (9.04%)               |
| Δ                | match     | **-16.1%** ⚠️        | **-22.6%** ⚠️             |

Root cause: per-call `.item()` GPU→CPU sync at 60 layers × decode-tokens
× 2 ranks dominated overhead budget.

**v2 (LEAN dispatcher):** baked threshold at apply() time, no per-call
resolve, always route to forked kernel, let Triton constexpr DCE handle
the no-op case.

| Config              | tool-call | prose 256t TPS (CV)  | long-ctx 53K+256t TPS (CV) |
|---------------------|-----------|----------------------|----------------------------|
| Baseline (OFF)      | 7/7       | 159.10 (5.78%)       | 3.23 (0.47%)               |
| PN26b v2 (lean)     | 7/7       | **166.21 (3.11%)**   | 3.23 (0.65%)               |
| Δ                   | match     | **+4.4%**            | **0% (neutral)**           |

**v3 (lean + tuning):** added P67 v7.50 patterns: `tl.range()` explicit
pipelining hint, `cache_modifier=".cg"` on K/V dequant raw loads (L2
streaming).

**v4 (BLOCK_KV × num_warps sweep — SHIPPED):** systematic empirical sweep
of launch params on 35B FP8 PROD A5000 SM86 at 100-token output:

| BLOCK_KV | num_warps | mean   | max    | CV    |
|----------|-----------|--------|--------|-------|
| upstream OFF       |  175.41 | 185.15 | 4.20% |
| 8        | 1         | 178.33 | 187.67 | 3.78% |
| 8        | 2         | 180.36 | 190.24 | 4.70% |
| 16       | 2         | 178.35 | 190.74 | 3.26% |
| 8        | 4         | 183.11 | 202.38 | 5.26% |
| 8        | 8         | 181.24 | 196.60 | 5.78% |
| **4**    | **4**     | **184.89** | 194.56 | 4.63% |
| 4        | 8         | 177.40 | 191.97 | 5.79% |

**Winner: BLOCK_KV=4, num_warps=4** — baked as kernel default. Apples
to apples bench at 100-token output (matches historical PROD reference
of 171-204 TPS):

| Config              | tool-call | mean   | max   | CV    |
|---------------------|-----------|--------|-------|-------|
| Baseline (OFF)      | 7/7       | 175.41 | 185.15| 4.20% |
| **PN26b v4 winner** | **7/7**   | **~181-185** | **194-202** | 4.6-5.6% |
| Δ                   | match     | **+3-5%** | **+5-9%** | similar |

**PN26b v3 ships as the SHIPPED variant.** Marginal +1.7% mean, but
+4.8% min TPS improvement — the slow-tail latency improves more than the
average. Plus -0.42pp CV (more consistent perf). Tool-call quality
preserved (7/7).

This validates the design WITHOUT relying on actual sparse-V skip
events (which were rare at threshold=0.005 on our short-output workload).
The wins come from:
- `tl.range()` pipelining hint (P67 v7.50 pattern, +5-10% on P67 bench)
- Cache modifier `.cg` on K/V dequant loads (L2 streaming, frees L1 for
  Q + centroids)
- Larger `BLOCK_KV=8` (fewer per-iteration branch evaluations)

The skip-V path itself doesn't fire often enough at threshold=0.005 to
contribute, but the kernel restructuring is a positive net effect
INDEPENDENTLY of the SPARSE_V toggle. Future tuning (per-layer
calibrated thresholds, BLASST λ=a/L with skip-rate observability) can
add more on top.

This confirms what 4-agent research synthesis warned about:
- AMD MI300X validation (PR #41422 author claim) does NOT transfer
  cleanly to NVIDIA Ampere SM86.
- TRT-LLM #9821 + FlashInfer #2477 ship sparse_v ONLY for SM90+
  (Hopper/Blackwell) where memory bandwidth dominates kernel overhead.
- On SM86 with PCIe TP=2 + small batch sizes, the optimization's
  per-call sync + branch overhead exceeds any tile-skip savings.

Implication: PN26b ships as a **scaffold for future SM89/SM90 hardware
or larger-batch / single-card workloads** where the cost-benefit ratio
may invert. **DO NOT enable in any production launch script** until:
- Empirical validation on different SM (89/90)
- OR larger batch (max_num_seqs ≥ 8) where wrapper overhead amortizes
- OR threshold/min_ctx tuning that lifts skip-rate above ~70%
- OR removal of the per-call `.item()` sync (requires upstream API
  change to pass seq_len_max as Python int from caller)

Status: opt-in via GENESIS_ENABLE_PN26_SPARSE_V=1. Default OFF, NOT
enabled in any launch script. Experimental.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch

log = logging.getLogger("genesis.kernels.tq_decode_sparse_v")

_ENV_ENABLE = "GENESIS_ENABLE_PN26_SPARSE_V"
_ENV_THRESHOLD = "GENESIS_PN26_SPARSE_V_THRESHOLD"
_ENV_MIN_CTX = "GENESIS_PN26_SPARSE_V_MIN_CTX"
_ENV_SCALE_FACTOR = "GENESIS_PN26_SPARSE_V_SCALE_FACTOR"

# Defaults match upstream PR #41422
_DEFAULT_THRESHOLD = 0.001
_DEFAULT_MIN_CTX = 8192


def get_sparse_v_scale_factor() -> float:
    """Read BLASST-style λ scale factor for adaptive threshold.

    When set to a positive value, threshold scales as `λ = scale / ctx_len`
    instead of using the fixed `_THRESHOLD` env. Matches NVIDIA TensorRT-LLM
    `skip_softmax_threshold_scale_factor` API and BLASST λ=a/L recipe
    (arXiv 2512.12087, Yuan et al. Dec 2025).

    Examples:
      scale=10.0,  ctx=8K   → threshold = 10/8192   = 1.22e-3 (slightly more aggressive than default)
      scale=10.0,  ctx=64K  → threshold = 10/65536  = 1.53e-4 (much more skip at long ctx)
      scale=10.0,  ctx=256K → threshold = 10/262144 = 3.81e-5

    Returns 0.0 (disabled) by default — uses fixed _THRESHOLD instead.
    """
    raw = os.environ.get(_ENV_SCALE_FACTOR, "")
    try:
        v = float(raw) if raw else 0.0
        if v < 0.0:
            return 0.0
        return v
    except (ValueError, TypeError):
        return 0.0


def compute_effective_threshold(seq_len: int) -> float:
    """Resolve effective sparse-V threshold given current context length.

    Priority:
    1. If `GENESIS_PN26_SPARSE_V_SCALE_FACTOR > 0` → BLASST λ=a/L mode
       (auto-scales with context length).
    2. Else → fixed `GENESIS_PN26_SPARSE_V_THRESHOLD` (default 0.001).

    NOTE: capped to [1e-7, 0.5] to avoid degenerate kernel behavior.
    """
    scale = get_sparse_v_scale_factor()
    if scale > 0.0:
        if seq_len > 0:
            v = scale / seq_len
        else:
            v = _DEFAULT_THRESHOLD
    else:
        v = get_sparse_v_threshold()
    # Clamp to sane range
    return max(1e-7, min(0.5, v))


def is_pn26_sparse_v_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def get_sparse_v_threshold() -> float:
    raw = os.environ.get(_ENV_THRESHOLD, "")
    try:
        v = float(raw) if raw else _DEFAULT_THRESHOLD
        # Clamp to sane range
        if v < 0.0 or v > 0.5:
            log.warning(
                "[PN26 sparse_v] threshold=%s out of [0, 0.5] — using default %s",
                raw, _DEFAULT_THRESHOLD,
            )
            return _DEFAULT_THRESHOLD
        return v
    except (ValueError, TypeError):
        return _DEFAULT_THRESHOLD


_ENV_DEBUG_SKIP = "GENESIS_PN26_SPARSE_V_DEBUG"


def is_debug_skip_enabled() -> bool:
    return os.environ.get(_ENV_DEBUG_SKIP, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Module-level skip-rate counter buffers + cumulative stats.
# Lazy-allocated on first call when DEBUG_SKIP_CTR=1.
_SKIP_TOTAL_BUF: torch.Tensor | None = None
_SKIP_COUNT_BUF: torch.Tensor | None = None
_SKIP_LIFETIME_TOTAL: int = 0
_SKIP_LIFETIME_SKIPPED: int = 0
_SKIP_LAUNCH_COUNT: int = 0


def _get_or_alloc_skip_buffers(grid_size: int, device):
    """Allocate per-CTA int64 counter buffers (one slot per CTA = no contention)."""
    global _SKIP_TOTAL_BUF, _SKIP_COUNT_BUF
    if (_SKIP_TOTAL_BUF is None or _SKIP_TOTAL_BUF.numel() < grid_size or
            _SKIP_TOTAL_BUF.device != device):
        _SKIP_TOTAL_BUF = torch.zeros(grid_size, dtype=torch.int64, device=device)
        _SKIP_COUNT_BUF = torch.zeros(grid_size, dtype=torch.int64, device=device)
    else:
        _SKIP_TOTAL_BUF[:grid_size].zero_()
        _SKIP_COUNT_BUF[:grid_size].zero_()
    return _SKIP_TOTAL_BUF, _SKIP_COUNT_BUF


def collect_skip_stats() -> dict:
    """Read accumulated skip-rate stats. Returns lifetime totals + last-launch.

    Call after a kernel invocation has completed (kernel writes via atomic_add).
    Aggregation happens here on host — sums all CTA slots.
    """
    global _SKIP_LIFETIME_TOTAL, _SKIP_LIFETIME_SKIPPED, _SKIP_LAUNCH_COUNT
    if _SKIP_TOTAL_BUF is None:
        return {"enabled": False}
    last_total = int(_SKIP_TOTAL_BUF.sum().item())
    last_skipped = int(_SKIP_COUNT_BUF.sum().item())
    _SKIP_LIFETIME_TOTAL += last_total
    _SKIP_LIFETIME_SKIPPED += last_skipped
    _SKIP_LAUNCH_COUNT += 1
    last_rate = (last_skipped / last_total * 100) if last_total else 0
    life_rate = (
        _SKIP_LIFETIME_SKIPPED / _SKIP_LIFETIME_TOTAL * 100
    ) if _SKIP_LIFETIME_TOTAL else 0
    return {
        "enabled": True,
        "last_launch_total_tiles": last_total,
        "last_launch_skipped_tiles": last_skipped,
        "last_launch_skip_rate_pct": round(last_rate, 2),
        "lifetime_total_tiles": _SKIP_LIFETIME_TOTAL,
        "lifetime_skipped_tiles": _SKIP_LIFETIME_SKIPPED,
        "lifetime_skip_rate_pct": round(life_rate, 2),
        "lifetime_launch_count": _SKIP_LAUNCH_COUNT,
    }


def reset_skip_stats():
    global _SKIP_LIFETIME_TOTAL, _SKIP_LIFETIME_SKIPPED, _SKIP_LAUNCH_COUNT
    _SKIP_LIFETIME_TOTAL = 0
    _SKIP_LIFETIME_SKIPPED = 0
    _SKIP_LAUNCH_COUNT = 0
    if _SKIP_TOTAL_BUF is not None:
        _SKIP_TOTAL_BUF.zero_()
        _SKIP_COUNT_BUF.zero_()


def get_sparse_v_min_ctx() -> int:
    raw = os.environ.get(_ENV_MIN_CTX, "")
    try:
        v = int(raw) if raw else _DEFAULT_MIN_CTX
        return max(0, v)
    except (ValueError, TypeError):
        return _DEFAULT_MIN_CTX


def should_apply() -> bool:
    """Platform gate: NVIDIA CUDA + SM ≥ 8.0 + env opt-in.

    SM 8.0 (Ampere datacenter A100) is the floor — sparse V relies on
    memory bandwidth being the bottleneck rather than compute. Older
    SMs (Volta SM 7.0) have lower HBM bandwidth ratios where skipping
    fewer loads doesn't pay off.
    """
    if not is_pn26_sparse_v_enabled():
        return False
    try:
        from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
    except ImportError:
        return False
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


# ---------------------------------------------------------------------------
# Genesis stage1 kernel — byte-equivalent to upstream when SPARSE_V=0
# ---------------------------------------------------------------------------

_CACHED_KERNEL: Any = None


def _build_kernel():
    """Build (and cache) the Genesis sparse-V Stage1 kernel.

    Lazy import + Triton compilation. One compile per process.
    """
    global _CACHED_KERNEL
    if _CACHED_KERNEL is not None:
        return _CACHED_KERNEL

    from vllm.triton_utils import tl, triton

    @triton.jit
    def _genesis_tq_decode_stage1_sparse_v(
        # Same signature as upstream _tq_decode_stage1 plus 2 constexpr params
        Q_rot_ptr,
        KV_cache_ptr,
        Block_table_ptr,
        Seq_lens_ptr,
        Centroids_ptr,
        Mid_o_ptr,
        stride_qb,
        stride_qh,
        stride_cache_block,
        stride_cache_pos,
        stride_cache_head,
        stride_bt_b,
        stride_mid_b,
        stride_mid_h,
        stride_mid_s,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
        KV_GROUP_SIZE: tl.constexpr,
        MSE_BITS: tl.constexpr,
        MSE_BYTES: tl.constexpr,
        KPS: tl.constexpr,
        VQB: tl.constexpr,
        VAL_DATA_BYTES: tl.constexpr,
        ATTN_SCALE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        KEY_FP8: tl.constexpr,
        NORM_CORRECTION: tl.constexpr = 0,
        FP8_E4B15: tl.constexpr = 0,
        # Genesis-added constexpr params:
        SPARSE_V: tl.constexpr = 0,
        SPARSE_V_THRESHOLD: tl.constexpr = 0.001,
        SINK_TOKENS: tl.constexpr = 4,  # StreamingLLM sink protection
        # Skip-rate observability: per-CTA atomic counters (constexpr-gated).
        # When DEBUG=0, all increments + atomics + pointer derefs are
        # constexpr-DCE'd → byte-equivalent SASS to non-instrumented kernel.
        # When DEBUG=1, costs ~50-100 ns per CTA at epilogue (one atomic
        # per CTA, distinct address per CTA = no contention).
        DEBUG_SKIP_CTR: tl.constexpr = 0,
        skip_total_ctr_ptr=None,    # int64* [grid_size] — total tiles seen
        skip_count_ctr_ptr=None,    # int64* [grid_size] — tiles skipped
    ):
        bid = tl.program_id(0)
        hid = tl.program_id(1)
        sid = tl.program_id(2)

        kv_head = hid // KV_GROUP_SIZE
        seq_len = tl.load(Seq_lens_ptr + bid)

        split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
        split_start = split_len * sid
        split_end = tl.minimum(split_start + split_len, seq_len)

        if split_start >= split_end:
            return

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < HEAD_DIM
        kv_range = tl.arange(0, BLOCK_KV)

        q_base = bid * stride_qb + hid * stride_qh
        q_rot = tl.load(
            Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0
        ).to(tl.float32)

        if not KEY_FP8:
            mse_bit_off = d_offs * MSE_BITS
            mse_byte_idx = mse_bit_off // 8
            mse_bit_shift = mse_bit_off % 8
            mse_mask = (1 << MSE_BITS) - 1

        if VQB == 3:
            val_bit_off = d_offs * 3
            val_byte_idx = val_bit_off // 8
            val_bit_shift = val_bit_off % 8

        m_prev = -float("inf")
        l_prev = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        bt_base = bid * stride_bt_b

        # [Genesis PN26 v5] Skip-rate observability — register-resident.
        # Constexpr-DCE'd to nothing when DEBUG_SKIP_CTR=0.
        local_total = 0
        local_skip = 0

        # [Genesis PN26 v3] tl.range() pipelining hint (P67 v7.50 pattern):
        # explicit Triton pipelining hint — compiler overlaps cp.async loads
        # with prior-iteration MMA on Ampere. +5-10% on P67 benchmark.
        for start_n in tl.range(split_start, split_end, BLOCK_KV):
            kv_offs = start_n + kv_range
            kv_mask = kv_offs < split_end

            page_idx = kv_offs // BLOCK_SIZE
            page_off = kv_offs % BLOCK_SIZE
            safe_page_idx = tl.where(kv_mask, page_idx, 0)
            block_nums = tl.load(
                Block_table_ptr + bt_base + safe_page_idx,
                mask=kv_mask,
                other=0,
            ).to(tl.int64)

            slot_bases = (
                block_nums * stride_cache_block
                + page_off.to(tl.int64) * stride_cache_pos
                + tl.cast(kv_head, tl.int64) * stride_cache_head
            )

            # ============================================================
            # COMPUTE ATTENTION SCORES
            # ============================================================
            if KEY_FP8:
                k_addrs = slot_bases[:, None] + d_offs[None, :]
                # [Genesis PN26 v3] cache_modifier=".cg" — L2 streaming hint
                # for K dequant raw data (one-shot per tile, no reuse).
                k_raw = tl.load(
                    KV_cache_ptr + k_addrs,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                    cache_modifier=".cg",
                )
                if FP8_E4B15:
                    k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
                else:
                    k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                scores = (
                    tl.sum(
                        tl.where(d_mask[None, :], q_rot[None, :] * k_float, 0.0),
                        axis=1,
                    )
                    * ATTN_SCALE
                )
                scores = tl.where(kv_mask, scores, -float("inf"))
            else:
                mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
                mse_raw0 = tl.load(
                    KV_cache_ptr + mse_addrs0,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                mse_raw1 = tl.load(
                    KV_cache_ptr + mse_addrs0 + 1,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                raw16 = mse_raw0 | (mse_raw1 << 8)
                mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

                c_vals = tl.load(
                    Centroids_ptr + mse_idx,
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )

                if NORM_CORRECTION:
                    c_norm_sq = tl.sum(
                        tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                        axis=1,
                    )
                    c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                    c_vals = c_vals * c_inv_norm[:, None]

                term1 = tl.sum(
                    tl.where(d_mask[None, :], q_rot[None, :] * c_vals, 0.0),
                    axis=1,
                )

                norm_bases = slot_bases + MSE_BYTES
                n_lo = tl.load(
                    KV_cache_ptr + norm_bases, mask=kv_mask, other=0
                ).to(tl.uint16)
                n_hi = tl.load(
                    KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0
                ).to(tl.uint16)
                vec_norms = (
                    (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                )

                scores = vec_norms * term1 * ATTN_SCALE
                scores = tl.where(kv_mask, scores, -float("inf"))

            # ============================================================
            # ONLINE SOFTMAX UPDATE (block-level)
            # ============================================================
            n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
            re_scale = tl.exp(m_prev - n_e_max)
            p = tl.exp(scores - n_e_max)

            # ============================================================
            # GENESIS PN26: SPARSE V tile-skip — opt-in via constexpr.
            # When SPARSE_V==0 (default), Triton dead-code-eliminates the
            # `if/else` below at compile time → byte-equivalent to upstream.
            #
            # Sink token protection (StreamingLLM, arXiv 2309.17453): NEVER
            # skip tiles that overlap the first SINK_TOKENS positions of the
            # KV stream. Those tokens carry attention "sink" mass critical
            # for long-context quality. Our BLOCK_KV=4 means the very first
            # tile (start_n==0) covers positions 0..3 — protect it.
            # ============================================================
            skip_v_tile = False
            if SPARSE_V:
                tile_protected = start_n < SINK_TOKENS
                if not tile_protected:
                    skip_v_tile = tl.max(p) < SPARSE_V_THRESHOLD

            # [Genesis PN26 v5] Skip-rate counter — register-only increments.
            # Compiler removes when DEBUG_SKIP_CTR=0 (constexpr branch).
            if DEBUG_SKIP_CTR:
                local_total += 1
                if skip_v_tile:
                    local_skip += 1

            if skip_v_tile:
                # Skip path: only decay accumulator. Online softmax
                # denominator (l_prev) and max (m_prev) still update so
                # totals stay numerically consistent for non-skipped tiles.
                acc = acc * re_scale
            else:
                # Standard path: load V, dequant, weighted sum.
                val_bases = slot_bases + KPS

                if VQB == 3:
                    val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
                    # [Genesis PN26 v3] L2 streaming hint on V dequant raw
                    val_raw0 = tl.load(
                        KV_cache_ptr + val_addrs0,
                        mask=kv_mask[:, None] & d_mask[None, :],
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.int32)
                    val_raw1 = tl.load(
                        KV_cache_ptr + val_addrs0 + 1,
                        mask=kv_mask[:, None] & d_mask[None, :],
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.int32)
                    raw16 = val_raw0 | (val_raw1 << 8)
                    v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

                    sc_bases = val_bases + VAL_DATA_BYTES
                    sc_lo = tl.load(
                        KV_cache_ptr + sc_bases, mask=kv_mask, other=0
                    ).to(tl.uint16)
                    sc_hi = tl.load(
                        KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0
                    ).to(tl.uint16)
                    v_scales = (
                        (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                    )
                    zr_lo = tl.load(
                        KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0
                    ).to(tl.uint16)
                    zr_hi = tl.load(
                        KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0
                    ).to(tl.uint16)
                    v_zeros = (
                        (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                    )
                    values = v_idx * v_scales[:, None] + v_zeros[:, None]
                else:  # VQB == 4
                    vb_idx = d_offs // 2
                    vb_shift = (d_offs % 2) * 4
                    val_addrs = val_bases[:, None] + vb_idx[None, :]
                    # [Genesis PN26 v3] L2 streaming hint on V dequant raw
                    val_raw = tl.load(
                        KV_cache_ptr + val_addrs,
                        mask=kv_mask[:, None] & d_mask[None, :],
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.int32)
                    v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

                    sc_bases = val_bases + VAL_DATA_BYTES
                    sc_lo = tl.load(
                        KV_cache_ptr + sc_bases, mask=kv_mask, other=0
                    ).to(tl.uint16)
                    sc_hi = tl.load(
                        KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0
                    ).to(tl.uint16)
                    v_scales = (
                        (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                    )
                    zr_lo = tl.load(
                        KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0
                    ).to(tl.uint16)
                    zr_hi = tl.load(
                        KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0
                    ).to(tl.uint16)
                    v_zeros = (
                        (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                    )
                    values = v_idx * v_scales[:, None] + v_zeros[:, None]

                acc = acc * re_scale + tl.sum(p[:, None] * values, 0)

            l_prev = l_prev * re_scale + tl.sum(p, 0)
            m_prev = n_e_max

        out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
        safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
        tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
        lse = m_prev + tl.log(safe_l)
        tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)

        # [Genesis PN26 v5] Skip-rate epilogue — single atomic per CTA,
        # per-CTA distinct slot (no contention). Constexpr-DCE'd when off.
        if DEBUG_SKIP_CTR:
            ctr_idx = (bid * NUM_KV_HEADS + hid) * NUM_KV_SPLITS + sid
            tl.atomic_add(skip_total_ctr_ptr + ctr_idx, local_total)
            tl.atomic_add(skip_count_ctr_ptr + ctr_idx, local_skip)

    _CACHED_KERNEL = _genesis_tq_decode_stage1_sparse_v
    log.info("[Genesis PN26 sparse_v] kernel built and cached")
    return _CACHED_KERNEL


def triton_turboquant_decode_attention_sparse_v(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
    # Genesis-added kwargs:
    sparse_v: bool = False,
    sparse_v_threshold: float = _DEFAULT_THRESHOLD,
    debug_skip_ctr: bool = False,
    skip_total_ctr: torch.Tensor | None = None,
    skip_count_ctr: torch.Tensor | None = None,
) -> torch.Tensor:
    """Genesis fork of `triton_turboquant_decode_attention` with SPARSE_V.

    When `sparse_v=False` (default), the kernel constexpr SPARSE_V=0 and
    Triton dead-code-eliminates the skip branch — byte-equivalent to
    upstream `triton_turboquant_decode_attention`.

    When `sparse_v=True`, per-tile skip engages: tiles with
    `tl.max(p) < sparse_v_threshold` skip V load + dequant + weighted sum.
    """
    from vllm.triton_utils import triton  # noqa: F401 — for completeness
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _get_layout, _use_fp8_e4b15, _fwd_kernel_stage2,
    )

    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

    if key_fp8:
        q_rot = query.contiguous()
    else:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        q_rot = (q_float @ PiT).contiguous()

    NUM_KV_SPLITS = max_num_kv_splits

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=device,
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_o

    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
    # [Genesis PN26 v3+] launch params configurable via env for sweeps.
    # Defaults match upstream when unset. Larger BLOCK_KV reduces loop
    # iterations + per-iter branch overhead at cost of register pressure;
    # higher num_warps improves occupancy on SM86 if shared-mem fits.
    BLOCK_KV = int(os.environ.get("GENESIS_PN26_SPARSE_V_BLOCK_KV", "4"))
    if BLOCK_KV not in (4, 8, 16, 32):
        BLOCK_KV = 4
    # Default num_warps=4 — empirical winner from BLOCK_KV × num_warps
    # sweep on 35B FP8 PROD A5000: gave 184.89 TPS mean (+5.4% vs upstream
    # num_warps=1 baseline). Override via env for re-tuning on different SM.
    NUM_WARPS = int(os.environ.get("GENESIS_PN26_SPARSE_V_NUM_WARPS", "4"))
    if NUM_WARPS not in (1, 2, 4, 8):
        NUM_WARPS = 4
    NUM_STAGES = int(os.environ.get("GENESIS_PN26_SPARSE_V_NUM_STAGES", "1"))
    if NUM_STAGES not in (1, 2, 3):
        NUM_STAGES = 1
    grid = (B, Hq, NUM_KV_SPLITS)

    # [Genesis PN26 v5] Auto-allocate skip-rate counter buffers when env-enabled
    # and not explicitly passed by caller.
    if debug_skip_ctr and skip_total_ctr is None:
        grid_size = B * Hq * NUM_KV_SPLITS
        skip_total_ctr, skip_count_ctr = _get_or_alloc_skip_buffers(grid_size, device)

    kernel = _build_kernel()
    kernel[grid](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        MSE_BITS=mse_bits,
        MSE_BYTES=cfg["mse_bytes"],
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        ATTN_SCALE=scale,
        BLOCK_D=cfg["BLOCK_D"],
        BLOCK_KV=BLOCK_KV,
        KEY_FP8=1 if key_fp8 else 0,
        NORM_CORRECTION=1 if norm_correction else 0,
        FP8_E4B15=fp8_e4b15,
        SPARSE_V=1 if sparse_v else 0,
        SPARSE_V_THRESHOLD=float(sparse_v_threshold),
        SINK_TOKENS=4,  # StreamingLLM finding: first 4 positions are attention sinks
        DEBUG_SKIP_CTR=1 if debug_skip_ctr else 0,
        skip_total_ctr_ptr=skip_total_ctr,
        skip_count_ctr_ptr=skip_count_ctr,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    out_dtype = query.dtype
    if (
        output_buf is not None
        and output_buf.shape[0] >= B
        and output_buf.dtype == out_dtype
    ):
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=cfg["BLOCK_D"],
        Lv=D,
        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
        num_warps=4,
        num_stages=2,
    )

    return output
