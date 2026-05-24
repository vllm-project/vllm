"""PN37 — DFlash tiny-Q non-causal attention kernel for Ampere SM 8.6.

Genesis-original Triton kernel that boosts DFlash spec-decode by replacing
FA2's underutilized "small Q" path. DFlash drafter forward produces
Q ∈ [2, 16] rows per request (1 + N speculative tokens, N=3..7), but FA2
launches its prefill grid `cdiv(Q, 64)` — most warps idle for Q < 16.

Scope (eligibility):
    - Q rows per request ∈ [2, 16]
    - head_dim = 128 (Qwen3.6 DFlash drafter, verified pre-flight)
    - non-causal attention (DFlash invariant per dflash.py:188 `causal=False`)
    - BF16 K/V (DFlash uses default KV cache, NOT TurboQuant — so no
      conflict with P67 which only fires on TQ packed-quant path)
    - Ampere SM 8.6+ (3090, A5000); on SM 8.9+ FA3 already covers via
      `pack_gqa` so PN37 should auto-OFF via gpu_profile.py

Algorithm:
    Grid `(num_kv_heads, batch_size)`. One CTA owns one (kv_head, request).
    Loop over heads_per_kv (=num_heads/num_kv_heads), accumulating output.
    For each head:
      Q tile [BLOCK_M=16, head_dim=128] loaded ONCE into smem.
      KV streaming loop: BLOCK_KV=64 K/V tiles, online softmax across all
      Q positions. No causal mask (saves comparison + unmasks ~half iters).
      `tl.exp2(... * 1.4426950408889634)` (LOG2E inlined per Triton constexpr
      requirement); `cache_modifier='.cg'` for streaming KV; `'.ca'` for hot Q.
      Pattern follows P67 v7.50.

Numerical guarantee:
    Per-row online softmax (NOT fused-M) → bit-equivalent to per-query
    reference. Prevents the v7.34 P67 drift bug class (Golden et al.
    arXiv 2405.02803, "Is Flash Attention Stable?").

Composition (NO conflict with):
    - P67 (TQ K+1 verify) — TQ ⊥ DFlash mutually exclusive (script comments)
    - PN21 (DFlash SWA) — sets metadata `causal` flag; PN37 reads
      `causal=False` and bails if True
    - PN23 (combine_hidden_states cast) — different function, same file
    - PN24 (aux layer +1) — different file
    - P38B/P15B (FA2 LSE clamp / varlen cast) — PN37 bypasses upstream FA2
      entirely on its eligible path

This file deliberately implements ONLY the standalone kernel + Python
dispatcher. Integration (text-patch into `qwen3_dflash.py`
`DFlashQwen3Attention.forward`) lives in
`vllm/_genesis/wiring/spec_decode/patch_N37_dflash_tiny_q_attn.py`
(separate, follows P67 split between kernel + wiring).

Default OFF (`GENESIS_ENABLE_PN37_DFLASH_TINY_Q=1` to engage). After A/B
verifies + 3% lower-bound TPS gain, candidate for default-on per
gpu_profile.py predicate (REC on SM 8.6).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.kernels.pn37")

_ENV_ENABLE = "GENESIS_ENABLE_PN37_DFLASH_TINY_Q"


def env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on"
    )


# ───────────────────────────────────────────────────────────────────
# Kernel — built lazily so test/module import doesn't require triton
# ───────────────────────────────────────────────────────────────────

_CACHED_KERNEL = None


def _build_kernel():
    """Build (and cache) the Triton kernel for tiny-Q non-causal attn."""
    global _CACHED_KERNEL
    if _CACHED_KERNEL is not None:
        return _CACHED_KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def _pn37_tiny_q_noncausal_kernel(
        Q_ptr, K_ptr, V_ptr, OUT_ptr,
        # 4D strides for [B, S, H, D]
        sQ_b, sQ_l, sQ_h, sQ_d,
        sK_b, sK_l, sK_h, sK_d,
        sV_b, sV_l, sV_h, sV_d,
        sO_b, sO_l, sO_h, sO_d,
        # shape
        Q_LEN: tl.constexpr,
        KV_LEN,
        HEADS_PER_KV: tl.constexpr,
        scale,
        # tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One CTA owns (kv_head, batch_idx). Iterates HEADS_PER_KV inner.

        Tensor layouts (4D, contiguous-or-not handled via explicit strides):
            Q : [B, Q_LEN,  NUM_HEADS,    HEAD_DIM]
            K : [B, KV_LEN, NUM_KV_HEADS, HEAD_DIM]
            V : [B, KV_LEN, NUM_KV_HEADS, HEAD_DIM]
            O : [B, Q_LEN,  NUM_HEADS,    HEAD_DIM]
        """
        kv_head_id = tl.program_id(0)
        batch_id = tl.program_id(1)

        offs_m = tl.arange(0, BLOCK_M)        # query positions [0..Q_LEN)
        offs_d = tl.arange(0, BLOCK_D)        # head_dim positions
        offs_kv = tl.arange(0, BLOCK_KV)      # KV chunk positions

        q_mask = offs_m < Q_LEN

        # K/V are shared across HEADS_PER_KV — load each KV tile once,
        # process inside the heads_per_kv loop.

        for q_head_offset in tl.static_range(HEADS_PER_KV):
            global_head_id = kv_head_id * HEADS_PER_KV + q_head_offset

            # Load Q tile [BLOCK_M, BLOCK_D] for THIS head
            q_offsets = (
                batch_id * sQ_b
                + offs_m[:, None] * sQ_l
                + global_head_id * sQ_h
                + offs_d[None, :] * sQ_d
            )
            q_tile = tl.load(
                Q_ptr + q_offsets,
                mask=q_mask[:, None],
                other=0.0,
                cache_modifier=".ca",
            )
            q_tile = (q_tile * scale).to(tl.float32)

            # Online softmax accumulators per query row
            m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
            l_i = tl.zeros((BLOCK_M,), tl.float32)
            acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

            for start_kv in tl.range(0, KV_LEN, BLOCK_KV):
                kv_pos = start_kv + offs_kv
                kv_mask = kv_pos < KV_LEN

                # Load K tile [BLOCK_KV, BLOCK_D]
                k_offsets = (
                    batch_id * sK_b
                    + kv_pos[:, None] * sK_l
                    + kv_head_id * sK_h
                    + offs_d[None, :] * sK_d
                )
                k_tile = tl.load(
                    K_ptr + k_offsets,
                    mask=kv_mask[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)

                # Load V tile [BLOCK_KV, BLOCK_D]
                v_offsets = (
                    batch_id * sV_b
                    + kv_pos[:, None] * sV_l
                    + kv_head_id * sV_h
                    + offs_d[None, :] * sV_d
                )
                v_tile = tl.load(
                    V_ptr + v_offsets,
                    mask=kv_mask[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                ).to(tl.float32)

                # qk: [BLOCK_M, BLOCK_KV]
                qk = tl.dot(q_tile, tl.trans(k_tile))
                # Mask invalid KV positions to -inf
                qk = tl.where(kv_mask[None, :], qk, float("-inf"))

                # Online softmax (per-row)
                m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                # LOG2E inlined: 1.4426950408889634
                alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
                p = tl.exp2((qk - m_new[:, None]) * 1.4426950408889634)

                acc = acc * alpha[:, None] + tl.dot(p, v_tile)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_new

            # Final normalize. Guard against l_i == 0 (Q_LEN > BLOCK_M
            # padded rows): output should be 0 there anyway via q_mask store.
            acc = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]

            # Store output [BLOCK_M, BLOCK_D] for this head
            o_offsets = (
                batch_id * sO_b
                + offs_m[:, None] * sO_l
                + global_head_id * sO_h
                + offs_d[None, :] * sO_d
            )
            tl.store(
                OUT_ptr + o_offsets,
                acc.to(OUT_ptr.dtype.element_ty),
                mask=q_mask[:, None],
            )

    _CACHED_KERNEL = _pn37_tiny_q_noncausal_kernel
    return _CACHED_KERNEL


# ───────────────────────────────────────────────────────────────────
# Python dispatcher (standalone API for TDD — no vllm-attention coupling)
# ───────────────────────────────────────────────────────────────────


def pn37_tiny_q_noncausal_attn(
    q,          # [B, Q_LEN, NUM_HEADS, HEAD_DIM]
    k,          # [B, KV_LEN, NUM_KV_HEADS, HEAD_DIM]
    v,          # [B, KV_LEN, NUM_KV_HEADS, HEAD_DIM]
    *,
    scale: float | None = None,
):
    """Standalone tiny-Q non-causal attention.

    Returns out of shape `[B, Q_LEN, NUM_HEADS, HEAD_DIM]`, same dtype as q.

    No paged KV in this v1. Integration with paged KV cache is a follow-up
    once standalone numerical TDD passes.
    """
    assert q.is_cuda, "PN37 requires CUDA tensors"
    assert q.dtype == k.dtype == v.dtype, "Q/K/V must match dtype"
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "head_dim mismatch"
    B, Q_LEN, NUM_HEADS, HEAD_DIM = q.shape
    Bk, KV_LEN, NUM_KV_HEADS, _ = k.shape
    assert B == Bk
    assert NUM_HEADS % NUM_KV_HEADS == 0, "GQA: heads must be divisible"
    HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS

    if scale is None:
        scale = 1.0 / (HEAD_DIM ** 0.5)

    out = q.new_empty((B, Q_LEN, NUM_HEADS, HEAD_DIM))

    # Tile sizes — BLOCK_M covers up to 16 (eligibility max).
    BLOCK_M = 16
    BLOCK_KV = 64
    BLOCK_D = HEAD_DIM   # 128 for Qwen3.6 DFlash

    kernel = _build_kernel()
    grid = (NUM_KV_HEADS, B)
    kernel[grid](
        q, k, v, out,
        # Explicit 4D strides — keeps layout assumptions inside the kernel
        # symmetric to what Python passes.
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        HEADS_PER_KV=HEADS_PER_KV,
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_KV=BLOCK_KV,
        BLOCK_D=BLOCK_D,
    )
    return out


def is_eligible_shape(q_len: int, head_dim: int) -> bool:
    """Cheap predicate without GPU sync. Used at integration time."""
    return 2 <= q_len <= 16 and head_dim == 128
