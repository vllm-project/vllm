# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused per-token RMSNorm + RoPE (+ optional FP8 per-row quant).

Replaces this 3-kernel sequence on the V4 decode path::

    q_flat, kv = qk_norm(q, kv_pre)          # triton: fused_qk_norm
    q = q_flat.view(T, H, D)
    rotary_emb(positions, q[..., -rd:],      # aiter: rope_cached_positions_2c
                          kv[..., -rd:])

with a single Triton kernel. The Q-side norm is *weightless* (V4's
``q_norm2`` has ``weight=None``) — the kernel hardcodes 1.0 on that side
and only loads ``kv_weight``. RoPE uses ``rotate_style=1`` (GPT-J
interleaved pairs) with ``reuse_freqs_front_part=True`` and
``nope_first=False`` to match ``_V4RoPE.forward``.

Optional FP8 outputs (``quant_q`` / ``quant_k``) emit per-row e4m3 + a
single fp32 ``amax/FP8_MAX`` scale per row. "1x128" blockscale = one
scale per (token, head) for Q, one per token for KV — head_dim is the
only contracted dim. Default off; plumbing for a future FP8 consumer
(sparse_attn FP8 path / FP8 swa_write). When the corresponding flag is
off the wrapper returns ``None`` for that scale and the fp8 output
buffer is not allocated.

Designed for the decode path only — prefill (large num_tokens) keeps the
3-kernel sequence where fusion savings are amortized over many GEMM-bound
ops anyway.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from vllm.models.deepseek_v4.amd.atom.model_ops.v4_kernels.state_writes import swa_write

# Lazy-imported flydsl path (optional dependency). Set to None when flydsl
# is unavailable; the dispatch in ``qk_norm_rope_maybe_quant`` will fall
# back to the Triton kernel.
try:
    import inspect as _inspect

    from aiter.ops.flydsl import flydsl_qk_norm_rope_quant

    _FLYDSL_AVAILABLE = True
    # Whether the installed flydsl fuses the SWA-ring scatter (``swa_kv`` kwarg).
    # Newer ATOM builds do; the flydsl here does not — in that case we still use
    # the fast flydsl norm+rope+quant and emit a separate ``swa_write`` (exactly
    # as ATOM's own decode trace shows: ``qk_norm_rope_..._flydsl`` +
    # ``_swa_write_kernel``). Keeping the fast flydsl path is the whole point —
    # the Triton fallback is much slower.
    _FLYDSL_HAS_SWA = "swa_kv" in _inspect.signature(
        flydsl_qk_norm_rope_quant
    ).parameters
except Exception:
    _FLYDSL_AVAILABLE = False
    _FLYDSL_HAS_SWA = False


# AMD MI3 native e4m3 variant. aiter's a8w8 path and the existing
# act_quant_inplace consumer agree on this dtype.
_FP8_DTYPE = torch.float8_e4m3fnuz
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
# Precomputed constants used by the fp8 quant fast-path. With the √2
# upper-bound for amax, scaling x_n by 1/scale algebraically equals
# scaling x (pre-norm) by FP8_MAX / (abs_max_x * √2) — `rstd` cancels.
# Folding into a single constant saves a multiply per row.
_SQRT2 = 1.4142135623730951
_INV_FP8_MAX_SQRT2 = _SQRT2 / _FP8_MAX
_FP8_MAX_OVER_SQRT2 = _FP8_MAX / _SQRT2


@triton.jit
def _gptj_rotate(x, x_rot_mask, BLOCK_M: tl.constexpr, RD: tl.constexpr):
    """GPT-J interleaved rotation on a [BLOCK_M, RD] tile.

    Returns ``(-x[2i+1], x[2i], -x[2i+3], x[2i+2], ...)`` so that
    ``x * cos + rotated * sin`` realizes the per-pair RoPE
    ``(e*c - o*s, e*s + o*c)``. cos/sin must be lane-duplicated
    (``cache[i]`` at lanes 2i and 2i+1), produced via
    ``d_cos_offs = d_pe_offs // 2``.
    """
    x_rot = tl.where(x_rot_mask, x, -x)
    x_rot = tl.reshape(x_rot, (BLOCK_M, RD // 2, 2))
    x_rot = tl.flip(x_rot, 2)
    return tl.reshape(x_rot, (BLOCK_M, RD))


@triton.jit
def _qk_norm_rope_maybe_quant_kernel(
    q_in_ptr,  # [T, H*D] bf16 (post wq_b, heads packed)
    kv_ptr,  # [T, D] bf16 (post wkv_a split)
    kv_weight_ptr,  # [D] bf16 (KV RMSNorm weight; Q weightless)
    cos_ptr,  # [..., rd/2] (REUSE_FREQS_FRONT_PART=True)
    sin_ptr,  # [..., rd/2]
    positions_ptr,  # [T] int64
    q_out_ptr,  # [T, H, D] bf16 or e4m3
    kv_out_ptr,  # [T, D] bf16 or e4m3
    q_scale_ptr,  # [T, H] fp32 (only when QUANT_Q)
    kv_scale_ptr,  # [T] fp32 (only when QUANT_K)
    eps: tl.constexpr,
    T,
    q_in_row_stride,
    kv_in_row_stride,
    cos_row_stride: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,  # head_dim — must be power of 2 (loaded as single tile)
    RD: tl.constexpr,  # rope_head_dim
    NOPE: tl.constexpr,  # D - RD
    NUM_PE_CHUNKS: tl.constexpr,  # D // RD — requires D % RD == 0 (V4: 512/64=8)
    FP8_MAX: tl.constexpr,
    INV_FP8_MAX_SQRT2: tl.constexpr,  # √2 / FP8_MAX, for `scale` compute
    FP8_MAX_OVER_SQRT2: tl.constexpr,  # FP8_MAX / √2, for `inv_scaled` (rstd-cancelled)
    BLOCK_M: tl.constexpr,
    QUANT_Q: tl.constexpr,
    QUANT_K: tl.constexpr,
):
    """Grid: ``(cdiv(T, BLOCK_M), H + 1)``.

    - ``pid_h < H`` → process Q-head ``pid_h`` (weightless RMSNorm + RoPE tail).
    - ``pid_h == H`` → process KV row (weighted RMSNorm + RoPE tail).

    Each program handles a ``BLOCK_M``-token tile. We load the full
    ``[BLOCK_M, D]`` tile, RMSNorm it, then extract the RoPE tail by
    ``tl.where(d >= NOPE, normed, 0)`` → reshape ``(BLOCK_M, NUM_PE_CHUNKS, RD)``
    → ``sum(axis=1)`` (only the last chunk is nonzero so the sum just
    selects that chunk). Stores nope to output positions, RoPE to tail.

    Q early-returns so the KV-only stores below see a single, non-divergent
    type for ``kv_out_ptr.dtype.element_ty`` (triton's IR cannot unify
    bf16 vs e4m3 store ops across an ``if/else`` branch).
    """
    pid_m = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    m_mask = m_offs < T

    d_offs = tl.arange(0, D)
    nope_d_mask = d_offs < NOPE

    rd_offs = tl.arange(0, RD).to(tl.int64)
    cos_d_offs = rd_offs // 2  # GPT-J + REUSE_FREQS_FRONT_PART: lane duplicate

    # positions/cos/sin are reused across all H+1 programs sharing this pid_m.
    # Tag them evict_last so the L2 keeps them hot for sibling head-tiles.
    pos = tl.load(
        positions_ptr + m_offs, mask=m_mask, other=0, eviction_policy="evict_last"
    ).to(tl.int64)
    cos_addr = pos[:, None] * cos_row_stride + cos_d_offs[None, :]
    cos = tl.load(
        cos_ptr + cos_addr,
        mask=m_mask[:, None],
        other=0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + cos_addr,
        mask=m_mask[:, None],
        other=0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    # Rotation mask: evens get +x, odds get -x → after pair-flip realizes
    # the (-o, e) pattern needed for x*c + rot*s == (e*c-o*s, e*s+o*c).
    x_rot_mask = (rd_offs % 2 == 0)[None, :]

    # ---- Q path (pid_h < H) ----
    if pid_h < H:
        h = pid_h.to(tl.int32)
        q_base = q_in_ptr + m_offs[:, None] * q_in_row_stride + h * D
        # Q tile is one-shot (no other program loads this head): evict_first.
        x = tl.load(
            q_base + d_offs[None, :],
            mask=m_mask[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        # Single pass over x: variance + (when quanting) input amax.
        # Triton fuses both reductions onto the same scan of x, so amax is
        # essentially free vs a second pass over `x_n`.
        sq = tl.sum(x * x, axis=1)
        if QUANT_Q:
            abs_max_x = tl.max(tl.abs(x), axis=1)
        rstd = tl.rsqrt(sq / D + eps)

        # RoPE input: re-load just the [BM, RD] rope tail (L2-hot from the
        # full-row load above) instead of extracting it via
        # `tl.where + reshape + sum` on [BM, D]. The extract path costs ~3us
        # at BM=8 D=512 because it touches the full 4096-elem tile; the
        # re-load hits L2 and is essentially free.
        pe_in = tl.load(q_base + NOPE + rd_offs[None, :], mask=m_mask[:, None]).to(
            tl.float32
        )

        q_out_base = q_out_ptr + m_offs[:, None] * (H * D) + h * D
        ot = q_out_ptr.dtype.element_ty
        if QUANT_Q:
            # Conservative √2 amax bound for fp8 scale (Q is weightless):
            #   |x_n[d]| = |x[d]| * rstd ≤ abs_max_x * rstd
            #   |pe[d]|  ≤ |x_rope[d]| * rstd * √2 (GPT-J rotation:
            #              |pe_even/odd| ≤ √(e²+o²) ≤ √2·max(|e|,|o|))
            # Bounded by `abs_max_x * rstd * √2`. Skipping the second-pass
            # `tl.max(tl.abs(x_n))` AND the [BM, RD] pe reduction. Cost:
            # ≤ 0.5 bits of fp8 precision (over-scale by ≤ √2).
            #
            # Algebraic fast-path: `x_n * inv == x * (rstd/scale)`, and
            # rstd/scale = rstd / (abs_max_x*rstd*INV_FP8_MAX_SQRT2)
            #            = FP8_MAX_OVER_SQRT2 / abs_max_x   (rstd cancels!)
            # So we skip materializing `x_n = x * rstd` as a separate fp32
            # tile, and apply a single multiplier directly to x before cast.
            # Same trick on pe via linearity of RoPE rotation:
            #   pe * inv = (pe_in * rstd * cos + rotate(...) * sin) * inv
            #            = (pe_in * inv_scaled) * cos + rotate(pe_in * inv_scaled) * sin
            inv_scaled = (FP8_MAX_OVER_SQRT2 / tl.maximum(abs_max_x, 1e-12))[:, None]
            pe_scaled = pe_in * inv_scaled
            pe_quant = (
                pe_scaled * cos + _gptj_rotate(pe_scaled, x_rot_mask, BLOCK_M, RD) * sin
            )
            # Scale to store (downstream consumer reconstructs via fp8*scale).
            scale = abs_max_x * rstd * INV_FP8_MAX_SQRT2
            tl.store(
                q_out_base + d_offs[None, :],
                (x * inv_scaled).to(ot),
                mask=m_mask[:, None] & nope_d_mask[None, :],
            )
            tl.store(
                q_out_base + NOPE + rd_offs[None, :],
                pe_quant.to(ot),
                mask=m_mask[:, None],
            )
            tl.store(q_scale_ptr + m_offs * H + h, scale, mask=m_mask)
        else:
            # bf16 path: still need to materialize x_n and pe in fp32.
            x_n = x * rstd[:, None]
            pe = pe_in * rstd[:, None]
            pe = pe * cos + _gptj_rotate(pe, x_rot_mask, BLOCK_M, RD) * sin
            tl.store(
                q_out_base + d_offs[None, :],
                x_n.to(ot),
                mask=m_mask[:, None] & nope_d_mask[None, :],
            )
            tl.store(
                q_out_base + NOPE + rd_offs[None, :],
                pe.to(ot),
                mask=m_mask[:, None],
            )
        return

    # ---- KV path (pid_h == H) ----
    kv_base = kv_ptr + m_offs[:, None] * kv_in_row_stride
    # KV tile is one-shot; weight is reused across all M-tiles.
    x = tl.load(
        kv_base + d_offs[None, :],
        mask=m_mask[:, None],
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    w = tl.load(kv_weight_ptr + d_offs, eviction_policy="evict_last").to(tl.float32)

    sq = tl.sum(x * x, axis=1)
    if QUANT_K:
        # Weighted amax: |x_n[d]| = |x[d]| * rstd * |w[d]|.
        # Pre-multiply x by abs(w) elementwise then take row-max.
        abs_max_xw = tl.max(tl.abs(x) * tl.abs(w)[None, :], axis=1)
    rstd = tl.rsqrt(sq / D + eps)

    # Reload rope tail from L2 (hot after the full-row load above) and apply
    # the per-rope-tail weight slice directly.
    pe_in = tl.load(kv_base + NOPE + rd_offs[None, :], mask=m_mask[:, None]).to(
        tl.float32
    )
    w_rope = tl.load(kv_weight_ptr + NOPE + rd_offs, eviction_policy="evict_last").to(
        tl.float32
    )

    kv_out_base = kv_out_ptr + m_offs[:, None] * D
    ot = kv_out_ptr.dtype.element_ty
    if QUANT_K:
        # Same √2 bound + rstd-cancellation fast-path as Q (see Q-path
        # comment). For KV with weighted norm:
        #   x_n_out = x * rstd * w * inv = (x * w) * (rstd / scale)
        #           = (x * w) * FP8_MAX_OVER_SQRT2 / abs_max_xw    (rstd cancels)
        # And pe_out via rope linearity:
        #   pe_out = (pe_in * inv_scaled * w_rope) * cos
        #          + rotate(pe_in * inv_scaled * w_rope) * sin
        inv_scaled = (FP8_MAX_OVER_SQRT2 / tl.maximum(abs_max_xw, 1e-12))[:, None]
        pe_scaled = pe_in * inv_scaled * w_rope[None, :]
        pe_quant = (
            pe_scaled * cos + _gptj_rotate(pe_scaled, x_rot_mask, BLOCK_M, RD) * sin
        )
        scale = abs_max_xw * rstd * INV_FP8_MAX_SQRT2
        tl.store(
            kv_out_base + d_offs[None, :],
            (x * inv_scaled * w[None, :]).to(ot),
            mask=m_mask[:, None] & nope_d_mask[None, :],
        )
        tl.store(
            kv_out_base + NOPE + rd_offs[None, :],
            pe_quant.to(ot),
            mask=m_mask[:, None],
        )
        tl.store(kv_scale_ptr + m_offs, scale, mask=m_mask)
    else:
        # bf16 path: materialize x_n and pe in fp32.
        x_n = x * rstd[:, None] * w[None, :]
        pe = pe_in * rstd[:, None] * w_rope[None, :]
        pe = pe * cos + _gptj_rotate(pe, x_rot_mask, BLOCK_M, RD) * sin
        tl.store(
            kv_out_base + d_offs[None, :],
            x_n.to(ot),
            mask=m_mask[:, None] & nope_d_mask[None, :],
        )
        tl.store(kv_out_base + NOPE + rd_offs[None, :], pe.to(ot), mask=m_mask[:, None])


def qk_norm_rope_maybe_quant(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    n_local_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    quant_q: bool = False,
    quant_k: bool = False,
    swa_kv: Optional[torch.Tensor] = None,
    state_slot_mapping: Optional[torch.Tensor] = None,
    batch_id_per_token: Optional[torch.Tensor] = None,
    swa_cu_seqlens_q: Optional[torch.Tensor] = None,
    swa_cache_size: Optional[int] = None,
    swa_write_per_batch: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused per-token RMSNorm + GPT-J interleaved RoPE (+ optional FP8 quant).

    Args:
        q: ``[T, H*D]`` bf16 — post-``wq_b`` Q (heads packed in last dim).
        kv: ``[T, D]`` bf16 — post-``wkv_a`` split KV row.
        kv_weight: ``[D]`` bf16 — KV-side RMSNorm weight. Q-side is weightless
            (kernel hardcodes 1.0).
        cos_cache, sin_cache: rope tables with ``rd/2`` columns on the inner-
            most axis (``reuse_freqs_front_part=True`` layout from
            ``_build_cos_sin_cache``). Higher-rank caches like
            ``[max_pos, 1, 1, rd/2]`` are tolerated — only the last-dim width
            and row-stride to (max_pos's) next index are read.
        positions: ``[T]`` int64 — absolute token positions.
        eps: RMSNorm epsilon.
        quant_q, quant_k: independently emit per-row FP8 + per-row fp32 scale.
            ``False`` keeps the bf16 output and returns ``None`` for that scale.
        swa_kv: ``[num_slots, cache_size, D]`` bf16 SWA ring buffer. When
            provided, the (bf16) KV row is also written into
            ``swa_kv[slot, pos % cache_size, :]`` where
            ``slot = state_slot_mapping[batch_id_per_token[t]]``. The flydsl
            path fuses this into the qk_norm launch; the Triton fallback emits
            a separate ``swa_write`` so both backends have identical side
            effects. Decode-only (prefill writes its SWA tail post-attention).
            BF16 only (requires ``quant_k=False``).
        state_slot_mapping: ``[bs]`` int32 — per-seq SWA ring slot. Required
            when ``swa_kv`` is set.
        batch_id_per_token: ``[T]`` int32, ``-1`` on CG-pad tokens — token→seq
            map for the fused (flydsl) SWA scatter. Required by the flydsl path.
        swa_cu_seqlens_q: ``[bs+1]`` int — per-seq cumulative seqlens used by
            the Triton-fallback ``swa_write``. Required only on the fallback
            path when ``swa_kv`` is set.
        swa_cache_size: SWA ring slot count (``swa_kv.shape[1]``); fallback only.
        swa_write_per_batch: ``min(max_seqlen_q, cache_size)``; fallback only.

    Returns:
        ``(q_out, kv_out, q_scale_or_None, k_scale_or_None)``:
          - ``q_out`` shape ``[T, H, D]``, dtype = ``float8_e4m3fnuz`` if
            ``quant_q`` else ``bf16``.
          - ``kv_out`` shape ``[T, D]``, dtype = ``float8_e4m3fnuz`` if
            ``quant_k`` else ``bf16``.
          - ``q_scale`` shape ``[T, H]`` fp32 if ``quant_q`` else ``None``.
          - ``k_scale`` shape ``[T]`` fp32 if ``quant_k`` else ``None``.
    """
    assert (
        q.dim() == 2 and kv.dim() == 2
    ), f"q/kv must be 2-D; got q={tuple(q.shape)} kv={tuple(kv.shape)}"
    T = q.shape[0]
    assert (
        q.shape[1] == n_local_heads * head_dim
    ), f"q last dim {q.shape[1]} != H*D = {n_local_heads * head_dim}"
    assert kv.shape == (
        T,
        head_dim,
    ), f"kv must be [T={T}, D={head_dim}]; got {tuple(kv.shape)}"
    assert (
        rope_head_dim <= head_dim and rope_head_dim % 2 == 0
    ), f"rope_head_dim must be even and ≤ head_dim; got {rope_head_dim}"
    # head_dim must be a power of 2 (loaded as a single triton tile) AND
    # divisible by rope_head_dim (the reshape+sum pe-extract trick requires
    # the rope tail to be the last `head_dim/rope_head_dim`-th chunk).
    assert (
        head_dim & (head_dim - 1)
    ) == 0, f"head_dim must be a power of 2; got {head_dim}"
    assert (
        head_dim % rope_head_dim == 0
    ), f"head_dim {head_dim} must be divisible by rope_head_dim {rope_head_dim}"
    assert (
        q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    ), f"q/kv must be bf16; got q={q.dtype} kv={kv.dtype}"
    assert cos_cache.shape[-1] == rope_head_dim // 2, (
        f"cos_cache last-dim {cos_cache.shape[-1]} != rope_head_dim/2 "
        f"{rope_head_dim // 2}"
    )
    assert sin_cache.stride(0) == cos_cache.stride(0), "sin/cos must share row stride"
    # Inner-dim stride must be 1 (dense). q.stride(0) and kv.stride(0) may
    # exceed H*D / D respectively when the caller passes a strided view of
    # a wider tensor (e.g. `kv_pre` from `torch.split(qkv_a, ...)` whose
    # row stride is `q_lora_rank + head_dim`).
    assert q.stride(-1) == 1 and kv.stride(-1) == 1, (
        f"q/kv must be dense in the last dim; got q.stride={q.stride()} "
        f"kv.stride={kv.stride()}"
    )

    q_out_dtype = _FP8_DTYPE if quant_q else torch.bfloat16
    kv_out_dtype = _FP8_DTYPE if quant_k else torch.bfloat16
    q_out = torch.empty(
        (T, n_local_heads, head_dim), dtype=q_out_dtype, device=q.device
    )
    kv_out = torch.empty((T, head_dim), dtype=kv_out_dtype, device=kv.device)

    # ------------------------------------------------------------------
    # flydsl dispatch (MVP hardcoded for V4-Pro decode shape). The combined
    # Q+KV single-launch kernel wins at all T (large for small T due to
    # halved launch overhead, large for big T due to better occupancy), so
    # "auto" picks flydsl whenever the shape matches.
    # ------------------------------------------------------------------
    # flydsl fast path — ATOM's production kernel. The installed flydsl may or
    # may not fuse the SWA-ring scatter; use it either way (a separate
    # ``swa_write`` covers the no-fusion case, matching ATOM's decode trace).
    # The single ``quant`` flag applies to both Q and KV, so only take this
    # path when quant_q == quant_k (V4-Pro decode: both False).
    if _FLYDSL_AVAILABLE and quant_q == quant_k:
        # flydsl requires bf16 RMSNorm weight; the module may store fp32.
        kv_weight = (
            kv_weight
            if kv_weight.dtype == torch.bfloat16
            else kv_weight.to(torch.bfloat16)
        )
        q_scale = (
            torch.empty((T, n_local_heads), dtype=torch.float32, device=q.device)
            if quant_q
            else None
        )
        kv_scale = (
            torch.empty((T,), dtype=torch.float32, device=kv.device)
            if quant_k
            else None
        )
        if _FLYDSL_HAS_SWA:
            q_out, kv_out, q_scale, kv_scale = flydsl_qk_norm_rope_quant(
                q, kv, kv_weight, cos_cache, sin_cache, positions,
                num_q_heads=n_local_heads, head_dim=head_dim,
                rope_head_dim=rope_head_dim, quant=quant_q,
                q_out=q_out, kv_out=kv_out, q_scale=q_scale, kv_scale=kv_scale,
                swa_kv=swa_kv, state_slot_mapping=state_slot_mapping,
                batch_id_per_token=batch_id_per_token,
                stream=torch.cuda.current_stream(),
            )
            return q_out, kv_out, q_scale, kv_scale
        q_out, kv_out, q_scale, kv_scale = flydsl_qk_norm_rope_quant(
            q, kv, kv_weight, cos_cache, sin_cache, positions,
            num_q_heads=n_local_heads, head_dim=head_dim,
            rope_head_dim=rope_head_dim, q_weight=None, quant=quant_q,
            quant_group_size=(head_dim if quant_q else None),
            q_out=q_out, kv_out=kv_out, q_scale=q_scale, kv_scale=kv_scale,
            stream=torch.cuda.current_stream(),
        )
        # Separate SWA-ring write (decode), matching ATOM's flydsl decode path.
        if swa_kv is not None:
            if (
                swa_cu_seqlens_q is None
                or swa_cache_size is None
                or swa_write_per_batch is None
            ):
                raise ValueError(
                    "swa_kv on the flydsl path requires swa_cu_seqlens_q, "
                    "swa_cache_size, and swa_write_per_batch"
                )
            swa_write(
                kv_out,
                positions,
                swa_cu_seqlens_q,
                state_slot_mapping,
                swa_kv,
                swa_cache_size,
                swa_write_per_batch,
            )
        return q_out, kv_out, q_scale, kv_scale

    q_scale = (
        torch.empty((T, n_local_heads), dtype=torch.float32, device=q.device)
        if quant_q
        else None
    )
    kv_scale = (
        torch.empty((T,), dtype=torch.float32, device=kv.device) if quant_k else None
    )

    # 1-element dummies so triton has concrete pointers when the QUANT_*
    # constexpr branch is off (kernel won't touch them).
    q_scale_arg = (
        q_scale if q_scale is not None else q.new_empty(1, dtype=torch.float32)
    )
    kv_scale_arg = (
        kv_scale if kv_scale is not None else q.new_empty(1, dtype=torch.float32)
    )

    # Tuned on V4-Pro decode shape (H=16, D=512, RD=64) on MI355. After
    # ditching the `tl.where + reshape + sum` pe-extract in favor of a direct
    # L2-hot reload of the rope tail, BM=8 NW=8 is within 0.1us of optimal
    # across the full T range (4..1024). The trailing shrink handles T<BM
    # cleanly (avoids tail-mask cost on a [BM, D] tile where BM is
    # over-provisioned).
    # Result vs baseline (fused_qk_norm + aiter rope) — median of 3 runs:
    #   T=4  1.92×   T=16 1.95×   T=32 1.92×   T=64 1.85×   T=128 1.81×
    #   T=256 1.66×  T=512 1.42×  T=1024 1.26×
    # At T=1024 we achieve 29.6% of MI355 HBM3e peak BW (2367 GB/s), vs the
    # baseline's 23.4% across the two-kernel chain.
    block_m, num_warps = 8, 8
    while block_m > 1 and block_m > T:
        block_m //= 2

    grid = (triton.cdiv(T, block_m), n_local_heads + 1)
    _qk_norm_rope_maybe_quant_kernel[grid](
        q,
        kv,
        kv_weight,
        cos_cache,
        sin_cache,
        positions,
        q_out,
        kv_out,
        q_scale_arg,
        kv_scale_arg,
        eps=float(eps),
        T=T,
        q_in_row_stride=q.stride(0),
        kv_in_row_stride=kv.stride(0),
        cos_row_stride=cos_cache.stride(0),
        H=n_local_heads,
        D=head_dim,
        RD=rope_head_dim,
        NOPE=head_dim - rope_head_dim,
        NUM_PE_CHUNKS=head_dim // rope_head_dim,
        FP8_MAX=_FP8_MAX,
        INV_FP8_MAX_SQRT2=_INV_FP8_MAX_SQRT2,
        FP8_MAX_OVER_SQRT2=_FP8_MAX_OVER_SQRT2,
        BLOCK_M=block_m,
        QUANT_Q=quant_q,
        QUANT_K=quant_k,
        num_warps=num_warps,
        waves_per_eu=1,
    )

    # Triton fallback does not fuse the SWA cache-write — emit it as a separate
    # launch so callers get identical side effects regardless of which kernel
    # backend ran (the flydsl path fuses it above). Only fires when the caller
    # requested it (swa_kv provided) AND supplied the fallback's cu_seqlens_q
    # path args.
    if swa_kv is not None:
        if (
            swa_cu_seqlens_q is None
            or swa_cache_size is None
            or swa_write_per_batch is None
        ):
            raise ValueError(
                "swa_kv requested on the Triton fallback path requires "
                "swa_cu_seqlens_q, swa_cache_size, and swa_write_per_batch"
            )
        swa_write(
            kv_out,
            positions,
            swa_cu_seqlens_q,
            state_slot_mapping,
            swa_kv,
            swa_cache_size,
            swa_write_per_batch,
        )

    return q_out, kv_out, q_scale, kv_scale


def qk_norm_rope_maybe_quant_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    n_local_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    quant_q: bool = False,
    quant_k: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Pure-torch reference. Matches the kernel modulo bf16 reduction-order
    noise. Performs RMSNorm (Q weightless, KV weighted), then a manual GPT-J
    interleaved RoPE on the tail ``rope_head_dim``, then optional per-row
    amax-based e4m3 quant.
    """
    T = q.shape[0]
    rd = rope_head_dim
    nope = head_dim - rd

    q_h = q.view(T, n_local_heads, head_dim).to(torch.float32)
    kv_f = kv.to(torch.float32)

    rstd_q = torch.rsqrt(q_h.pow(2).mean(-1, keepdim=True) + eps)
    q_h = q_h * rstd_q  # weightless
    rstd_kv = torch.rsqrt(kv_f.pow(2).mean(-1, keepdim=True) + eps)
    kv_f = kv_f * rstd_kv * kv_weight.to(torch.float32)

    cos = cos_cache.index_select(0, positions).view(T, rd // 2).to(torch.float32)
    sin = sin_cache.index_select(0, positions).view(T, rd // 2).to(torch.float32)

    def _rope_tail(x: torch.Tensor) -> torch.Tensor:
        head_shape = x.shape[:-1]
        tail = x[..., nope:].reshape(*head_shape, rd // 2, 2)
        c = cos.reshape((T,) + (1,) * (tail.ndim - 3) + (rd // 2,))
        s = sin.reshape((T,) + (1,) * (tail.ndim - 3) + (rd // 2,))
        even, odd = tail[..., 0], tail[..., 1]
        new_even = even * c - odd * s
        new_odd = even * s + odd * c
        tail_new = torch.stack([new_even, new_odd], dim=-1).reshape(*head_shape, rd)
        return torch.cat([x[..., :nope], tail_new], dim=-1)

    # Compute amax for quant BEFORE applying rope: the kernel uses the
    # `abs_max_x * rstd * √2` upper bound (saves a full-tile reduction).
    # Reproduce that bound here so kernel and reference quantize to the same
    # values bit-for-bit (modulo bf16 noise).
    SQRT2 = 1.4142135623730951
    if quant_q:
        # Q is weightless: x_n = x * rstd. amax bound from input.
        x_q_in = q.view(T, n_local_heads, head_dim).to(torch.float32)
        abs_max_x_q = x_q_in.abs().amax(dim=-1, keepdim=True)
        amax_q = abs_max_x_q * rstd_q * SQRT2
        q_scale_t = (amax_q / _FP8_MAX).clamp_min(1e-12)

    if quant_k:
        # KV is weighted: x_n = x * rstd * w. amax bound from |x*w|.
        x_kv_in = kv.to(torch.float32)
        abs_max_xw_kv = (x_kv_in.abs() * kv_weight.to(torch.float32).abs()).amax(
            dim=-1, keepdim=True
        )
        amax_k = abs_max_xw_kv * rstd_kv * SQRT2
        kv_scale_t = (amax_k / _FP8_MAX).clamp_min(1e-12)

    q_h = _rope_tail(q_h)
    kv_f = _rope_tail(kv_f)

    if quant_q:
        q_out = (q_h / q_scale_t).to(_FP8_DTYPE)
        q_scale = q_scale_t.squeeze(-1).contiguous()
    else:
        q_out = q_h.to(torch.bfloat16)
        q_scale = None

    if quant_k:
        kv_out = (kv_f / kv_scale_t).to(_FP8_DTYPE)
        kv_scale = kv_scale_t.squeeze(-1).contiguous()
    else:
        kv_out = kv_f.to(torch.bfloat16)
        kv_scale = None

    return q_out, kv_out, q_scale, kv_scale
