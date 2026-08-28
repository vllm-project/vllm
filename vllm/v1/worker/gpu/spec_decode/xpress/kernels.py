# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernel for the XPress K-pass Jacobi refinement.

One program per request runs ALL K passes on-chip: the refined latent
x [B, r] and the block tokens never leave registers between passes (tokens
round-trip through a tiny [N, B] scratch so pass p+1 can read slot b-1's
token written by pass p without register shuffles). Scoring is
candidate-restricted (base top-C, columns pre-gathered) — same semantics as
XPressRefinerHead.jacobi_refine_greedy(candidate_topc=C).

Per pass, per slot k (block slot 0 = anchor, never rewritten):
    lat      = W1[prev]                       # [B, r] row gather
    x        = xh + lat @ W_lat               # xh = hcache-half, pass-invariant
    u[k, c]  = sum_j L_kjc * x[j, c]          # causal mix, j-loop of FMAs
    x        = u + MLP_swiglu(u)
    score    = base_cand + w2_cand @ x        # [C] per slot
    blk[k]   = cand[argmax score]             # k >= 1

Targets the launch-latency-bound small-batch regime (bs ~ 1-4); large batches
keep the torch einsum path where cuBLAS weight reuse wins.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _xpress_jacobi_kernel(
    # tokens / scratch
    blk_ptr,            # [N, B] int64  in: seed (slot0=anchor), out: refined
    am1_ptr,            # [N] int64     token before the anchor
    # pass-invariant activations
    xh_ptr,             # [N, B, R] hcache @ W_hc  (precomputed, bf16/fp32)
    base_cand_ptr,      # [N, B, C] fp32/bf16 base logits of candidates
    cand_ptr,           # [N, B, C] int64 candidate token ids
    w2c_ptr,            # [N, B, C, R] gathered w2 rows
    # weights
    w1_ptr,             # [V, R] prev-token embedding
    wlat_ptr,           # [R, R]   in_proj latent half, transposed (in @ W -> r)
    mixl_ptr,           # [B, B, R] folded mixer, layout [k, j, c]
    wg_ptr, wu_ptr,     # [R, H] mlp gate/up, transposed
    wd_ptr,             # [H, R] mlp down, transposed
    num_passes,
    B: tl.constexpr, R: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
    BP: tl.constexpr,   # B padded to pow2
    HT: tl.constexpr,   # mlp H-chunk (shared-memory bound)
    CT: tl.constexpr,   # candidate C-chunk (shared-memory bound)
):
    n = tl.program_id(0)
    offs_b = tl.arange(0, BP)
    offs_r = tl.arange(0, R)
    mask_b = offs_b < B

    # pass-invariant hidden half of in_proj
    xh = tl.load(
        xh_ptr + n * B * R + offs_b[:, None] * R + offs_r[None, :],
        mask=mask_b[:, None], other=0.0,
    ).to(tl.float32)

    wlat = tl.load(wlat_ptr + offs_r[:, None] * R + offs_r[None, :])

    for _p in range(num_passes):
        # prev[k] = blk[k-1], prev[0] = tok_am1 (blk slot 0 = anchor)
        prev = tl.load(
            blk_ptr + n * B + offs_b - 1,
            mask=mask_b & (offs_b >= 1), other=0,
        )
        am1 = tl.load(am1_ptr + n)
        prev = tl.where(offs_b == 0, am1, prev)

        lat = tl.load(
            w1_ptr + prev[:, None] * R + offs_r[None, :],
            mask=mask_b[:, None], other=0.0,
        )
        x = xh + tl.dot(lat, wlat, out_dtype=tl.float32)   # [BP, R]

        # causal mix: u[k, c] = sum_j L[k, j, c] * x[j, c]
        u = tl.zeros([BP, R], dtype=tl.float32)
        for j in tl.static_range(B):
            lj = tl.load(
                mixl_ptr + offs_b[:, None] * B * R + j * R + offs_r[None, :],
                mask=mask_b[:, None], other=0.0,
            ).to(tl.float32)
            xj = tl.sum(tl.where(offs_b[:, None] == j, x, 0.0), axis=0)
            u += lj * xj[None, :]

        # swiglu mlp, residual — chunked over H to bound shared memory
        ub = u.to(wg_ptr.dtype.element_ty)
        x = u
        for h0 in range(0, H, HT):
            offs_ht = h0 + tl.arange(0, HT)
            g = tl.dot(ub, tl.load(wg_ptr + offs_r[:, None] * H + offs_ht[None, :]),
                       out_dtype=tl.float32)
            v = tl.dot(ub, tl.load(wu_ptr + offs_r[:, None] * H + offs_ht[None, :]),
                       out_dtype=tl.float32)
            m = (g * tl.sigmoid(g) * v).to(wd_ptr.dtype.element_ty)
            x += tl.dot(
                m, tl.load(wd_ptr + offs_ht[:, None] * R + offs_r[None, :]),
                out_dtype=tl.float32,
            )

        # candidate scoring + argmax per draft slot (slots 1..B-1),
        # chunked over C with a running max
        for b in tl.static_range(1, B):
            xb = tl.sum(tl.where(offs_b[:, None] == b, x, 0.0), axis=0)  # [R]
            best_val = float("-inf")
            best_tok = tl.zeros([], dtype=tl.int64)
            for c0 in range(0, C, CT):
                offs_ct = c0 + tl.arange(0, CT)
                w2b = tl.load(
                    w2c_ptr
                    + ((n * B + b) * C + offs_ct[:, None]) * R
                    + offs_r[None, :]
                ).to(tl.float32)
                sc = tl.sum(w2b * xb[None, :], axis=1) + tl.load(
                    base_cand_ptr + (n * B + b) * C + offs_ct
                ).to(tl.float32)
                val = tl.max(sc, axis=0)
                arg = tl.argmax(sc, axis=0)
                tok = tl.load(cand_ptr + (n * B + b) * C + c0 + arg)
                take = val > best_val
                best_val = tl.where(take, val, best_val)
                best_tok = tl.where(take, tok, best_tok)
            tl.store(blk_ptr + n * B + b, best_tok)


def xpress_jacobi_fused(
    *,
    blk: torch.Tensor,        # [N, B] int64, seeded (slot0=anchor, 1..=base argmax)
    tok_am1: torch.Tensor,    # [N] int64
    xh: torch.Tensor,         # [N, B, R] pass-invariant in_proj hidden half
    base_cand: torch.Tensor,  # [N, B, C]
    cand: torch.Tensor,       # [N, B, C] int64
    w2_cand: torch.Tensor,    # [N, B, C, R]
    w1_weight: torch.Tensor,  # [V, R]
    wlat_t: torch.Tensor,     # [R, R] in_proj latent half, pre-transposed
    mix_l_kjc: torch.Tensor,  # [B, B, R] folded mixer in [k, j, c] layout
    wg_t: torch.Tensor, wu_t: torch.Tensor, wd_t: torch.Tensor,
    num_passes: int,
) -> None:
    """Runs all K refine passes in one launch; writes draft ids into blk[:, 1:]."""
    N, B = blk.shape
    R = xh.shape[-1]
    C = cand.shape[-1]
    H = wg_t.shape[-1]
    _xpress_jacobi_kernel[(N,)](
        blk, tok_am1,
        xh, base_cand, cand, w2_cand,
        w1_weight, wlat_t, mix_l_kjc,
        wg_t, wu_t, wd_t,
        num_passes,
        B=B, R=R, C=C, H=H, BP=triton.next_power_of_2(B),
        HT=64, CT=128,
        num_warps=8, num_stages=1,
    )


# ---------------------------------------------------------------------------
# Fused add+argmax epilogue for the FULL-VOCAB greedy path.
#
# Replaces `(base + bias).argmax(-1)` without materializing the sum tensor:
# per refine pass this removes one write and one read of the [rows, V] sum
# (~40% of the vocab-level traffic after the bias GEMM). BIT-IDENTICAL to the
# eager path: the sum is rounded back to the input dtype (bf16) before
# comparison, and ties resolve to the FIRST index (torch.argmax semantics).
# ---------------------------------------------------------------------------


@triton.jit
def _xpress_add_argmax_partial_kernel(
    base_ptr, bias_ptr, out_val_ptr, out_idx_ptr,
    V,
    stride_base_r, stride_bias_r,
    stride_ov_r, stride_oi_r,
    BLOCK_V: tl.constexpr,
):
    pid_v = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)  # int64: row*stride overflows int32 past ~14k rows
    offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    a = tl.load(base_ptr + row * stride_base_r + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + row * stride_bias_r + offs, mask=mask, other=0.0)
    # round the sum back to the storage dtype -> bit-identical to eager bf16 add
    s = (a + b).to(base_ptr.dtype.element_ty).to(tl.float32)
    s = tl.where(mask, s, -float("inf"))
    val = tl.max(s, axis=0)
    idx = tl.argmax(s, axis=0)          # first max within the block
    tl.store(out_val_ptr + row * stride_ov_r + pid_v, val)
    tl.store(out_idx_ptr + row * stride_oi_r + pid_v, pid_v * BLOCK_V + idx)


@triton.jit
def _xpress_add_argmax_reduce_kernel(
    out_val_ptr, out_idx_ptr, token_ptr,
    N,
    stride_ov_r, stride_oi_r,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    best_val = -float("inf")
    best_idx = tl.zeros([], dtype=tl.int64)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        vals = tl.load(out_val_ptr + row * stride_ov_r + offs,
                       mask=mask, other=-float("inf"))
        pos = tl.argmax(vals, axis=0)
        val = tl.max(vals, axis=0)
        idx = tl.load(out_idx_ptr + row * stride_oi_r + start + pos)
        take = val > best_val               # strict > keeps the EARLIEST block
        best_val = tl.where(take, val, best_val)
        best_idx = tl.where(take, idx, best_idx)
    tl.store(token_ptr + row, best_idx)


def fused_add_argmax(
    base: torch.Tensor,   # [rows, V] bf16 (contiguous rows)
    bias: torch.Tensor,   # [rows, V] bf16
    out_val: torch.Tensor,   # [rows, num_v_blocks] fp32 scratch
    out_idx: torch.Tensor,   # [rows, num_v_blocks] int64 scratch
    tokens: torch.Tensor,    # [rows] int64 destination
    block_v: int = 4096,
) -> None:
    rows, v = base.shape
    num_v_blocks = (v + block_v - 1) // block_v
    _xpress_add_argmax_partial_kernel[(num_v_blocks, rows)](
        base, bias, out_val, out_idx, v,
        base.stride(0), bias.stride(0),
        out_val.stride(0), out_idx.stride(0),
        BLOCK_V=block_v, num_warps=8,
    )
    _xpress_add_argmax_reduce_kernel[(rows,)](
        out_val, out_idx, tokens, num_v_blocks,
        out_val.stride(0), out_idx.stride(0),
        BLOCK_N=64, num_warps=1,
    )


# ---------------------------------------------------------------------------
# Fused single-pass LATENT kernel for the FULL-VOCAB path.
#
# One launch computes the whole refine pass up to (not including) the w2
# readout, for every request: prev tokens read straight from blk (no roll),
# in_proj + folded mixer + SwiGLU MLP on-chip, latent stored for draft slots
# 1..B-1 only (slot 0's output is never read). Together with mm(lat, w2_t)
# and the add+argmax epilogue writing back into blk, a K-pass refine is
# 3*K + 2 kernels instead of ~15*K -- the launch-count (and CUDA-graph
# node-count) reduction is the point. Float accumulation orders differ from
# the eager torch ops (same reassociation class as torch.compile).
# ---------------------------------------------------------------------------


@triton.jit
def _xpress_latent_kernel(
    blk_ptr,            # [N, B] int64  in: current block tokens (slot 0 = anchor)
    am1_ptr,            # [N] int64     token BEFORE the anchor
    xh_ptr,             # [N, B, R]     hcache @ W_hc (pass-invariant)
    lat_out_ptr,        # [N, B-1, R]   OUT: refined latent for draft slots 1..B-1
    w1_ptr,             # [V, R] prev-token embedding
    wlat_ptr,           # [R, R] in_proj latent half, pre-transposed
    mixl_ptr,           # [B, B, R] FOLDED mixer in [k, j, c] layout
    wg_ptr, wu_ptr,     # [R, H] mlp gate/up, pre-transposed
    wd_ptr,             # [H, R] mlp down, pre-transposed
    B: tl.constexpr, R: tl.constexpr, H: tl.constexpr,
    BP: tl.constexpr, HT: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    offs_b = tl.arange(0, BP)
    offs_r = tl.arange(0, R)
    mask_b = offs_b < B

    xh = tl.load(
        xh_ptr + n * B * R + offs_b[:, None] * R + offs_r[None, :],
        mask=mask_b[:, None], other=0.0,
    ).to(tl.float32)
    wlat = tl.load(wlat_ptr + offs_r[:, None] * R + offs_r[None, :])

    prev = tl.load(blk_ptr + n * B + offs_b - 1, mask=mask_b & (offs_b >= 1), other=0)
    am1 = tl.load(am1_ptr + n)
    prev = tl.where(offs_b == 0, am1, prev)

    lat = tl.load(
        w1_ptr + prev[:, None] * R + offs_r[None, :],
        mask=mask_b[:, None], other=0.0,
    )
    x = xh + tl.dot(lat, wlat, out_dtype=tl.float32)          # in_proj(cat) split

    # folded causal mix: u[k, c] = sum_j L_fold[k, j, c] * x[j, c]
    u = tl.zeros([BP, R], dtype=tl.float32)
    for j in tl.static_range(B):
        lj = tl.load(
            mixl_ptr + offs_b[:, None] * B * R + j * R + offs_r[None, :],
            mask=mask_b[:, None], other=0.0,
        ).to(tl.float32)
        xj = tl.sum(tl.where(offs_b[:, None] == j, x, 0.0), axis=0)
        u += lj * xj[None, :]

    # x = u + SwiGLU_MLP(u), H chunked to bound shared memory
    ub = u.to(wg_ptr.dtype.element_ty)
    x = u
    for h0 in range(0, H, HT):
        offs_ht = h0 + tl.arange(0, HT)
        g = tl.dot(ub, tl.load(wg_ptr + offs_r[:, None] * H + offs_ht[None, :]),
                   out_dtype=tl.float32)
        v = tl.dot(ub, tl.load(wu_ptr + offs_r[:, None] * H + offs_ht[None, :]),
                   out_dtype=tl.float32)
        m = (g * tl.sigmoid(g) * v).to(wd_ptr.dtype.element_ty)
        x += tl.dot(
            m, tl.load(wd_ptr + offs_ht[:, None] * R + offs_r[None, :]),
            out_dtype=tl.float32,
        )

    # store draft slots only (k >= 1), packed as [N, B-1, R]
    tl.store(
        lat_out_ptr + n * (B - 1) * R + (offs_b[:, None] - 1) * R + offs_r[None, :],
        x.to(lat_out_ptr.dtype.element_ty),
        mask=mask_b[:, None] & (offs_b[:, None] >= 1),
    )


def xpress_latent_pass(blk, tok_am1, xh, lat_out, w1_weight, wlat_t, mix_kjc,
                       wg_t, wu_t, wd_t) -> None:
    """One refine pass's latent for all requests, single launch. See kernel doc."""
    N, B = blk.shape
    R = xh.shape[-1]
    H = wg_t.shape[-1]
    _xpress_latent_kernel[(N,)](
        blk, tok_am1, xh, lat_out,
        w1_weight, wlat_t, mix_kjc, wg_t, wu_t, wd_t,
        B=B, R=R, H=H, BP=triton.next_power_of_2(B), HT=64,
        num_warps=8, num_stages=1,
    )


@triton.jit
def _xpress_add_argmax_reduce_to_blk_kernel(
    out_val_ptr, out_idx_ptr, blk_ptr,
    N, Bm1,
    stride_ov_r, stride_oi_r,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    best_val = -float("inf")
    best_idx = tl.zeros([], dtype=tl.int64)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        vals = tl.load(out_val_ptr + row * stride_ov_r + offs,
                       mask=mask, other=-float("inf"))
        pos = tl.argmax(vals, axis=0)
        val = tl.max(vals, axis=0)
        idx = tl.load(out_idx_ptr + row * stride_oi_r + start + pos)
        take = val > best_val               # strict > keeps the EARLIEST block
        best_val = tl.where(take, val, best_val)
        best_idx = tl.where(take, idx, best_idx)
    # write straight into blk[n, 1 + b]  (row = n * Bm1 + b)
    n = row // Bm1
    b = row % Bm1
    tl.store(blk_ptr + n * (Bm1 + 1) + 1 + b, best_idx)


def fused_add_argmax_to_blk(base, bias, out_val, out_idx, blk, block_v: int = 4096) -> None:
    """fused_add_argmax variant writing the winning tokens straight into blk[:, 1:]."""
    rows, v = base.shape
    num_v_blocks = (v + block_v - 1) // block_v
    _xpress_add_argmax_partial_kernel[(num_v_blocks, rows)](
        base, bias, out_val, out_idx, v,
        base.stride(0), bias.stride(0),
        out_val.stride(0), out_idx.stride(0),
        BLOCK_V=block_v, num_warps=8,
    )
    _xpress_add_argmax_reduce_to_blk_kernel[(rows,)](
        out_val, out_idx, blk, num_v_blocks, blk.shape[1] - 1,
        out_val.stride(0), out_idx.stride(0),
        BLOCK_N=64, num_warps=1,
    )
