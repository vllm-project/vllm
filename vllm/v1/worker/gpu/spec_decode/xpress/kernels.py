# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the XPress refine step.

``xpress_latent_pass`` computes one Jacobi pass up to the readout (prev-token
gather, hcache add, causal mix, SwiGLU) and ``fused_add_argmax_to_blk`` folds
``base + bias`` and the argmax into one pass that writes straight into the block
buffer, so the [N, B, V] sum is never materialized. Together they keep a refine
pass at three launches, which matters because the whole draft step is captured
in a CUDA graph. ``fused_add_argmax`` is the standalone epilogue used when the
fused-latent path is unavailable.
"""

import torch
import triton
import triton.language as tl


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
