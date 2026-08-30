# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _xpress_add_argmax_partial_kernel(
    base_ptr,
    bias_ptr,
    out_val_ptr,
    out_idx_ptr,
    V,
    stride_base_r,
    stride_bias_r,
    stride_ov_r,
    stride_oi_r,
    BLOCK_V: tl.constexpr,
):
    pid_v = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    a = tl.load(base_ptr + row * stride_base_r + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + row * stride_bias_r + offs, mask=mask, other=0.0)
    s = (a + b).to(base_ptr.dtype.element_ty).to(tl.float32)
    s = tl.where(mask, s, -float("inf"))
    val = tl.max(s, axis=0)
    idx = tl.argmax(s, axis=0)
    tl.store(out_val_ptr + row * stride_ov_r + pid_v, val)
    tl.store(out_idx_ptr + row * stride_oi_r + pid_v, pid_v * BLOCK_V + idx)


@triton.jit
def _xpress_add_argmax_reduce_kernel(
    out_val_ptr,
    out_idx_ptr,
    token_ptr,
    N,
    stride_ov_r,
    stride_oi_r,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    best_val = -float("inf")
    best_idx = tl.zeros([], dtype=tl.int64)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        vals = tl.load(
            out_val_ptr + row * stride_ov_r + offs, mask=mask, other=-float("inf")
        )
        pos = tl.argmax(vals, axis=0)
        val = tl.max(vals, axis=0)
        idx = tl.load(out_idx_ptr + row * stride_oi_r + start + pos)
        take = val > best_val
        best_val = tl.where(take, val, best_val)
        best_idx = tl.where(take, idx, best_idx)
    tl.store(token_ptr + row, best_idx)


# base + bias and the argmax in one pass, so the [rows, V] sum is never materialized.
def fused_add_argmax(
    base: torch.Tensor,
    bias: torch.Tensor,
    out_val: torch.Tensor,
    out_idx: torch.Tensor,
    tokens: torch.Tensor,
    block_v: int = 4096,
) -> None:
    rows, v = base.shape
    num_v_blocks = (v + block_v - 1) // block_v
    _xpress_add_argmax_partial_kernel[(num_v_blocks, rows)](
        base,
        bias,
        out_val,
        out_idx,
        v,
        base.stride(0),
        bias.stride(0),
        out_val.stride(0),
        out_idx.stride(0),
        BLOCK_V=block_v,
        num_warps=8,
    )
    _xpress_add_argmax_reduce_kernel[(rows,)](
        out_val,
        out_idx,
        tokens,
        num_v_blocks,
        out_val.stride(0),
        out_idx.stride(0),
        BLOCK_N=64,
        num_warps=1,
    )


@triton.jit
def _xpress_latent_kernel(
    blk_ptr,
    am1_ptr,
    xh_ptr,
    lat_out_ptr,
    w1_ptr,
    wlat_ptr,
    mixl_ptr,
    wg_ptr,
    wu_ptr,
    wd_ptr,
    B: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
    BP: tl.constexpr,
    HT: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    offs_b = tl.arange(0, BP)
    offs_r = tl.arange(0, R)
    mask_b = offs_b < B

    xh = tl.load(
        xh_ptr + n * B * R + offs_b[:, None] * R + offs_r[None, :],
        mask=mask_b[:, None],
        other=0.0,
    ).to(tl.float32)
    wlat = tl.load(wlat_ptr + offs_r[:, None] * R + offs_r[None, :])

    prev = tl.load(blk_ptr + n * B + offs_b - 1, mask=mask_b & (offs_b >= 1), other=0)
    am1 = tl.load(am1_ptr + n)
    prev = tl.where(offs_b == 0, am1, prev)

    lat = tl.load(
        w1_ptr + prev[:, None] * R + offs_r[None, :],
        mask=mask_b[:, None],
        other=0.0,
    )
    x = xh + tl.dot(lat, wlat, out_dtype=tl.float32)

    u = tl.zeros([BP, R], dtype=tl.float32)
    for j in tl.static_range(B):
        lj = tl.load(
            mixl_ptr + offs_b[:, None] * B * R + j * R + offs_r[None, :],
            mask=mask_b[:, None],
            other=0.0,
        ).to(tl.float32)
        xj = tl.sum(tl.where(offs_b[:, None] == j, x, 0.0), axis=0)
        u += lj * xj[None, :]

    ub = u.to(wg_ptr.dtype.element_ty)
    x = u
    for h0 in range(0, H, HT):
        offs_ht = h0 + tl.arange(0, HT)
        g = tl.dot(
            ub,
            tl.load(wg_ptr + offs_r[:, None] * H + offs_ht[None, :]),
            out_dtype=tl.float32,
        )
        v = tl.dot(
            ub,
            tl.load(wu_ptr + offs_r[:, None] * H + offs_ht[None, :]),
            out_dtype=tl.float32,
        )
        m = (g * tl.sigmoid(g) * v).to(wd_ptr.dtype.element_ty)
        x += tl.dot(
            m,
            tl.load(wd_ptr + offs_ht[:, None] * R + offs_r[None, :]),
            out_dtype=tl.float32,
        )

    tl.store(
        lat_out_ptr + n * (B - 1) * R + (offs_b[:, None] - 1) * R + offs_r[None, :],
        x.to(lat_out_ptr.dtype.element_ty),
        mask=mask_b[:, None] & (offs_b[:, None] >= 1),
    )


# One Jacobi pass up to the readout: prev gather, hcache add, causal mix, SwiGLU.
def xpress_latent_pass(
    blk, tok_am1, xh, lat_out, w1_weight, wlat_t, mix_kjc, wg_t, wu_t, wd_t
) -> None:
    N, B = blk.shape
    R = xh.shape[-1]
    H = wg_t.shape[-1]
    _xpress_latent_kernel[(N,)](
        blk,
        tok_am1,
        xh,
        lat_out,
        w1_weight,
        wlat_t,
        mix_kjc,
        wg_t,
        wu_t,
        wd_t,
        B=B,
        R=R,
        H=H,
        BP=triton.next_power_of_2(B),
        HT=64,
        num_warps=8,
        num_stages=1,
    )


@triton.jit
def _xpress_add_argmax_reduce_to_blk_kernel(
    out_val_ptr,
    out_idx_ptr,
    blk_ptr,
    N,
    Bm1,
    stride_ov_r,
    stride_oi_r,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    best_val = -float("inf")
    best_idx = tl.zeros([], dtype=tl.int64)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        vals = tl.load(
            out_val_ptr + row * stride_ov_r + offs, mask=mask, other=-float("inf")
        )
        pos = tl.argmax(vals, axis=0)
        val = tl.max(vals, axis=0)
        idx = tl.load(out_idx_ptr + row * stride_oi_r + start + pos)
        take = val > best_val
        best_val = tl.where(take, val, best_val)
        best_idx = tl.where(take, idx, best_idx)
    n = row // Bm1
    b = row % Bm1
    tl.store(blk_ptr + n * (Bm1 + 1) + 1 + b, best_idx)


# As fused_add_argmax, but writes the winning ids straight into the block buffer,
# which keeps a pass at three launches inside the captured graph.
def fused_add_argmax_to_blk(
    base, bias, out_val, out_idx, blk, block_v: int = 4096
) -> None:
    rows, v = base.shape
    num_v_blocks = (v + block_v - 1) // block_v
    _xpress_add_argmax_partial_kernel[(num_v_blocks, rows)](
        base,
        bias,
        out_val,
        out_idx,
        v,
        base.stride(0),
        bias.stride(0),
        out_val.stride(0),
        out_idx.stride(0),
        BLOCK_V=block_v,
        num_warps=8,
    )
    _xpress_add_argmax_reduce_to_blk_kernel[(rows,)](
        out_val,
        out_idx,
        blk,
        num_v_blocks,
        blk.shape[1] - 1,
        out_val.stride(0),
        out_idx.stride(0),
        BLOCK_N=64,
        num_warps=1,
    )
