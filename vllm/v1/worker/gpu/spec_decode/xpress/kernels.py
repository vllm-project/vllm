# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for one XPress Jacobi pass.

A pass over a batch of N blocks (B slots each, slot 0 = the verified anchor) is:

    latent   u[n, b, :] = causal_mix(hcache[n, b] + embed(prev[n, b]) @ W_lat)
             x[n, b, :] = u[n, b, :] + SwiGLU_MLP(u[n, b, :])
    readout  logits[n, b, :] = base[n, b, :] + x[n, b, :] @ W2      (a matmul)
    argmax   blk[n, b] = argmax_v logits[n, b, v]

Only the latent and argmax stages are custom. The latent stage is a dozen tiny
ops per block that torch launches separately; fusing them removes the launch
floor, which is what bounds a small head. The argmax stage would otherwise
materialize [N * (B - 1), V] bf16 logits per pass (~146 MB at N = 32, V = 152k)
only to reduce them; fusing the add into the reduction never writes them.
"""

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
    """Stage 1 of the fused argmax: per (row, vocab chunk) max and its index.

    Grid: (ceil(V / BLOCK_V), rows). Each program sums one BLOCK_V-wide chunk of
    ``base[row] + bias[row]`` in registers and reduces it, so the full [rows, V]
    sum is never stored.

    Args:
        base_ptr: [rows, V] base logits (bf16), row stride ``stride_base_r``.
        bias_ptr: [rows, V] refiner bias, same layout.
        out_val_ptr: [rows, num_chunks] fp32 chunk maxima (written).
        out_idx_ptr: [rows, num_chunks] int64 chunk argmax, in vocab ids (written).
    """
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
    """The latent stage of one Jacobi pass for one block, fused end to end.

    Grid: (N,), one program per block. For every slot b it gathers the previous
    slot's token embedding (slot 0 reads the anchor's predecessor ``am1``),
    adds the pass-invariant hidden cache, applies the folded causal mixer
    ``u[b] = sum_{j <= b} L[b, j, :] * x[j]``, then a SwiGLU MLP with a
    residual, and writes the latent for slots 1..B-1 (slot 0 is the anchor and
    is never read out).

    Args:
        blk_ptr: [N, B] int64 current block tokens; slot b reads ``blk[b - 1]``.
        am1_ptr: [N] int64 token preceding each block's anchor.
        xh_ptr: [N, B, R] hidden cache, ``down_h(h) ++ down_g(mean h)`` projected
            to R, constant across passes.
        lat_out_ptr: [N, B - 1, R] latent output (written).
        w1_ptr: [V, R] token embedding.
        wlat_ptr: [R, R] embedding-to-latent projection (transposed).
        mixl_ptr: [B, B, R] folded mixer ``L * tril + I``, per channel.
        wg_ptr, wu_ptr: [R, H] SwiGLU gate / up (transposed).
        wd_ptr: [H, R] SwiGLU down (transposed).
        B, R, H: block size, latent rank, MLP hidden.
        BP: B rounded up to a power of two (tile size).
        HT: MLP hidden tile size.
    """
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


def xpress_latent_pass(
    blk, tok_am1, xh, lat_out, w1_weight, wlat_t, mix_kjc, wg_t, wu_t, wd_t
) -> None:
    """Run ``_xpress_latent_kernel`` over N blocks; see its docstring for shapes.

    The caller follows this with the readout matmul ``lat_out @ W2`` and one of
    the fused argmax entry points below.
    """
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
    """Stage 2 of the fused argmax: reduce one row's chunk maxima into ``blk``.

    Grid: (rows,). Row r is slot ``1 + r % (B - 1)`` of block ``r // (B - 1)``;
    the winner is written to ``blk[n, slot]`` so the next pass reads it in place.
    """
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


def fused_add_argmax_to_blk(
    base, bias, out_val, out_idx, blk, block_v: int = 4096
) -> None:
    """``blk[n, 1 + b] = argmax_v(base[r, v] + bias[r, v])`` for row r = n * (B-1) + b.

    The [rows, V] sum is never materialized: stage 1 reduces it chunk-wise in
    registers, stage 2 reduces the chunk maxima and scatters the winner into the
    block buffer, so one pass stays at three launches inside the captured graph.

    Args:
        base: [N * (B - 1), V] base logits for the draft slots.
        bias: [N * (B - 1), V] refiner bias for the same rows.
        out_val: [rows, ceil(V / block_v)] fp32 scratch for the chunk maxima.
        out_idx: [rows, ceil(V / block_v)] int64 scratch for the chunk argmax.
        blk: [N, B] int64 block buffer; slots 1..B-1 are written.
    """
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
