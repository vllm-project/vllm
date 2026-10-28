# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused four-stream mHC kernels ported from the Raon serving kit."""

import torch

from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as tld


@triton.jit
def _mhc_site_mix(
    s_ptr,
    y_ptr,
    head_ptr,
    e_ptr,
    h0_ptr,
    negh0_ptr,
    geye_ptr,
    ggate_ptr,
    out_ptr,
    C: tl.constexpr,
    N: tl.constexpr,
    BC: tl.constexpr,
    MIX_ONE: tl.constexpr,
):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    i4 = tl.arange(0, 4)
    j4 = tl.arange(0, 4)
    e = tl.load(e_ptr + t * N * N + i4[:, None] * N + j4[None, :])
    negh0 = tl.load(negh0_ptr + i4[:, None] * N + j4[None, :])
    rho = tl.maximum(tl.max(tl.max(tld.div_rn(e, negh0), axis=1), axis=0), 0.0)
    h0 = tl.load(h0_ptr + i4[:, None] * N + j4[None, :])
    phim = h0 + tld.div_rn(e, tl.full(e.shape, 1.0, tl.float32) + rho)
    geye = tl.load(geye_ptr + i4[:, None] * N + j4[None, :])
    g = tl.load(ggate_ptr)
    hres = tld.add_rn(geye, tld.mul_rn(g, phim))

    hb = head_ptr + t * (2 * N)
    hpost = tl.load(hb + N + i4)
    y = tl.load(y_ptr + t * C + cols).to(tl.float32)
    h0 = tl.sum(tl.where(j4 == 0, hres, 0.0), axis=1)
    h1 = tl.sum(tl.where(j4 == 1, hres, 0.0), axis=1)
    h2 = tl.sum(tl.where(j4 == 2, hres, 0.0), axis=1)
    h3 = tl.sum(tl.where(j4 == 3, hres, 0.0), axis=1)
    s0 = tl.load(s_ptr + t * N * C + 0 * C + cols).to(tl.float32)
    s1 = tl.load(s_ptr + t * N * C + 1 * C + cols).to(tl.float32)
    s2 = tl.load(s_ptr + t * N * C + 2 * C + cols).to(tl.float32)
    s3 = tl.load(s_ptr + t * N * C + 3 * C + cols).to(tl.float32)
    if MIX_ONE:
        p0 = tld.mul_rn(h0[:, None], s0[None, :])
        p1 = tld.mul_rn(h1[:, None], s1[None, :])
        p2 = tld.mul_rn(h2[:, None], s2[None, :])
        p3 = tld.mul_rn(h3[:, None], s3[None, :])
        acc = tld.add_rn(tld.add_rn(tld.add_rn(p0, p1), p2), p3)
    else:
        acc = tld.fma(
            h3[:, None],
            s3[None, :],
            tld.fma(
                h2[:, None],
                s2[None, :],
                tld.fma(h1[:, None], s1[None, :], tld.mul_rn(h0[:, None], s0[None, :])),
            ),
        )
    o = tld.add_rn(acc, tld.mul_rn(hpost[:, None], y[None, :]))
    tl.store(
        out_ptr + t * N * C + i4[:, None] * C + cols[None, :],
        o.to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _mhc_halve_x(v, H: tl.constexpr, w: tl.constexpr):
    return tl.sum(tl.reshape(v, [H, 2, w // 2]), axis=1)


@triton.jit
def _mhc_halve_y(v, h: tl.constexpr):
    return tl.sum(tl.reshape(v, [2, h // 2]), axis=0)


@triton.jit
def _mhc_rms_mean(
    v_base, NC: tl.constexpr, H: tl.constexpr, W: tl.constexpr, FACTOR: tl.constexpr
):
    L = tl.arange(0, W)[None, :] + tl.arange(0, H)[:, None] * W
    a0 = tl.zeros([H, W], tl.float32)
    a1 = tl.zeros([H, W], tl.float32)
    a2 = tl.zeros([H, W], tl.float32)
    a3 = tl.zeros([H, W], tl.float32)
    for k in tl.static_range(NC // 2048):
        off = (L + k * 512) * 4
        v0 = tl.load(v_base + off + 0).to(tl.float32)
        v1 = tl.load(v_base + off + 1).to(tl.float32)
        v2 = tl.load(v_base + off + 2).to(tl.float32)
        v3 = tl.load(v_base + off + 3).to(tl.float32)
        a0 = tld.add_rn(a0, tld.mul_rn(v0, v0))
        a1 = tld.add_rn(a1, tld.mul_rn(v1, v1))
        a2 = tld.add_rn(a2, tld.mul_rn(v2, v2))
        a3 = tld.add_rn(a3, tld.mul_rn(v3, v3))
    v = tld.add_rn(tld.add_rn(tld.add_rn(a0, a1), a2), a3)
    if W >= 512:
        v = _mhc_halve_x(v, H, 512)
    if W >= 256:
        v = _mhc_halve_x(v, H, 256)
    if W >= 128:
        v = _mhc_halve_x(v, H, 128)
    if W >= 64:
        v = _mhc_halve_x(v, H, 64)
    v = _mhc_halve_x(v, H, 32)
    v = _mhc_halve_x(v, H, 16)
    v = _mhc_halve_x(v, H, 8)
    v = _mhc_halve_x(v, H, 4)
    v = _mhc_halve_x(v, H, 2)
    v = tl.reshape(v, [H])
    if H >= 16:
        v = _mhc_halve_y(v, 16)
    if H >= 8:
        v = _mhc_halve_y(v, 8)
    if H >= 4:
        v = _mhc_halve_y(v, 4)
    if H >= 2:
        v = _mhc_halve_y(v, 2)
    return tld.mul_rn(tl.sum(v, axis=0), FACTOR)


@triton.jit
def _mhc_dot3(a0, a1, a2, b0, b1, b2, FMA: tl.constexpr):
    if FMA:
        r = tld.fma(a2, b2, tld.fma(a1, b1, tld.mul_rn(a0, b0)))
    else:
        r = tld.add_rn(
            tld.add_rn(tld.mul_rn(a0, b0), tld.mul_rn(a1, b1)), tld.mul_rn(a2, b2)
        )
    return r


@triton.jit
def _mhc_site_head_v2(
    s_ptr,
    mm_ptr,
    mean_ptr,
    arow_ptr,
    brow_ptr,
    head_ptr,
    e_ptr,
    q_ptr,
    agg_ptr,
    NC: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    MMS: tl.constexpr,
    BC: tl.constexpr,
    EPS: tl.constexpr,
    AGG_TREE: tl.constexpr,
    RMS_IN: tl.constexpr,
    RH: tl.constexpr,
    RW: tl.constexpr,
    FACTOR: tl.constexpr,
    CHART_IN: tl.constexpr,
    CHART_FMA2: tl.constexpr,
    SPLIT: tl.constexpr,
):
    t = tl.program_id(0)
    v_base = s_ptr + t * NC
    r = tl.arange(0, 32)
    rmask = r < R
    acc = tl.load(mm_ptr + t * MMS + r, mask=rmask, other=0.0).to(tl.float32)
    if RMS_IN:
        mean = _mhc_rms_mean(v_base, NC, RH, RW, FACTOR)
    else:
        mean = tl.load(mean_ptr + t)
    inv = tld.rsqrt(mean + EPS)
    a = tl.load(arow_ptr + r, mask=rmask, other=0.0)
    b = tl.load(brow_ptr + r, mask=rmask, other=0.0)
    pre = tld.add_rn(tld.mul_rn(a, tld.mul_rn(acc, inv)), b)
    sig = tld.div_rn(tl.full(pre.shape, 1.0, tl.float32), 1.0 + tld.exp(-pre))
    hvec = tl.where(r < N, sig, 2.0 * sig)
    hb = head_ptr + t * (2 * N)
    # Split CTAs keep routing intermediates in registers. Only one CTA writes
    # the shared head/chart, so there are no cross-CTA read/write races.
    owner = tl.program_id(1) == 0
    tl.store(hb + r, hvec, mask=(r < 2 * N) & owner)

    if CHART_IN:
        eb = e_ptr + t * 16
        for i in tl.static_range(4):
            q0 = tl.load(q_ptr + i * 3 + 0)
            q1 = tl.load(q_ptr + i * 3 + 1)
            q2 = tl.load(q_ptr + i * 3 + 2)
            t0 = _mhc_dot3(
                q0,
                q1,
                q2,
                tl.sum(tl.where(r == 8, pre, 0.0), 0),
                tl.sum(tl.where(r == 11, pre, 0.0), 0),
                tl.sum(tl.where(r == 14, pre, 0.0), 0),
                False,
            )
            t1 = _mhc_dot3(
                q0,
                q1,
                q2,
                tl.sum(tl.where(r == 9, pre, 0.0), 0),
                tl.sum(tl.where(r == 12, pre, 0.0), 0),
                tl.sum(tl.where(r == 15, pre, 0.0), 0),
                False,
            )
            t2 = _mhc_dot3(
                q0,
                q1,
                q2,
                tl.sum(tl.where(r == 10, pre, 0.0), 0),
                tl.sum(tl.where(r == 13, pre, 0.0), 0),
                tl.sum(tl.where(r == 16, pre, 0.0), 0),
                False,
            )
            for j in tl.static_range(4):
                ev = _mhc_dot3(
                    t0,
                    t1,
                    t2,
                    tl.load(q_ptr + j * 3 + 0),
                    tl.load(q_ptr + j * 3 + 1),
                    tl.load(q_ptr + j * 3 + 2),
                    CHART_FMA2,
                )
                tl.store(eb + i * 4 + j, ev, mask=owner)

    h0 = tl.sum(tl.where(r == 0, hvec, 0.0), 0)
    h1 = tl.sum(tl.where(r == 1, hvec, 0.0), 0)
    h2 = tl.sum(tl.where(r == 2, hvec, 0.0), 0)
    h3 = tl.sum(tl.where(r == 3, hvec, 0.0), 0)
    if SPLIT:
        c_lo = tl.program_id(1) * BC
        c_hi = c_lo + BC
    else:
        c_lo = 0
        c_hi = C
    for c0 in range(c_lo, c_hi, BC):
        cols = c0 + tl.arange(0, BC)
        s0 = tl.load(v_base + 0 * C + cols).to(tl.float32)
        s1 = tl.load(v_base + 1 * C + cols).to(tl.float32)
        s2 = tl.load(v_base + 2 * C + cols).to(tl.float32)
        s3 = tl.load(v_base + 3 * C + cols).to(tl.float32)
        if AGG_TREE:
            p01 = tld.fma(h1, s1, tld.mul_rn(h0, s0))
            p23 = tld.fma(h3, s3, tld.mul_rn(h2, s2))
            aggv = tld.add_rn(p01, p23)
        else:
            aggv = tld.fma(h3, s3, tld.fma(h2, s2, tld.fma(h1, s1, tld.mul_rn(h0, s0))))
        tl.store(agg_ptr + t * C + cols, aggv.to(agg_ptr.dtype.element_ty))


@torch.no_grad()
def prepare_mhc(site):
    """Refresh packed constants after loading, preserving captured buffer pointers."""
    phi = torch.cat((site.phi_pre, site.phi_post, site.res.phi_z)).detach()
    if phi.dtype != torch.bfloat16 or not phi.is_cuda:
        raise ValueError("Raon FAST mHC requires CUDA BF16 weights")
    values = {
        "fast_alpha": torch.cat(
            (
                site.alpha_pre.float().expand(4),
                site.alpha_post.float().expand(4),
                site.res.alpha_z.float().expand(9),
            )
        ),
        "fast_bias": torch.cat(
            (site.b_pre.float(), site.b_post.float(), site.res.b_z.float())
        ),
        "fast_gamma": torch.sigmoid(site.res.eta.float()),
        "fast_negative_chart": -site.chart,
    }
    values["fast_identity"] = (1 - values["fast_gamma"]) * site.eye
    for width in (17, 18, 20):
        padded = phi.new_zeros((width, phi.shape[1]))
        padded[:17].copy_(phi)
        values[f"fast_phi{width}"] = padded
    for name, value in values.items():
        previous = getattr(site, name, None)
        if previous is None:
            site.register_buffer(name, value, persistent=False)
        else:
            if previous.shape != value.shape or previous.dtype != value.dtype:
                raise ValueError(f"Cannot change captured mHC buffer {name}")
            previous.copy_(value)


def mhc_forward(site, stream, branch):
    """Run packed projection, fused routing/aggregation, and residual mixing."""
    from .phi_gemm import DynMTactic, run_direct_dense_dynm

    count, n, hidden = stream.shape
    if count == 0:
        return stream
    tiers = (
        (8, 256, 1, 1, 17),
        (16, 256, 1, 2, 17),
        (64, 256, 2, 2, 18),
        (128, 128, 2, 4, 18),
        (256, 64, 2, 4, 18),
        (512, 64, 2, 8, 18),
        (float("inf"), 128, 4, 4, 20),
    )
    _, block, outputs, rows, width = next(t for t in tiers if count <= t[0])
    projected = stream.new_empty((count, width))
    run_direct_dense_dynm(
        stream.flatten(1),
        getattr(site, f"fast_phi{width}").T,
        projected,
        DynMTactic(block, outputs, rows),
    )
    head = torch.empty(count, 8, device=stream.device, dtype=torch.float32)
    chart = torch.empty(count, 4, 4, device=stream.device, dtype=torch.float32)
    aggregate = stream.new_empty((count, hidden))
    height = min(1 << (count.bit_length() - 1), 16)
    split = count <= 32
    _mhc_site_head_v2[(count, hidden // 512) if split else (count,)](
        stream,
        projected,
        projected,
        site.fast_alpha,
        site.fast_bias,
        head,
        chart,
        site.helmert,
        aggregate,
        NC=n * hidden,
        C=hidden,
        N=n,
        R=17,
        MMS=width,
        BC=512,
        EPS=site.eps,
        AGG_TREE=count <= 8,
        RMS_IN=True,
        RH=height,
        RW=512 // height,
        FACTOR=1 / (n * hidden),
        CHART_IN=True,
        CHART_FMA2=count >= 5,
        SPLIT=split,
        num_warps=4,
    )
    output = branch(aggregate)
    result = torch.empty_like(stream)
    block_c = 128 if split else 512
    _mhc_site_mix[(count, hidden // block_c)](
        stream,
        output,
        head,
        chart,
        site.chart,
        site.fast_negative_chart,
        site.fast_identity,
        site.fast_gamma,
        result,
        C=hidden,
        N=n,
        BC=block_c,
        MIX_ONE=count == 1,
        num_warps=4,
    )
    return result
