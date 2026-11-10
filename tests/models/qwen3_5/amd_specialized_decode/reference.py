# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Float64 references of the wide decode's stages with the stock path's bf16
rounding points (``bf``), at one TP8 rank's Qwen3.8-2.4T shapes. Tensors stay
on the device they come on."""

import torch
import torch.nn.functional as F

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = torch.tensor(_E2M1 + [-v for v in _E2M1])


def bf(t):
    return t.to(torch.bfloat16).double()


def mxfp4(n, k, gen, lo=118, hi=124, device="cpu"):
    """Random e2m1 codes and E8M0 scales: (packed [n, k / 2] uint8, scales
    [n, k / 32] uint8)."""
    codes = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.uint8, device=device)
    sc = torch.randint(lo, hi, (n, k // 32), generator=gen, dtype=torch.uint8, device=device)
    return codes[..., 0::2] | (codes[..., 1::2] << 4), sc


def shuffle_data(packed):
    """AITER's ``shuffle_weight(layout=(16, 16))`` of fp4 codes [.., n, k / 2]."""
    *lead, n, kb = packed.shape
    v = packed.reshape(-1, n // 16, 16, kb // 16, 16).permute(0, 1, 3, 2, 4)
    return v.contiguous().view(*lead, n, kb)


def shuffle_scale(sc):
    """AITER's ``e8m0_shuffle`` of [rows, cols] scales (cols padded to 8)."""
    rows, cols = sc.shape
    pad = (cols + 7) // 8 * 8
    if pad != cols:
        fill = torch.full((rows, pad - cols), 127, dtype=torch.uint8, device=sc.device)
        sc = torch.cat([sc, fill], 1)
    return (
        sc.view(rows // 32, 2, 16, pad // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view(rows, pad)
    )


def dequant_mx(q, s):
    """Dequantized float64 [.., k] of packed e2m1 codes ``q`` [.., k / 2] and
    E8M0 scales ``s`` [.., >= k / 32]."""
    q = q.long()
    lut = E2M1.to(q.device, torch.float64)
    vals = torch.stack([lut[q & 15], lut[q >> 4]], -1).flatten(-2)
    k = vals.shape[-1]
    scale = torch.pow(2.0, s[..., : k // 32].double() - 127)
    return vals * scale.repeat_interleave(32, -1)


def hadamard32(device):
    h = torch.ones(1, 1, dtype=torch.float64, device=device)
    while h.shape[0] < 32:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / 32**0.5


def rotate32(x):
    """``block_hadamard_rotate`` of bf16-valued ``x``: f32 math, bf16 out."""
    h = hadamard32(x.device).float()
    return bf((x.float().unflatten(-1, (-1, 32)) @ h).flatten(-2))


def quant_act(x):
    """AITER's dynamic MXFP4 quantization of activation rows -> dequantized."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    q, s = dynamic_mxfp4_quant(x.to(torch.bfloat16).contiguous())
    return dequant_mx(q, s)


def linear(x, w, fmt: str):
    """``x @ W.T`` with the stock rounding: ``w`` a bf16 weight, or for
    "mxfp4" / "mxfp4_h32" the dequantized float64 weight (of the rotated
    weight for h32); the input quantized (after its rotation) as AITER does.
    Returns bf16-rounded float64."""
    if fmt == "bf16":
        return bf(x.double() @ w.double().T)
    if fmt == "mxfp4_h32":
        x = rotate32(x)
    return bf(quant_act(x) @ w.T)


def gemma_norm(z, w, eps):
    """``GemmaRMSNorm`` of bf16-valued rows ``z``: (1 + w) weights."""
    rr = torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)
    return bf(z * rr * (1 + w.double()))


def gdn(x, w, st_idx, conv_state, ssm, eps: float, fmts=("bf16", "bf16")):
    """The GDN block of rows ``x`` with state slots ``st_idx`` (<= 0: none).
    ``w``: qkvz, ba, conv, a_log, dt_bias, norm, out (dequantized float64
    for MXFP4 formats ``fmts`` = (in_proj_qkvz's, out_proj's)). Returns
    (conv output q | k | v, z, new conv states, new recurrent states, core
    [rows, NV * 128], out_proj output)."""
    rows = x.shape[0]
    nv = w["a_log"].numel()
    dk = 128
    key_dim = (w["qkvz"].shape[0] - 2 * nv * dk) // 2
    nk = key_dim // dk
    conv_dim = 2 * key_dim + nv * dk
    qkvz = linear(x, w["qkvz"], fmts[0])
    bav = bf(x.double() @ w["ba"].double().T)
    mixed, z = qkvz[:, :conv_dim], qkvz[:, conv_dim:]
    conv_state = conv_state.double().clone()
    ssm = ssm.double().clone()
    out = torch.zeros(rows, conv_dim, dtype=torch.float64, device=x.device)
    wconv = w["conv"].double().view(conv_dim, 4)
    for r in range(rows):
        sl = int(st_idx[r])
        s = (
            conv_state[sl]
            if sl > 0
            else torch.zeros(3, conv_dim, dtype=torch.float64, device=x.device)
        )
        win = torch.cat([s, mixed[r : r + 1]], 0)
        out[r] = bf(F.silu((win.T * wconv).sum(1)))
        if sl > 0:
            conv_state[sl] = torch.cat([s[1:], mixed[r : r + 1]], 0)
    q = out[:, :key_dim].view(rows, nk, dk)
    k = out[:, key_dim : 2 * key_dim].view(rows, nk, dk)
    v = out[:, 2 * key_dim :].view(rows, nv, dk)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * dk**-0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(bav[:, :nv])
    g = -torch.exp(w["a_log"].double()) * F.softplus(bav[:, nv:] + w["dt_bias"].double())
    core = torch.zeros(rows, nv, dk, dtype=torch.float64, device=x.device)
    per_k = nv // nk
    zz = z.view(rows, nv, dk)
    for r in range(rows):
        sl = int(st_idx[r])
        if sl <= 0:
            continue
        kh = torch.arange(nv, device=x.device) // per_k
        h = ssm[sl] * torch.exp(g[r])[:, None, None]  # [NV, V, K]
        kk, qq = k[r, kh], q[r, kh]  # [NV, K]
        dv = (v[r] - (h @ kk[..., None])[..., 0]) * beta[r][:, None]
        h = h + dv[..., None] * kk[:, None, :]
        o = bf((h @ qq[..., None])[..., 0])  # [NV, V]
        ssm[sl] = h
        rr = torch.rsqrt((o * o).mean(-1, keepdim=True) + eps)
        core[r] = bf(o * rr * w["norm"].double() * F.silu(zz[r]))
    core = core.view(rows, nv * dk)
    return out, z, conv_state, ssm, core, linear(core, w["out"], fmts[1])


def route(x, w_gate, experts: int, topk: int, prefer=None, margin: float = 1e-2):
    """The fused router's logits (bf16) -> (expert ids [rows, topk + 1], f64
    weights): softmax top-k of the routed experts, then the shared expert
    (index ``experts``) at sigmoid of its logit. ``prefer``: a kernel's
    picks [rows, topk], taken for a row when they are a top-k within
    ``margin`` (relative to the row's largest logit) of the boundary, as
    inputs a few bf16 ulps apart flip near-tied experts."""
    rows = x.shape[0]
    logits = bf(x.double() @ w_gate.double().T)
    routed = logits[:, :experts]
    top_v, top_i = torch.topk(routed, topk, dim=-1)
    if prefer is not None:
        prefer = prefer.long().to(x.device)
        tol = margin * routed.abs().amax(-1)
        picked = routed.gather(1, prefer)
        ok = (
            (prefer < experts).all(-1)
            & (prefer.sort(-1).values.diff(dim=-1) > 0).all(-1)
            & (picked.amin(-1) >= top_v[:, -1] - tol)
        )
        top_i = torch.where(ok[:, None], prefer, top_i)
        top_v = routed.gather(1, top_i)
    shared = torch.full((rows, 1), experts, device=x.device)
    ids = torch.cat([top_i, shared], 1)
    ws = torch.cat(
        [torch.softmax(top_v, -1), torch.sigmoid(logits[:, experts])[:, None]], 1
    )
    return ids, ws, logits


def moe(x, w_gate, w13, w13s, w2, w2s, experts: int, topk: int, ri: int, rip: int,
        prefer=None, xd=None):
    """The MoE of rows ``x`` (one rank's intermediate slice; unshuffled MXFP4
    expert codes and scales), activations quantized to MXFP4 as AITER does
    (the input and the gate-up product). Returns (y f64, ids, weights);
    ``prefer``: as ``route``; ``xd``: the experts' dequantized input rows
    (default ``quant_act(x)``)."""
    ids, ws, _ = route(x, w_gate, experts, topk, prefer)
    if xd is None:
        xd = quant_act(x)
    y = torch.zeros_like(xd)
    for e in ids.unique().tolist():
        w13d = dequant_mx(w13[e], w13s[e])
        w2d = dequant_mx(w2[e], w2s[e])[:, :ri]
        for b, j in (ids == e).nonzero().tolist():
            gu = w13d @ xd[b]
            act = F.silu(gu[:ri]) * gu[rip : rip + ri]
            y[b] += ws[b, j] * (w2d @ quant_act(act[None].float())[0])
    return y, ids, ws
