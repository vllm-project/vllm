# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Float64 references of the wide decode's stages with the stock path's bf16
rounding points (`bf`), at Qwen3.8-Flash-Next's TP1 or TP2 shapes."""

import torch
import torch.nn.functional as F

E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)


def bf(t):
    return t.to(torch.bfloat16).double()


def mxfp4(n, k, gen, lo=118, hi=124):
    """Random e2m1 codes and E8M0 scales: (packed [n, k / 2], scales [n, k /
    32], the dequantized float32 [n, k])."""
    codes = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.uint8)
    sc = torch.randint(lo, hi, (n, k // 32), generator=gen, dtype=torch.uint8)
    deq = E2M1[codes.long()] * torch.pow(2.0, sc.float() - 127).repeat_interleave(32, 1)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), sc, deq


def shuffle_data(packed):
    """AITER's `shuffle_weight(layout=(16, 16))` of fp4 codes [.., n, k / 2]."""
    *lead, n, kb = packed.shape
    v = packed.reshape(-1, n // 16, 16, kb // 16, 16).permute(0, 1, 3, 2, 4)
    return v.contiguous().view(*lead, n, kb)


def shuffle_scale(sc):
    """AITER's `e8m0_shuffle` of [rows, cols] scales (cols padded to 8)."""
    rows, cols = sc.shape
    pad = (cols + 7) // 8 * 8
    if pad != cols:
        sc = torch.cat([sc, torch.full((rows, pad - cols), 127, dtype=torch.uint8)], 1)
    return (
        sc.view(rows // 32, 2, 16, pad // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view(rows, pad)
    )


def dequant_mx(q, s):
    """Dequantized [.., k] of packed e2m1 codes `q` [.., k / 2] and E8M0
    scales `s` [.., >= k / 32]."""
    q = q.cpu().long()
    vals = torch.stack([E2M1[q & 15], E2M1[q >> 4]], -1).flatten(-2)
    return (
        vals
        * torch.pow(2.0, s.cpu().float() - 127).repeat_interleave(32, -1)[
            ..., : vals.shape[-1]
        ]
    )


def gated_residual(state, norm_w, w_di, w_up, y, hc: int, lowrank: int, eps: float):
    """The gated residual's mix of `state` [rows, hc * hidden] and the
    combine of a block output `y` [rows, hidden]: (mixed x, injection
    scales, new state, rrms of state, rrms of the new state)."""
    rows = state.shape[0]
    hidden = y.shape[1]
    x = state.double().view(rows, hc, hidden)
    rrms = torch.rsqrt(x.pow(2).mean(-1) + eps)
    yn = x * rrms[..., None]
    xn = bf(yn + yn * norm_w.double().view(hc, hidden)).view(rows, hc * hidden)
    dn = bf(xn @ w_di.double().T)
    low = bf(F.silu(dn[:, :lowrank] / hc))
    inj = 2 * torch.sigmoid(dn[:, lowrank : lowrank + hc] / hc)
    gate = bf(low @ w_up.double().T)
    mixed = bf((torch.sigmoid(gate) * xn).view(rows, hc, hidden).sum(1) / hc)
    new = bf(x + bf(y.double())[:, None, :] * inj[..., None]).view(rows, hc * hidden)
    rrms2 = torch.rsqrt(new.view(rows, hc, hidden).pow(2).mean(-1) + eps)
    return mixed, inj, new, rrms, rrms2


def gated_delta_net(
    x, wq, wba, wconv, a_log, dt_bias, normw, wo, st_idx, conv_state, ssm, eps: float
):
    """The GDN block of rows `x` with state slots `st_idx` (<= 0: none):
    (conv output q | k | v, z, new conv states, new recurrent states, core
    [rows, NV * 128], out_proj output). Dims from the weights: `wq` [q | k |
    v | z rows, hidden], `wba` [b | a rows, hidden]."""
    rows = x.shape[0]
    nv = a_log.numel()
    dk = 128
    key_dim = (wq.shape[0] - 2 * nv * dk) // 2
    nk = key_dim // dk
    conv_dim = 2 * key_dim + nv * dk
    qkvz = bf(x.double() @ wq.double().T)
    bav = bf(x.double() @ wba.double().T)
    mixed, z = qkvz[:, :conv_dim], qkvz[:, conv_dim:]
    conv_state = conv_state.double().clone()
    ssm = ssm.double().clone()
    out = torch.zeros(rows, conv_dim, dtype=torch.float64)
    for r in range(rows):
        sl = int(st_idx[r])
        s = conv_state[sl] if sl > 0 else torch.zeros(3, conv_dim, dtype=torch.float64)
        win = torch.cat([s, mixed[r : r + 1]], 0)
        out[r] = bf(F.silu((win.T * wconv.double().view(conv_dim, 4)).sum(1)))
        if sl > 0:
            conv_state[sl] = torch.cat([s[1:], mixed[r : r + 1]], 0)
    q = out[:, :key_dim].view(rows, nk, dk)
    k = out[:, key_dim : 2 * key_dim].view(rows, nk, dk)
    v = out[:, 2 * key_dim :].view(rows, nv, dk)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * dk**-0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(bav[:, :nv])
    g = -torch.exp(a_log.double()) * F.softplus(bav[:, nv:] + dt_bias.double())
    core = torch.zeros(rows, nv, dk, dtype=torch.float64)
    per_k = nv // nk
    for r in range(rows):
        sl = int(st_idx[r])
        if sl <= 0:
            continue
        for vh in range(nv):
            kh = vh // per_k
            h = ssm[sl, vh] * torch.exp(g[r, vh])  # [V, K]
            dv = (v[r, vh] - h @ k[r, kh]) * beta[r, vh]
            h = h + torch.outer(dv, k[r, kh])
            o = bf(h @ q[r, kh])
            ssm[sl, vh] = h
            gate = torch.sigmoid(z[r].view(nv, dk)[vh])
            core[r, vh] = bf(
                o * torch.rsqrt((o * o).mean() + eps) * normw.double() * gate
            )
    core = core.view(rows, nv * dk)
    return out, z, conv_state, ssm, core, bf(core @ wo.double().T)


def moe(x, w_gate, w13d, w2d, experts: int, topk: int, ri: int, rip: int):
    """The MoE of rows `x`: softmax top-k over the bf16 router logits of the
    routed experts plus the shared expert (index `experts`, sigmoid
    weight), activations quantized to MXFP4 as AITER does (input and the
    gate-up product). Returns (y f64, expert ids, weights)."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    rows, hidden = x.shape
    logits = bf(x.double() @ w_gate.double().T)
    top_v, top_i = torch.topk(logits[:, :experts], topk, dim=-1)
    ids = torch.cat([top_i, torch.full((rows, 1), experts)], 1)
    ws = torch.cat(
        [torch.softmax(top_v, -1), torch.sigmoid(logits[:, experts])[:, None]], 1
    )
    xq, xs = dynamic_mxfp4_quant(x.cuda())
    xd = dequant_mx(xq, xs).double()
    y = torch.zeros(rows, hidden, dtype=torch.float64)
    for b in range(rows):
        for j in range(topk + 1):
            e = int(ids[b, j])
            gu = w13d[e].double() @ xd[b]
            act = (F.silu(gu[:ri]) * gu[rip : rip + ri]).float()
            aq, as_ = dynamic_mxfp4_quant(act[None].cuda())
            y[b] += ws[b, j] * (
                w2d[e][:, :ri].double() @ dequant_mx(aq, as_)[0].double()
            )
    return y, ids, ws


def int4_roundtrip(y):
    """The TP exchange's INT4 line codec on `y` [rows, cols] (f32 math as
    the kernels do): groups of 16 columns, `scale = bf16(extremum / -8)`,
    codes `roundeven(y * (1 / scale))` clamped to [-8, 7]."""
    v = y.float().view(y.shape[0], -1, 16)
    hi, lo = v.amax(-1, keepdim=True), v.amin(-1, keepdim=True)
    ext = torch.where(hi > -lo, hi, lo)
    scale = (ext * -0.125).to(torch.bfloat16).float()
    inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    q = torch.round(v * inv).clamp(-8, 7)
    return (q * scale).view_as(y)


def int4_lines(words, rows: int, groups: int, lpg: int, hidden: int):
    """Decode an exchange row of INT4 lines (i32 `words`) to [rows, hidden]
    f32: a line's 64 code dwords (token `t`, segment `s` at `8 t + s`),
    then each token's two scale dwords (bf16 pairs of its four groups)."""
    lines = words.view(groups, lpg, 80).cpu().long() & 0xFFFFFFFF
    codes = lines[..., :64].view(groups, lpg, 8, 8)  # [g, blk, token, seg]
    nib = torch.stack([(codes >> (4 * e)) & 15 for e in range(8)], -1) - 8
    sc = lines[..., 64:].view(groups, lpg, 8, 2)
    halves = torch.stack([sc & 0xFFFF, sc >> 16], -1).view(groups, lpg, 8, 4)
    scale = (halves << 16).to(torch.int32).view(torch.float32)  # [g, blk, token, group]
    vals = nib.float() * scale.repeat_interleave(2, -1)[..., None]
    out = vals.view(groups, lpg, 8, 64).permute(1, 2, 0, 3).reshape(lpg * 8, hidden)
    return out[:rows]
