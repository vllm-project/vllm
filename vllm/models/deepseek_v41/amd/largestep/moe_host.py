# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side reference pieces of the gfx942 large-step MoE, for the tests:
the pick groups (`pick_groups`, the table `moe_sort` writes), the
activation quantization the kernels use (`quant_rows`), and torch
references of the routed and shared experts.
"""

import torch

GROUP = 16  # picks a group: one MFMA's N
HIDDEN = 5120
LIMIT = 10.0  # DeepSeek-V4.1's swiglu_limit
EXPERTS, TOPK = 384, 6  # the target model's routing
# 8-value chunk orders: even K first, then odd (`weights942.EVEN_ODD`)
EVEN_ODD = [0, 2, 4, 6, 1, 3, 5, 7]


def pick_groups(topk_ids: torch.Tensor, experts: int) -> tuple[torch.Tensor, int]:
    """The decode step's picks in groups of GROUP picks of one expert.

    topk_ids: int [S, K], the experts each token picked.
    Returns (table int32 [G, 4 + GROUP], G). Row g: expert, 0, 0, 0, then
    the GROUP picks p = token * K + k of the group, -1 past the expert's
    last pick. Groups are expert-major; an expert's picks keep their order."""
    s, k = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat * (s * k) + torch.arange(s * k, device=flat.device))
    ex = flat[order]
    counts = torch.bincount(flat, minlength=experts)
    rows = []
    start = 0
    for e in torch.nonzero(counts).flatten().tolist():
        n = int(counts[e])
        picks = order[start : start + n]
        start += n
        for g0 in range(0, n, GROUP):
            row = torch.full((4 + GROUP,), -1, dtype=torch.int32)
            row[:4] = torch.tensor([e, 0, 0, 0], dtype=torch.int32)
            chunk = picks[g0 : g0 + GROUP]
            row[4 : 4 + chunk.numel()] = chunk.to(torch.int32).cpu()
            rows.append(row)
    assert start == ex.numel()
    table = torch.stack(rows).to(topk_ids.device)
    return table, table.shape[0]


def code_ceil(amax: torch.Tensor) -> torch.Tensor:
    """The smallest E8M0 code whose power of two is >= amax / 448, clamped
    to [1, 254] (`mono.common.mx.code_ceil` on amax / FP8_MAX)."""
    q = amax.float() / 448.0
    bits = q.view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return e.clamp(1, 254)


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 rows [S, H] -> (FNUZ e4m3 bytes uint8 [S, H] with every 8-value
    chunk even K first, FNUZ codes uint8 [S, H / 32]).

    The OCP bytes for x / 2^(code - 127) are the FNUZ bytes for half that
    value, so the FNUZ code is the OCP code plus one."""
    s, h = x.shape
    v = x.float().view(s, h // 32, 32)
    amax = v.abs().amax(-1).clamp_min(1e-10)
    code = code_ceil(amax)
    scale = torch.exp2(code.float() - 127.0)
    q = (v / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    b = q.view(torch.uint8).view(s, h)
    b = torch.where(b == 0x80, torch.zeros_like(b), b)
    b = b.view(s, h // 8, 8)[..., EVEN_ODD].reshape(s, h)
    return b.contiguous(), (code + 1).to(torch.uint8)


def dequant_rows(b: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """`quant_rows`' output back to float32 values in natural K order."""
    s, h = b.shape
    inv = torch.argsort(torch.tensor(EVEN_ODD))
    nat = b.view(s, h // 8, 8)[..., inv].reshape(s, h)
    vals = nat.view(torch.float8_e4m3fnuz).float().view(s, h // 32, 32)
    return (vals * torch.exp2(codes.float() - 127.0)[..., None]).view(s, h)


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


def dequant_fp4(w: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Plain packed e2m1 rows [..., K / 2] (low nibble the lower K) and E8M0
    codes [..., K / 32] -> float32 [..., K]."""
    lo = (w & 0x0F).long()
    hi = (w >> 4).long()
    vals = E2M1.to(w.device)[torch.stack([lo, hi], -1).flatten(-2)]
    k = vals.shape[-1]
    sc = torch.exp2(codes.float() - 127.0).repeat_interleave(32, -1)[..., :k]
    return vals * sc


def ref_stage1(
    xb: torch.Tensor,
    xs: torch.Tensor,
    w13: torch.Tensor,
    w13_s: torch.Tensor,
    topk_ids: torch.Tensor,
    inter: int,
) -> torch.Tensor:
    """Stage 1 in float32: for pick p = t * K + k, expert e = topk_ids[t, k],
    g = W13_gate[e] x_t and u = W13_up[e] x_t over the quantized operands,
    then bf16(silu(min(g, 10)) * clamp(u, -10, 10)) -> float32 [P, inter].

    w13 / w13_s are the plain tensors vLLM loads (gate rows, then up rows,
    each padded to Np): [E, 2 Np, H / 2] and [E, 2 Np, H / 32]."""
    s, k = topk_ids.shape
    x = dequant_rows(xb, xs)
    np_ = w13.shape[1] // 2
    out = torch.empty(s * k, inter, dtype=torch.float32, device=x.device)
    for e in topk_ids.unique().tolist():
        gate = dequant_fp4(w13[e, :inter], w13_s[e, :inter])
        up = dequant_fp4(w13[e, np_ : np_ + inter], w13_s[e, np_ : np_ + inter])
        for p in (topk_ids.reshape(-1) == e).nonzero().flatten().tolist():
            xt = x[p // k]
            g, u = gate @ xt, up @ xt
            g = g.clamp(max=LIMIT)
            u = u.clamp(-LIMIT, LIMIT)
            out[p] = (g * torch.sigmoid(g) * u).to(torch.bfloat16).float()
    return out


def ref_routed(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_s: torch.Tensor,
    w2: torch.Tensor,
    w2_s: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    inter: int,
) -> torch.Tensor:
    """The routed MoE of one TP rank in float32 with the kernels' rounding
    points: x to MXFP8 (`quant_rows`), stage 1 (`ref_stage1`, bf16 h),
    h to MXFP8, down projection, times the routing weight, bf16 a pick, then
    the token's 6 picks summed in f32 in top-k order and rounded to bf16.
    -> float32 [S, H]. w2 / w2_s are vLLM's plain tensors [E, H, Np / 2] and
    [E, H, Np / 32]."""
    s, k = topk_ids.shape
    xb, xs = quant_rows(x)
    h = ref_stage1(xb, xs, w13, w13_s, topk_ids, inter)
    hb, hs = quant_rows(h.to(torch.bfloat16))
    hq = dequant_rows(hb, hs)
    y = torch.empty(s * k, x.shape[1], dtype=torch.float32, device=x.device)
    for e in topk_ids.unique().tolist():
        down = dequant_fp4(w2[e, :, : inter // 2], w2_s[e, :, : inter // 32])
        for p in (topk_ids.reshape(-1) == e).nonzero().flatten().tolist():
            y[p] = (topk_w.reshape(-1)[p] * (down @ hq[p])).to(torch.bfloat16).float()
    return y.view(s, k, -1).sum(1).to(torch.bfloat16).float()


def quant_rows_natural(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`quant_rows` with natural chunk order: the dense (shared expert) copies
    keep K in order (`weights942.linear_copy`)."""
    s, h = x.shape
    b, codes = quant_rows(x)
    inv = torch.argsort(torch.tensor(EVEN_ODD))
    return b.view(s, h // 8, 8)[..., inv].reshape(s, h).contiguous(), codes


def dequant_mxfp8_blocks(w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """OCP e4m3 bytes [N, K] with E8M0 codes a 32 x 32 block [N / 32, K / 32]
    -> float32 [N, K]."""
    n, k = w8.shape
    sc = torch.exp2(scale.float() - 127.0)
    sc = sc.repeat_interleave(32, 0).repeat_interleave(32, 1)
    return w8.view(torch.float8_e4m3fn).float() * sc


def ref_shared(
    x: torch.Tensor,
    gu8: torch.Tensor,
    gu_s: torch.Tensor,
    d8: torch.Tensor,
    d_s: torch.Tensor,
) -> torch.Tensor:
    """The shared expert with the kernels' rounding points: x to MXFP8, gate
    and up rounded to bf16, vLLM's silu_and_mul_with_clamp (gate <= 10, silu
    rounded to bf16, up in [-10, 10]), bf16, to MXFP8, down, bf16.
    gu8 [2 I, H] / d8 [H, I] are OCP e4m3 bytes with 32 x 32 block codes."""
    i2 = gu8.shape[0]
    xb, xs = quant_rows_natural(x)
    xq = dequant_rows_natural(xb, xs)
    gu = (xq @ dequant_mxfp8_blocks(gu8, gu_s).T).to(torch.bfloat16).float()
    g, u = gu[:, : i2 // 2], gu[:, i2 // 2 :]
    g = g.clamp(max=LIMIT).to(torch.bfloat16).float()
    act = (g * torch.sigmoid(g)).to(torch.bfloat16).float()
    sh = (act * u.clamp(-LIMIT, LIMIT)).to(torch.bfloat16)
    hb, hs = quant_rows_natural(sh)
    hq = dequant_rows_natural(hb, hs)
    return (hq @ dequant_mxfp8_blocks(d8, d_s).T).to(torch.bfloat16).float()


def dequant_rows_natural(b: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """`quant_rows_natural`' output back to float32."""
    s, h = b.shape
    vals = b.view(torch.float8_e4m3fnuz).float().view(s, h // 32, 32)
    return (vals * torch.exp2(codes.float() - 127.0)[..., None]).view(s, h)
