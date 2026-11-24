# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-only references for the gfx942 large-step MoE tests.

Nothing here calls a vLLM or AITER op, so a change in vLLM's kernels or their
precision cannot move the reference.

Two kinds of reference:

- High precision (`*_fp64`): float64 throughout, with no activation
  quantization. The weights are the exact values of the MXFP4 (routed) or
  FP8 (shared) checkpoint tensors, which float64 holds without loss. Tests
  measure vLLM's MoE and the large-step block against these.
- Kernel layout helpers (`pick_groups`, `quant_rows`, `dequant_rows`): the
  exact table and bytes the kernels write, for bit-exact tests of the sort
  and the MXFP8 quantization.
"""

import torch

# DeepSeek-V4.1-Flash's MoE (config.json of deepseek-ai/DeepSeek-V4.1-Flash).
EXPERTS = 384
TOPK = 6
HIDDEN = 5120
LIMIT = 10.0  # swiglu_limit
GROUP = 16  # picks a group in the kernels' table

# 8-value chunk order of the kernels' MXFP8 bytes: even K first, then odd.
EVEN_ODD = [0, 2, 4, 6, 1, 3, 5, 7]

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) + tuple(
    -v for v in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
)


# ---------------------------------------------------------------- weights


def dequant_fp4(w: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 rows [..., K / 2] (low nibble the lower K) and E8M0 codes
    [..., K / 32] -> float64 [..., K]. Exact."""
    lut = torch.tensor(E2M1, dtype=torch.float64, device=w.device)
    lo = (w & 0x0F).long()
    hi = (w >> 4).long()
    vals = lut[torch.stack([lo, hi], -1).flatten(-2)]
    k = vals.shape[-1]
    sc = torch.exp2(codes.double() - 127.0).repeat_interleave(32, -1)[..., :k]
    return vals * sc


def dequant_fp8_blocks(w: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """FP8 weight [N, K] with E8M0 codes per 32 x 32 block ([N / 32, K / 32],
    or [N, K / 32] with each row's block code) -> float64 [N, K]. Exact."""
    n, k = w.shape
    c = codes.double()
    if c.shape[0] != n:
        c = c.repeat_interleave(32, 0)[:n]
    return w.double() * torch.exp2(c - 127.0).repeat_interleave(32, -1)[:, :k]


# ------------------------------------------------------------ fp64 model


def route_fp64(
    x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek-V4 routing in float64: score = sqrt(softplus(x W^T)), the top
    TOPK of score + bias, weights the unbiased scores renormalized to `scale`.

    Returns (ids [S, TOPK] sorted by expert id, weights [S, TOPK] in the same
    order, margin [S]): the margin is the gap between the 6th and 7th biased
    score. A row with a margin near zero is a tie, where any implementation
    may pick either expert."""
    logits = x.double() @ gate_w.double().T
    s = torch.sqrt(torch.nn.functional.softplus(logits))
    biased = s + bias.double()
    top = torch.topk(biased, TOPK + 1, dim=-1)
    ids = top.indices[:, :TOPK]
    margin = top.values[:, TOPK - 1] - top.values[:, TOPK]
    w = torch.gather(s, 1, ids)
    w = w / w.sum(-1, keepdim=True) * scale
    order = ids.argsort(-1)
    return ids.gather(1, order), w.gather(1, order), margin


def swiglu_fp64(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """silu(min(g, LIMIT)) * clamp(u, -LIMIT, LIMIT)."""
    g = g.clamp(max=LIMIT)
    return g * torch.sigmoid(g) * u.clamp(-LIMIT, LIMIT)


def stage1_fp64(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_s: torch.Tensor,
    topk_ids: torch.Tensor,
    inter: int,
) -> torch.Tensor:
    """Routed stage 1 of every pick p = token * TOPK + k: swiglu(W_gate x,
    W_up x) of expert topk_ids[token, k] -> float64 [S * TOPK, inter].

    x: [S, H], any float dtype (taken as is). w13 / w13_s: vLLM's plain loaded
    tensors [E, 2 Np, H / 2] and [E, 2 Np, H / 32] (gate rows, then up rows,
    each padded to Np)."""
    s, k = topk_ids.shape
    xd = x.double()
    np_ = w13.shape[1] // 2
    out = torch.empty(s * k, inter, dtype=torch.float64, device=x.device)
    flat = topk_ids.reshape(-1)
    for e in flat.unique().tolist():
        gate = dequant_fp4(w13[e, :inter], w13_s[e, :inter])
        up = dequant_fp4(w13[e, np_ : np_ + inter], w13_s[e, np_ : np_ + inter])
        picks = (flat == e).nonzero().flatten()
        xt = xd[picks // k]
        out[picks] = swiglu_fp64(xt @ gate.T, xt @ up.T)
    return out


def stage2_fp64(
    h: torch.Tensor,
    w2: torch.Tensor,
    w2_s: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    inter: int,
) -> torch.Tensor:
    """Routed stage 2 of every pick: routing weight x (W_down h) -> float64
    [S * TOPK, H]. h: [S * TOPK, inter], any float dtype (taken as is).
    w2 / w2_s: vLLM's plain tensors [E, H, Np / 2] and [E, H, Np / 32]."""
    hd = h.double()
    flat = topk_ids.reshape(-1)
    wts = topk_w.reshape(-1).double()
    out = torch.empty(h.shape[0], w2.shape[1], dtype=torch.float64, device=h.device)
    for e in flat.unique().tolist():
        down = dequant_fp4(w2[e, :, : inter // 2], w2_s[e, :, : inter // 32])
        picks = (flat == e).nonzero().flatten()
        out[picks] = wts[picks, None] * (hd[picks] @ down.T)
    return out


def routed_fp64(x, w13, w13_s, w2, w2_s, topk_ids, topk_w, inter) -> torch.Tensor:
    """One TP rank's routed MoE output in float64, no activation
    quantization -> [S, H]."""
    h = stage1_fp64(x, w13, w13_s, topk_ids, inter)
    y = stage2_fp64(h, w2, w2_s, topk_ids, topk_w, inter)
    return y.view(x.shape[0], TOPK, -1).sum(1)


def shared_fp64(
    x: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    """One TP rank's shared expert in float64: W_down swiglu(W_gate x, W_up x).
    w_gate_up: float64 [2 I, H] (gate rows, then up rows); w_down: [H, I]."""
    gu = x.double() @ w_gate_up.T
    i = gu.shape[-1] // 2
    return swiglu_fp64(gu[:, :i], gu[:, i:]) @ w_down.T


# ------------------------------------------------- kernel layout helpers


def pick_groups(topk_ids: torch.Tensor, experts: int) -> tuple[torch.Tensor, int]:
    """The step's picks in groups of GROUP picks of one expert, as the pick
    sort writes them.

    topk_ids: int [S, K]. Returns (table int32 [G, 4 + GROUP], G). Row g:
    expert, 0, 0, 0, then the GROUP picks p = token * K + k of the group, -1
    past the expert's last pick. Groups are expert-major; an expert's picks
    keep their order."""
    s, k = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat * (s * k) + torch.arange(s * k, device=flat.device))
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
    assert start == flat.numel()
    table = torch.stack(rows).to(topk_ids.device)
    return table, table.shape[0]


def code_ceil(amax: torch.Tensor) -> torch.Tensor:
    """The smallest E8M0 code whose power of two is >= amax / 448, clamped to
    [1, 254]."""
    q = amax.float() / 448.0
    bits = q.view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return e.clamp(1, 254)


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 rows [S, K] -> the kernels' MXFP8: (FNUZ e4m3 bytes uint8 [S, K],
    every 8-value chunk even K first; FNUZ codes uint8 [S, K / 32]).

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
