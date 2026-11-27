# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp64 references of the GLM-5.3-Flash decode layer, in plain torch.

No vLLM op and no activation quantization unless asked for: each function is the
model's math at float64 on the inputs it is given. The tests feed every stage the
MonoKernel's own input to that stage, so an error measured at a stage is that
stage's own, and run vLLM's ops on the same inputs for the baseline.

Errors are relative L2 (`rel`): ||got - ref|| / ||ref|| over the whole tensor. One
bf16 rounding of a tensor is about 0.0016 of it (8-bit significand), so a stage
that rounds to bf16 a few times sits at a few 1e-3."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

F64 = torch.float64
_FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


@dataclass(frozen=True)
class Bound:
    """A stage's bound: the MonoKernel's error to the fp64 reference is below
    `ref`, and, when `vllm` is set, at most `vllm` times vLLM's own error."""

    ref: float
    vllm: float | None = None


def rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative L2 error of `got` to `ref`, in fp64."""
    a, b = got.to(F64).reshape(-1), ref.to(F64).reshape(-1)
    return ((a - b).norm() / b.norm()).item()


def dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """MXFP4 nibbles `q` (two a byte, low first) with E8M0 scales `s` per 32."""
    fp4 = torch.tensor(_FP4, device=q.device, dtype=F64)
    v = torch.stack([fp4[(q & 15).long()], fp4[(q >> 4).long()]], -1).flatten(-2)
    return v * torch.exp2(s.to(F64) - 127).repeat_interleave(32, -1)


def mxfp8(a: torch.Tensor) -> torch.Tensor:
    """The MonoKernel's expert-input rounding (AITER's W4A8) of fp32 values: per 32
    an E8M0 scale, amax / 448 rounded up to a power of two, values to E4M3."""
    shp, a = a.shape, a.float().reshape(-1, 32)
    bits = (a.abs().amax(-1, keepdim=True) / 448.0).view(torch.int32)
    e = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()).clamp(1, 254)
    scale = torch.exp2(e.to(F64) - 127)
    q = (a / scale.float()).clamp(-448, 448).to(torch.float8_e4m3fn).to(F64)
    return (q * scale).reshape(shp)


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.to(F64)
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.to(F64)


def mhc_post(x, res, post, comb) -> torch.Tensor:
    """The mHC post: R' = comb^T res + post * x, per token. x [N, H], res [N, 4, H],
    post [N, 4, 1], comb [N, 4, 4]."""
    mixed = torch.einsum("nij,nih->njh", comb.to(F64), res.to(F64))
    return mixed + post.to(F64) * x.to(F64).unsqueeze(-2)


def mhc_pre(r, fn, scale, base, gamma):
    """The mHC pre and the layer's input RMSNorm on the residual streams r [N, 4, H]:
    (post [N, 4, 1], comb [N, 4, 4], normed layer input [N, H])."""
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        EPS,
        HC_EPS,
        HC_MULT,
        HC_POST_MULT,
        SINKHORN_ITERS,
    )

    m, r = HC_MULT, r.to(F64)
    x, scale, base = r.flatten(1), scale.to(F64), base.to(F64)
    mixes = (x @ fn.to(F64).T) * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    pre = torch.sigmoid(mixes[:, :m] * scale[0] + base[:m]) + HC_EPS
    post = torch.sigmoid(mixes[:, m : 2 * m] * scale[1] + base[m : 2 * m])
    comb = mixes[:, 2 * m :].view(-1, m, m) * scale[2] + base[2 * m :].view(1, m, m)
    comb = torch.softmax(comb, -1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    li = (pre.unsqueeze(-1) * r).sum(1)
    return (post * HC_POST_MULT).unsqueeze(-1), comb, _rms_norm(li, gamma, EPS)


def route(
    x: torch.Tensor, w_r: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The router on x [N, H]: the top-8 expert ids [N, 8] of the sigmoid scores
    plus bias, and each row's margin between its 8th and 9th biased score (how
    close the 8th pick was to a tie)."""
    from vllm.models.glm5next.amd.specialized_decode.mono.config import TOP_K

    biased = torch.sigmoid(x.to(F64) @ w_r.to(F64).T) + bias.to(F64)
    order = torch.argsort(-biased, dim=-1, stable=True)[:, : TOP_K + 1]
    top = torch.gather(biased, 1, order)
    return order[:, :TOP_K], top[:, TOP_K - 1] - top[:, TOP_K]


def moe(
    x: torch.Tensor,
    w_r: torch.Tensor,
    bias: torch.Tensor,
    expert: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    shared: int,
    order: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The MoE on x [N, H]: sigmoid router scores, the bias only to pick the top 8,
    their unbiased scores renormalized and x2.5 as weights, the shared expert
    `shared` at weight 1, SwiGLU with the gate clamped above and up both sides.

    `expert(e)` gives expert e's fp64 (gate_up [2n, H], down [H, n]); `order`
    [N, 8], experts to run instead of the top 8 (a run's own routing; their
    scores still weigh them). Returns (out with exact expert inputs, out with
    both expert GEMMs' inputs rounded as `mxfp8`, the expert ids [N, 8])."""
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        ROUTE_SCALE,
        SWIGLU_LIMIT,
    )

    x = x.to(F64)
    scores = torch.sigmoid(x @ w_r.to(F64).T)
    if order is None:
        order = route(x, w_r, bias)[0]
    order = order.long()
    picked = torch.gather(scores, 1, order)
    weights = picked / picked.sum(-1, keepdim=True) * ROUTE_SCALE
    users: dict[int, list[tuple[int, float]]] = {shared: []}
    for n in range(x.shape[0]):
        users[shared].append((n, 1.0))
        for e, wt in zip(order[n].tolist(), weights[n].tolist()):
            users.setdefault(e, []).append((n, wt))
    out, out8 = torch.zeros_like(x), torch.zeros_like(x)
    for e, use in users.items():
        w13, w2 = expert(e)
        rows = torch.tensor([n for n, _ in use], device=x.device)
        wt = torch.tensor([w for _, w in use], device=x.device, dtype=F64)[:, None]
        for acc, act in ((out, None), (out8, mxfp8)):
            xin = x[rows] if act is None else act(x[rows])
            gu = xin @ w13.T
            half = gu.shape[-1] // 2
            gate = gu[:, :half].clamp(max=SWIGLU_LIMIT)
            up = gu[:, half:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            mid = torch.nn.functional.silu(gate) * up
            acc.index_add_(0, rows, wt * ((mid if act is None else act(mid)) @ w2.T))
    return out, out8, order


def kda_decode(proj, gates, conv, rec, idx, weights, dim_first: bool):
    """One decode token a row through the short conv, the gated delta rule and the
    gated RMSNorm, on the projection rows proj [N, 3P + H + ...] and the f_b | g_b
    gate rows gates [N, 2P]. Updates the fp64 states `conv` (slot, 3P, 3), or
    (slot, 3, 3P) when not `dim_first`, and `rec` (slot, H, D, D) in place and
    returns the normed output [N, P]. A row of state index < 0 is padding: no
    output, no state write. `weights`: conv_w [3P, 4], a_log [H], dt_bias [P],
    w_on [D] of this rank."""
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        KDA_DIM,
        KDA_LOWER_BOUND,
        O_NORM_EPS,
    )

    conv_w, a_log, dt_bias, w_on = (t.to(F64) for t in weights)
    D, H = KDA_DIM, a_log.shape[0]
    P = H * D
    proj, gates = proj.to(F64), gates.to(F64)
    states = conv if dim_first else conv.transpose(-1, -2)
    out = torch.zeros(proj.shape[0], P, dtype=F64, device=proj.device)
    for n, slot in enumerate(idx.tolist()):
        if slot < 0:
            continue
        x, st = proj[n, : 3 * P], states[slot]
        y = (conv_w[:, :3] * st).sum(-1) + conv_w[:, 3] * x
        st.copy_(torch.cat([st[:, 1:], x[:, None]], -1))
        q, k, v = torch.nn.functional.silu(y).view(3, H, D)
        if slot == 0:  # vLLM's null block: the recurrence skips it
            continue
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * D**-0.5
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        g = gates[n, :P].view(H, D) + dt_bias.view(H, D)
        g = KDA_LOWER_BOUND / (1 + torch.exp(-(torch.exp(a_log)[:, None] * g)))
        beta = torch.sigmoid(proj[n, 3 * P : 3 * P + H])
        h = rec[slot] * torch.exp(g)[:, None, :]  # [H, V, K]
        v = (v - (h * k[:, None, :]).sum(-1)) * beta[:, None]
        h = h + v[:, :, None] * k[:, None, :]
        rec[slot] = h
        o = (h * q[:, None, :]).sum(-1)  # [H, V]
        o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + O_NORM_EPS)
        out[n] = (o * w_on * torch.sigmoid(gates[n, P:].view(H, D))).reshape(P)
    return out


@dataclass
class Numerics:
    """Errors of the MonoKernel (and of vLLM, where it has the same stage) to the
    fp64 reference, a row a stage and width. `check` prints them as one table and
    then fails on every row out of its `Bound` at once."""

    title: str
    bounds: dict[str, Bound]
    rank: int = 0
    rows: list[tuple[str, int, float, float | None]] = field(default_factory=list)

    def add(self, stage: str, width: int, mono, ref, vllm=None) -> None:
        e_v = None if vllm is None else rel(vllm, ref)
        self.add_errors(stage, width, rel(mono, ref), e_v)

    def add_errors(
        self, stage: str, width: int, e_mono: float, e_vllm: float | None = None
    ) -> None:
        assert stage in self.bounds, f"no bound for {stage!r}"
        self.rows.append((stage, width, e_mono, e_vllm))

    def failures(self) -> list[str]:
        out = []
        for stage, width, e_m, e_v in self.rows:
            b = self.bounds[stage]
            if not e_m < b.ref:
                out.append(f"{stage} M={width}: mono {e_m:.3g} >= {b.ref:g}")
            if b.vllm is not None and e_v is not None and not e_m <= b.vllm * e_v:
                out.append(
                    f"{stage} M={width}: mono {e_m:.3g} > {b.vllm:g} x vLLM {e_v:.3g}"
                )
        return out

    def table(self) -> str:
        head = f"{'stage':<26}{'M':>4}{'mono':>11}{'vLLM':>11}{'bound':>9}{'x vLLM':>8}"
        lines = [self.title, head]
        for stage, width, e_m, e_v in self.rows:
            b = self.bounds[stage]
            v = "-" if e_v is None else f"{e_v:.3e}"
            k = "-" if b.vllm is None else f"{b.vllm:g}"
            lines.append(f"{stage:<26}{width:>4}{e_m:>11.3e}{v:>11}{b.ref:>9.0e}{k:>8}")
        return "\n".join(lines)

    def check(self) -> None:
        if self.rank == 0:
            print("\n" + self.table(), flush=True)
        bad = self.failures()
        assert not bad, f"{self.title}:\n" + "\n".join(bad)
