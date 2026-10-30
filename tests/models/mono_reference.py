# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slow references for the AMD monokernel I/O tests: plain torch in float64,
written from the model definitions, never from a vLLM, AITER or Triton kernel.

Rounding points a kernel documents are explicit calls here (``bf16_rne``,
``to_e4m3_rne``, the E8M0 code rules), so a test reproduces them on purpose.
``tests/kernels/models/amd/test_mono_primitives.py`` proves each rounding and
code rule bit-exact against the device helpers the kernels use."""

import torch
import torch.nn.functional as F

F64 = torch.float64
FLT_MIN = 2.0**-126
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1 = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), dtype=F64)
RCP6_F32 = torch.tensor(1 / 6, dtype=torch.float32).item()


# ---------------------------------------------------------------- formats
def e8m0(codes: torch.Tensor) -> torch.Tensor:
    return torch.exp2(codes.double() - 127)


def _expand(scale: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    return scale.repeat_interleave(block[0], -2).repeat_interleave(block[1], -1)


def dequant_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp4x2 bytes [..., K / 2] (low nibble the lower K) with E8M0 codes
    [..., K / 32] -> float64 [..., K]."""
    e2m1 = E2M1.to(codes.device)
    v = torch.stack([e2m1[(codes & 15).long()], e2m1[(codes >> 4).long()]], -1)
    return v.flatten(-2) * e8m0(scales).repeat_interleave(32, -1)


def dequant_fp8(q, scales, block=(32, 32), fnuz=False) -> torch.Tensor:
    """E4M3 [..., N, K] (OCP, or FNUZ: the same bytes at half the value) with
    E8M0 codes [..., N / block[0], K / block[1]] -> float64 [..., N, K]."""
    dt = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    return q.view(torch.uint8).view(dt).double() * _expand(e8m0(scales), block)


NOPE, ROPE, RECORD = 448, 64, 576


def read_records(cache, block: int, slots, fnuz=False) -> torch.Tensor:
    """fp8_ds_mla records at ``slots`` -> float64 [n, 512]. A page holds
    ``block`` records of 576 bytes (448 E4M3, 64 bf16 rope), then ``block``
    times 8 scale bytes (a UE8M0 per 64 nope dims)."""
    flat = torch.empty(0, dtype=torch.uint8, device=cache.device)
    flat.set_(cache.untyped_storage())
    slots = slots.to(cache.device).long()
    esz = cache.element_size()
    page = (cache.storage_offset() + slots // block * cache.stride(0)) * esz
    d0 = page + slots % block * RECORD
    s0 = page + block * RECORD + slots % block * 8
    data = flat[d0[:, None] + torch.arange(RECORD, device=cache.device)]
    codes = flat[s0[:, None] + torch.arange(7, device=cache.device)]
    nope = dequant_fp8(data[:, :NOPE], codes, (1, 64), fnuz)
    rope = data[:, NOPE:].contiguous().view(torch.bfloat16).double()
    return torch.cat([nope, rope], 1)


def pairs(region: torch.Tensor, n: int, tag: int | None = None) -> torch.Tensor:
    """The first ``n`` (value, tag) int32 mailbox pairs of a byte region ->
    the int32 values, after checking every tag is ``tag`` (the launch's)."""
    p = region[: n * 8].view(torch.int32).view(n, 2)
    if tag is not None:
        stale = (p[:, 1] != tag).nonzero()
        assert stale.numel() == 0, f"{stale.numel()} stale pairs, first {stale[0]}"
    return p[:, 0].clone()


def words(region: torch.Tensor, n: int, dtype: torch.dtype) -> torch.Tensor:
    """The first ``n`` elements of ``dtype`` of a byte region (a copy)."""
    return region[: n * dtype.itemsize].view(dtype).clone()


# ---------------------------------------------------------------- rounding
def round_float(x, mant: int, emin: int, fmax: float | None = None):
    """``x`` rounded to nearest-even onto a binary format with ``mant``
    mantissa bits and normal exponents >= ``emin`` (subnormals below),
    saturating at ``fmax`` if given, else overflowing to inf at 2^(emax+1)."""
    x = x.double()
    if fmax is not None:
        x = x.clamp(-fmax, fmax)
    e = torch.floor(torch.log2(x.abs())).clamp(min=emin)
    q = torch.exp2(e - mant)
    r = torch.where(torch.isfinite(x) & (x != 0), torch.round(x / q) * q, x)
    if fmax is None:
        r = torch.where(r.abs() >= 2.0**128, r * float("inf"), r)
    return r


def bf16_rne(x):
    return round_float(x, 7, -126)


def f32_rne(x):
    return round_float(x, 23, -126)


def to_e4m3_rne(x, fnuz=False, fmax=None):
    """The E4M3 value of ``x`` clamped to +-fmax (448 OCP, 224 FNUZ), RNE."""
    if fmax is None:
        fmax = 224.0 if fnuz else 448.0
    return round_float(x, 3, -7 if fnuz else -6, fmax)


def to_e2m1_rne(x):
    return round_float(x, 1, 0, 6.0)


def _ceil_log2(v):
    m, e = torch.frexp(v.double())
    return torch.where(v > 0, e.long() - (m == 0.5).long(), -(1 << 20))


def ceil_code(v):
    """The E8M0 round-up rule on an f32 value v >= 0: its exponent field, plus
    one if its mantissa field is not zero (the smallest power of two >= v for
    a normal v; 1 for a subnormal, 0 for zero)."""
    v = v.double()
    return torch.where(v < FLT_MIN, (v > 0).long(), _ceil_log2(v) + 127)


def mx_code_ceil(amax, fmax=448.0, floor=0.0):
    """ATOM / AITER MXFP8 rule: ceil code of max(amax, floor) / fmax, [1, 254]."""
    return ceil_code(amax.double().clamp(min=floor) / fmax).clamp(1, 254)


def mx_code_vllm(amax, fmax=448.0):
    """VLLM's ``mxfp8_e4m3_quantize``: ceil(log2(RN_f32(max(amax, FLT_MIN)
    / fmax))) + 127, clamped to [0, 254] (0 for an all-zero group)."""
    q = f32_rne(amax.double().clamp(min=FLT_MIN) / fmax)
    return (_ceil_log2(q) + 127).clamp(0, 254)


def e8m0_roundup6(amax):
    """AITER's inline MXFP4 scale: ceil code of RN_f32(amax RN_f32(1 / 6)), <= 254."""
    return ceil_code(f32_rne(amax.double() * RCP6_F32)).clamp(max=254)


def mx_quantize(x, code_rule, block=32, fnuz=False):
    """``x`` [..., K] per ``block`` -> (E4M3 values [..., K], codes
    [..., K / block]) with ``code_rule(amax)``: x * 2^(127 - code), clamped, RNE."""
    xb = x.double().unflatten(-1, (-1, block))
    codes = code_rule(xb.abs().amax(-1))
    q = to_e4m3_rne(xb * torch.exp2(127 - codes.double())[..., None], fnuz)
    return q.flatten(-2), codes


# ---------------------------------------------------------------- math
def rmsnorm(x, w, eps, gemma=False):
    x = x.double()
    w = w.double() + 1 if gemma else w.double()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w


def gated_rmsnorm(x, gate, w, eps):
    return rmsnorm(x, w, eps) * torch.sigmoid(gate.double())


def rope_gptj(x, cos, sin, inverse=False):
    """GPT-J (interleaved pair) rotation of the last dim of ``x`` by
    cos / sin [..., D / 2]; ``inverse`` rotates back."""
    x, cos, sin = x.double(), cos.double(), sin.double()
    sin = -sin if inverse else sin
    e, o = x[..., 0::2], x[..., 1::2]
    return torch.stack([e * cos - o * sin, e * sin + o * cos], -1).flatten(-2)


def mhc_mixes(x, fn, scale, base, hc, rms_eps, pre_eps, sk_eps, post_mult, iters):
    """DeepSeek-V4 mHC coefficients of the flattened residual x [M, hc D]
    -> (pre [M, hc], post [M, hc], comb [M, hc, hc]); ``fn`` [hc (2 + hc),
    hc D] as the kernel reads it (e.g. bf16 hi + lo, summed in float64)."""
    x = x.double()
    mixes = (x @ fn.double().T) * torch.rsqrt(x.square().mean(-1, True) + rms_eps)
    scale, base = scale.double(), base.double()
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + pre_eps
    post = torch.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[:, 2 * hc :].view(-1, hc, hc) * scale[2] + base[2 * hc :].view(hc, hc)
    comb = torch.softmax(comb, -1) + sk_eps
    comb = comb / (comb.sum(-2, keepdim=True) + sk_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + sk_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + sk_eps)
    return pre, post * post_mult, comb


def mhc_post(x, residual, post, comb):
    """New streams [M, hc, D]: post_j x + sum_i comb[i, j] residual_i."""
    mixed = torch.einsum("mij,mid->mjd", comb.double(), residual.double())
    return mixed + post.double()[..., None] * x.double()[:, None]


def mhc_collapse(residual, pre):
    """The sublayer input sum_i pre_i residual_i [M, D]."""
    return (pre.double()[..., None] * residual.double()).sum(1)


def sparse_attention_sink(q, keys, scale, sink=None, valid=None, values=None):
    """Softmax attention of q [M, H, D] over each row's keys [M, N, D] (MLA:
    the values are the keys unless given), ``valid`` [M, N] masking keys, a
    per-head ``sink`` logit [H] in the denominator only -> [M, H, Dv]."""
    q, keys = q.double(), keys.double()
    values = keys if values is None else values.double()
    s = torch.einsum("mhd,mnd->mhn", q, keys) * scale
    if valid is not None:
        s = s.masked_fill(~valid[:, None, :], float("-inf"))
    if sink is not None:
        sk = sink.double().view(1, -1, 1).expand(s.shape[0], -1, 1)
        s = torch.cat([s, sk], -1)
    p = torch.softmax(s, -1)[..., : keys.shape[1]]
    return torch.einsum("mhn,mnd->mhd", p, values)


def silu(x):
    return F.silu(x.double())


def swiglu_clamped(g, u, limit=10.0):
    """DeepSeek's routed / shared activation: silu(min(g, limit)) clamp(u)."""
    return silu(g.double().clamp(max=limit)) * u.double().clamp(-limit, limit)


def swiglu_oai(g, u, limit=7.0, alpha=1.702, beta=1.0):
    g = g.double().clamp(max=limit)
    return g * torch.sigmoid(alpha * g) * (u.double().clamp(-limit, limit) + beta)


def situ(g, u, beta=4.0, lbeta=25.0):
    """Kimi SiTU: beta tanh(g / beta) sigmoid(g) lbeta tanh(u / lbeta)."""
    g, u = g.double(), u.double()
    return (
        beta * torch.tanh(g / beta) * torch.sigmoid(g) * lbeta * torch.tanh(u / lbeta)
    )


def router_scores(logits, kind: str):
    """'sqrt_softplus' (DeepSeek, x itself past 20) or 'sigmoid'."""
    x = logits.double()
    return torch.sqrt(F.softplus(x)) if kind == "sqrt_softplus" else torch.sigmoid(x)


def route(logits, bias, k, kind, scale, ids=None):
    """Top-k of score + bias (ties to the lower id; ``ids`` given: hash
    routing), weights the picked scores times scale / their sum.
    -> (ids [M, k], weights [M, k], margin [M]: k-th minus (k+1)-th value)."""
    s = router_scores(logits, kind)
    sel = s if bias is None else s + bias.double()
    order = torch.sort(sel, dim=-1, descending=True, stable=True)
    margin = order.values[:, k - 1] - order.values[:, k]
    if ids is None:
        ids = order.indices[:, :k]
    w = s.gather(1, ids.long())
    return ids, w * scale / w.sum(-1, keepdim=True), margin


def kda_step(S, q, k, v, g, beta, a_log, dt_bias, lower_bound, scale):
    """One token of the KDA gated delta rule, per head h: S [H, V, K], q, k, g
    [H, K], v [H, V], beta [H] (pre-sigmoid), a_log [H], dt_bias [H, K]
    -> (S', o [H, V])."""
    S, q, k, v = S.double(), q.double(), k.double(), v.double()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = g.double() + dt_bias.double()
    gate = lower_bound * torch.sigmoid(torch.exp(a_log.double())[:, None] * gate)
    S = S * torch.exp(gate)[:, None, :]
    v = (v - torch.einsum("hvk,hk->hv", S, k)) * torch.sigmoid(beta.double())[:, None]
    S = S + v[:, :, None] * k[:, None, :]
    return S, torch.einsum("hvk,hk->hv", S, q)


def causal_conv1d(state, x, w, bias=None, act=True):
    """Depthwise causal conv over [state | x]: state [C, W - 1] (oldest
    first), x [T, C], w [C, W] -> (y [T, C], SiLU if ``act``; the last W - 1
    columns as the new state)."""
    seq = torch.cat([state.double(), x.double().T], 1)
    W = w.shape[1]
    y = torch.stack(
        [(seq[:, t : t + W] * w.double()).sum(1) for t in range(x.shape[0])]
    )
    if bias is not None:
        y = y + bias.double()
    return (silu(y) if act else y), seq[:, -(W - 1) :]


def attn_res(prefix, blocks, w_norm, w_qk, w_out, eps, out_eps):
    """Kimi AttnRes of one row set: softmax over the sources (``blocks``
    [M, B, D], then ``prefix`` [M, D]) of rmsnorm-scaled logits
    v . (w_norm w_qk), the mix of the sources, then RMSNorm with ``w_out``."""
    src = torch.cat([blocks.double(), prefix.double()[:, None]], 1)
    rstd = torch.rsqrt(src.square().mean(-1) + eps)
    logits = (src @ (w_norm.double() * w_qk.double())) * rstd
    mixed = (torch.softmax(logits, -1)[..., None] * src).sum(1)
    return rmsnorm(mixed, w_out, out_eps)


# ---------------------------------------------------------------- metrics
_FMT = {torch.bfloat16: (7, -126), torch.float32: (23, -126)}


def ulp(x, dtype=torch.bfloat16):
    """The ``dtype`` (bf16 / f32) spacing at |x| (subnormal spacing below)."""
    mant, emin = _FMT[dtype]
    e = torch.floor(torch.log2(x.double().abs())).clamp(min=emin)
    return torch.exp2(e - mant)


def ulps(got, ref, dtype=torch.bfloat16, floor=0.0):
    """|got - ref| in ``dtype`` ulps of max(|ref|, floor)."""
    ref = ref.double()
    return (got.double() - ref).abs() / ulp(ref.abs().clamp(min=floor), dtype)


def err_over_cond(got, ref, a, b):
    """|got - ref| / (|a| @ |b|^T) for ``got`` a computed a @ b^T: the dot
    product's error relative to its conditioning scale."""
    return (got.double() - ref.double()).abs() / (a.double().abs() @ b.double().abs().T)


def blockmax_cond(a, b, block=32):
    """Sum over the K blocks of max_k |a_k b_k|, for a [..., M, K] times
    b [..., N, K]^T: the scale of a gfx950 FP8 MFMA's error, which sums a
    block's products at the precision of its largest one (a product 2^-14
    below it is dropped), not at fp32."""
    a, b = a.double().abs(), b.double().abs()
    out = 0
    for k in range(0, a.shape[-1], block):
        aj, bj = a[..., k : k + block], b[..., k : k + block]
        out = out + (aj.unsqueeze(-2) * bj.unsqueeze(-3)).amax(-1)
    return out


def boundary(fn, x, d):
    """Where ``fn`` (monotone) is not constant on [x - d, x + d] -> (mask,
    fn(x - d), fn(x + d)): the elements a reference error of d may flip."""
    lo, hi = fn(x - d), fn(x + d)
    return lo != hi, lo, hi


def assert_quant_equal(got_q, got_codes, x, d, code_rule, block=32, fnuz=False):
    """MXFP8 output (E4M3 values [..., K] as float64, codes [..., K / block])
    against the reference values ``x`` known to +-``d``: exact outside the
    boundary set, one of its two ends inside, and the set under 1 %."""
    xb = x.double().unflatten(-1, (-1, block))
    db = torch.as_tensor(d, dtype=F64, device=x.device).expand_as(x)
    db = db.unflatten(-1, (-1, block))
    amax = xb.abs().amax(-1)
    bc, clo, chi = boundary(code_rule, amax, db.amax(-1))
    got_codes = got_codes.long()
    assert torch.equal(got_codes[~bc], clo[~bc]), "codes differ"
    assert ((got_codes == clo) | (got_codes == chi))[bc].all(), "codes off by > 1"
    inv = torch.exp2(127 - got_codes.double())[..., None]
    bv, vlo, vhi = boundary(lambda y: to_e4m3_rne(y, fnuz), xb * inv, db * inv)
    gq = got_q.double().unflatten(-1, (-1, block))
    assert torch.equal(gq[~bv], vlo[~bv]), "values differ"
    assert ((gq == vlo) | (gq == vhi))[bv].all(), "values off by > 1 step"
    frac = (bv.sum() + bc.sum() * block).item() / x.numel()
    assert frac < 0.01, f"boundary set {frac:.2%} of the elements"
    return frac


def assert_topk(ids, ref_ids, margin, bound):
    """Picked ids as a set equal the reference's on every row whose k-th /
    (k+1)-th margin exceeds ``bound`` -> rows checked."""
    clear = margin > bound
    got = ids.long()[clear].sort(-1).values
    assert torch.equal(got, ref_ids.long()[clear].sort(-1).values), "top-k ids"
    return int(clear.sum())
