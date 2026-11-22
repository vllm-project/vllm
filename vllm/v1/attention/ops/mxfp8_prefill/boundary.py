# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal MXFP8 prefill forward: preparation, normal phase, boundary merge.

One launch prepares the MXFP8 operands. The normal phase (`bulk`, cuDNN)
covers every completed 32-token block before the query. A boundary kernel
covers each query's own block and merges the two softmax partials:
`_boundary`, or `boundary_gluon` for long GQA. All matrix products use
block-scaled E4M3 MMAs with FP32 accumulation.
"""

from functools import lru_cache

import torch

from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr
from vllm.triton_utils.tensor_descriptor import get_tensor_descriptor

from . import boundary_gluon
from .bulk import execute


@tr.jit
def _scale_exponent(amax):
    # Round amax/448 upward to E8M0, matching TE's round-up conversion.
    value = amax * (1.0 / 448.0)
    bits = value.to(tl.int32, bitcast=True)
    exp, mant = (bits >> 23) & 255, bits & 0x7FFFFF
    return exp + ((mant > 0) & (exp != 254) & ~((exp == 0) & (mant <= 0x400000))).to(
        tl.int32
    )


@tr.jit
def _reciprocal(e):
    return tl.where(e == 254, 0x00400000, (254 - e) << 23).to(tl.float32, bitcast=True)


@tr.jit
def _rowwise(
    X,
    X8,
    SX,
    XB,
    XH,
    XN,
    N,
    H: tl.constexpr,
    TOKENS: tl.constexpr,
    tile,
    head,
    batch,
):
    # TE's default rowwise MXFP8 cast: one E8M0 scale per token and 32 features.
    i = tl.arange(0, TOKENS)
    g = tl.arange(0, 4)
    d = g[None, :, None] * 32 + tl.arange(0, 32)[None, None, :]
    r = tile * TOKENS + i
    x = tl.load(X + batch * XB + head * XH + r[:, None, None] * XN + d).to(tl.float32)
    e = _scale_exponent(tl.max(tl.abs(x), 2))
    out = ((batch * N + r[:, None, None]) * H + head) * 128 + d
    tl.store(X8 + out, (x * _reciprocal(e)[:, :, None]).to(tl.float8e4nv))
    sf = ((batch * H + head) * (N // 128) + r[:, None] // 128) * 512
    sf += (r[:, None] % 32) * 16 + (r[:, None] % 128 // 32) * 4 + g[None, :]
    tl.store(SX + sf, e.to(tl.uint8))


@tr.jit
def _columnwise(
    V,
    V8,
    SV,
    U,
    AE,
    VB,
    VH,
    VN,
    N,
    H: tl.constexpr,
    TOKENS: tl.constexpr,
    TRANSPOSED_U: tl.constexpr,
    tile,
    head,
    batch,
):
    # Ordinary columnwise V8 (32-token scales) plus token-normalized U and AE.
    GROUPS: tl.constexpr = TOKENS // 32
    s = tl.arange(0, GROUPS)
    d = tl.arange(0, 128)
    r = tile * TOKENS + s[:, None] * 32 + tl.arange(0, 32)[None, :]
    v = tl.load(V + batch * VB + head * VH + r[:, :, None] * VN + d[None, None, :]).to(
        tl.float32
    )
    e = _scale_exponent(tl.max(tl.abs(v), 1))
    out = ((batch * N + r[:, :, None]) * H + head) * 128 + d[None, None, :]
    tl.store(V8 + out, (v * _reciprocal(e)[:, None, :]).to(tl.float8e4nv))
    block = tile * GROUPS + s
    sf = ((batch * H + head) * (N // 128) + block[:, None] // 4) * 512
    sf += (d[None, :] % 32) * 16 + (d[None, :] // 32) * 4 + block[:, None] % 4
    tl.store(SV + sf, e.to(tl.uint8))
    maximum = tl.max(tl.abs(v), 2)
    # Use exponent bits for normal magnitudes; as in the CPU oracle,
    # subnormal amplitudes are bounded below by 2**-126.
    bits = maximum.to(tl.int32, bitcast=True)
    ae = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    # Exact v * 2**(8 - ae) as two normal power-of-two factors per token
    # (8 - ae lies in [-120, 135]); libdevice.ldexp costs three exp2 per element.
    shift = 8 - ae
    first = (((shift >> 1) + 127) << 23).to(tl.float32, bitcast=True)
    second = ((shift - (shift >> 1) + 127) << 23).to(tl.float32, bitcast=True)
    u = (v * first[:, :, None] * second[:, :, None]).to(tl.float8e4nv)
    u = tl.where(maximum[:, :, None] == 0, 0, u)
    if TRANSPOSED_U:
        # [B, H, 128, N]: the K-contiguous B operand of the Gluon W·U MMA.
        tl.store(
            U + ((batch * H + head) * 128 + d[None, None, :]) * N + r[:, :, None], u
        )
    else:
        tl.store(U + out, u)
    tl.store(AE + (batch * H + head) * N + r, tl.where(maximum == 0, -1000, ae))


@tr.jit
def _prepare_all(
    Q,
    K,
    V,
    Q8,
    K8,
    V8,
    SQ,
    SK,
    SV,
    U,
    AE,
    QB,
    QH,
    QN,
    KB,
    KH,
    KN,
    VB,
    VH,
    VN,
    N,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    TOKENS: tl.constexpr,
    TRANSPOSED_U: tl.constexpr,
):
    # One launch for every operand: slots [0, HQ) cast Q heads, [HQ, HQ + HK)
    # cast K heads, and the last HK slots prepare V heads.
    # Slots vary fastest: concurrent programs cover every head of the same
    # token rows, so BSHD reads and writes are contiguous across the wave.
    slot, tile, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if slot < HQ:
        _rowwise(Q, Q8, SQ, QB, QH, QN, N, HQ, TOKENS, tile, slot, batch)
    elif slot < HQ + HK:
        _rowwise(K, K8, SK, KB, KH, KN, N, HK, TOKENS, tile, slot - HQ, batch)
    else:
        _columnwise(
            V,
            V8,
            SV,
            U,
            AE,
            VB,
            VH,
            VN,
            N,
            HK,
            TOKENS,
            TRANSPOSED_U,
            tile,
            slot - HQ - HK,
            batch,
        )


@tr.jit
def _zero(M: tl.constexpr, N: tl.constexpr):
    return tl.inline_asm_elementwise(
        "mov.b32 $0, $1;",
        "=f,f",
        [tl.full((M, N), 0, tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@tr.jit(do_not_specialize=["B"])
def _boundary(
    Q,
    K,
    SQ,
    SK,
    U,
    AE,
    P_DESC,
    LSE,
    O_DESC,
    N,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    B,
    PACK: tl.constexpr,
    STAGES: tl.constexpr,
):
    # Persistent inference kernel with token-major rows (row = token * PACK +
    # head), so the BF16 partial and output move as TMA boxes [KEYS, PACK, 64].
    KEYS: tl.constexpr = 128 // PACK
    GROUPS: tl.constexpr = KEYS // 32
    TILES = N // KEYS
    GROUPS_H: tl.constexpr = HQ // PACK
    total = TILES * GROUPS_H * B
    row = tl.arange(0, 128)
    token = row // PACK
    lane = row % PACK
    d = tl.arange(0, 128)
    g = tl.arange(0, 4)
    j = tl.arange(0, 32)
    pick = tl.arange(0, GROUPS)[None, :, None] == (token // 32)[:, None, None]
    usf = tl.full((128, GROUPS), 119, tl.uint8)
    for t in tl.range(tl.program_id(0), total, tl.num_programs(0), num_stages=STAGES):
        # Head groups vary fastest: concurrent programs cover whole token rows.
        packed_h = t % GROUPS_H
        tile = (t // GROUPS_H) % TILES
        b = t // (TILES * GROUPS_H)
        kh = packed_h // ((HQ // HK) // PACK)
        h = packed_h * PACK + lane
        r = tile * KEYS + token
        kr = tile * KEYS + tl.arange(0, KEYS)
        row0 = b * N + tile * KEYS
        bulk = tl.reshape(P_DESC.load([row0, packed_h * PACK, 0]), (128, 128))
        bulk_lse = tl.load(LSE + (b * HQ + h) * N + r)
        qo = ((b * N + r[:, None]) * HQ + h[:, None]) * 128 + d[None, :]
        ko = ((b * N + kr[:, None]) * HK + kh) * 128 + d[None, :]
        q, k = tl.load(Q + qo), tl.load(K + ko)
        sf = (r[:, None] % 32) * 16 + (r[:, None] % 128 // 32) * 4 + g[None, :]
        sq = tl.load(
            SQ + ((b * HQ + h[:, None]) * (N // 128) + r[:, None] // 128) * 512 + sf
        )
        ksf = (kr[:, None] % 32) * 16 + (kr[:, None] % 128 // 32) * 4 + g[None, :]
        sk = tl.load(SK + ((b * HK + kh) * (N // 128) + kr[:, None] // 128) * 512 + ksf)
        score = tl.dot_scaled(q, sq, "e4m3", tl.trans(k), sk, "e4m3", _zero(128, KEYS))
        score *= 128**-0.5
        # Each query row sees only its own 32-key group.
        if GROUPS == 1:
            own_score = score
            ae = tl.load(AE + (b * HK + kh) * N + kr)[None, :]
        else:
            grouped_score = tl.reshape(score, (128, GROUPS, 32))
            own_score = tl.sum(tl.where(pick, grouped_score, 0.0), 1)
            ae = tl.load(AE + (b * HK + kh) * N + (r[:, None] // 32) * 32 + j[None, :])
        visible = j[None, :] <= (r % 32)[:, None]
        own_score = tl.where(visible, own_score, -float("inf"))
        m = tl.max(own_score, 1)
        p = tl.exp(own_score - m[:, None])
        denominator = tl.sum(p, 1)
        logw = (own_score - m[:, None]) * 1.4426950408889634 + ae
        logw = tl.where(visible & (ae != -1000), logw, -float("inf"))
        wmax = tl.max(logw, 1)
        we = tl.where(wmax == -float("inf"), 0, tl.ceil(wmax - 8.807354922057604))
        we = tl.minimum(127, tl.maximum(-127, we))
        own_w = tl.exp2(logw - we[:, None])
        if GROUPS == 1:
            w = own_w.to(tl.float8e4nv)
        else:
            w = tl.where(pick, own_w[:, None, :], 0.0)
            w = tl.reshape(w, (128, KEYS)).to(tl.float8e4nv)
        ws = tl.broadcast_to((we + 127).to(tl.uint8)[:, None], (128, GROUPS))
        bulk_lse = tl.where(r >= 32, bulk_lse, -float("inf"))
        common = tl.maximum(m, bulk_lse)
        a, z = tl.exp(bulk_lse - common), tl.exp(m - common)
        inverse = 1.0 / (a + z * denominator)
        # One full-width W·U product and one 128-wide TMA store per tile.
        numerator = tl.dot_scaled(
            w, ws, "e4m3", tl.load(U + ko), usf, "e4m3", _zero(128, 128)
        )
        bulk = tl.where(r[:, None] >= 32, bulk.to(tl.float32), 0.0)
        result = (a[:, None] * bulk + z[:, None] * numerator) * inverse[:, None]
        O_DESC.store(
            [row0, packed_h * PACK, 0],
            tl.reshape(result.to(tl.bfloat16), (KEYS, PACK, 128)),
        )


def prepare(q, k, v, tokens=32, transposed_u=False):
    """TE-equivalent Q/K casts and causal V preparation in one launch.

    Returns `(q8, k8, v8, sq, sk, sv, u, ae)`. With `transposed_u` U is
    written only as `ut` in `[B, HK, 128, N]` for the Gluon inference
    kernel, and the result is `(q8, k8, v8, sq, sk, sv, None, ae, ut)`.
    """
    q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v)]
    b, hq, n, _ = q.shape
    hk = k.shape[1]

    def payload(h):
        return torch.empty(
            (b, n, h, 128), device=q.device, dtype=torch.float8_e4m3fn
        ).transpose(1, 2)

    def scales(h):
        return torch.empty((b, h, n // 128, 512), device=q.device, dtype=torch.uint8)

    q8, k8, v8 = payload(hq), payload(hk), payload(hk)
    if transposed_u:
        u = torch.empty((b, hk, 128, n), device=q.device, dtype=torch.float8_e4m3fn)
    else:
        u = payload(hk)
    sq, sk, sv = scales(hq), scales(hk), scales(hk)
    ae = torch.empty((b, hk, n), device=q.device, dtype=torch.int32)
    _prepare_all[(hq + 2 * hk, n // tokens, b)](
        q,
        k,
        v,
        q8,
        k8,
        v8,
        sq,
        sk,
        sv,
        u,
        ae,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        n,
        hq,
        hk,
        tokens,
        transposed_u,
        num_warps=2,
    )
    if transposed_u:
        return q8, k8, v8, sq, sk, sv, None, ae, u
    return q8, k8, v8, sq, sk, sv, u, ae


PERSISTENT_CTAS = 2
PERSISTENT_STAGES = 2


@lru_cache(maxsize=8)
def _sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _pack_heads(hq, hk):
    """Fill 128 MMA rows with heads sharing K/V, using 32/64/128 keys."""
    return next(p for p in (4, 2, 1) if (hq // hk) % p == 0)


def _uses_gluon(q, k, pack_heads):
    b, hq, n, _ = q.shape
    return pack_heads in (None, boundary_gluon.PACK) and boundary_gluon.preferred(
        b, hq, k.shape[1], n, q.device
    )


def merge_boundary(prepared, partial, lse, out, pack_heads):
    """Inference boundary launch: own-block attention merged with the bulk partial."""
    q, k, v, sq, sk, sv, u, ae = prepared[:8]
    ut = prepared[8] if len(prepared) > 8 else None
    b, hq, n, _ = q.shape
    hk = k.shape[1]
    if ut is not None and pack_heads == boundary_gluon.PACK:
        # Hand-scheduled warp-specialized kernel; bitwise equal to _boundary.
        return boundary_gluon.boundary(q, k, sq, sk, ut, ae, partial, lse, out)
    if u is None:
        raise ValueError(
            "This path needs token-major U: prepare(..., transposed_u=False)"
        )
    total = (n // (128 // pack_heads)) * (hq // pack_heads) * b
    programs = min(total, _sms(q.device) * PERSISTENT_CTAS)
    keys = 128 // pack_heads

    def boxes(x):
        rows = x.transpose(1, 2).reshape(b * n, hq, 128)
        return get_tensor_descriptor().from_tensor(rows, [keys, pack_heads, 128])

    return _boundary[(programs,)](
        q,
        k,
        sq,
        sk,
        u,
        ae,
        boxes(partial),
        lse,
        boxes(out),
        n,
        hq,
        hk,
        b,
        pack_heads,
        # Two pipeline stages of 128-key K/U tiles need ~117 KB of shared
        # memory and leave one program per SM; one stage keeps two.
        PERSISTENT_STAGES if pack_heads == 4 else 1,
        num_warps=4,
    )


def forward_prepared(prepared, pack_heads=None):
    q, k, v, sq, sk, sv = prepared[:6]
    hq, hk = q.shape[1], k.shape[1]
    if pack_heads is None:
        pack_heads = _pack_heads(hq, hk)
    if pack_heads not in (1, 2, 4) or (hq // hk) % pack_heads:
        raise ValueError("Boundary packing must divide the GQA ratio")
    partial, lse = execute(q, k, v, sq, sk, sv)
    out = torch.empty_like(partial)
    merge_boundary(prepared, partial, lse, out, pack_heads)
    return out


def _validate(q, k, v):
    if any(x.ndim != 4 or x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError(
            "Expected BF16 inputs in [batch, heads, sequence, features] layout."
        )
    if q.device.type != "cuda" or any(x.device != q.device for x in (k, v)):
        raise ValueError("Inputs must share one CUDA device.")
    if torch.cuda.get_device_capability(q.device)[0] != 10:
        raise ValueError("Native MXFP8 requires SM100/SM103.")
    if q.shape[-1] != 128 or not q.shape[-2] or q.shape[-2] % 128:
        raise ValueError(
            "Expected D=128 and a positive sequence length divisible by 128."
        )
    if (
        not k.shape[1]
        or not q.shape[1]
        or q.shape[1] % k.shape[1]
        or k.shape != v.shape
        or q.shape[0] != k.shape[0]
        or q.shape[2:] != k.shape[2:]
    ):
        raise ValueError(
            "Expected matching self-attention shapes and an integral GQA ratio."
        )


def forward(q, k, v, pack_heads=None):
    """Causal MXFP8 attention only; this experimental entry is forward-only."""
    _validate(q, k, v)
    if any(x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError("Features must have unit stride")
    prepared = prepare(q, k, v, transposed_u=_uses_gluon(q, k, pack_heads))
    return forward_prepared(prepared, pack_heads)
