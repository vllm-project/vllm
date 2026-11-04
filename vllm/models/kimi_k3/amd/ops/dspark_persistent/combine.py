# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1b `mla_post`: one Kimi-K3 MLA layer from AITER's decode output to
o_proj's input, at TP8 + DCP8 (the DCP group the TP group) or TP8 + DCP1.

AITER leaves every rank the partial attention of its KV shard for all the
gathered heads (`o` [S, ranks x 12, 512] bf16 and `lse` [S, ranks x 12]
f32, strided views). As `dcp_a2a_lse_reduce` -> `_v_up_proj` -> the gate:

    push (a wave a (token, head), over every CTA)
                        o / lse of rank p's heads -> rank p's CMB: the o row
                        untagged, then the LSE pair as its flag (an LSE of
                        -inf for a row with no local KV or past the batch, as
                        ``_dcp_a2a_pack_send_kernel`` masks it)
    combine (a CTA a (head, 4 tokens))
                        the ranks' partials in rank order, LSE-weighted in
                        fp32 -> bf16 latents in LDS
    W_UV (a wave 16 of the head's 128 rows)
                        the latents -> MXFP4 (aiter's even scale), four FP4
                        MFMAs against W_V -> bf16, times the gate -> ``core``

At DCP1 (`ranks` 1) `o` [S, 12, 512] is already the attention's output, as
vLLM hands it to `_v_up_proj`: no push, no LSE; the combine stage loads it.
With `splits` (M1c `mla_attn`'s partials: `o` / `lse` split 0 of dense
[splits, S, 12, 512] bf16 / [splits, S, 12] f32) the combine stage merges a
token's request's splits, LSE-weighted.

K2 (`layer`, `K2Build.m1b_qlen`) runs these stages in its own launch.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
    LOG2E,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
    KVL,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    bf16_pair,
    bf_hi,
    bf_lo,
    butterfly,
    hw_exp2,
    hw_rcp,
    rsrc,
    traced,
    uniform,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    THREADS,
    WAVES,
    KernelAbi,
)

NH = 8
VD = 128  # v_head_dim
CW = KVL // 2 + 2  # CMB pairs a (source, token, head): 256 o words, the LSE, pad
OU = KVL // 4  # the LSE's pair in a CMB row
TT = 4  # tokens a combine task
KS = KVL // 128  # W_UV's 128-k MFMA steps
LROW = KVL // 2 + 4  # an LDS latent row (words, padded)
WMAX = 128  # M1c splits a request at most (an LDS weight row)
CB = 8  # combine_splits' o loads in flight a thread (1 with M1c in the launch)
# timeline points: start, push, combine, W_UV
TL_POINTS = 4

assert VD == WAVES * 16 and THREADS == TT * (KVL // 4) and KVL // 2 == 64 * 4


REGIONS = ("cmb",)


ABI = KernelAbi(
    (
        "o",
        "o_sb",
        "o_sh",
        "lse",
        "lse_sb",
        "lse_sh",
        "seq_lens",
        "qsl",
        "w_v",
        "wv_sh",
        "wv_sn",
        "w_vs",
        "vs_sh",
        "vs_sn",
        "vs_sk",
        "gate",
        "core",
        "peers",
        "rank",
        "layer",
        "epoch",
        "health",
        "alarm",
        "tl",
    )
)


# ---------------------------------------------------------------- push


# ---------------------------------------------------------------- combine
def sane(lse):
    """NaN / +inf -> -inf (`_dcp_a2a_unpack_combine_kernel`)."""
    bad = (lse != lse) | (lse == fx.Float32(float("inf")))
    return bad.select(fx.Float32(float("-inf")), lse)


@traced
def combine_t1(c, t, h):
    """`combine_splits` for one token a task (the D1 back kernel, M1c in the
    launch): token t, head h. Wave w sums splits w, w + 8, .. (as
    `combine_splits` at tc 1); every wave's part_o loads go out right after
    the done flags, behind the LSE loads, so the LSE round trip and the
    partials' overlap. Wave 0 turns the LSEs into weights (LDS `cw`)."""
    s, q, n, a, tid = c["S"], c["Q"], c["splits"], c["args"], c["tid"]
    per = n // (s // q)
    sp0 = t // q * per
    wv = uniform(tid // 64)
    lane = tid % 64
    ht = c["m1c_ht"]
    if tid < per:
        c["poll"]([(c["ardy"], (sp0 + tid) * (NH // ht) + h // ht, 1)])
    gpu.barrier()
    raw = [
        fx.Int32(
            bo.buffer_load(
                rsrc(a["lse"]),
                ((sp0 + fx.min(lane + 64 * k, per - 1)) * s + t) * NH + h,
                vec_width=1,
                dtype=T.i32,
            )
        )
        for k in range((per + 63) // 64)
    ]
    nj = per // WAVES
    src = (((sp0 + wv) * s + t) * NH + h) * (KVL // 2) + 4 * lane
    stride = WAVES * s * NH * KVL * 2  # bytes between a wave's splits
    os_ = [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["o"]), src, vec_width=4, dtype=T.i32, soffset_bytes=j * stride
            )
        )
        for j in range(nj)
    ]
    if wv == 0:
        lse = [
            (lane + 64 * k < per).select(
                sane(raw[k].bitcast(fx.Float32)), fx.Float32(float("-inf"))
            )
            for k in range(len(raw))
        ]
        m = lse[0]
        for k in range_constexpr(1, len(lse)):
            m = fx.max(m, lse[k])
        m = butterfly(m, (32, 16, 8, 4, 2, 1), fx.max)
        m = (m == fx.Float32(float("-inf"))).select(fx.Float32(0.0), m)
        e = [hw_exp2((x - m) * fx.Float32(LOG2E)) for x in lse]
        tot = e[0]
        for k in range_constexpr(1, len(e)):
            tot = tot + e[k]
        tot = butterfly(tot, (32, 16, 8, 4, 2, 1))
        inv = (tot == fx.Float32(0.0)).select(fx.Float32(0.0), hw_rcp(tot))
        for k in range_constexpr(len(e)):
            fx.ptr_store(e[k] * inv, c["cw"] + lane + 64 * k)
    gpu.barrier()
    acc = [fx.Float32(0.0)] * 8
    for j in range_constexpr(nj):
        o = os_[j]
        wt = fx.ptr_load(c["cw"] + wv + WAVES * j)
        for k in range_constexpr(4):
            acc[2 * k] = acc[2 * k] + bf_lo(o[k]) * wt
            acc[2 * k + 1] = acc[2 * k + 1] + bf_hi(o[k]) * wt
    # waves p >= hf hand their sums to p - hf, halving (``combine_splits``)
    red = c["cw"] + TT * WMAX + lane * 8
    for hf in [WAVES >> k for k in range(1, WAVES.bit_length())]:
        if (wv >= hf) & (wv < 2 * hf):
            at = red + (wv - hf) * (64 * 8)
            for k in range_constexpr(2):
                fx.ptr_store(
                    fx.Vector.from_elements(acc[4 * k : 4 * k + 4], fx.Float32),
                    at + 4 * k,
                )
        gpu.barrier()
        if const_expr(hf > 1):
            at = red + fx.min(wv, hf - 1) * (64 * 8)
            acc = [
                (wv < hf).select(acc[i] + fx.ptr_load(at + i), acc[i]) for i in range(8)
            ]
            gpu.barrier()  # read before the next half's stores
        else:
            acc = [acc[i] + fx.ptr_load(red + i) for i in range(8)]
    if wv == 0:
        for k in range_constexpr(4):
            fx.ptr_store(
                bf16_pair(acc[2 * k], acc[2 * k + 1]).bitcast(fx.Int32),
                c["lat"] + 4 * lane + k,
            )
    gpu.barrier()


# ---------------------------------------------------------------- W_UV


# ---------------------------------------------------------------- kernel
