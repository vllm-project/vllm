# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2(split) combine (DESIGN_BC 7.4, Phase C): the split-KV units' partials of
one (token t, q head h) merged into the attention output, its output norm and
gate, the gated row -> ``xo`` and the ``crdy`` flag.

Combine k = 16 t + h runs on CTA ``C0(S) + k`` (``K2Build.c0``); kvh, j =
divmod(h, 8): it reads q head j of the N = ``K2Build.nsplit()`` units u
= (2 t + kvh) N + p of its (t, kvh), each published by its split unit
(``srdy[u]``, ``attention/split.py``) as plain fp32 regions, device scope:

    pstat[u, j]   (m, l): the unit's running max of the base-2 logits and the
                  sum of the bf16 P its PV MFMA consumed (E9)
    opart[u, j]   O[64]: the unit's unnormalized output, sum of P V

    M   = max_p m_p                        over the live units p < live
    w_p = l_p > 0 ? exp2(m_p - M) : 0      (an empty unit's (NEG, 0) weighs 0)
    L   = sum_p l_p w_p
    o   = bf16(L > 0 ? (sum_p O_p w_p) / L : 0)      the stock attn output
    y   = o_gate(o, og)                    ``gate.o_gate``: rms_norm_gated

``live`` is the split's own unit count of this (t, kvh) (``unit_plan`` on
``seq_lens[t]``, the build's window): units p >= live hold no key, so their
flags are not polled and their partials never loaded (masked buffer loads
return zeros without a memory access). A pad row (or L = 0) gives zeros.

Program order (rule 8: the CTA holds no o_proj weights here; they are issued
after ``crdy``'s publish, DESIGN_BC 7.2):

    entry     pad word, seq_lens[t] and the og element (``combine_loads``,
              plain: earlier launches' outputs)
    poll      thread p < live polls srdy[base + p]; barrier
    loads     at once, CM_DEV only (rule 9: written in this launch, on other
              XCDs): the stats lane-parallel (lane p, a dwordx2), the
              partials wave-parallel (wave w: p = w + 8 i, lane = dim)
    merge     M by ``wave_max``; w_p; L by ``wave_sum`` of l_p w_p (one fixed
              tree, the same in every wave); wave w's sum_i O_p w_p (fma, i
              order) -> LDS ``ow[w]``; barrier; wave 0 sums the 8 in wave
              order, divides (``div_rn``), rounds to bf16, gates
    out       wave 0: ``xo[t, 64 h + d]`` (bf16, CM_DEV), then ``publish``
              (vmcnt(0), barrier, ``crdy[k]``)

The merge's sums are in a fixed order, so the output is the same in every
launch; the stats' and partials' fp32 sums differ from a sequential sum by
fp32 rounding only (DESIGN_BC 4: the core's target is ulps vs stock).

LDS: ``lds_struct(S)`` under the stage key ``"combine"`` (``ow``, 2 KiB).
``publish``'s barrier follows the last LDS read (the hand-over rule).

Pattern: DSv4.1's ``stage_combine`` (``deepseek_v41/amd/mono/attention/
back.py``) with lane-parallel stats, a wave a group of partials and the gate
fused (gqa_attention.md 7.6).
"""

import functools

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.mono.attention import gate
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    CM_DEV,
    bf16_round,
    div_rn,
    hw_exp2,
    ld_i32,
    readlane,
    rsrc,
    traced,
    uniform,
    wave_max,
    wave_sum,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    WAVES,
    cdiv,
    sel,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
    publish,
    uniform_addr,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    HEAD_DIM,
    HEADS,
    KV_BLOCK,
    MIN_PER,
    OPART,
    PSTAT,
    Q_PER_KV,
    QKV_ROWS,
    ROW_G,
    WINDOW,
    XO,
    K2Build,
    crdy_pair,
    opart_word,
    pstat_word,
    scratch_layout,
    srdy_pair,
    xo_elem,
)

STAGE = "combine"

NEG = -3.4028234663852886e38  # the split's empty max (-FLT_MAX)
_LAY = scratch_layout()
PSTAT_BYTES = _LAY[PSTAT][1]  # the masked loads' buffer sizes: an index
OPART_BYTES = _LAY[OPART][1]  # past live reads zeros, no memory access
assert HEAD_DIM == 64 and Q_PER_KV == WAVES  # lane = dim; the LDS sum by wave
assert HEADS == 2 * Q_PER_KV  # h = 8 kvh + j


@functools.cache
def lds_struct(s: int):
    """The stage's LDS: ``ow`` f32 [WAVES, 64], a wave's partial sum of the
    merged row (2 KiB, any width). Cached: one struct type a width."""

    @fx.struct
    class CombineLds:
        ow: fx.Array[fx.Float32, WAVES * HEAD_DIM, 16]

    return CombineLds


def _imax(a, b):
    return sel(a > b, a, b)


def unit_plan(L, window: bool, n: int):
    """A (token, KV head)'s split at context length L (0 on a pad row,
    clamped to the table): (lo, b0, nb, per, live), as the split unit
    computes it. Python ints or traced Int32 (``sel``)."""
    lo = _imax(L - WINDOW, 0) if window else 0
    b0 = lo // KV_BLOCK
    nb = sel(L > 0, (L - 1) // KV_BLOCK - b0 + 1, 0)
    per = _imax(cdiv(nb, n), MIN_PER)
    return lo, b0, nb, per, cdiv(nb, per)


def _key(c) -> K2Build:
    key = c["key"]
    if not isinstance(key, K2Build) or key.core != "split":
        raise ValueError(f"the combine runs in a K2(split) build, not {key!r}")
    return key


def _ld_bf16(addr, i):
    """bf16 element i of the buffer at ``addr``: one 2-byte load, raw."""
    return bo.buffer_load(rsrc(addr), i, vec_width=1, dtype=T.bf16)


def _f32(v):
    return fx.BFloat16(v).to(fx.Float32)


def combine_loads(c, k) -> dict:
    """Combine k's inputs that need no wait, its first loads (raw): the pad
    word of row t, ``seq_lens[t]`` and the lane's o_gate element (every
    wave: no wave-dependent branch around a load)."""
    a = c["args"]
    t, h = k // HEADS, k % HEADS
    return {
        "pad": ld_i32(a["pad"], t // 4),
        "seq": ld_i32(a["seq_lens"], t),
        "og": _ld_bf16(a["qkvo"], t * QKV_ROWS + ROW_G + h * HEAD_DIM + c["lane"]),
    }


@traced
def stage_combine(c, k, pre=None):
    """Combine k = 16 t + h (CTA ``key.c0() + k``): the (t, kvh) units'
    partials of q head j merged, gated -> ``xo`` row (t, h), then ``crdy[k]``
    published. ``pre``: ``combine_loads(c, k)`` (issued here if None)."""
    tid, lane = c["tid"], c["lane"]
    a, scr = c["args"], c["scr"]
    key = _key(c)
    window, n = key.window, key.nsplit()
    if const_expr(pre is None):
        pre = combine_loads(c, k)
    t, h = k // HEADS, k % HEADS
    kvh, j = h // Q_PER_KV, h % Q_PER_KV
    base = (2 * t + kvh) * n
    w = uniform(c["wave"])  # an SGPR: the readlanes' lane select
    dead = ((pre["pad"] >> (8 * (t % 4))) & 0xFF) != 0
    cap = a["bt_stride"] * KV_BLOCK
    L = dead.select(fx.Int32(0), fx.min(fx.max(pre["seq"], fx.Int32(0)), cap))
    live = unit_plan(L, window, n)[4]
    # the live units' flags (the base re-read: this CTA's split unit put srdy
    # from one lane, uniform_addr's GVN-PRE case)
    if tid < live:
        srdy = uniform_addr(scr["srdy"])
        c["poll"]([(srdy, srdy_pair(base + tid), 1)])
    gpu.barrier()
    # every load of the merge at once, device scope; a unit p >= live loads
    # zeros (masked: no access)
    r_st = rsrc(scr[PSTAT], PSTAT_BYTES)
    r_op = rsrc(scr[OPART], OPART_BYTES)
    st = []
    for i in range_constexpr(cdiv(n, 64)):
        p = lane + 64 * i
        st.append(
            fx.Vector(
                bo.buffer_load(
                    r_st,
                    pstat_word(base + p, j, 0),
                    vec_width=2,
                    dtype=T.f32,
                    mask=p < live,
                    cache_modifier=CM_DEV,
                )
            )
        )
    xs = []
    for i in range_constexpr(cdiv(n, WAVES)):
        p = w + WAVES * i
        xs.append(
            fx.Float32(
                bo.buffer_load(
                    r_op,
                    opart_word(base + p, j, lane),
                    vec_width=1,
                    dtype=T.f32,
                    mask=p < live,
                    cache_modifier=CM_DEV,
                )
            )
        )
    zero = fx.Float32(0.0)
    neg = fx.Float32(NEG)
    mp, lp = [], []
    for i in range_constexpr(len(st)):
        ok = (lane + 64 * i) < live
        mp.append(fx.Float32(ok.select(st[i][0], neg)))
        lp.append(fx.Float32(ok.select(st[i][1], zero)))
    m = mp[0]
    for i in range_constexpr(1, len(mp)):
        m = fx.max(m, mp[i])
    big = wave_max(m)
    wq = [
        fx.Float32((lp[i] > zero).select(hw_exp2(mp[i] - big), zero))
        for i in range(len(mp))
    ]
    lw = lp[0] * wq[0]
    for i in range_constexpr(1, len(lp)):
        lw = fx.Float32(fmath.fma(lp[i], wq[i], lw))
    tot = wave_sum(lw)
    # wave w's partials p = w + 8 i: the weight of p is lane p % 64 of wq[p // 64]
    ow = zero
    for i in range_constexpr(len(xs)):
        wt = readlane(wq[i // WAVES], w + WAVES * (i % WAVES))
        ow = fx.Float32(fmath.fma(xs[i], wt, ow))
    buf = c["lds"][STAGE]["ow"]
    fx.ptr_store(ow, buf + (c["wave"] * HEAD_DIM + lane))
    gpu.barrier()
    if c["wave"] == 0:
        o = fx.ptr_load(buf + lane)
        for v in range_constexpr(1, WAVES):
            o = o + fx.ptr_load(buf + (v * HEAD_DIM + lane))
        has = tot > zero
        d = fx.Float32(has.select(tot, fx.Float32(1.0)))
        o = fx.Float32(has.select(div_rn(o, d, fx.Float32(1.0) / d), zero))
        o = fx.Float32(dead.select(zero, bf16_round(o)))
        y = fx.Float32(dead.select(zero, gate.o_gate(o, _f32(pre["og"]))))
        bo.buffer_store(
            y.to(fx.BFloat16),
            rsrc(scr[XO]),
            xo_elem(t, h, lane),
            cache_modifier=CM_DEV,
        )
    publish(c["put"], scr["crdy"], crdy_pair(t, h), fx.Int32(1), tid == 0)
