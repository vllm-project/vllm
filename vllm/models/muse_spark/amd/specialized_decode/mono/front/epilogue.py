# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1's epilogue (DESIGN_BC 5.6, Phase B1) on the epilogue CTAs u in [0,
E_UNITS S): the qkv rows out of the GEMV partials, the weightless QK-norm,
GPT-J RoPE on sliding layers, and this step's K / V into the layer's cache.

Stock reference: ``fusedMuseSparkQKNormRopeKVInsertKernel``
(``csrc/fused_muse_spark_qknorm_rope_kv_insert_kernel.cu``), the instance
decode runs (HEADS_PER_GROUP 1, bf16 KV, fp32 table; its gfx950 ISA,
DESIGN_BC 1.3), bit for bit on the same bf16 GEMV rows. A head is 8 lanes
there, lane li dims 8 li .. 8 li + 7; here a head is a wave, lane d dim d::

    x    = f32(bf16(p_0 + p_1))           the two contributors' partials
    tot  = ((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7)),  s_li the serial sum
           RN(x_8li^2) + ... + RN(x_8li+7^2)        (``ops.seq8_sum``)
    x    = RN(x rsq(fma(tot, 2^-6, eps)))           q and k heads only
    rope (sliding layers; pair i of chunk li: dims 2i, 2i + 1, c = cos[pos,
          4 li + i], s = sin[pos, 4 li + i]):
           dim 0      RN(RN(x0 c) - RN(x1 s))       two rounded products
           1, 3, 5    fma(x_odd, c, RN(x_even s))
           2, 4, 6    fma(x_even, c, -RN(x_odd s))
           7          fma(x6, s, RN(x7 c))          the fused product is x6 s
    out  = bf16(x)                        one rounding; v and og verbatim

Full layers are NoPE (``K1Build.rope`` False: no table read). The squares
and the products are separate multiplies (FlyDSL emits no ``contract``
flag, so LLVM fuses nothing it is not given as ``fmath.fma``).

Unit u = token t = u // 5, group g5 = u % 5; wave w is a head slot, lane d a
dim (DESIGN_BC 5.6's table, ``unit_row``): g5 0, 1 q heads 8 g5 + w; g5 2
waves 0, 1 the k heads, 2, 3 the v heads (4-7 idle); g5 3, 4 og heads 8
(g5 - 3) + w. Every live wave polls its row's two ``qpart`` pairs (one
batch), then stores ``qkvo`` (a pad row as zeros) and, on the k / v waves,
the cache: slot >= 0 only (the stock rule: MRV2 pad rows are -1), never a
block at or past ``kv_blocks`` (a garbage slot stays in bounds), never once
this rank has FAILED (a rank that gave up must not write stale K / V for the
steps until the engine stops). A pad row's position is 0 (its K, if its slot
were live, is roped there).

Every scalar of the step (pad, position, slot) is loaded at the CTA's entry
(``boundary.epilogue_inputs``, with the boundary's loads; made wave-uniform
here) and the cos / sin row before the poll: the poll holds only this task's
operands (rule 8). The cache page is a 64-bit address (a block stride times
a block id passes 4 GiB; LHBNC's head stride is ~5.5 GB) built from
wave-uniform values, so its buffer resource stays scalar.

The stage uses no LDS (registers and DPP only) and no barrier.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.mono.abi import FAILED
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    bf16_round,
    hw_rsq,
    rsrc,
    seq8_sum,
    traced,
    uniform,
    xshfl,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    sel,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
    ld_health,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    E_UNITS,
    HEAD_DIM,
    KV_BLOCK,
    KV_HEADS,
    QKV_ROWS,
    RMS_EPS,
    ROW_G,
    ROW_K,
    ROW_Q,
    ROW_V,
    ROWS,
    V_OFF,
    WAVES,
    qpart_pair,
)

INV_D = 2.0**-6  # 1 / HEAD_DIM: the .cu's ``sumsq / 64`` is fma(tot, 2^-6, eps)
ROPE_COLS = HEAD_DIM  # a cos_sin_cache row: cos in 0 .. 31, sin in 32 .. 63
KV_WAVES = 2 * KV_HEADS  # g5 2: the k heads, then the v heads
assert HEAD_DIM == 64 and ROW_K + KV_HEADS * HEAD_DIM == ROW_V and V_OFF == HEAD_DIM
assert 2 * WAVES * HEAD_DIM == ROW_K and 2 * WAVES * HEAD_DIM == QKV_ROWS - ROW_G


def unit_row(g5, w):
    """Unit group ``g5``'s first row of wave ``w`` (the head's dim 0) and
    whether the wave is live, a q / k head (normalized) and a k / v head
    (inserted): (row0, live, qk, kv). Python ints or wave-uniform
    traced Int32 alike."""
    rq = ROW_Q + HEAD_DIM * (WAVES * g5 + w)
    rk = ROW_K + HEAD_DIM * w  # k heads 0, 1 then v heads 0, 1: contiguous
    rg = ROW_G + HEAD_DIM * (WAVES * (g5 - 3) + w)
    row0 = sel(g5 < 2, rq, sel(g5 == 2, rk, rg))
    is_kv = g5 == 2
    kv = is_kv & (w < KV_WAVES)
    live = (g5 != 2) | (w < KV_WAVES)
    qk = (g5 < 2) | (is_kv & (w < KV_HEADS))
    return row0, live, qk, kv


def _uniform64(v):
    """An Int64 every lane holds the same value of (a loaded position or
    slot), wave-uniform: each half through ``uniform``."""
    lo = uniform(fx.Int32(v))
    hi = uniform(fx.Int32(v >> 32))
    return (fx.Int64(hi) << 32) | fx.Int64(fx.Uint32(lo))


def rope_loads(c, pos, lane):
    """The cos / sin of dim ``lane``'s pair at row ``pos`` (Int32, clamped)
    of the layer's fp32 ``cos_sin_cache``: cos[pos, d // 2], sin[pos, d // 2]."""
    r = rsrc(c["args"]["cos_sin"])
    i = pos * ROPE_COLS + lane // 2
    cs = fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))
    sn = fx.Float32(bo.buffer_load(r, i + ROPE_COLS // 2, vec_width=1, dtype=T.f32))
    return cs, sn


def qk_norm(x, lane):
    """A head's weightless RMS norm in the fused insert's order (1.3 steps
    2-5): RN(x rsq(fma(tot, 2^-6, eps))), ``tot`` the 8-lane chains and tree
    of the rounded squares. Every lane of the wave must be active."""
    tot = seq8_sum(x * x, lane)
    q = fx.Float32(fmath.fma(tot, fx.Float32(INV_D), fx.Float32(RMS_EPS)))
    return x * hw_rsq(q)


def rope(x, cs, sn, lane):
    """GPT-J RoPE of a normalized head, the instance's per-dim arithmetic
    (1.3 step 6, review 0 D1): the partner dim d ^ 1 by DPP; both products
    rounded in every lane, then per d % 8 the sub of two rounded products
    (dim 0), the fma with the fused product x6 s (dim 7), or fma(x, c, -+
    RN(x_partner s)) (dims 1-6)."""
    pr = xshfl(x, 1)
    a = pr * sn
    b = x * cs
    k8 = lane % 8
    even = fx.Float32(fmath.fma(x, cs, -a))
    odd = fx.Float32(fmath.fma(x, cs, a))
    d7 = fx.Float32(fmath.fma(pr, sn, b))
    mid = fx.Float32(((lane % 2) == 0).select(even, odd))
    return fx.Float32((k8 == 0).select(b - a, (k8 == 7).select(d7, mid)))


def _partials(c, tile, t, r16):
    """The fp32 sum of row r16 of ``tile``'s two contributors' partials for
    token t, in contributor order."""
    scr = c["scr"]
    got = c["poll"](
        [
            (scr["qpart"], qpart_pair(tile, 0, t, r16), 1),
            (scr["qpart"], qpart_pair(tile, 1, t, r16), 1),
        ]
    )
    return got[0][0].bitcast(fx.Float32) + got[1][0].bitcast(fx.Float32)


def stage_epilogue(c, u, inputs) -> None:
    """Unit u (module docstring): the partials -> ``qkvo``, the k / v heads
    into the cache. RoPE iff the context's build key is a ``K1Build`` with
    ``rope``."""
    rope_on = bool(getattr(c["key"], "rope", False))
    _stage(c, u, inputs, rope_on)


@traced
def _stage(c, u, inputs, rope_on):
    lane = c["lane"]
    a = c["args"]
    t = u // E_UNITS
    g5 = u % E_UNITS
    w = uniform(c["wave"])
    row0, live, qk, kv = unit_row(g5, w)
    row = row0 + lane
    dead = uniform(fx.Int32(inputs["pad"].select(fx.Int32(1), fx.Int32(0)))) != 0
    sl = _uniform64(inputs["slot"])
    if const_expr(rope_on):
        # clamped in Int64, then narrowed (rule 6); issued before the poll
        pos = fx.Int64(dead.select(fx.Int64(0), _uniform64(inputs["pos"])))
        top = fx.Int64(fx.Int32(a["max_pos"])) - 1
        pos = fx.Int64(fx.max(fx.min(pos, top), fx.Int64(0)))
        cs, sn = rope_loads(c, fx.Int32(pos), lane)
    if live:
        tile = row // ROWS
        r16 = row % ROWS
        x = bf16_round(_partials(c, tile, t, r16))
        y = qk_norm(x, lane)
        if const_expr(rope_on):
            y = rope(y, cs, sn, lane)
        yb = bf16_round(fx.Float32(qk.select(y, x)))
        out = fx.Float32(dead.select(fx.Float32(0.0), yb))
        bo.buffer_store(out.to(fx.BFloat16), rsrc(a["qkvo"]), t * QKV_ROWS + row)
        if kv:
            _insert(c, sl, w, yb.to(fx.BFloat16), lane)


@traced
def _insert(c, sl, w, y, lane):
    """K / V head w % 2 (w < 2: K, else V) of slot ``sl`` := ``y`` (lane =
    dim), skipped for a negative slot, a block past ``kv_blocks`` or a FAILED
    rank. The page base (block, head) is 64-bit and wave-uniform."""
    a = c["args"]
    failed = uniform(ld_health(a["health"], FAILED))
    blk = sl // KV_BLOCK
    ok = (sl >= 0) & (blk < fx.Int64(fx.Int32(a["kv_blocks"]))) & (failed == 0)
    if ok:
        head = w % KV_HEADS
        page = (
            a["kv"] + blk * fx.Int64(a["kv_sb"]) + fx.Int64(head) * fx.Int64(a["kv_sh"])
        )
        st = fx.Int32(a["kv_st"])
        n = fx.Int32(sl % KV_BLOCK)
        off = n * st + 2 * (lane + (w // KV_HEADS) * V_OFF)
        bo.buffer_store(
            y,
            rsrc(page, fx.Int64(KV_BLOCK * st)),
            off,
            offset_is_bytes=True,
        )
