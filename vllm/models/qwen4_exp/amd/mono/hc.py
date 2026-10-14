# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hyper-connection (HC) phases of the Flash-Next mono kernels: a
``GatedResidual``'s combine, its per-stream norm and mix, as ``ops/hc.py``
computes them, on the 4-stream state [S, 4 H] every rank holds whole.

    msq      (``NRED`` tasks, 128 columns of every stream): the state's sums of
             squares -> MSQ (j, t, k)
    combine  (``NRED`` tasks): a block output's TP partials in rank order ->
             bf16 block; state_k + block 2 sigmoid(inj_k / 4) -> bf16 state;
             its sums of squares -> MSQ
    down     (21 row groups x 8 K parts of 1280): xn = bf16(state rrms_k (1 +
             w)) built by every task, bf16 GEMV -> fp32 partials; part 0 sums
             the parts in part order -> bf16; the 320 low-rank rows through
             hc_silu -> LORA, the 4 injection rows -> ``inj_out``
    up       (160 tasks: H columns 16 u .. of the 4 streams): LORA -> the gate
             rows (bf16), the gate-mix bf16(mean_k sigmoid(gate_k) xn_k) ->
             BIN, then BINF[u]

A consumer of the block input waits on every up task's flag (``bin_to_lds``).
Every rank runs the same arithmetic on the same state, so the ranks' states
stay equal.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
    CM_NT,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    hw_rsq,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    wave_sum,
)
from vllm.models.kimi_k3.amd.mono.common.plan import THREADS
from vllm.models.kimi_k3.amd.mono.common.ranks import sum_partials
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono.moe import ksplit_rows, sigmoid
from vllm.models.qwen4_exp.amd.mono import layout as L

H, HH, HC, ROWS = L.HIDDEN, L.HH, L.HC, L.ROWS
COLS = 128  # columns of each stream a msq / combine task
NRED = H // COLS  # 20
DOWN_GROUPS = L.DOWN_ROWS // ROWS  # 21: 20 low-rank, 1 injection
INJ_GROUP = L.LORA // ROWS  # 20
DPARTS = 8
DK = HH // DPARTS  # 1280: a down part's K, inside one stream
DOWN_TASKS = DOWN_GROUPS * DPARTS  # 168
UP_TASKS = H // ROWS  # 160
UP_STEPS = L.LORA // 32  # 10 MFMA K steps
UP_HALF = UP_STEPS // 2  # a wave's: wave w takes stream w / 2, half w % 2
LROW = L.LORA + gemv.LDS_PAD  # a LORA row in LDS
DROW = DK + gemv.LDS_PAD  # a down part's xn row in LDS

assert DK % H == 0 or H % DK == 0
assert H % COLS == 0 and L.LORA % 32 == 0 and UP_STEPS % 2 == 0


def f32_of(w):
    return w.bitcast(fx.Float32)


def regions(s):
    """The HC phases' scratch mailboxes, pairs: name -> count."""
    return {
        "msq": NRED * s * HC,
        "dpart": DOWN_TASKS * s * ROWS,
        "lora": s * L.LORA // 2,
        "binf": UP_TASKS,
    }


def bin_bytes(s):
    """The plain block-input rows [S, H] bf16."""
    return s * H * 2


def gemma(x, r, w):
    """``GroupedGemmaRMSNorm``'s element: y = x rrms; y + y w (fp32)."""
    y = x * r
    return y + y * w


# ---------------------------------------------------------------- msq
@traced
def stage_msq(c, j):
    """Columns 128 j .. of each stream: the state's sums of squares (fp32 of the
    bf16 state) -> MSQ (j, t, k). Wave t token t."""
    s, tid, lane = c["S"], c["tid"], c["lane"]
    h = c["hc"]
    mine = tid < s * 64
    t = fx.min(tid // 64, s - 1)
    col = j * COLS + 2 * (tid % 64)
    sqs = []
    for k in range_constexpr(HC):
        w = fx.Int32(
            bo.buffer_load(
                rsrc(h["state"]),
                (t * HH + k * H + col) // 2,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=h["state_cm"],
            )
        )
        z0, z1 = bf_lo(w), bf_hi(w)
        sqs.append(wave_sum(z0 * z0 + z1 * z1))
    if mine & (lane == 0):
        for k in range_constexpr(HC):
            c["put"](h["msq"], (j * s + t) * HC + k, sqs[k])


# ---------------------------------------------------------------- combine
@traced
def stage_combine(c, j, own, inj, src_state, src_cm, dst, dst_cm, msq_put):
    """Columns 128 j ..: the TP ranks' bf16 partials of a block output at peer
    region ``own`` (rank order, fp32) -> bf16 block; state_k + block 2
    sigmoid(inj_k / 4) (fp32) -> bf16 -> ``dst``; their sums of squares ->
    ``msq_put(j, t, k, v)``, after ``dst``'s stores drained. ``inj(t)``: the
    token's 4 injection logits (fp32 of bf16). Wave t token t."""
    s, tid, lane = c["S"], c["tid"], c["lane"]
    mine = tid < s * 64
    t = fx.min(tid // 64, s - 1)
    col = j * COLS + 2 * (tid % 64)
    lo, hi = sum_partials(
        c["poll"], own, lambda src: ((src * s + t) * H + col) // 2, c["TP"]
    )
    b0, b1 = bf16_round(lo), bf16_round(hi)
    lg = inj(t)
    sqs = []
    for k in range_constexpr(HC):
        m = fx.Float32(2.0) * sigmoid(lg[k] * fx.Float32(1.0 / HC))
        at = (t * HH + k * H + col) // 2
        sw = fx.Int32(
            bo.buffer_load(
                rsrc(src_state), at, vec_width=1, dtype=T.i32, cache_modifier=src_cm
            )
        )
        z0 = bf16_round(bf_lo(sw) + b0 * m)
        z1 = bf16_round(bf_hi(sw) + b1 * m)
        if mine:
            bo.buffer_store(
                bf16_pair(z0, z1).bitcast(fx.Int32),
                rsrc(dst),
                at,
                cache_modifier=dst_cm,
            )
        sqs.append(wave_sum(z0 * z0 + z1 * z1))
    rocdl.s_waitcnt(vmcnt=0)
    if mine & (lane == 0):
        for k in range_constexpr(HC):
            msq_put(j, t, k, sqs[k])


# ---------------------------------------------------------------- norm
@traced
def load_rrms(c):
    """Every column task's sums of squares -> rrms (t, k) = rsqrt(mean + eps)
    -> LDS ``rs`` [t HC + k]. Wave t token t, lane j task j's; ``hc["msq_in"]``:
    a previous launch's plain [NRED, S, HC] fp32, else the MSQ mailbox."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    h = c["hc"]
    if wave < s:
        j = fx.min(lane, NRED - 1)
        at = (j * s + wave) * HC
        if const_expr(h["msq_in"] is not None):
            v = fx.Vector(
                bo.buffer_load(rsrc(h["msq_in"]), at, vec_width=4, dtype=T.f32)
            )
            vals = [v[k] for k in range(HC)]
        else:
            got = c["poll"]([(h["msq"], at, 2), (h["msq"], at + 2, 2)])
            vals = [f32_of(got[k // 2][k % 2]) for k in range(HC)]
        for k in range_constexpr(HC):
            x = (lane < NRED).select(vals[k], fx.Float32(0.0))
            r = hw_rsq(wave_sum(x) * fx.Float32(1.0 / H) + fx.Float32(c["eps"]))
            if lane == 0:
                fx.ptr_store(r, c["rs"] + (wave * HC + k))
    gpu.barrier()


@traced
def xn_to_lds(c, k0, klen, xrow):
    """Xn columns k0 .. k0 + klen of the state (inside stream k0 / H) ->
    bf16 LDS rows (``xrow`` a row), 8 columns a lane; ``load_rrms`` first."""
    s, tid = c["S"], c["tid"]
    h = c["hc"]
    k = k0 // H
    words = s * klen // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (klen // 8)
        kk = e % (klen // 8) * 8
        sv = fx.Vector(
            bo.buffer_load(
                rsrc(h["state"]),
                (t * HH + k0 + kk) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=h["state_cm"],
            )
        )
        wv = fx.Vector(
            bo.buffer_load(rsrc(h["w_norm"]), (k0 + kk) // 2, vec_width=4, dtype=T.i32)
        )
        r = fx.ptr_load(c["rs"] + (t * HC + k))
        ys = [
            bf16_pair(
                gemma(bf_lo(sv[d]), r, bf_lo(wv[d])),
                gemma(bf_hi(sv[d]), r, bf_hi(wv[d])),
            ).bitcast(fx.Int32)
            for d in range(4)
        ]
        if tid + THREADS * i < words:
            fx.ptr_store(
                fx.Vector.from_elements(ys, fx.Int32), c["xl"] + (t * xrow + kk) // 2
            )
    gpu.barrier()


# ---------------------------------------------------------------- down
def down_loads(c, d):
    """Down task d's (group d / 8, part d % 8) weights for this wave."""
    g, q = d // DPARTS, d % DPARTS
    return gemv.loads(
        c["lane"], c["wave"], c["hc"]["w_down"], g * ROWS, DK, ldk=HH, k0=q * DK
    )


def hc_silu(v):
    """``hc_silu`` on a bf16 down output: x = v / HC; bf16(x sigmoid(x))."""
    x = v * fx.Float32(1.0 / HC)
    return bf16_round(x * sigmoid(x))


@traced
def stage_down(c, d, ops):
    """Down task d: rows 16 g .. of [down; inject] over K part q (xn built
    here; ``load_rrms`` first). Parts q > 0 publish fp32 partials; part 0 adds
    them in part order, rounds to bf16 (the GEMM's output), and puts the
    low-rank rows through ``hc_silu`` to LORA or the injection rows to
    ``hc["inj_out"](t, r, (v0, v1))`` (rows r, r + 1 < 4)."""
    s, lane, wave, red = c["S"], c["lane"], c["wave"], c["red"]
    g, q = d // DPARTS, d % DPARTS
    xn_to_lds(c, q * DK, DK, DROW)
    acc = gemv.mfmas(lane, wave, c["xl"], DROW, DK, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    ksplit_rows(
        c,
        c["hc"]["dpart"],
        lambda qq, t, r: ((g * DPARTS + qq) * s + t) * ROWS + r,
        q,
        DPARTS,
        lambda t, r, vs: _down_out(c, g, t, r, vs),
    )


@traced
def _down_out(c, g, t, r, vs):
    """Down group g's rows r, r + 1 of token t: low-rank -> LORA through
    ``hc_silu``, injection -> ``hc["inj_out"]``."""
    h = c["hc"]
    if g < INJ_GROUP:
        c["put_bf"](
            h["lora"], t * L.LORA + g * ROWS + r, [hc_silu(vs[0]), hc_silu(vs[1])]
        )
    if (g == INJ_GROUP) & (r < L.INJ):
        h["inj_out"](t, r, vs)


# ---------------------------------------------------------------- up
def up_loads(c, u):
    """Up task u's gate rows for this wave: stream w / 2's rows k H + 16 u ..,
    K steps 5 (w % 2) .. (a step: 32 K, lane group g's 8 at 32 st + 8 g)."""
    lane, wave = c["lane"], c["wave"]
    k = wave // 2
    row = k * H + u * ROWS + lane % ROWS
    g = lane // ROWS
    r_w = rsrc(c["hc"]["w_up"])
    return [
        fx.Vector(
            bo.buffer_load(
                r_w,
                (row * L.LORA + 32 * (UP_HALF * (wave % 2) + i) + 8 * g) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )
        for i in range(UP_HALF)
    ]


@traced
def lora_to_lds(c):
    """Every down group's LORA (bf16 pairs) -> LDS rows; one poll batch."""
    s, tid = c["S"], c["tid"]
    h = c["hc"]
    n = s * L.LORA // 2
    per = (n + THREADS - 1) // THREADS
    got = c["poll"](
        [(h["lora"], fx.min(tid + THREADS * i, n - 1), 1) for i in range(per)]
    )
    for i in range_constexpr(per):
        p = fx.min(tid + THREADS * i, n - 1)
        t = p // (L.LORA // 2)
        cp = p % (L.LORA // 2)
        if tid + THREADS * i < n:
            fx.ptr_store(got[i][0], c["ll"] + (t * LROW // 2 + cp))
    gpu.barrier()


@traced
def stage_up(c, u, ops):
    """H columns 16 u ..: the 4 streams' gate rows against the LORA rows ->
    bf16 gate (the GEMM's output); the gate-mix bf16(sum_k sigmoid(gate_k)
    xn_k / HC) (fp32, stream order) -> BIN; then BINF[u] (``hc["binf"]`` None:
    BIN is the launch's output, no flag)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    h = c["hc"]
    lora_to_lds(c)
    t = fx.min(lane % 16, s - 1)
    g = lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(UP_HALF):
        st = UP_HALF * (wave % 2) + i
        xb = fx.Vector(
            fx.ptr_load(
                c["ll"] + (t * LROW + 32 * st + 8 * g) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        acc = mfma_bf16(ops[i].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        col = u * ROWS + 2 * rp
        y = [fx.Float32(0.0), fx.Float32(0.0)]
        for k in range_constexpr(HC):
            sw = fx.Int32(
                bo.buffer_load(
                    rsrc(h["state"]),
                    (tt * HH + k * H + col) // 2,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=h["state_cm"],
                )
            )
            ww = fx.Int32(
                bo.buffer_load(
                    rsrc(h["w_norm"]), (k * H + col) // 2, vec_width=1, dtype=T.i32
                )
            )
            r = fx.ptr_load(c["rs"] + (tt * HC + k))
            xs = (bf_lo(sw), bf_hi(sw))
            ws = (bf_lo(ww), bf_hi(ww))
            for e in range_constexpr(2):
                gate = bf16_round(
                    row_sum(red, 2 * rp + e, tt, waves=(2 * k, 2 * k + 1))
                )
                xn = bf16_round(gemma(xs[e], r, ws[e]))
                y[e] = y[e] + sigmoid(gate) * xn
        q = fx.Float32(1.0 / HC)
        bo.buffer_store(
            bf16_pair(y[0] * q, y[1] * q).bitcast(fx.Int32),
            rsrc(h["bin"]),
            (tt * H + col) // 2,
            cache_modifier=CM_DEV,
        )
    if const_expr(h["binf"] is not None):
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid == 0:
            c["put"](h["binf"], u, fx.Int32(1))
    gpu.barrier()


@traced
def bin_to_lds(c, xrow):
    """Every up task's flag in -> the block-input rows [S, H] into LDS ``xl``."""
    h = c["hc"]
    if c["tid"] < UP_TASKS:
        c["poll"]([(h["binf"], c["tid"], 1)])
    gpu.barrier()
    gemv.dev_rows_to_lds(c["tid"], h["bin"], H, c["S"], c["xl"], xrow)
