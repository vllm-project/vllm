# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the Qwen mono K2s share (Qwen3.8's ``layer``, Qwen3.8-Flash-Next's
``qwen4_exp.amd.mono.k2``): the MXFP4 arithmetic, the online-MXFP4 GEMV and
o_proj, the route (the top-10 of 512 logits, the union U of the step's picks)
and the routed experts (ug, INTER through MXFP4, down), at the rank's shapes
``c["moe"]`` (``layout.MoeDims``).

A stage reads the kernel context ``c``: ``S``, ``TP``, ``KT`` (picks a token),
the thread ids, ``rank``, the mailbox (``put``, ``put_bf``, ``put_words``,
``poll``), LDS (``red``, ``xl``, ``il``, ``contrib``, ``rt``), the scratch
regions ``sgate`` / ``route`` / ``inter``, the peer regions ``ar`` / ``final``,
``args`` and the variant flags ``fse`` / ``oq`` / ``og`` / ``u1``. Where the
router logits are is the model's: ``rlog_at(t, lane, i)`` is token t's logit
of expert lane + 64 i as a poll spec, ``rlog_last`` (n, spec of i) the n
logits that land last, and CTA ``route_cta`` also puts ROUTE (for the tests).
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_NT,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    block_excl_scan,
    div_rn,
    hw_exp2,
    ld_i32,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_max,
    xred,
)
from vllm.models.kimi_k3.amd.mono.common.plan import BLOCKS, THREADS, WAVES
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono import layout as L

E = 512
TOPK = 10
MAX_TOKENS = 8
MAX_U = MAX_TOKENS * TOPK + 1  # distinct experts a step can touch + the fused shared
ROWS = L.ROWS
PS_MIN = 4  # rows from which the fused shared expert's ug runs before the route
UG_AHEAD = 1  # ug tasks a CTA prefetches past the running one
DB = 4  # a wave's down slots whose w2 is in flight together
UG_COLS = 16  # intermediate columns an ug task
LOG2E = 1.0 / math.log(2.0)
TAG_SLOTS = 256


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch (bumped once a step, in the
    graph) and the launch's slot, so no pair of another step or launch passes."""
    e = uniform(ld_i32(epoch, 0))
    return (e % (1 << 22)) * TAG_SLOTS + slot + 1


def f32_of(w):
    return w.bitcast(fx.Float32)


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def silu_mul(g, u):
    return g * sigmoid(g) * u


def pow2(code):
    """The fp32 power of two an e8m0 code stands for."""
    return (code << 23).bitcast(fx.Float32)


def mx_scale(amax):
    """A 1 x 32 block's MXFP4 e8m0 code from its abs max: ceil_pow2(amax / 6)
    (RCEIL, AITER's activation quant), clamped to codes 1 .. 253."""
    d = div_rn(amax, fx.Float32(6.0), fx.Float32(1.0 / 6.0))
    bits = d.bitcast(fx.Int32)
    e = (bits >> 23) & 0xFF
    e = e + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
    return fx.min(fx.max(e, fx.Int32(1)), fx.Int32(253))


def pk_fp4_bf16(old, word, scale, j):
    """A packed bf16 pair / ``scale`` -> e2m1 codes into byte j of ``old``
    (nearest even, saturating at 6)."""
    v = fx.Vector.from_elements([fx.Int32(word)], fx.Int32).bitcast(fx.BFloat16)
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_bf16(
            T.i32,
            as_ir_value(fx.Int32(old)),
            as_ir_value(v),
            as_ir_value(fx.Float32(scale)),
            j,
        )
    )


def pk_fp4_f32(old, a, b, scale, j):
    """Two fp32 / ``scale`` -> e2m1 codes into byte j of ``old``."""
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32,
            as_ir_value(fx.Int32(old)),
            as_ir_value(fx.Float32(a)),
            as_ir_value(fx.Float32(b)),
            as_ir_value(fx.Float32(scale)),
            j,
        )
    )


def pk_bf16_fp4(word, scale, j):
    """Byte j of ``word``'s two e2m1 codes times ``scale`` -> a bf16 pair."""
    v = rocdl.cvt_scalef32_pk_bf16_fp4(
        T.vec(2, T.bf16),
        as_ir_value(fx.Int32(word)),
        as_ir_value(fx.Float32(scale)),
        j,
    )
    return fx.Vector(v).bitcast(fx.Int32)[0]


def mfma_f4(a, b, acc, sa, sb):
    """16x16x128 fp4 x fp4 (e8m0 scales ``sa``, ``sb``: a lane's 32-block)."""
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [a, b, acc, 4, 4, 0, sa, 0, sb]
        )
    )


def fp4x8_bf16(word, scale):
    """The 8 fp4 of ``word`` times ``scale`` -> an MFMA operand (8 bf16 as 4 i32,
    nibble order: element 2 j, 2 j + 1 from byte j)."""
    out = []
    for j in range(4):
        v = rocdl.cvt_scalef32_pk_bf16_fp4(
            T.vec(2, T.bf16),
            as_ir_value(fx.Int32(word)),
            as_ir_value(fx.Float32(scale)),
            j,
        )
        out.append(fx.Vector(v).bitcast(fx.Int32)[0])
    return fx.Vector.from_elements(out, fx.Int32)


def fp4_mfmas(tile, scale, xl, xoff, acc):
    """One 128-K step: a lane's 16 B of fp4 (4 dwords, K 32 (lane / 16) + 8 q ..
    for dword q) against the bf16 B rows at LDS ``xl`` + ``xoff`` (the same K)."""
    for q in range_constexpr(4):
        xb = fx.Vector(
            fx.ptr_load(
                xl + (xoff + 8 * q) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        a = fp4x8_bf16(tile[q], scale)
        acc = mfma_bf16(a.bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc)
    return acc


def fp4_tile(r_w, at):
    return fx.Vector(
        bo.buffer_load(r_w, at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )


def scale_word(r_s, byte_index):
    return fx.Int32(bo.buffer_load(r_s, byte_index // 4, vec_width=1, dtype=T.i32))


# ---------------------------------------------------------------- fp4 GEMV
def _paired(k):
    """K's steps split over the waves in even runs: a wave's run contiguous, a
    scale word two steps. Else wave w takes steps w, w + 8, ..., a word each."""
    return k // L.FP4_STEP % (2 * WAVES) == 0


def fp4_gemv_loads(lane, wave, w, ws, rg, k):
    """An online-MXFP4 linear's rows 16 rg .. (K ``k``, the ASM FP4 GEMM's
    shuffled weight and scales) for this wave: its K steps' tiles and their
    scale words (``_paired``; past the last step clamped, ``fp4_gemv`` zeroes
    those)."""
    n = k // L.FP4_STEP
    r_w, r_s = rsrc(w), rsrc(ws)
    row = rg * ROWS + lane % ROWS
    cols = L.scale_cols(k)
    if const_expr(_paired(k)):
        per = n // WAVES
        tiles = [
            fp4_tile(r_w, L.fp4_tile_dword(0, rg, k, wave * per + i, lane))
            for i in range(per)
        ]
        words = [
            scale_word(
                r_s, L.scale_index(row, 4 * (wave * per + 2 * i) + lane // 16, cols)
            )
            for i in range(per // 2)
        ]
        return tiles + words
    per = (n + WAVES - 1) // WAVES
    sts = [fx.min(wave + WAVES * i, n - 1) for i in range(per)]
    tiles = [fp4_tile(r_w, L.fp4_tile_dword(0, rg, k, st, lane)) for st in sts]
    words = [
        scale_word(r_s, L.scale_index(row, 4 * st + lane // 16, cols)) for st in sts
    ]
    return tiles + words


def fp4_gemv(lane, wave, xl, xrow, k, s, rg, ops):
    """The ``fp4_gemv_loads`` operands against the MXFP4-exact bf16 rows at LDS
    ``xl`` (row stride ``xrow``) -> this wave's fp32 partial MFMA tile."""
    n = k // L.FP4_STEP
    t = fx.min(lane % 16, s - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    if const_expr(_paired(k)):
        per = n // WAVES
        for i in range_constexpr(per):
            code = (ops[per + i // 2] >> L.scale_code_shift(i, rg)) & 0xFF
            xoff = t * xrow + (wave * per + i) * L.FP4_STEP + 32 * (lane // 16)
            acc = fp4_mfmas(ops[i], pow2(code), xl, xoff, acc)
        return acc
    per = (n + WAVES - 1) // WAVES
    for i in range_constexpr(per):
        raw = wave + WAVES * i
        st = fx.min(raw, n - 1)
        code = (ops[per + i] >> L.scale_code_shift(st, rg)) & 0xFF
        scale = pow2(code)
        if const_expr(per * WAVES != n):
            scale = (raw < n).select(scale, fx.Float32(0.0))
        xoff = t * xrow + st * L.FP4_STEP + 32 * (lane // 16)
        acc = fp4_mfmas(ops[i], scale, xl, xoff, acc)
    return acc


@traced
def ksplit_rows(c, region, at, q, parts, out):
    """A K-split GEMV task's 16 rows (``red``; part q of ``parts``): parts
    q > 0 publish fp32 partials to ``region`` at ``at(qq, t, r)``; part 0 adds
    them in part order, rounds to bf16 (the GEMM's output) and passes token t's
    rows r, r + 1 to ``out(t, r, vs)``."""
    s, tid, red = c["S"], c["tid"], c["red"]
    if q > 0:  # noqa: SIM102 (traced: `and` does not combine traced values)
        if tid < ROWS * s:
            r = tid % ROWS
            t = tid // ROWS
            c["put"](region, at(q, t, r), row_sum(red, r, t))
    if q == 0:  # noqa: SIM102
        if tid < (ROWS // 2) * s:
            r = 2 * (tid % (ROWS // 2))
            t = tid // (ROWS // 2)
            got = c["poll"](
                [
                    (region, at(qq, t, r + d), 1)
                    for d in range(2)
                    for qq in range(1, parts)
                ]
            )
            vs = []
            for d in range_constexpr(2):
                v = row_sum(red, r + d, t)
                for qq in range_constexpr(parts - 1):
                    v = v + f32_of(got[d * (parts - 1) + qq][0])
                vs.append(bf16_round(v))
            out(t, r, vs)
    gpu.barrier()


# ---------------------------------------------------------------- x through MXFP4
def _quad_scale(ptr):
    """A lane's 8 bf16 at LDS ``ptr`` (4 words) and their 1 x 32 block's e8m0
    code (the amax over the block's 4 lanes)."""
    wv = fx.Vector(fx.ptr_load(ptr, result_type=fx.Vector.make_type(4, fx.Int32)))
    vs = []
    for d in range_constexpr(4):
        vs += [bf_lo(wv[d]), bf_hi(wv[d])]
    amax = fx.Float32(0.0)
    for j in range_constexpr(8):
        amax = fx.max(amax, fx.Float32(fmath.absf(vs[j])))
    for off in range_constexpr(2):
        amax = xred(amax, 1 << off, fx.max)
    return wv, mx_scale(amax)


@traced
def quant_x(c, k):
    """The rows of K ``k`` in LDS through MXFP4 in place, 1 x 32 blocks (RCEIL
    scale), 8 elements a lane (a block on 4 adjacent lanes: one 16 B LDS access
    a lane, conflict-free): an fp4 linear's input rows, as
    ``per_1x32_f4_quant_hip`` quantizes the ASM FP4 GEMM's."""
    s, tid, xl = c["S"], c["tid"], c["xl"]
    xrow = k + gemv.LDS_PAD
    nq = s * k // 8
    for i in range_constexpr((nq + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, nq - 1)
        at = ((e // (k // 8)) * xrow + 8 * (e % (k // 8))) // 2
        wv, code = _quad_scale(xl + at)
        sc = pow2(code)
        q = fx.Int32(0)
        for d in range_constexpr(4):
            q = pk_fp4_bf16(q, wv[d], sc, d)
        ys = [pk_bf16_fp4(q, sc, d) for d in range(4)]
        if tid + THREADS * i < nq:
            fx.ptr_store(fx.Vector.from_elements(ys, fx.Int32), xl + at)
    gpu.barrier()


def xq_row(k):
    """Bytes a token's fp4 x row of K ``k`` takes in LDS (padded)."""
    return k // 2 + 16


def xq_scales(s, k):
    """Dword of LDS ``xl`` where the fp4 x rows' e8m0 codes start (a dword a
    1 x 32 block, token-major)."""
    return s * xq_row(k) // 4


@traced
def quant_x_fp4(c, k):
    """The x rows in LDS (bf16, K ``k``) -> MXFP4 codes over them,
    ``quant_x``'s arithmetic and lanes (the routed experts' input, as the stock
    decode MoE quantizes it): token t's block b at byte t ``xq_row`` + 16 b
    (element 2 j in byte j's low nibble; a lane's 8 elements one word), its
    code at dword ``xq_scales`` + t (k / 32) + b. Every thread reads before
    any writes."""
    s, tid, xl = c["S"], c["tid"], c["xl"]
    xrow = k + gemv.LDS_PAD
    kq = k // 8
    nq = s * kq
    rounds = (nq + THREADS - 1) // THREADS
    # four rounds read, then their words written: a word lands below every
    # later round's x, so only its own group's reads must precede it; the
    # codes (four rounds' a register) after every read
    codes = []
    for g0 in range_constexpr(0, rounds, 4):
        words = []
        for i in range_constexpr(g0, min(g0 + 4, rounds)):
            e = fx.min(tid + THREADS * i, nq - 1)
            wv, code = _quad_scale(xl + ((e // kq) * xrow + 8 * (e % kq)) // 2)
            w = fx.Int32(0)
            for d in range_constexpr(4):
                w = pk_fp4_bf16(w, wv[d], pow2(code), d)
            words.append(w)
            if const_expr(i % 4 == 0):
                codes.append(code)
            else:
                codes[-1] = codes[-1] | (code << (8 * (i % 4)))
        gpu.barrier()
        for i in range_constexpr(g0, min(g0 + 4, rounds)):
            e = fx.min(tid + THREADS * i, nq - 1)
            if tid + THREADS * i < nq:
                fx.ptr_store(words[i - g0], xl + ((e // kq) * xq_row(k) // 4 + e % kq))
    gpu.barrier()
    for i in range_constexpr(rounds):
        e = fx.min(tid + THREADS * i, nq - 1)
        if (tid + THREADS * i < nq) & (e % 4 == 0):
            code = (codes[i // 4] >> (8 * (i % 4))) & 0xFF
            fx.ptr_store(code, xl + (xq_scales(s, k) + e // 4))
    gpu.barrier()


# ---------------------------------------------------------------- o_proj
def o_loads(c, rg, k):
    """o_proj rows 16 rg .. (K ``k``): this wave's weights."""
    lane, wave, a = c["lane"], c["wave"], c["args"]
    if const_expr(c["oq"]):
        return fp4_gemv_loads(lane, wave, a["w_o"], a["w_os"], rg, k)
    return gemv.loads(lane, wave, a["w_o"], rg * ROWS, k)


@traced
def stage_o(c, rg, ops, k):
    """Rows 16 rg ..: GEMV of the LDS core rows -> bf16 partials -> every rank's
    AR region, this rank's slot (thread: token, row pair)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    if const_expr(c["oq"]):
        acc = fp4_gemv(lane, wave, c["xl"], k + gemv.LDS_PAD, k, s, rg, ops)
    else:
        acc = gemv.mfmas(lane, wave, c["xl"], k + gemv.LDS_PAD, k, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * (ROWS // 2):
        tt = tid // (ROWS // 2)
        rp = tid % (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, tt))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
        at = (c["rank"] * s + tt) * c["moe"].hidden + rg * ROWS + 2 * rp
        for p in range_constexpr(c["TP"]):
            c["put_bf"](c["peer"]["ar"][p], at, [v0, v1])
    gpu.barrier()


@traced
def gated_rows_to_lds(c, k):
    """``og``: bf16(core x sigmoid(gate)) (fp32 math, as the compiled
    ``o * torch.sigmoid(gate)``) -> the o_proj input rows in LDS."""
    tid, s, xl = c["tid"], c["S"], c["xl"]
    a = c["args"]
    words = s * k // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        at = (t * k + kk) // 2
        v = fx.Vector(bo.buffer_load(rsrc(a["core"]), at, vec_width=4, dtype=T.i32))
        g = fx.Vector(bo.buffer_load(rsrc(a["gate"]), at, vec_width=4, dtype=T.i32))
        y = fx.Vector.from_elements(
            [
                bf16_pair(
                    bf_lo(v[d]) * sigmoid(bf_lo(g[d])),
                    bf_hi(v[d]) * sigmoid(bf_hi(g[d])),
                ).bitcast(fx.Int32)
                for d in range(4)
            ],
            fx.Int32,
        )
        if tid + THREADS * i < words:
            fx.ptr_store(y, xl + (t * (k + gemv.LDS_PAD) + kk) // 2)
    gpu.barrier()


@traced
def run_oproj(c, oops, k):
    """This CTA's o_proj tasks (rows 16 (bid + 256 j) .., K ``k``) on the core
    rows (plain loads: the previous launch wrote them), their weights ``oops``
    already in flight."""
    core = c["args"]["core"]
    if const_expr(c["og"]):
        gated_rows_to_lds(c, k)
    else:
        gemv.rows_to_lds(c["tid"], core, k, c["S"], c["xl"], k + gemv.LDS_PAD)
    if const_expr(c["oq"]):
        quant_x(c, k)
    for j in range_constexpr(len(oops)):
        stage_o(c, c["bid"] + BLOCKS * j, oops[j], k)


# ---------------------------------------------------------------- route
_SORT8 = (
    (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7), (0, 1),
    (2, 3), (4, 5), (6, 7), (2, 4), (3, 5), (1, 4), (3, 6), (1, 2), (3, 4), (5, 6),
)  # fmt: skip


def _route_mp(c, lgs):
    """The top-10 of a token's logits as 10 wave max pops, softmax-renormalized
    over the 10. A pick's key is its logit's order-preserving key (16 bits:
    bf16) over 511 - expert, sign-flipped for a signed max: distinct, equal
    logits ordered by the lower expert, so the pops come out ranked; each
    lane's 8 keys sorted first, a pop shifts the winner's list."""
    lane = c["lane"]
    imin = fx.Int32(-2147483648)
    hi16 = fx.Int32(-65536)
    ks = []
    for i in range_constexpr(E // 64):
        b = lgs[i].bitcast(fx.Int32)
        ok = (b < fx.Int32(0)).select(~b, b ^ imin) & hi16
        ks.append((ok | (fx.Int32(E - 1) - (lane + 64 * i))) ^ imin)
    for i, j in _SORT8:
        ks[i], ks[j] = fx.max(ks[i], ks[j]), fx.min(ks[i], ks[j])

    def logit(k):
        u = k ^ imin
        h = u & hi16
        return (h < fx.Int32(0)).select(h ^ imin, ~h & hi16).bitcast(fx.Float32), u

    mine = imin
    top = fx.Float32(0.0)
    total = fx.Float32(0.0)
    mex = fx.Float32(0.0)
    for j in range_constexpr(TOPK):
        m = wave_max(ks[0])
        hit = ks[0] == m
        for i in range_constexpr(E // 64 - 1):
            ks[i] = hit.select(ks[i + 1], ks[i])
        ks[E // 64 - 1] = hit.select(imin, ks[E // 64 - 1])
        lg, _ = logit(m)
        if j == 0:
            top = lg
        ex = hw_exp2((lg - top) * fx.Float32(LOG2E))
        total = total + ex
        sel = lane == j
        mine = sel.select(m, mine)
        mex = sel.select(ex, mex)
    _, u = logit(mine)
    e = fx.Int32(E - 1) - (u & fx.Int32(0xFFFF))
    inv = fx.Float32(1.0) / total
    return fx.min(lane, TOPK - 1), fx.min(fx.max(e, 0), fx.Int32(E - 1)), mex * inv


@traced
def route_token(c, t):
    """Token t on this wave: the 512 bf16 logits (packed pairs), 8 a lane (expert
    lane + 64 i; a NaN logit, a warmup's garbage row, ranks last) -> the top-10
    (``_route_mp``). Returns (rank, expert, weight), lanes < ``TOPK``."""
    lane = c["lane"]
    per = E // 64
    specs = [c["rlog_at"](t, lane, i) for i in range(per)]
    got = c["poll"](specs, batch=len(specs))
    odd = (lane % 2) == 1
    lgs = []
    for i in range_constexpr(per):
        w = got[i][0]
        lg = odd.select(bf_hi(w), bf_lo(w))
        lgs.append((lg == lg).select(lg, fx.Float32(-(2.0**127))))
    return _route_mp(c, lgs)


def put_route(c, t, rank, e, w):
    c["put"](c["route"], t * 2 * TOPK + rank, e)
    c["put"](c["route"], t * 2 * TOPK + TOPK + rank, w)


@traced
def local_route(c):
    """Every CTA routes its tokens itself, wave t token t, into its LDS route
    (``load_route``'s layout; ``load_route``'s barrier follows). Waits on the
    logits that land last first, so the full polls don't spin."""
    lane, wave, rt, kt, s = c["lane"], c["wave"], c["rt"], c["KT"], c["S"]
    n, last = c["rlog_last"]
    if c["tid"] < n:
        c["poll"]([last(c["tid"])])
    gpu.barrier()
    if wave < s:
        rank, e, w = route_token(c, wave)
        if lane < TOPK:
            fx.ptr_store(e, rt["route"] + (wave * 2 * kt + rank))
            fx.ptr_store(w.bitcast(fx.Int32), rt["route"] + (wave * 2 * kt + kt + rank))
            if c["bid"] == c["route_cta"]:
                put_route(c, wave, rank, e, w)


@traced
def load_route(c):
    """The LDS route (ids, weights; a token's ``KT`` picks) -> the union U in
    expert order: flag[e] -> slot, expert[slot], kof[slot][t] (the token's pick
    index, -1 if not picked); fused, the shared expert is U's last slot, every
    token's pick ``TOPK`` with its SGATE weight. Returns |U|."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt, kt = c["rt"], c["KT"]
    if const_expr(c["fse"]):  # noqa: SIM102 (traced: `and` does not combine traced values)
        if tid < s:
            g = c["poll"]([(c["sgate"], tid, 1)])[0][0]
            fx.ptr_store(fx.Int32(E), rt["route"] + (tid * 2 * kt + TOPK))
            fx.ptr_store(g, rt["route"] + (tid * 2 * kt + kt + TOPK))
    if const_expr(c["u1"]):
        # local_route's stores are this wave's (wave 0): no barrier first
        if tid < TOPK:
            e = fx.max(fx.min(fx.ptr_load(rt["route"] + tid), E - 1), 0)
            fx.ptr_store(e, rt["expert"] + tid)
            fx.ptr_store(fx.Int32(tid), rt["kof"] + tid * 8)
        if const_expr(c["fse"]):  # noqa: SIM102
            if tid == 0:
                fx.ptr_store(fx.Int32(E), rt["expert"] + TOPK)
                fx.ptr_store(fx.Int32(TOPK), rt["kof"] + TOPK * 8)
        gpu.barrier()
        return fx.Int32(kt)
    for i in range_constexpr((E + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < E:
            fx.ptr_store(fx.Int32(0), rt["flag"] + e)
    for i in range_constexpr((MAX_U * 8 + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < MAX_U * 8:
            fx.ptr_store(fx.Int32(-1), rt["kof"] + j)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * kt + k)), E - 1), 0)
        fx.ptr_store(fx.Int32(1), rt["flag"] + e)
    gpu.barrier()
    # E = 512 = THREADS: an expert a thread
    f = fx.ptr_load(rt["flag"] + tid)
    off, tot = block_excl_scan(f, lane, wave, rt["scan"])
    gpu.barrier()
    if f == 1:
        fx.ptr_store(off, rt["flag"] + tid)
        fx.ptr_store(fx.Int32(tid), rt["expert"] + off)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * kt + k)), E - 1), 0)
        slot = fx.ptr_load(rt["flag"] + e)
        fx.ptr_store(fx.Int32(k), rt["kof"] + (slot * 8 + t))
    if const_expr(c["fse"]):
        if tid < s:
            fx.ptr_store(fx.Int32(TOPK), rt["kof"] + (tot * 8 + tid))
        if tid == 0:
            fx.ptr_store(fx.Int32(E), rt["expert"] + tot)
        tot = tot + 1
    gpu.barrier()
    return uniform(tot)


# ---------------------------------------------------------------- ug
def ug_steps(d):
    """A wave's K quarter of w13's steps."""
    return d.hidden // L.FP4_STEP // 4


def ug_loads(c, task, live, shared=False):
    """Task ``task``'s w13 operands for this wave (gate or up by wave % 2, K
    quarter wave / 2) and its scale words (a word two steps where a quarter's
    steps pair up, else a word a step); out of bounds (no memory access,
    zeros) where ``live`` is false. ``shared``: the fused shared expert's task
    ``task`` (no union yet)."""
    lane, wave = c["lane"], c["wave"]
    a, rt, d = c["args"], c["rt"], c["moe"]
    groups = d.ri // UG_COLS
    n = ug_steps(d)
    slot = task // groups
    g = task % groups
    gu = wave % 2
    kq = wave // 2
    rg = L.w13_group(d, gu, g)
    wb, sb = d.rip * d.hidden, 2 * d.rip * d.hidden // 32
    if const_expr(shared):
        e = 0
        r_w = rsrc(a["w13"] + fx.Int64(E * wb), live.select(fx.Int32(wb), fx.Int32(0)))
        r_s = rsrc(a["w13s"] + fx.Int64(E * sb), live.select(fx.Int32(sb), fx.Int32(0)))
    else:
        e = uniform(fx.ptr_load(rt["expert"] + slot))
        # + the fused shared expert, expert E
        n_w, n_s = (E + 1) * wb, (E + 1) * sb
        r_w = rsrc(a["w13"], live.select(fx.Int32(n_w), fx.Int32(0)))
        r_s = rsrc(a["w13s"], live.select(fx.Int32(n_s), fx.Int32(0)))
    base = L.w13_base(d, e)
    tiles = [
        fp4_tile(r_w, L.fp4_tile_dword(base, rg, d.hidden, kq * n + i, lane))
        for i in range(n)
    ]
    row = rg * ROWS + lane % ROWS
    pair = 2 if n % 2 == 0 else 1
    words = [
        scale_word(
            r_s, L.w13_scale_index(d, e, row, 4 * (kq * n + pair * i) + lane // 16)
        )
        for i in range(n // pair)
    ]
    return tiles + words


@traced
def stage_ug(c, task, ops, shared=False):
    """Slot task / groups, columns 16 (task % groups) ..: the gate and up rows
    against the x rows' MXFP4 codes in LDS (``ops``: ``ug_loads``); silu(g) u ->
    fp32 -> INTER of each picking token (``shared``: every token's pick
    ``TOPK``)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt, d = c["rt"], c["moe"]
    groups = d.ri // UG_COLS
    n = ug_steps(d)
    pair = 2 if n % 2 == 0 else 1
    slot = task // groups
    g = task % groups
    kq = wave // 2
    rg = L.w13_group(d, wave % 2, g)
    tiles = ops[:n]
    words = ops[n:]
    t = fx.min(lane % 16, s - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(n):
        st = kq * n + i
        code = (words[i // pair] >> L.scale_code_shift(st, rg)) & 0xFF
        xq = (t * xq_row(d.hidden) + st * (L.FP4_STEP // 2)) // 4 + 4 * (lane // 16)
        xb = fx.Vector(
            fx.ptr_load(c["xl"] + xq, result_type=fx.Vector.make_type(4, fx.Int32))
        )
        sb = fx.ptr_load(
            c["xl"]
            + (xq_scales(s, d.hidden) + t * (d.hidden // 32) + 4 * st + lane // 16)
        )
        acc = mfma_f4(tiles[i], xb, acc, code, sb)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    # thread (token, column pair): g, u summed over the K quarters
    if tid < s * (UG_COLS // 2):
        tt = tid // (UG_COLS // 2)
        cp = tid % (UG_COLS // 2)
        if const_expr(shared):
            k = fx.Int32(TOPK)
        else:
            k = fx.ptr_load(rt["kof"] + (slot * 8 + tt))
        gw = (0, 2, 4, 6)
        uw = (1, 3, 5, 7)
        v = [
            silu_mul(
                row_sum(red, 2 * cp + d_, tt, waves=gw),
                row_sum(red, 2 * cp + d_, tt, waves=uw),
            ).bitcast(fx.Int32)
            for d_ in range(2)
        ]
        if k >= 0:
            pick = tt * c["KT"] + k
            at = pick * d.ri + g * UG_COLS + 2 * cp
            c["put_words"](c["inter"], at, v)
    gpu.barrier()


@traced
def run_ug(c, ps, ps0, x_to_lds):
    """The routed experts' gate | up: x into LDS (``x_to_lds``) and through
    MXFP4 while the logits come in, the route, then the union's ug tasks over
    every CTA. ``ps``: the fused shared expert's tasks run first on CTAs
    ``ps0`` .. (they need no route). Returns |U|."""
    bid, d = c["bid"], c["moe"]
    groups = d.ri // UG_COLS
    is_ps = (bid >= ps0) & (bid < ps0 + groups)
    if const_expr(ps):
        sops = ug_loads(c, fx.min(fx.max(bid - ps0, 0), groups - 1), is_ps, True)
    x_to_lds()
    quant_x_fp4(c, d.hidden)
    if const_expr(ps):  # noqa: SIM102 (traced: `and` does not combine traced values)
        if is_ps:
            stage_ug(c, bid - ps0, sops, True)
    local_route(c)
    nu = load_route(c)
    total = (nu - int(ps)) * groups
    last = total - 1
    # reversed placement: the round's leftover tasks go to the high CTAs,
    # past the reduction ones; a prefetch past the last task loads nothing
    task = BLOCKS - 1 - bid

    def task_loads(n):
        return ug_loads(c, fx.min(n, last), n <= last)

    ops = [task_loads(task + BLOCKS * j) for j in range(UG_AHEAD)]
    while task < total:
        ops_n = task_loads(task + BLOCKS * UG_AHEAD)
        stage_ug(c, task, ops[0])
        task = task + BLOCKS
        ops = ops[1:] + [ops_n]
    return nu


# ---------------------------------------------------------------- down
def irow(d):
    """Bytes a pick's fp4 INTER row (RIP) takes in LDS (padded)."""
    return d.rip // 2 + 16


def iq_scales(d, s, kt):
    """Dword of LDS ``il`` where the fp4 INTER rows' e8m0 codes start (a dword
    a 1 x 32 block, RIP / 32 a pick)."""
    return s * kt * irow(d) // 4


@traced
def load_inter(c):
    """Every pick's INTER (fp32, RI columns) through MXFP4 by 32 columns (RCEIL
    scale, the block's 16 column pairs on 16 lanes of a row), as the stock
    decode kernel quantizes its intermediate -> LDS (fp4 rows of RIP, padded,
    the pad columns zero codes; then the e8m0 codes); one poll batch."""
    s, tid, d, kt = c["S"], c["tid"], c["moe"], c["KT"]
    ri, rip = d.ri, d.rip
    ir = irow(d)
    nb = rip // 32
    n = s * kt * ri // 2
    per = (n + THREADS - 1) // THREADS
    got = c["poll"](
        [(c["inter"], 2 * fx.min(tid + THREADS * i, n - 1), 2) for i in range(per)]
    )
    for i in range_constexpr(per):
        p = fx.min(tid + THREADS * i, n - 1)
        pick = p // (ri // 2)
        cp = p % (ri // 2)
        h0, h1 = f32_of(got[i][0]), f32_of(got[i][1])
        amax = fx.max(fx.Float32(fmath.absf(h0)), fx.Float32(fmath.absf(h1)))
        for off in range_constexpr(4):
            amax = xred(amax, 1 << off, fx.max)
        e = mx_scale(amax)
        # a lane's byte; four lanes' bytes -> a word, the block's code
        v = pk_fp4_f32(fx.Int32(0), h0, h1, pow2(e), 0)
        v = v << (8 * (cp % 4))
        for off in range_constexpr(2):
            v = xred(v, 1 << off, lambda a, b: a | b)
        ok = tid + THREADS * i < n
        if ok & (cp % 4 == 0):
            fx.ptr_store(v, c["il"] + (pick * ir + cp) // 4)
        if ok & (cp % 16 == 0):
            fx.ptr_store(e, c["il"] + (iq_scales(d, s, kt) + pick * nb + cp // 16))
    if const_expr(rip > ri):
        pw = (rip - ri) // 8  # pad dwords a pick
        pb = (rip - ri) // 32  # pad blocks a pick
        for i in range_constexpr((s * kt * pw + THREADS - 1) // THREADS):
            j = tid + THREADS * i
            if j < s * kt * pw:
                fx.ptr_store(
                    fx.Int32(0), c["il"] + ((j // pw) * ir + ri // 2) // 4 + j % pw
                )
        if tid < s * kt * pb:
            fx.ptr_store(
                fx.Int32(127),
                c["il"]
                + (iq_scales(d, s, kt) + (tid // pb) * nb + ri // 32 + tid % pb),
            )
    gpu.barrier()


def down_loads(c, rgd, nu, b0):
    """The w2 tiles and scale words of a wave's batch from slot b0 (slots b0,
    b0 + 8, ...; past |U| the last slot's)."""
    lane, d = c["lane"], c["moe"]
    a, rt = c["args"], c["rt"]
    r_w = rsrc(a["w2"])
    r_s = rsrc(a["w2s"])
    row = rgd * ROWS + lane % ROWS
    w2_steps = d.rip // L.FP4_STEP
    tiles, words = [], []
    for j in range(DB):
        sl = fx.min(b0 + WAVES * j, nu - 1)
        e = uniform(fx.ptr_load(rt["expert"] + sl))
        base = L.w2_base(d, e)
        tiles += [
            fp4_tile(r_w, L.fp4_tile_dword(base, rgd, d.rip, st, lane))
            for st in range(w2_steps)
        ]
        words += [
            scale_word(r_s, L.w2_scale_index(d, e, row, 4 * st + lane // 16))
            for st in range(0, w2_steps, 2)
        ]
    return tiles + words


@traced
def stage_down(c, rgd, nu, ops0, rgd_next=None):
    """Hidden rows 16 rgd ..: every slot's w2 against its picks' INTER (wave w
    the slots w, w + 8, ...; a batch in flight under the last one's MFMAs)
    -> fp32 (t, k) contributions times the route weight in LDS -> summed in
    top-k order -> bf16; unfused, + the shared down rows against h, times
    SGATE -> bf16 partial -> every rank's FINAL. The last batch's prefetch is
    row group ``rgd_next``'s first batch, returned (the loop runs: |U| >=
    TOPK > WAVES)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt, kt, fse, d = c["rt"], c["KT"], c["fse"], c["moe"]
    t = fx.min(lane % 16, s - 1)
    w2_steps = d.rip // L.FP4_STEP
    nw = (w2_steps + 1) // 2
    nt = DB * w2_steps
    if const_expr(not fse):
        # the shared down's operands first: K 256 is two chunks (waves 0, 1)
        sops = gemv.loads(lane, wave, c["args"]["w_sd"], rgd * ROWS, d.si)
    for i in range_constexpr((s * kt * ROWS + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < s * kt * ROWS:
            fx.ptr_store(fx.Float32(0.0), c["contrib"] + j)
    gpu.barrier()
    b0 = wave
    ops = ops0
    while b0 < nu:
        nb = b0 + WAVES * DB
        if const_expr(rgd_next is None):
            ops_n = down_loads(c, rgd, nu, fx.min(nb, nu - 1))
        else:
            more = nb < nu
            ops_n = down_loads(
                c,
                more.select(rgd, rgd_next),
                nu,
                more.select(nb, fx.min(wave, nu - 1)),
            )
        for j in range_constexpr(DB):
            sl = fx.min(b0 + WAVES * j, nu - 1)
            live = (b0 + WAVES * j) < nu
            k = fx.ptr_load(rt["kof"] + (sl * 8 + t))
            picked = (k >= 0) & (lane % 16 < s)
            pick = t * kt + fx.max(k, 0)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(w2_steps):
                word = ops[nt + j * nw + st // 2]
                code = (word >> L.scale_code_shift(st, rgd)) & 0xFF
                tile = ops[j * w2_steps + st]
                xq = (pick * irow(d) + st * (L.FP4_STEP // 2)) // 4 + 4 * (lane // 16)
                xb = fx.Vector(
                    fx.ptr_load(
                        c["il"] + xq, result_type=fx.Vector.make_type(4, fx.Int32)
                    )
                )
                xb = fx.Vector.from_elements(
                    [picked.select(xb[q], fx.Int32(0)) for q in range(4)], fx.Int32
                )
                sb = fx.ptr_load(
                    c["il"]
                    + (iq_scales(d, s, kt) + pick * (d.rip // 32) + 4 * st + lane // 16)
                )
                acc = mfma_f4(tile, xb, acc, code, sb)
            if live & picked:
                wt = f32_of(fx.ptr_load(rt["route"] + (t * 2 * kt + kt + fx.max(k, 0))))
                for i in range_constexpr(4):
                    row = 4 * (lane // 16) + i
                    fx.ptr_store(
                        acc[i] * wt,
                        c["contrib"] + ((t * kt + fx.max(k, 0)) * ROWS + row),
                    )
        b0 = nb
        ops = ops_n
    if const_expr(not fse):
        # the shared down: rows 16 rgd .. against h
        acc = gemv.mfmas(lane, wave, c["hl"], d.si + gemv.LDS_PAD, d.si, s, sops)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        if const_expr(not fse):
            sg = f32_of(c["poll"]([(c["sgate"], tt, 1)])[0][0])
        ys = []
        for q in range_constexpr(2):
            r = 2 * rp + q
            y = fx.Float32(0.0)
            for k in range_constexpr(kt):
                y = y + fx.ptr_load(c["contrib"] + ((tt * kt + k) * ROWS + r))
            if const_expr(fse):
                ys.append(bf16_round(y))
            else:
                sd = bf16_round(bf16_round(row_sum(red, r, tt)) * sg)
                ys.append(bf16_round(bf16_round(y) + sd))
        for p in range_constexpr(c["TP"]):
            c["put_bf"](
                c["peer"]["final"][p],
                (c["rank"] * s + tt) * d.hidden + rgd * ROWS + 2 * rp,
                ys,
            )
    gpu.barrier()
    return ops
