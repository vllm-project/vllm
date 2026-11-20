# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device helpers of the DSpark drafter kernels: memory, bounded waits, the MXFP8
GEMV and the row norms.

Every function is traced inside a `@flyc.kernel` body; the ones with control
flow over traced values are `@traced` (FlyDSL rewrites only the kernel body's
own AST). Conventions:

- a global `base` is a wave-uniform Int64 address (a buffer resource); offsets
  are per lane, in elements;
- tagged pairs are 8 bytes: (value word, tag word);
- data other CTAs read after a flag is stored device-coherent (`CM_DEV`) and
  published by `publish`.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.minimax_m3.amd.mono.drafter import layout
from vllm.models.minimax_m3.amd.mono.kernels.common import (
    CM_DEV,
    CM_SYS,
    bf16_pair,
    bo,
    butterfly,
    fp8_pack4,
    hw_exp2,
    memrealtime,
    rsrc,
    traced,
    xshfl,
)

CHECK_EVERY = 256  # poll rounds between deadline / health checks
POLL_SLEEP = 1
LOG2E = 1.4426950408889634
I32 = fx.Int32
F32 = fx.Float32
I64 = fx.Int64


def hw_sqrt(x):
    """v_sqrt_f32 (~1 ulp), not the correctly rounded libm expansion."""
    return F32(
        llvm.call_intrinsic(T.f32, "llvm.amdgcn.sqrt.f32", [F32(x).ir_value()], [], [])
    )


def exp(x):
    """The natural exponent as exp2 of x log2(e) (denormal-safe llvm.exp2)."""
    return fmath.exp2(F32(x) * LOG2E)


def opaque(v, loop_var):
    """`v` through a side-effecting asm that also reads `loop_var`.

    Per-lane values derived from it are not loop-invariant for LLVM: hoisted out of
    the layer loop they would stay live across it and spill.
    """
    return I32(
        llvm.inline_asm(
            ir.IntegerType.get_signless(32),
            [I32(v).ir_value(), I32(loop_var).ir_value()],
            "v_mov_b32 $0, $1 ; $2",
            "=v,v,v",
            has_side_effects=True,
        )
    )


# ---------------------------------------------------------------- memory
def _ld(rs, idx, n, cm, dtype):
    dt = T.i32 if dtype is None else dtype
    v = bo.buffer_load(rs, idx, vec_width=n, dtype=dt, cache_modifier=cm)
    if n == 1:
        return F32(v) if dt == T.f32 else I32(v)
    return fx.Vector(v)


def ld(base, idx, n=1, cm=0, dtype=None):
    """Load `n` elements (i32 unless `dtype`) at element `idx` of `base`."""
    return _ld(rsrc(base), idx, n, cm, dtype)


def ldn(base, nbytes, idx, n=1, cm=0, dtype=None):
    """`ld` through a resource of `nbytes`: elements past it read 0 without a
    memory access."""
    return _ld(rsrc(base, nbytes), idx, n, cm, dtype)


def ld8(base, idx):
    """Byte `idx` of `base`, zero-extended."""
    b = bo.buffer_load(rsrc(base), idx, vec_width=1, dtype=T.i8)
    return I32(fx.Uint8(b).to(fx.Uint32))


def st(base, idx, v, cm=0):
    """Store `v` (a Numeric or a Vector) at element `idx` of `base`."""
    bo.buffer_store(v, rsrc(base), idx, cache_modifier=cm)


def st8(base, idx, v, cm=0):
    bo.buffer_store(I32(v).to(fx.Int8).ir_value(), rsrc(base), idx, cache_modifier=cm)


def st16(base, idx, v, cm=0):
    bo.buffer_store(I32(v).to(fx.Int16).ir_value(), rsrc(base), idx, cache_modifier=cm)


def gst32(addr, v):
    """A 32-bit store at a per-lane 64-bit address."""
    ptr_type = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
    fx.generic_store(fx.inttoptr(ptr_type, addr), I32(v))


def pair(v, tag):
    return fx.Vector.from_elements([I32(v), I32(tag)], fx.Int32)


def bf_lo(w):
    return (I32(w) << 16).bitcast(F32)


def bf_hi(w):
    return (I32(w) & I32(-65536)).bitcast(F32)


def bf16r(x):
    """`x` rounded to bf16 (RNE), as f32."""
    return F32(F32(x).to(fx.BFloat16))


def bf16_bits(x):
    """bf16 bits of `x` (RNE) in the low half of an Int32."""
    v = fx.Vector.from_elements([F32(x), F32(0.0)], fx.Float32)
    return v.to(fx.BFloat16).bitcast(fx.Int32)[0] & 0xFFFF


def bfpair(a, b):
    """bf16(a) | bf16(b) << 16 as an Int32."""
    return bf16_pair(F32(a), F32(b)).bitcast(fx.Int32)


def fence_release():
    fx.memory_fence(
        syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Release
    )


def fence_acquire():
    fx.memory_fence(
        syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Acquire
    )


# ---------------------------------------------------------------- bounded waits
@traced
def _spin(load_all, pending, dl, health):
    """Re-issue `load_all` until `pending` clears.

    Every CHECK_EVERY rounds the clock is checked against `dl` and the health
    word read. Returns (values, failed): failed is 1 when the wait timed out, 2 when
    it gave up on a failure reported elsewhere.
    """
    v = load_all()
    n = I32(0)
    fl = I32(0)
    while pending(v) & (fl == 0):
        rocdl.s_sleep(POLL_SLEEP)
        v = load_all()
        n = n + 1
        if n >= CHECK_EVERY:
            n = I32(0)
            hv = ld(health, 0, 1, CM_SYS)
            if hv != 0:
                fl = I32(2)
            if memrealtime() > dl:
                fl = I32(1)
    return v, fl


@traced
def fail(health, fl, rank, site):
    """A timed-out wait sets the health word to [1, rank, site]."""
    if fl == 1:
        st(health, 1, I32(rank), CM_SYS)
        st(health, 2, I32(site), CM_SYS)
        st(health, 0, I32(1), CM_SYS)


def poll_pairs(base, idx, n, tag, dl, health, cm=CM_DEV):
    """Wait for the `n` pairs (1, 2, 4 or 8) from pair `idx` of `base` to carry
    `tag`. Returns ([n value words], failed)."""
    tag = I32(tag)

    def load_all():
        words = []
        for q in range_constexpr(0, 2 * n, 4 if n >= 2 else 2):
            w = min(4, 2 * n)
            v = ld(base, I32(idx) * 2 + q, w, cm)
            words += [v[e] for e in range(w)]
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        bad = v[1] != tag
        for e in range_constexpr(3, 2 * n, 2):
            bad = bad | (v[e] != tag)
        return bad

    v, fl = _spin(load_all, pending, dl, health)
    return [v[2 * q] for q in range(n)], fl


def poll_rows(bases, idx, n, tag, dl, health, cm=CM_SYS):
    """`poll_pairs` of every base in `bases` in one batch (one round trip a
    round). Returns ([[n value words] a base], failed)."""
    tag = I32(tag)

    def load_all():
        words = []
        for b in bases:
            for q in range_constexpr(0, 2 * n, 4):
                v = ld(b, I32(idx) * 2 + q, 4, cm)
                words += [v[e] for e in range(4)]
        return fx.Vector.from_elements(words, fx.Int32)

    nw = 2 * n * len(bases)

    def pending(v):
        bad = v[1] != tag
        for e in range_constexpr(3, nw, 2):
            bad = bad | (v[e] != tag)
        return bad

    v, fl = _spin(load_all, pending, dl, health)
    return [[v[2 * n * j + 2 * q] for q in range(n)] for j in range(len(bases))], fl


def wait_flags(base, n, tag, tid, dl, health, rank, site):
    """Wait for flags 0..n-1 of `base` (n <= 512), then a CTA barrier."""
    _, fl = poll_pairs(base, fx.min(tid, fx.max(I32(n) - 1, 0)), 1, tag, dl, health)
    fail(health, fl, rank, site)
    gpu.barrier()
    fence_acquire()


@traced
def publish(base, i, tag, tid):
    """Drain the CTA's stores, then set flag `i` of `base` to `tag`.

    The drain is an explicit vmcnt(0): on gfx950 a workgroup-scope release fence
    emits no wait for global stores, and the flag must not overtake another wave's
    data.
    """
    rocdl.s_waitcnt(vmcnt=0)
    fence_release()
    gpu.barrier()
    if tid == 0:
        st(base, I32(i) * 2, pair(0, tag), CM_DEV)


# ---------------------------------------------------------------- MXFP8 GEMV
# Weights stay in registers. For a 16-row tile and chunk c of a K-wide linear,
# wave w covers k in [w K / 8 + 128 c, + 128). Lane (r = l % 16, g = l / 16) holds
# row r's 16-byte units g and g + 4 of the chunk (the scaled MFMA's k layout) and
# the e8m0 scale of the chunk's 32-k block g. Token rows (x8 [16][K] + xs
# [16][K / 32]) use the same loader. The waves' partials are summed in order 0..7
# through LDS.
def w_chunk(w, s, row, K, wave, g4, c):
    """MFMA operand (8 words) and scale of row `row`, chunk `c` of `wave`."""
    kw = K // 8
    byte = row * K + wave * kw + 128 * c + 16 * g4
    a = ld(w, byte // 4, 4)
    b = ld(w, (byte + 64) // 4, 4)
    words = [a[e] for e in range(4)] + [b[e] for e in range(4)]
    sc = ld8(s, row * (K // 32) + wave * (kw // 32) + 4 * c + g4)
    return fx.Vector.from_elements(words, fx.Int32), sc


def mma8(a, sa, b, sb, acc):
    """Acc += A B, 16x16x128 e4m3 with e8m0 scales (in the low byte)."""
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [a, b, acc, 0, 0, 0, sa, 0, sb]
        )
    )


def zero4():
    return fx.Vector.filled(4, 0.0, fx.Float32)


def mma_chain(ws, xs, nc):
    """One tile's partials over chunks 0..nc-1, in order."""
    acc = zero4()
    for c in range_constexpr(nc):
        acc = mma8(ws[c][0], ws[c][1], xs[c][0], xs[c][1], acc)
    return acc


def red_store(red, t, acc, wave, g4, l16):
    """Wave `wave`'s partial of tile `t` -> `red[8 t + wave][row][token]`."""
    for e in range_constexpr(4):
        fx.ptr_store(acc[e], red + ((t * 8 + wave) * 256 + (g4 * 4 + e) * 16 + l16))


def red_sum(red, t, row, tok):
    """Tile `t`'s (row, token) summed over waves 0..7, in order."""
    s = fx.ptr_load(red + (t * 2048 + row * 16 + tok))
    for w in range_constexpr(1, 8):
        s = s + fx.ptr_load(red + (t * 2048 + w * 256 + row * 16 + tok))
    return s


def mx_scale(am):
    """VLLM's MXFP8 block scale of amax `am`: (e8m0 exponent as f32, 1 / scale)."""
    am = fx.max(am, F32(layout.TINY))
    sb = fmath.ceil(fmath.log2(am / layout.FP8_MAX)) + 127.0
    sb = fx.min(fx.max(sb, F32(0.0)), F32(254.0))
    return sb, hw_exp2(127.0 - sb)  # an integer exponent: exact


def xq_chunk(x, row, K, wave, g4, l16, c):
    """bf16 row `row` of `x[16][K]`, chunk `c` of `wave`, quantized to MXFP8 in
    the operand layout. A 32-k block is units 2b and 2b + 1, held by lanes 16
    apart. Returns (operand: 8 words, scale of block g)."""
    kw = K // 8
    units = []
    for h in range_constexpr(2):
        e0 = row * K + wave * kw + 128 * c + 16 * (g4 + 4 * h)
        v0 = ld(x, e0 // 2, 4)
        v1 = ld(x, e0 // 2 + 4, 4)
        units.append([v0[e] for e in range(4)] + [v1[e] for e in range(4)])
    sbs = []
    words = []
    for h in range_constexpr(2):
        vals = []
        for wd in units[h]:
            vals += [bf_lo(wd), bf_hi(wd)]
        am = fmath.absf(vals[0])
        for e in range_constexpr(1, 16):
            am = fx.max(am, fmath.absf(vals[e]))
        am = fx.max(am, xshfl(am, 16))
        sb, rs = mx_scale(am)
        sbs.append(sb)
        for m in range_constexpr(4):
            q = [vals[4 * m + e] * rs for e in range(4)]
            words.append(fp8_pack4(q[0], q[1], q[2], q[3]))
    # lane group g supplies block g's scale: group 2 g's first unit for g < 2,
    # group 2 (g - 2)'s second unit otherwise
    packed = I32(sbs[0].to(fx.Int32)) | (I32(sbs[1].to(fx.Int32)) << 8)
    src = l16 + 16 * (2 * (g4 % 2))
    got = I32(rocdl.ds_bpermute(T.i32, I32(src) * 4, packed.ir_value()))
    sc = (g4 < 2).select(got & 0xFF, (got >> 8) & 0xFF)
    return fx.Vector.from_elements(words, fx.Int32), sc


# ---------------------------------------------------------------- row norms
# Thread i < 384 owns columns 16 i .. 16 i + 15 of a 6144-wide row; the residual row
# stays in LDS as bf16 pairs.
def norm_w(w, tid):
    """A thread's 16 norm weights (8 bf16-pair words)."""
    v0 = ld(w, fx.min(tid, 383) * 8, 4)
    v1 = ld(w, fx.min(tid, 383) * 8 + 4, 4)
    return [v0[e] for e in range(4)] + [v1[e] for e in range(4)]


def row_x(x_in, t, tid):
    """A thread's 16 values of row `t` of bf16 `x_in[rows][6144]`."""
    v0 = ld(x_in, t * 3072 + fx.min(tid, 383) * 8, 4)
    v1 = ld(x_in, t * 3072 + fx.min(tid, 383) * 8 + 4, 4)
    out = []
    for wd in [v0[e] for e in range(4)] + [v1[e] for e in range(4)]:
        out += [bf_lo(wd), bf_hi(wd)]
    return out


def row_sum4(sym, Y, t, tid, tag, dl, health, rank, site):
    """A thread's 16 values of row `t`: bf16 of the 4 source ranks' bf16 partials
    summed in rank order, polled from the symmetric buffer at `Y`."""
    bases = [sym + I64(Y + src * layout.RANK_STRIDE) for src in range(4)]
    idx = t * layout.ROW_PAIRS + fx.min(tid, 383) * 8
    got, fl = poll_rows(bases, idx, 8, tag, dl, health)
    fail(health, fl, rank, site)
    vals = []
    for k in range_constexpr(8):
        lo = 0.0 + bf_lo(got[0][k])
        hi = 0.0 + bf_hi(got[0][k])
        for r in range_constexpr(1, 4):
            lo = lo + bf_lo(got[r][k])
            hi = hi + bf_hi(got[r][k])
        vals += [bf16r(lo), bf16r(hi)]
    return vals


@traced
def norm_row(v, res, w, add, red, tid, lane, wave):
    """Gemma RMSNorm of a row: s = v (+ the residual in LDS `res`) becomes the
    residual; returns bf16(s rstd (1 + w)), 16 values a thread. `red`: 8 LDS
    floats."""
    act = tid < 384
    ri = fx.min(tid, 383) * 8
    s = []
    for k in range_constexpr(8):
        if const_expr(add):
            r = fx.ptr_load(res + (ri + k))
            s += [bf16r(v[2 * k] + bf_lo(r)), bf16r(v[2 * k + 1] + bf_hi(r))]
        else:
            s += [v[2 * k], v[2 * k + 1]]
    if act:
        for k in range_constexpr(8):
            fx.ptr_store(bfpair(s[2 * k], s[2 * k + 1]), res + (ri + k))
    ss = F32(0.0)
    for e in range_constexpr(16):
        ss = ss + act.select(s[e] * s[e], F32(0.0))
    ss = butterfly(ss, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(ss, red + wave)
    gpu.barrier()
    tot = fx.ptr_load(red + 0)
    for k in range_constexpr(1, 8):
        tot = tot + fx.ptr_load(red + k)
    rstd = 1.0 / hw_sqrt(tot / 6144.0 + layout.EPS)
    o = []
    for k in range_constexpr(8):
        o += [
            bf16r(s[2 * k] * rstd * (1.0 + bf_lo(w[k]))),
            bf16r(s[2 * k + 1] * rstd * (1.0 + bf_hi(w[k]))),
        ]
    return o


@traced
def quant_store(o, t, x8, xs8, tid):
    """MXFP8 of a thread's 16 values -> e4m3 `x8[t][16 tid ..]`, e8m0
    `xs8[t][tid / 2]`."""
    am = fmath.absf(o[0])
    for e in range_constexpr(1, 16):
        am = fx.max(am, fmath.absf(o[e]))
    am = fx.max(am, xshfl(am, 1))
    sb, rs = mx_scale(am)
    wd = []
    for m in range_constexpr(4):
        q = [o[4 * m + e] * rs for e in range(4)]
        wd.append(fp8_pack4(q[0], q[1], q[2], q[3]))
    if tid < 384:
        words = fx.Vector.from_elements(wd, fx.Int32)
        st(x8, (t * 6144 + tid * 16) // 4, words, CM_DEV)
        if tid % 2 == 0:
            st8(xs8, t * 192 + tid // 2, sb.to(fx.Int32), CM_DEV)


@traced
def store_bf(o, t, out, tid):
    if tid < 384:
        wd = [bfpair(o[2 * k], o[2 * k + 1]) for k in range(8)]
        st(out, t * 3072 + tid * 8, fx.Vector.from_elements(wd[:4], fx.Int32))
        st(out, t * 3072 + tid * 8 + 4, fx.Vector.from_elements(wd[4:], fx.Int32))
