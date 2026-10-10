# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device primitives of the DeepSeek-V4.1 attention mega kernel.

Wave reductions and shuffles, hardware math, bf16 / fp8 / MX packing, the
scaled MFMA, buffer and 64-bit global accesses, and the tagged-pair mailbox.
Adapted from ATOM's mono decode framework (``atom/mono/device``), itself ported
from the FlyDSL GLM-5 layer kernel.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

# The execution model: one CTA a CU of an MI355X, all resident.
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64

# gfx95x cache policy bits (LLVM CPol): SC0 = 1, SC1 = 16, NT = 2. SC1 alone is
# device scope (past the per-XCD L2s); NT streams data read once.
CM_DEV = 16
CM_NT = 2

FP8_MAX = 448.0
FP8, FP4 = 0, 4  # f8f6f4 MFMA operand formats
UNIT_SCALE = 127  # E8M0 code of 1.0
FLT_MIN = 1.1754943508222875e-38

POLL_MAX = 12  # mailbox pairs polled a batch


def traced(fn):
    """FlyDSL's AST rewrite for a helper, so its ``if`` / ``while`` / ``for`` on
    traced values lower to scf ops (kernels rewrite only their own body)."""
    return ASTRewriter.transform(fn)


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``aiter::<stem>_<params>``."""
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN5aiter{len(name)}{name}E"


def first_task(bid, base):
    """CTA ``bid``'s first task of a stage placed from CTA ``base``, then every
    BLOCKS-th."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS


# ----------------------------------------------------------------------------- memory


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def ld_i32(ptr, i, cm=0):
    return fx.Int32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32, cache_modifier=cm)
    )


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def ld_bf(ptr, i):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
    )


def ld_i64(ptr, i):
    """Element i of an i64 buffer (two dwords)."""
    w = fx.Vector(bo.buffer_load(rsrc(ptr), i * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(w[1]) << 32) | fx.Int64(fx.Uint32(w[0]))


def _gptr(addr):
    return llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr))
    ).result


def gload(addr, words=4, nt=False):
    """``words`` i32 (1, 2 or 4) at a 64-bit global address: a cache past 4 GiB is
    reachable only this way (a buffer offset is 32 bits)."""
    i32 = ir.IntegerType.get_signless(32)
    ty = i32 if words == 1 else ir.VectorType.get([words], i32)
    v = llvm.LoadOp(ty, _gptr(addr), alignment=4 * words, nontemporal=nt or None).result
    return fx.Int32(v) if words == 1 else fx.Vector(v)


def atomic_add_dev(addr, value):
    """Device-scope atomic add on the global i32 at ``addr``; the old value.
    Relaxed: an acquire / release here would write back and invalidate L2."""
    return fx.Int32(
        fx.atomic_add(
            _gptr(addr),
            fx.Int32(value),
            syncscope="agent",
            ordering=fx.AtomicOrdering.Monotonic,
        )
    )


def gstore(addr, value, words=4):
    """Store ``words`` i32 (an Int32 or a Vector) at a 64-bit global address."""
    v = as_ir_value(fx.Int32(value) if words == 1 else value)
    llvm.StoreOp(v, _gptr(addr), alignment=4 * words)


# ----------------------------------------------------------------------------- math


def fresh(v):
    """``v`` through an opaque move, so the lane math built on it is computed at
    its use instead of being hoisted out of a task loop and held in registers."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [fx.Int32(v).ir_value()],
            "v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


def tile_map(tid, width):
    """(tid // width, tid % width) for 0 <= tid < THREADS without a division:
    shifts for a power of two, else a multiply-shift exact on that range."""
    if width & (width - 1) == 0:
        sh = width.bit_length() - 1
        return tid >> sh, tid & (width - 1)
    m = -(-(1 << 20) // width)
    assert all((t * m) >> 20 == t // width for t in range(THREADS)), width
    row = (tid * m) >> 20
    return row, tid - row * width


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def _hw_f32(name, x):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], [])
    )


def hw_rsq(x):
    return _hw_f32("llvm.amdgcn.rsq.f32", x)


def hw_rcp(x):
    return _hw_f32("llvm.amdgcn.rcp.f32", x)


def hw_exp2(x):
    return _hw_f32("llvm.amdgcn.exp2.f32", x)


def fma(a, b, c):
    return fx.Float32(fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


def div_rn(x, d, r):
    """``x / d`` rounded to nearest (IEEE), given ``r = RN(1 / d)``."""
    q = x * r
    return fma(fma(-q, d, x), r, q)


def permlane_swap(off, x, y):
    swap = rocdl.permlane32_swap if off == 32 else rocdl.permlane16_swap
    pr = swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(fx.Int32(x)),
        as_ir_value(fx.Int32(y)),
        False,
        False,
    )
    return tuple(fx.Int32(llvm.extractvalue(T.i32, pr, [j])) for j in range(2))


def lane_gather(v, src_lane):
    """Lane ``src_lane``'s value of v (ds_bpermute)."""
    return fx.Int32(
        rocdl.ds_bpermute(T.i32, fx.Int32(src_lane) * 4, fx.Int32(v).ir_value())
    )


def lane_f32(v, src_lane):
    return lane_gather(v.bitcast(fx.Int32), src_lane).bitcast(fx.Float32)


def readlane(v, src_lane):
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    r = fx.Int32(rocdl.readlane(T.i32, bits, fx.Int32(src_lane)))
    return r.bitcast(fx.Float32) if isinstance(v, fx.Float32) else r


def xshfl(v, off):
    """Value of lane ``lane ^ off`` (DPP in a row, permlane swaps across)."""
    assert off in (1, 2, 4, 8, 16, 32), off
    if off >= 16:
        return v.shuffle_xor(off, 64)
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    if off == 8:
        y = fx.Int32(update_dpp_i32(x, x, 0x118, 0xF, 0xC, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x108, 0xF, 0x3, False))
    elif off == 4:
        y = fx.Int32(update_dpp_i32(x, x, 0x114, 0xF, 0xA, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x104, 0xF, 0x5, False))
    elif off == 2:
        y = fx.Int32(update_dpp_i32(x, x, 0x4E, 0xF, 0xF, False))
    else:
        y = fx.Int32(update_dpp_i32(x, x, 0xB1, 0xF, 0xF, False))
    return y.bitcast(fx.Float32) if is_f else y


def xred(v, off, op):
    if off < 16:
        return op(v, xshfl(v, off))
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    a, b = permlane_swap(off, x, x)
    if is_f:
        return op(a.bitcast(fx.Float32), b.bitcast(fx.Float32))
    return op(type(v)(a), type(v)(b))


def _add(a, b):
    return a + b


def butterfly(v, offsets, op=_add):
    """Butterfly reduction over the xor ``offsets``, in that order."""
    for off in offsets:
        v = xred(v, off, op)
    return v


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


def wave_max(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1), fx.max)


def row_shr(v, lane, k):
    """Lane - k of the same 16-lane row, else 0 (DPP row_shr)."""
    moved = lane_gather(v.bitcast(fx.Int32), fx.max(lane - k, 0)).bitcast(fx.Float32)
    return fx.Float32((lane % 16 >= k).select(moved, fx.Float32(0.0)))


# ----------------------------------------------------------------------------- packing


def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf16_pair(a, b):
    """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Float32)[0]
    )


def bf_lo(w):
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def fp8_pack4(a, b, c, d):
    """Four f32 (in range) -> one dword of four E4M3 bytes, RNE."""
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))


def fp8x8_bf16(w0, w1, scale):
    """Eight E4M3 of two dwords times an f32 ``scale`` -> eight bf16 as a
    4-dword Vector (exact for a power-of-two scale in range)."""
    out = []
    for w in (w0, w1):
        for hi in (False, True):
            p = rocdl.cvt_scalef32_pk_bf16_fp8(
                T.bf16x2, as_ir_value(fx.Int32(w)), as_ir_value(fx.Float32(scale)), hi
            )
            out.append(fx.Vector(p).bitcast(fx.Int32)[0])
    return fx.Vector.from_elements(out, fx.Int32)


def clamp_fp8(v):
    return fx.max(fx.min(v, fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX))


def pow2(code):
    """The fp32 power of two a biased exponent (1..254) stands for."""
    return (code << 23).bitcast(fx.Float32)


def rcp_pow2(code):
    """1 / ``pow2(code)``, exact for code in [1, 253]."""
    return ((254 - code) << 23).bitcast(fx.Float32)


def ceil_exp(v):
    """Biased exponent of the smallest power of two >= v (v > 0 finite, normal)."""
    bits = v.bitcast(fx.Int32)
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(
        fx.Int32(1), fx.Int32(0)
    )


def mx_code(amax):
    """VLLM's ROCm MXFP8 scale code of a 32-group: clamp(ceil(log2(amax / 448))
    + 127, 0, 254), amax floored at FLT_MIN."""
    a = fx.max(amax, fx.Float32(FLT_MIN))
    q = div_rn(a, fx.Float32(FP8_MAX), fx.Float32(1.0 / FP8_MAX))
    return fx.min(fx.max(ceil_exp(q), fx.Int32(0)), fx.Int32(254))


def mx_mul(code):
    """2^(127 - code) as fp32, for code in [0, 254] (the quant multiplier)."""
    return ((254 - code) << 23).bitcast(fx.Float32)


# ----------------------------------------------------------------------------- MFMA


def mfma_scaled(a, b, c, sa, sb, a_fmt=FP8, b_fmt=FP8):
    """16x16x128 scaled MFMA (gfx950 f8f6f4): lane l (row l % 16) holds K bytes
    16 (l // 16) .. +16 and 64 + 16 (l // 16) .. +16 of the 128-step, and the
    E8M0 of K block l // 16."""
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [a, b, c, a_fmt, b_fmt, 0, sa, 0, sb]
        )
    )


def mfma_bf16(a, b, c):
    """16x16x32 bf16 MFMA: lane l (row / col l % 16) holds K 8 (l // 16) .. +8."""
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


# ----------------------------------------------------------------------------- mailbox


@traced
def _spin(load_all, pending):
    v = load_all()
    while pending(v):
        rocdl.s_sleep(1)
        v = load_all()
    return v


def spin_until(load, pending):
    """``load()`` again (s_sleep between) while ``pending(value)``; the value."""
    return _spin(load, pending)


class Mailbox:
    """Tagged-pair hand-off between the CTAs of one launch (device scope).

    Every 32-bit value is stored next to the launch's tag in one 8-byte store; a
    consumer polls until every tag matches, so a hand-off is one round trip with
    no separate flag. The tag is ``epoch * 64 + layer + 1`` (``epoch`` a device
    counter bumped once a step): a pair from an earlier layer or step never
    carries a live tag, and nothing is cleared between launches."""

    def __init__(self, tag):
        self.tag = tag

    def put(self, base, i, v):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, self.tag], fx.Int32),
            rsrc(base),
            i * 2,
            cache_modifier=CM_DEV,
        )

    def put_bf(self, base, i, vs):
        """Elements i .. i + len(vs) (2 or 4, i aligned) as bf16 pairs, one store."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(base),
            i,
            cache_modifier=CM_DEV,
        )

    def put_words(self, base, i, words):
        """Raw 32-bit words (1 or 2) at pairs i.. (fp8 words must come this way)."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(base),
            i * 2,
            cache_modifier=CM_DEV,
        )

    def poll(self, specs, batch=POLL_MAX):
        """Batched poll of ``specs`` = [(base, pair index, npairs in {1, 2})]: all
        re-loaded while any tag is stale. Returns the value words."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        tag = self.tag

        def load_all():
            words = []
            for b, i, n in specs:
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        v = _spin(load_all, pending)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs


LOAD_BATCH = 4


@traced
def batched_rounds(tid, n, load, use, batch=LOAD_BATCH):
    """A CTA's strided loop over v = tid + THREADS i < n: ``batch`` rounds' loads
    issued before their uses; a round past n loads element n - 1, uses nothing."""
    rounds = -(-n // THREADS)
    for b0 in range_constexpr(0, rounds, batch):
        idx = range(b0, min(b0 + batch, rounds))
        got = [load(fx.min(tid + THREADS * i, n - 1)) for i in idx]
        for i, g in zip(idx, got):
            v = tid + THREADS * i
            if v < n:
                use(v, g)


def barrier():
    gpu.barrier()


# ----------------------------------------------------------------------------- timeline


def memrealtime():
    """s_memrealtime: the 100 MHz device clock."""
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


@traced
def stamp(c, point):
    """A timeline build's stamp: thread 0 of the CTA stores the clock at
    ``point`` (``c["tl"]`` [BLOCKS, points] i64; None: not a timeline build)."""
    if const_expr(c.get("tl") is None):
        return
    if c["tid"] == 0:
        t = memrealtime()
        w = fx.Vector.from_elements([fx.Int32(t), fx.Int32(t >> 32)], fx.Int32)
        bo.buffer_store(w, rsrc(c["tl"]), (c["bid"] * c["tl_points"] + point) * 2)
