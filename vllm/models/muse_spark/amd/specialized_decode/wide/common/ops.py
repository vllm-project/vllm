# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the Qwen3.8-Flash-Next wide decode's ``wide/common/ops.py``
# (gfx950) for gfx942.
"""Device primitives of the wide kernels on gfx942: global accesses through
buffer descriptors with explicit cache policies, LDS vectors, wave reductions,
activations, bf16 and FP8 packing, the MFMAs and the clock.

Kernels take raw `Int64` byte addresses. A buffer descriptor's offset is 32
bits, so a tensor past 4 GiB (a layer's experts are 1.6 GB, so not yet) is
addressed from a 64-bit base per slice. Offsets here count elements of the
access's dtype.

Cache policy (GFX940-family `CPol`, the same bits on gfx942 and gfx950):
data one CTA writes and another CTA reads in the same launch goes through
`CM_DEV` (sc1) on both sides, past the per-XCD L2s; peer data through
`CM_SYS` (sc0 | sc1). The dense weights stream with `CM_NT`; the experts
with plain loads (`moe.EXPERT_CM`).

gfx942 has no permlane16/32 swaps (cross-row reductions use `ds_bpermute`),
no 16x16x32 bf16 MFMA (`mfma_bf16` is the 16x16x16 one) and no scaled FP4
MFMA; its FP8 is e4m3fnuz.
"""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import math as fmath
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

CM_DEV = 16
CM_SYS = 17
CM_NT = 2

LOG2E = 1.4426950408889634


def traced(fn):
    """FlyDSL's AST rewriting on a helper, so its `while` / `if` over
    traced values lower to scf ops (kernels rewrite only their own body)."""
    return ASTRewriter.transform(fn)


# --------------------------------------------------------------------------
# global memory
# --------------------------------------------------------------------------


def rsrc(addr, nbytes=None):
    """A buffer descriptor at byte address `addr`, which must be wave-uniform:
    a lane-varying base makes the compiler loop over the wave's distinct
    descriptors one at a time. Lane variation goes in the access's offset."""
    return bo.create_buffer_resource_from_addr(fx.Int64(addr), num_records_bytes=nbytes)


def ld(addr, i, dtype=fx.Int32, n=1, cm=0, nbytes=None, mask=None):
    """`n` (1, 2, 4) elements of `dtype` at element `i` of address `addr`.
    With `mask` (and the descriptor bounded by `nbytes`), a lane whose mask
    is false reads zeros without a memory request."""
    v = bo.buffer_load(
        rsrc(addr, nbytes),
        fx.Int32(i),
        vec_width=n,
        dtype=dtype,
        cache_modifier=cm,
        mask=mask,
    )
    return fx.Vector(v) if n > 1 else dtype(v)


def st(addr, i, value, cm=0, nbytes=None):
    """`value` (a Numeric or a Vector) at element `i` of address `addr`; with
    `nbytes`, a store past the extent is dropped by the hardware."""
    bo.buffer_store(
        fx.Vector(value).ir_value() if isinstance(value, fx.Vector) else value,
        rsrc(addr, nbytes),
        fx.Int32(i),
        cache_modifier=cm,
    )


def ld_i32x4(addr, i, cm=0, nbytes=None, mask=None):
    """Four i32 words at word `i`."""
    return ld(addr, i, fx.Int32, 4, cm=cm, nbytes=nbytes, mask=mask)


def gptr(addr):
    """An i32 global pointer at byte address `addr` (for atomics)."""
    return fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), fx.Int64(addr)
    )


def atomic_add(addr, v):
    """Agent-scope atomic add on the i32 at byte address `addr`; the old value."""
    return fx.Int32(fx.llvm.atomic_add(gptr(addr), fx.Int32(v), syncscope="agent"))


def atomic_max_pos(addr, v):
    """Agent-scope max of a non-negative f32 `v` into the f32 at byte address
    `addr` (zeroed first): their bits order as signed ints."""
    bits = fx.Float32(v).bitcast(fx.Int32)
    fx.llvm.atomic_max(gptr(addr), bits, syncscope="agent")


def lds_atomic_or(ptr, v):
    fx.llvm.atomic_or(ptr, fx.Int32(v), syncscope="workgroup")


def drain_stores():
    """Wait for this thread's outstanding memory operations. A workgroup-scope
    release fence alone lowers to no vmcnt wait, so it cannot order stores to
    another CTA before a flag or an arrival."""
    fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Release)
    rocdl.s_waitcnt(vmcnt=0)


def acquire_device():
    """An agent-scope acquire (`buffer_inv sc1`): drops the XCD's L2 lines
    that other XCDs may have written since, so plain (cached) loads after a
    grid barrier read other CTAs' device-scope stores, and the CTAs of an XCD
    share those lines in its L2 instead of each going to memory (`CM_DEV`)."""
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)


def memrealtime():
    """s_memrealtime: the 100 MHz device clock."""
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def uniform(v):
    """The first active lane's Int32, wave-uniform in an SGPR."""
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def ld_ptr64(table, i):
    """Entry `i` of an i64 table in memory, wave-uniform."""
    w = ld(table, fx.Int32(i) * 2, fx.Int32, 2)
    lo, hi = uniform(w[0]), uniform(w[1])
    return (fx.Int64(hi) << fx.Int64(32)) | fx.Int64(fx.Uint32(lo))


# --------------------------------------------------------------------------
# LDS
# --------------------------------------------------------------------------


def lds_ld(ptr, n=1, dtype=fx.Int32):
    """`n` elements at LDS pointer `ptr` (an `fx.Array` iterator plus an
    element offset): one ds_read of `n` x width bits."""
    if n == 1:
        return fx.ptr_load(ptr)
    return fx.Vector(fx.ptr_load(ptr, result_type=fx.Vector.make_type(n, dtype)))


def lds_st(ptr, value):
    fx.ptr_store(value, ptr)


# --------------------------------------------------------------------------
# wave reductions (DPP inside a row of 16, ds_bpermute across rows)
# --------------------------------------------------------------------------


def _xshfl(v, off):
    """Lane `lane ^ off`'s value: DPP for 1..8, `ds_bpermute` for 16, 32."""
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


def _add(a, b):
    return a + b


def butterfly(v, offsets, op=_add):
    """Butterfly reduction over the xor `offsets` in order: every lane of the
    covered group receives the result."""
    for off in offsets:
        v = op(v, _xshfl(v, off))
    return v


def wave_sum(v):
    """The wave's sum on every lane; the row (16-lane) sums first, so the
    order is fixed."""
    return butterfly(v, (1, 2, 4, 8, 16, 32))


def wave_max(v):
    return butterfly(v, (1, 2, 4, 8, 16, 32), fx.max)


# --------------------------------------------------------------------------
# math
# --------------------------------------------------------------------------


def _hw_f32(name, x):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], [])
    )


def hw_exp2(x):
    return _hw_f32("llvm.amdgcn.exp2.f32", x)


def hw_rcp(x):
    return _hw_f32("llvm.amdgcn.rcp.f32", x)


def sigmoid(x):
    """1 / (1 + exp(-x)) on v_exp_f32 and v_rcp_f32: an underflowing exp gives 1,
    an overflowing one 0."""
    return hw_rcp(fx.Float32(1.0) + hw_exp2(x * fx.Float32(-LOG2E)))


def silu(x):
    return x * sigmoid(x)


def rsqrt(x):
    return fmath.rsqrt(x)


def fabs(x):
    return fmath.absf(x)


def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf_lo(w):
    """The f32 of a packed bf16 pair word's low half."""
    return (fx.Int32(w) << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (fx.Int32(w) & fx.Int32(-65536)).bitcast(fx.Float32)


def pack_bf16(a, b):
    """Two f32 rounded to bf16, packed in one Int32 (a low)."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
    )


def pack_bf16x8(values):
    """Eight f32 rounded to bf16 as four packed Int32 words."""
    return (
        fx.Vector.from_elements(list(values), fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)
    )


def unpack_bf16x8(words):
    """Four packed Int32 words -> eight f32."""
    out = []
    for w in range(4):
        out += [bf_lo(words[w]), bf_hi(words[w])]
    return out


def zeros(n, dtype=fx.Float32):
    return fx.Vector.from_elements([dtype(0)] * n, dtype)


# --------------------------------------------------------------------------
# FP8 (e4m3fnuz)
# --------------------------------------------------------------------------


def cvt_fp8x4(a, b, c, d):
    """Four f32 to e4m3fnuz codes in one Int32 (`a` the low byte), RNE with
    saturation (`v_cvt_pk_fp8_f32`); the caller has scaled and clamped them
    to +-FP8_MAX."""
    lo = rocdl.cvt_pk_fp8_f32(
        T.i32,
        fx.Float32(a).ir_value(),
        fx.Float32(b).ir_value(),
        fx.Int32(0).ir_value(),
        False,
    )
    return fx.Int32(
        rocdl.cvt_pk_fp8_f32(
            T.i32, fx.Float32(c).ir_value(), fx.Float32(d).ir_value(), lo, True
        )
    )


def quant_scale(amax, quant):
    """The per-token scale of a row with absolute max `amax` under the
    `layout.Quant` `quant`."""
    if quant.divide:
        scale = amax / fx.Float32(quant.fp8_max)
    else:
        scale = amax * fx.Float32(1.0 / quant.fp8_max)
    if quant.min_scale:
        scale = fx.max(scale, fx.Float32(quant.min_scale))
    return scale


def quant_factor(scale, quant):
    """What `fp8_quant8` multiplies by: `1 / scale` (v_rcp_f32). For a
    dividing quant a code differs from `x / scale`'s only where the
    quotient is within an f32 ulp of an FP8 rounding midpoint; a true
    division an element would cost the down GEMM's staging ~10x its loads."""
    return hw_rcp(scale)


def fp8_quant8(values, factor, quant):
    """Eight f32 times `factor` (`quant_factor`), clamped to +-fp8_max, as
    two Int32 of e4m3fnuz codes."""
    lim = fx.Float32(quant.fp8_max)
    v = [fx.min(fx.max(x * factor, -lim), lim) for x in values]
    return [cvt_fp8x4(*v[0:4]), cvt_fp8x4(*v[4:8])]


# --------------------------------------------------------------------------
# MFMA
# --------------------------------------------------------------------------


def mfma_bf16(a, b, acc):
    """`acc + A @ B.T` over 16 K: `a`, `b` a lane's 4 bf16 as Int16
    (16x16x16): lane `l` holds row `l % 16`, K `4 (l / 16) + [0, 4)`."""
    return fx.Vector(
        rocdl.mfma_f32_16x16x16bf16_1k(
            T.vec(4, T.f32),
            [
                fx.Vector(a).ir_value(),
                fx.Vector(b).ir_value(),
                fx.Vector(acc).ir_value(),
                0,
                0,
                0,
            ],
        )
    )


def mfma_fp8(a, b, acc):
    """`acc + A @ B.T` over 32 K of e4m3fnuz: `a`, `b` a lane's 8 codes as
    Int64 (16x16x32): lane `l` holds row `l % 16`, K `8 (l / 16) + [0, 8)`.
    The accumulator lane `l` holds rows `4 (l / 16) + [0, 4)` of column
    `l % 16`."""
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_fp8_fp8(
            T.vec(4, T.f32),
            [
                fx.Int64(a).ir_value(),
                fx.Int64(b).ir_value(),
                fx.Vector(acc).ir_value(),
                0,
                0,
                0,
            ],
        )
    )


def i64_of(lo, hi):
    """Two Int32 words as one Int64 (`lo` the low word)."""
    return (fx.Int64(fx.Uint32(hi)) << fx.Int64(32)) | fx.Int64(fx.Uint32(lo))


def sched_barrier():
    """Keep issued loads where they are (the scheduler would sink them to their
    use)."""
    rocdl.sched_barrier(0)


def unlikely(cond):
    """`cond` marked cold (llvm.expect)."""
    return fx.Boolean(
        llvm.call_intrinsic(
            ir.IntegerType.get_signless(1),
            "llvm.expect.i1",
            [as_ir_value(cond), as_ir_value(fx.Boolean(False))],
            [],
            [],
        )
    )
