# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ops.py
"""Device primitives of the kernels: global accesses through buffer
descriptors with explicit cache policies, LDS vectors, wave reductions and
shuffles, activations, bf16 packing and the clock. Every function is traced
inside a `@flyc.kernel` body.

Kernels take raw `Int64` byte addresses. A buffer descriptor's offset is 32
bits, so a pool past 4 GiB (the GDN state) is addressed from a 64-bit base per
slot. Offsets here count elements of the access's dtype.

Cache policy (gfx950 `CPol`): data one CTA writes and another CTA reads in
the same launch goes through `CM_DEV` (sc1) on both sides, past the per-XCD
caches; peer data through `CM_SYS` (sc0 | sc1). Weights a launch reads once
stream with `CM_NT`.
"""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
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


def ld_i32(addr, i):
    """Word `i`, a plain Int32 load."""
    return ld(addr, i)


def ld_dev(addr, i, words=1):
    """`words` i32 (1, 2 or 4) from word `i` at device scope, as the buffer
    load returns them."""
    return bo.buffer_load(
        rsrc(addr), i, vec_width=words, dtype=T.i32, cache_modifier=CM_DEV
    )


def ld_i32x4(addr, i, cm=0):
    """Four i32 words at word `i`."""
    return fx.Vector(
        bo.buffer_load(
            rsrc(addr), fx.Int32(i), vec_width=4, dtype=T.i32, cache_modifier=cm
        )
    )


def _gptr(addr, dtype, n):
    return fx.inttoptr(
        fx.PointerType.get(
            dtype.ir_type, fx.AddressSpace.Global, 4 * max(1, n * dtype.width // 32)
        ),
        fx.Int64(addr),
    )


def gld(addr, dtype=fx.Int32, n=1):
    """`n` elements at a per-lane 64-bit byte address (no descriptor): a
    gather whose base differs by lane and lies past a descriptor's 4 GiB."""
    v = fx.generic_load(_gptr(addr, dtype, n), dtype=dtype, count=n)
    return fx.Vector(v) if n > 1 else v


def gst(addr, value):
    """`value` (a Numeric or Vector) at a per-lane 64-bit byte address."""
    v = fx.Vector(value) if isinstance(value, fx.Vector) else value
    dtype = v.dtype if isinstance(v, fx.Vector) else type(v)
    n = v.numel if isinstance(v, fx.Vector) else 1
    fx.generic_store(_gptr(addr, dtype, n), v)


def gptr(addr):
    """An i32 global pointer at byte address `addr` (for atomics)."""
    return fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), fx.Int64(addr)
    )


def atomic_add(addr, v):
    """Agent-scope atomic add on the i32 at byte address `addr`; the old value."""
    return fx.Int32(fx.llvm.atomic_add(gptr(addr), fx.Int32(v), syncscope="agent"))


def drain_stores():
    """Wait for this thread's outstanding memory operations. A workgroup-scope
    release fence alone lowers to no vmcnt wait on gfx950, so it cannot order
    stores to another CTA before a flag or an arrival."""
    fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Release)
    rocdl.s_waitcnt(vmcnt=0)


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
# wave reductions (DPP inside a row, permlane swaps across rows)
# --------------------------------------------------------------------------


def permlane_swap(off, x, y):
    """v_permlane{32,16}_swap on registers (x, y): off 32 trades x's upper-half
    lanes with y's lower-half lanes; off 16 trades x's odd 16-lane rows with y's
    even rows (each within its half). -> (x', y')."""
    swap = rocdl.permlane32_swap if off == 32 else rocdl.permlane16_swap
    pr = swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(fx.Int32(x)),
        as_ir_value(fx.Int32(y)),
        False,
        False,
    )
    return tuple(fx.Int32(llvm.extractvalue(T.i32, pr, [j])) for j in range(2))


def xshfl(v, off):
    """Lane `lane ^ off`'s value: offsets 1..8 via DPP (VALU latency, not
    LDS), 16 / 32 via shuffle. Powers of two only."""
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
    """op(v, lane ^ off) for a symmetric op; 32 / 16 take both halves of one swap."""
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
    """Butterfly reduction over the xor `offsets` in order: every lane of the
    covered group receives the result. The order is the numerical contract:
    the aiter per-head norm reduces 32 lanes with offsets 16, 8, 4, 2, 1, and
    matching it keeps rstd bit-exact."""
    for off in offsets:
        v = xred(v, off, op)
    return v


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


def wave_max(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1), fx.max)


def _xpartner(v, off, lane):
    """Lane `lane ^ off`'s v: DPP within a row, v_permlane*_swap across rows
    (`xshfl` takes `shuffle_xor` there)."""
    if off < 16:
        return xshfl(v, off)
    a, b = permlane_swap(off, v, v)
    return ((lane & off) != 0).select(a, b)


def wave_incl_scan(v, lane):
    """Inclusive prefix sum of an Int32 over the wave, in lane order."""
    blk = v
    for off in (1, 2, 4, 8, 16, 32):
        o = _xpartner(blk, off, lane)
        v = v + ((lane & off) != 0).select(o, 0)
        blk = blk + o
    return v


def readlane(v, src_lane):
    """Lane `src_lane`'s value of v (wave-uniform)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    r = fx.Int32(rocdl.readlane(T.i32, bits, fx.Int32(src_lane)))
    return r.bitcast(fx.Float32) if isinstance(v, fx.Float32) else r


def ballot(pred):
    """The wave's lanes where `pred` holds, a bit a lane."""
    return fx.Int64(rocdl.ballot(T.i64, as_ir_value(pred)))


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


def hw_rsq(x):
    """v_rsq_f32 (~1 ulp, no libm range fixups)."""
    return _hw_f32("llvm.amdgcn.rsq.f32", x)


def div_rn(x, d, r):
    """`x / d` rounded to nearest, bit-identical to the IEEE division, given
    `r = 1.0 / d` from one true division per divisor (Markstein: a quotient
    within 1 ulp plus its exact fma residual times RN(1/d) rounds correctly)."""
    q = x * r
    return fx.Float32(fmath.fma(fx.Float32(fmath.fma(-q, d, x)), r, q))


def sigmoid(x):
    """1 / (1 + exp(-x)) on v_exp_f32 and v_rcp_f32: an underflowing exp gives 1,
    an overflowing one 0."""
    return hw_rcp(fx.Float32(1.0) + hw_exp2(x * fx.Float32(-LOG2E)))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    """log(1 + exp(x)), x past 20 as is (vLLM's GDN gate)."""
    return (x > fx.Float32(20.0)).select(x, fmath.log1p(fmath.exp(x)))


def exp(x):
    return fmath.exp(fx.Float32(x))


def rsqrt(x):
    return fmath.rsqrt(x)


def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf_lo(w):
    """The f32 of a packed bf16 pair word's low half."""
    return (fx.Int32(w) << 16).bitcast(fx.Float32)


def bf_hi(w):
    """The f32 of a packed bf16 pair word's high half."""
    return (fx.Int32(w) & fx.Int32(-65536)).bitcast(fx.Float32)


def bf2_f32(w):
    """Packed bf16 pair word -> (f32 low, f32 high)."""
    return bf_lo(w), bf_hi(w)


def bf16_pair(a, b):
    """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Float32)[0]
    )


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


def mfma_bf16(a, b, acc):
    """`acc + A @ B.T` over 32 K: `a`, `b` a lane's 8 bf16 (16x16x32)."""
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_bf16(
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


def mfma_f4(a, b, acc, sa, sb, opsel_a=0):
    """`acc + A @ B.T` over 128 K of e2m1: `a` / `b` a lane's four Int32 (32
    codes), `sa` / `sb` its E8M0 scale codes (`opsel_a` picks A's byte of
    the `sa` word; B's is the low byte)."""
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32),
            [
                fx.Vector(a).ir_value(),
                fx.Vector(b).ir_value(),
                fx.Vector(acc).ir_value(),
                4,
                4,
                opsel_a,
                fx.Int32(sa).ir_value(),
                0,
                fx.Int32(sb).ir_value(),
            ],
        )
    )


def lds_barrier():
    """A workgroup barrier ordering LDS only: `fx.barrier`'s fences also
    wait out every global load in flight (`vmcnt(0)`), draining a load
    pipeline at each barrier. The memory clobber keeps the compiler from
    moving LDS accesses across it."""
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
        constraints="~{memory}",
        has_side_effects=True,
    )


def sched_barrier():
    """Keep issued loads where they are (the scheduler would sink them to their
    use)."""
    rocdl.sched_barrier(0)
