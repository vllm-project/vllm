# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ops.py
"""Device-side primitives of the mono kernels: wave reductions and shuffles, the
approximate hardware math, bf16 / fp8 packing, buffer resources, the
execution-model constants and the kernel symbol naming.

Ported from the FlyDSL GLM-5 layer kernel (ROCm/FlyDSL branch
``glm5-mla-moe-layer-monokernel`` @ 1e28fd8a,
``kernels/mla_moe_layer/shared_reuse_moe_kernel.py``). Only published flydsl APIs
and aiter's vendored ``buffer_ops`` / ``dpp_utils`` are used.

Every function but ``kernel_symbol`` is traced inside a ``@flyc.kernel`` body.

Raw boundaries kept, for want of an equivalent on the pinned FlyDSL:
``buffer_ops`` loads / stores (the kernels take raw Int64 addresses, no tensors
for ``make_buffer_tensor``); raw MFMA intrinsics (hand-laid operands: permlane
transposes, fp8 packing, MXFP4 scales); the LLVM intrinsics here (``expect``,
``s_memrealtime``, the approximate rsq / rcp / exp2) and the struct unpacking of
``permlane*_swap``.
"""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from vllm.models.kimi_k3.amd.mono.common.plan import GFX942, THREADS, WAVES

# gfx94x / gfx95x cache policy bits (LLVM CPol): SC0 = 1, SC1 = 16. SC1 alone is
# device scope (past the per-XCD caches); SC0 | SC1 is system scope (peers over
# XGMI).
CM_DEV = 16


CM_SYS = 17


# NT = 2: nontemporal. A weight a layer reads once streams past the caches it
# would only evict the step's reused data from (K2b at 48 rows: 143 -> 133 us)
CM_NT = 2


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``atom::<stem>_<params>``. The
    amdhsa assembler rejects ``::`` in a symbol, so it is Itanium-mangled, as
    aiter's ``_ZN5aiter...E`` kernels are."""
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN4atom{len(name)}{name}E"


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


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


def lane_gather(v, src_lane):
    """Lane ``src_lane``'s value of v, a source lane a lane (ds_bpermute)."""
    return fx.Int32(
        rocdl.ds_bpermute(T.i32, fx.Int32(src_lane) * 4, fx.Int32(v).ir_value())
    )


def readlane(v, src_lane):
    """Lane ``src_lane``'s value of v (wave-uniform)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    r = fx.Int32(rocdl.readlane(T.i32, bits, fx.Int32(src_lane)))
    return r.bitcast(fx.Float32) if isinstance(v, fx.Float32) else r


def memrealtime():
    """s_memrealtime: the 100 MHz device clock (no FlyDSL wrapper)."""
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def unlikely(cond):
    """``cond`` marked cold (llvm.expect): LLVM lays the branch out of the hot
    path. A rare path's code inline among the hot stages costs them even when it
    never runs (the long-context selection: S = 16, 3k context +1.4 us)."""
    return fx.Boolean(
        llvm.call_intrinsic(
            ir.IntegerType.get_signless(1),
            "llvm.expect.i1",
            [as_ir_value(cond), as_ir_value(fx.Boolean(False))],
            [],
            [],
        )
    )


def fresh(v):
    """``v`` (a per-lane Int32) through an opaque move, here: the lane math built
    on it is computed at its use, where LLVM would otherwise hoist it out of a task
    loop or CSE it across stages into registers held for the whole kernel (spills,
    and a kernel with scratch is refused: ``runtime.widths``)."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32, [fx.Int32(v).ir_value()], "v_mov_b32 $0, $1", "=v,v",
            has_side_effects=True,
        )
    )  # fmt: skip


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def _hw_f32(name, x):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], [])
    )


def hw_rsq(x):
    """v_rsq_f32 (~1 ulp, no libm range fixups)."""
    return _hw_f32("llvm.amdgcn.rsq.f32", x)


def hw_rcp(x):
    return _hw_f32("llvm.amdgcn.rcp.f32", x)


def hw_exp2(x):
    return _hw_f32("llvm.amdgcn.exp2.f32", x)


def div_rn(x, d, r):
    """``x / d`` rounded to nearest, bit-identical to the IEEE division, given
    ``r = 1.0 / d`` from one true division per divisor (Markstein: a quotient
    within 1 ulp plus its exact fma residual times RN(1/d) rounds correctly)."""
    q = x * r
    return fx.Float32(fmath.fma(fx.Float32(fmath.fma(-q, d, x)), r, q))


def xshfl(v, off):
    """Value of lane ``lane ^ off``: offsets 32 / 16 via v_permlane*_swap, in-row
    offsets via DPP (VALU latency, not LDS). Powers of two only: any other
    offset would take the lane ^ 1 path."""
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
    """op(v, lane ^ off) for a symmetric op; 32 / 16 take both halves of one swap
    (gfx942, no permlane swap: the partner by ``shuffle_xor``, the same value
    as op is symmetric)."""
    if off < 16 or GFX942:
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
    """Butterfly reduction over the given xor offsets, in the given order.

    The order is the numerical contract: the aiter per-head norm reduces 32
    lanes with offsets 16, 8, 4, 2, 1, and matching it keeps rstd bit-exact."""
    for off in offsets:
        v = xred(v, off, op)
    return v


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


def traced(fn):
    """Apply flydsl's AST rewriting to a helper, so its ``while`` / ``if`` over
    traced values lower to scf ops. Kernels rewrite only their own body."""
    return ASTRewriter.transform(fn)


def wave_max(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1), fx.max)


@traced
def block_max(v, lane, wave, red):
    """Max of v over the CTA's WAVES waves; ``red`` needs WAVES words."""
    wv = wave_max(v)
    gpu.barrier()
    if lane == 0:
        fx.ptr_store(wv, red + wave)
    gpu.barrier()
    t = fx.ptr_load(red + 0)
    for w in range_constexpr(1, WAVES):
        t = fx.max(t, fx.ptr_load(red + w))
    gpu.barrier()
    return t


def row_shr(v, lane, k):
    """DPP row_shr:k of an f32: lane - k of the same 16-lane row, else 0."""
    moved = lane_gather(v.bitcast(fx.Int32), fx.max(lane - k, 0)).bitcast(fx.Float32)
    return fx.Float32((lane % 16 >= k).select(moved, fx.Float32(0.0)))


def _xpartner(v, off, lane):
    """Lane ``lane ^ off``'s v: DPP within a row, v_permlane*_swap across rows
    (``xshfl`` takes ``shuffle_xor`` there, as gfx942 does)."""
    if off < 16 or GFX942:
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


@traced
def block_excl_scan(v, lane, wave, buf):
    """(exclusive prefix of v over the CTA in thread order, CTA total); ``buf``
    needs WAVES words."""
    incl = wave_incl_scan(v, lane)
    gpu.barrier()
    if lane == 63:
        fx.ptr_store(incl, buf + wave)
    gpu.barrier()
    off = fx.Int32(0)
    tot = fx.Int32(0)
    for w in range_constexpr(WAVES):
        cnt = fx.ptr_load(buf + w)
        off = off + (w < wave).select(cnt, 0)
        tot = tot + cnt
    return incl - v + off, tot


def global_load(addr, words=1, nt=False):
    """``words`` i32 (1, 2 or 4) at the 64-bit global address ``addr``. A pool
    past 4 GiB is reached only this way: a buffer load's offset (index x stride
    included) is 32 bits. ``nt``: nontemporal."""
    i32 = ir.IntegerType.get_signless(32)
    ty = i32 if words == 1 else ir.VectorType.get([words], i32)
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr)))
    return llvm.LoadOp(
        ty, ptr.result, alignment=4 * words, nontemporal=nt or None
    ).result


def ld_i32(ptr, i):
    """Word ``i`` of a buffer, a plain Int32 load."""
    return fx.Int32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32))


def row_live(batch_ids, k):
    """Whether row k of the step is a request's: its ``batch_id_per_q_token``
    entry, -1 on a CUDA-graph pad row (wave-uniform for a uniform k). A pad row
    runs like any other but joins no decision another row's work depends on."""
    return uniform(ld_i32(batch_ids, k)) >= 0


def ld_dev(ptr, i, words=1):
    """``words`` i32 (1, 2 or 4) of a buffer from word ``i``, at device scope
    (past the per-XCD L2)."""
    return bo.buffer_load(
        rsrc(ptr), i, vec_width=words, dtype=T.i32, cache_modifier=CM_DEV
    )


LOAD_BATCH = 4  # rounds of a strided loop whose loads go out together


@traced
def batched_rounds(tid, n, load, use, batch=LOAD_BATCH):
    """A CTA's strided loop over elements v = tid + THREADS i < n (n a Python
    int): ``load(v)`` of ``batch`` rounds issued before any ``use(v, loaded)``.
    A round whose ``use`` stores to global memory keeps the next round's loads
    behind it (they may alias), so the rounds' loads would go out in series. A
    round past n loads element n - 1 (in range) and uses nothing."""
    rounds = -(-n // THREADS)
    for b0 in range_constexpr(0, rounds, batch):
        idx = range(b0, min(b0 + batch, rounds))
        got = [load(fx.min(tid + THREADS * i, n - 1)) for i in idx]
        for i, g in zip(idx, got):
            v = tid + THREADS * i
            if v < n:
                use(v, g)


def rows_to_lds(tid, src, rows, row_words, dst, dst_row):
    """``rows`` rows of ``row_words`` words (a multiple of 4) read at device scope
    from address ``src`` -> LDS ``dst``, a row every ``dst_row`` words: 16 B a
    lane, the loads batched (``batched_rounds``)."""
    assert row_words % 4 == 0, row_words
    batched_rounds(
        tid, rows * row_words // 4,
        lambda v: fx.Vector(ld_dev(src, v * 4, 4)),
        lambda v, w: fx.ptr_store(
            w, dst + (v * 4 // row_words * dst_row + v * 4 % row_words)
        ),
    )  # fmt: skip


def _bf16x4(v8, half):
    """Elements 4 half .. 4 half + 3 of a v8bf16, as the v4i16 the gfx942 MFMA
    takes."""
    w = v8.bitcast(fx.Int32)
    return fx.Vector.from_elements(
        [w[2 * half], w[2 * half + 1]], fx.Int32
    ).bitcast(fx.Int16)


def mfma_bf16(a, b, c):
    """16x16x32 bf16 MFMA of v8bf16 operands; gfx942 (no K=32 form) runs it as
    two 16x16x16 on the operands' halves into the same accumulator."""
    if GFX942:
        for h in range(2):
            c = fx.Vector(
                rocdl.mfma_f32_16x16x16bf16_1k(
                    T.vec(4, T.f32), [_bf16x4(a, h), _bf16x4(b, h), c, 0, 0, 0]
                )
            )
        return c
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


def row_sum(red, r, t, waves=None):
    """Row r, token t of the waves' 16x16 MFMA accumulators (wave w's at ``red``
    + (w * 64 + lane) * 4), summed in wave order from 0 (``waves``: which, in
    order; all of them by default)."""
    tot = fx.Float32(0.0)
    for w in waves if waves is not None else range(WAVES):
        tot = tot + fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4))
    return tot


def mfma_fp8(a, b, c):
    """16x16x32 fp8 x fp8 MFMA, unscaled: ``a`` / ``b`` an i64 of 8 E4M3 each."""
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_fp8_fp8(T.vec(4, T.f32), [a, b, c, 0, 0, 0])
    )


def ballot(pred):
    """The wave's lanes where ``pred`` holds, a bit a lane."""
    return fx.Int64(rocdl.ballot(T.i64, as_ir_value(pred)))


def popcount(x):
    return fx.Int32(fx.ctpop(x))


def lanes_below(bal, lane):
    """How many of the lanes in ballot ``bal`` are below ``lane``."""
    return popcount(bal & ((fx.Int64(1) << fx.Int64(lane)) - 1))


def load_ptr64(table, i):
    """Entry ``i`` of an i64 table in memory, wave-uniform (each half through
    ``uniform``): a peer's buffer base, a kernel argument block's pointer."""
    pw = fx.Vector(bo.buffer_load(rsrc(table), i * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))


def fp8x8_bf16_pk(words):
    """Eight fp8 of two dwords -> bf16 MFMA operand (exact) by
    v_cvt_scalef32_pk_bf16_fp8 (scale 1), two values an instruction."""
    out = []
    for w in words:
        for hi in (False, True):
            p = rocdl.cvt_scalef32_pk_bf16_fp8(
                T.bf16x2, as_ir_value(fx.Int32(w)), as_ir_value(fx.Float32(1.0)), hi
            )
            out.append(fx.Vector(p).bitcast(fx.Int32)[0])
    return fx.Vector.from_elements(out, fx.Int32).bitcast(fx.BFloat16)


def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf16_pair(a, b):
    """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Float32)[0]
    )


def lds_bytes(struct) -> int:
    """The LDS an ``fx.struct`` class takes, alignment included."""
    return struct.__dsl_size_of__()


def bf_lo(w):
    """The f32 of a packed bf16 pair word's low half."""
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    """The f32 of a packed bf16 pair word's high half."""
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def bf2_f32(w):
    """Packed bf16 pair word -> (f32 low, f32 high)."""
    return bf_lo(w), bf_hi(w)


def fp8_pack4(a, b, c, d):
    """Four f32 (already in range) -> one dword of four E4M3 bytes, RNE."""
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))
