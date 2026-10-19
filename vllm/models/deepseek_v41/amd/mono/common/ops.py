# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ops.py
# atom/mono/device/ranks.py
# atom/mono/device/stamps.py
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

Also the TP group seen from inside a kernel (every rank's peer buffer, an
all-reduce's rank-ordered sum) and a timeline build's phase stamps."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from vllm.models.deepseek_v41.amd.mono.common.plan import THREADS, WAVES

# ---------------------------------------------------------------- primitives
# gfx95x cache policy bits (LLVM CPol): SC0 = 1, SC1 = 16. SC1 alone is device
# scope (past the per-XCD caches); SC0 | SC1 is system scope (peers over XGMI).
CM_DEV = 16


CM_SYS = 17


# NT = 2: nontemporal. A weight a layer reads once streams past the caches it
# would only evict the step's reused data from (the MoE at 48 rows: 143 -> 133 us)
CM_NT = 2


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


def memrealtime():
    """s_memrealtime: the 100 MHz device clock (no FlyDSL wrapper)."""
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def fresh(v):
    """``v`` (a per-lane Int32) through an opaque move, here: the lane math built
    on it is computed at its use, where LLVM would otherwise hoist it out of a task
    loop or CSE it across stages into registers held for the whole kernel (spills,
    and a kernel with scratch is refused: ``runtime.widths``)."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [fx.Int32(v).ir_value()],
            "v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


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
    """Butterfly reduction over the given xor offsets, in the given order.

    The order is the numerical contract: the aiter per-head norm reduces 32
    lanes with offsets 16, 8, 4, 2, 1, and matching it keeps rstd bit-exact."""
    for off in offsets:
        v = xred(v, off, op)
    return v


def traced(fn):
    """Apply flydsl's AST rewriting to a helper, so its ``while`` / ``if`` over
    traced values lower to scf ops. Kernels rewrite only their own body."""
    return ASTRewriter.transform(fn)


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


def mfma_bf16(a, b, c):
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


def row_sum(red, r, t, waves=None):
    """Row r, token t of the waves' 16x16 MFMA accumulators (wave w's at ``red``
    + (w * 64 + lane) * 4), summed in wave order from 0 (``waves``: which, in
    order; all of them by default)."""
    tot = fx.Float32(0.0)
    for w in waves if waves is not None else range(WAVES):
        tot = tot + fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4))
    return tot


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


# ---------------------------------------------------------------- TP ranks
def peer_bases(peers, tp):
    """Every rank's peer buffer base from the ``peers`` table (an i64 a rank),
    loaded once: a put's address must not wait on a load of its own."""
    return [load_ptr64(peers, p) for p in range(tp)]


def sum_partials(poll, own, pair_of, tp):
    """The ``tp`` ranks' bf16 pair partials at ``pair_of(src)`` of the mailbox
    region ``own``, summed in fp32 in rank order 0 .. tp - 1: one poll batch ->
    (lo sum, hi sum). Every rank sums in the same order, so they agree."""
    ws = poll([(own, pair_of(src), 1) for src in range(tp)])
    acc0, acc1 = bf2_f32(ws[0][0])
    for src in range_constexpr(1, tp):
        lo, hi = bf2_f32(ws[src][0])
        acc0 = acc0 + lo
        acc1 = acc1 + hi
    return acc0, acc1


# ---------------------------------------------------------------- timeline stamps
def stamp(on, tls, tid, k):
    # compile-time gate outside, traced condition inside: they cannot be one `and`
    if const_expr(on):  # noqa: SIM102
        if tid == 0:
            fx.ptr_store(memrealtime(), tls + k)
