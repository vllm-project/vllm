# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL kernel definitions for ll_router_gemm."""

import operator

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op

# ---------------------------------------------------------------------------
# fp8 pair conversion via PTX
# ---------------------------------------------------------------------------


@dsl_user_op
def fp8x2_cvt(packed_i16, *, loc=None, ip=None):
    """Convert packed Int16 (2x fp8 e4m3) -> 2x Float32 via PTX."""
    i16_ir = packed_i16.ir_value(loc=loc, ip=ip)
    i32_f16x2 = _llvm.inline_asm(
        _ir.IntegerType.get_signless(32),
        [i16_ir],
        "cvt.rn.f16x2.e4m3x2 $0, $1;",
        "=r,h",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    lo16 = _arith.trunci(_ir.IntegerType.get_signless(16), i32_f16x2, loc=loc, ip=ip)
    f32_lo = _llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [lo16],
        "cvt.f32.f16 $0, $1;",
        "=f,h",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    hi32 = _arith.shrui(
        i32_f16x2,
        _arith.constant(_ir.IntegerType.get_signless(32), 16, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    hi16 = _arith.trunci(_ir.IntegerType.get_signless(16), hi32, loc=loc, ip=ip)
    f32_hi = _llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [hi16],
        "cvt.f32.f16 $0, $1;",
        "=f,h",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(f32_lo), cutlass.Float32(f32_hi)


# ---------------------------------------------------------------------------
# bf16 kernel
# ---------------------------------------------------------------------------


@cute.kernel
def dotprod_bf16(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    M: cutlass.Constexpr,
    K_dim: cutlass.Int32,
    N_dim: cutlass.Int32,
):
    cute.arch.setmaxregister_increase(128)
    tidx = cute.arch.thread_idx()[0]
    n_idx = cute.arch.block_idx()[0]

    VPT: cutlass.Constexpr = 16
    BS: cutlass.Constexpr = 128
    KPI: cutlass.Constexpr = VPT * BS
    VPT_T: cutlass.Constexpr = 8
    KPT: cutlass.Constexpr = VPT_T * BS

    k_main = K_dim // KPI
    k_rem = K_dim - k_main * KPI
    k_tail = k_rem // KPT

    elem = gB.element_type
    acc = cute.make_rmem_tensor((M,), cutlass.Float32)
    for m in cutlass.range_constexpr(M):
        acc[m] = cutlass.Float32(0.0)

    if k_main > 0:
        kb0 = tidx * VPT
        bp0 = (gB.iterator + (n_idx * K_dim + kb0)).align(32)
        bt0 = cute.make_tensor(bp0, cute.make_layout((VPT,)))
        br0 = cute.make_rmem_tensor((VPT,), elem)
        cute.autovec_copy(bt0, br0)

        cute.arch.griddepcontrol_wait()
        
        for m in cutlass.range_constexpr(M):
            ap0 = (gA.iterator + (m * K_dim + kb0)).align(32)
            at0 = cute.make_tensor(ap0, cute.make_layout((VPT,)))
            ar0 = cute.make_rmem_tensor((VPT,), elem)
            cute.autovec_copy(at0, ar0)
            for v in cutlass.range_constexpr(VPT):
                acc[m] = acc[m] + ar0[v].to(cutlass.Float32) * br0[v].to(cutlass.Float32)

        for ki in cutlass.range(k_main - 1, unroll=4):
            kb = (ki + 1) * KPI + tidx * VPT
            bp = (gB.iterator + (n_idx * K_dim + kb)).align(32)
            bt = cute.make_tensor(bp, cute.make_layout((VPT,)))
            br = cute.make_rmem_tensor((VPT,), elem)
            cute.autovec_copy(bt, br)
            for m in cutlass.range_constexpr(M):
                ap = (gA.iterator + (m * K_dim + kb)).align(32)
                at = cute.make_tensor(ap, cute.make_layout((VPT,)))
                ar = cute.make_rmem_tensor((VPT,), elem)
                cute.autovec_copy(at, ar)
                for v in cutlass.range_constexpr(VPT):
                    acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br[v].to(cutlass.Float32)
    else:
        cute.arch.griddepcontrol_wait()

    # Tail: 128-bit loads
    for ti in cutlass.range(k_tail):
        kb = k_main * KPI + ti * KPT + tidx * VPT_T
        bp = (gB.iterator + (n_idx * K_dim + kb)).align(16)
        bt = cute.make_tensor(bp, cute.make_layout((VPT_T,)))
        br = cute.make_rmem_tensor((VPT_T,), elem)
        cute.autovec_copy(bt, br)
        for m in cutlass.range_constexpr(M):
            ap = (gA.iterator + (m * K_dim + kb)).align(16)
            at = cute.make_tensor(ap, cute.make_layout((VPT_T,)))
            ar = cute.make_rmem_tensor((VPT_T,), elem)
            cute.autovec_copy(at, ar)
            for v in cutlass.range_constexpr(VPT_T):
                acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br[v].to(cutlass.Float32)

    # Scalar tail for non-aligned K
    kp = k_main * KPI + k_tail * KPT
    kr = K_dim - kp
    ks_full = kr // BS
    ks_part = kr - ks_full * BS
    for si in cutlass.range(ks_full):
        ko = kp + si * BS + tidx
        bp = (gB.iterator + (n_idx * K_dim + ko)).align(2)
        bt = cute.make_tensor(bp, cute.make_layout((1,)))
        br = cute.make_rmem_tensor((1,), elem)
        cute.autovec_copy(bt, br)
        bv = br[0].to(cutlass.Float32)
        for m in cutlass.range_constexpr(M):
            ap = (gA.iterator + (m * K_dim + ko)).align(2)
            at = cute.make_tensor(ap, cute.make_layout((1,)))
            ar = cute.make_rmem_tensor((1,), elem)
            cute.autovec_copy(at, ar)
            acc[m] = acc[m] + ar[0].to(cutlass.Float32) * bv
    if tidx < ks_part:
        ko = kp + ks_full * BS + tidx
        bp = (gB.iterator + (n_idx * K_dim + ko)).align(2)
        bt = cute.make_tensor(bp, cute.make_layout((1,)))
        br = cute.make_rmem_tensor((1,), elem)
        cute.autovec_copy(bt, br)
        bv = br[0].to(cutlass.Float32)
        for m in cutlass.range_constexpr(M):
            ap = (gA.iterator + (m * K_dim + ko)).align(2)
            at = cute.make_tensor(ap, cute.make_layout((1,)))
            ar = cute.make_rmem_tensor((1,), elem)
            cute.autovec_copy(at, ar)
            acc[m] = acc[m] + ar[0].to(cutlass.Float32) * bv

    # Reduction
    WS: cutlass.Constexpr = 32
    NW: cutlass.Constexpr = BS // WS
    for m in cutlass.range_constexpr(M):
        acc[m] = cute.arch.warp_reduction(acc[m], operator.add)
    wid = tidx // WS
    lid = tidx % WS
    sp = cute.arch.alloc_smem(cutlass.Float32, M * NW, alignment=16)
    sm = cute.make_tensor(sp, cute.make_layout((M, NW)))
    for m in cutlass.range_constexpr(M):
        if lid == 0:
            sm[m, wid] = acc[m]
    cute.arch.sync_threads()
    if tidx == 0:
        for m in cutlass.range_constexpr(M):
            t = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(NW):
                t = t + sm[m, w]
            gC[m * N_dim + n_idx] = t.to(cutlass.Float32)
    cute.arch.griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------
# fp8 kernel (receives pre-packed Int16 data)
# ---------------------------------------------------------------------------


@cute.kernel
def dotprod_fp8(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    M: cutlass.Constexpr,
    K_pairs: cutlass.Int32,
    N_dim: cutlass.Int32,
):
    cute.arch.setmaxregister_increase(128)
    tidx = cute.arch.thread_idx()[0]
    n_idx = cute.arch.block_idx()[0]

    VPT: cutlass.Constexpr = 8
    F_PER_T: cutlass.Constexpr = 16
    BS: cutlass.Constexpr = 128
    KPI: cutlass.Constexpr = VPT * BS
    VPT_T: cutlass.Constexpr = 4
    F_PER_T_T: cutlass.Constexpr = 8
    KPT: cutlass.Constexpr = VPT_T * BS

    k_main = K_pairs // KPI
    k_rem = K_pairs - k_main * KPI
    k_tail = k_rem // KPT

    acc = cute.make_rmem_tensor((M,), cutlass.Float32)
    for m in cutlass.range_constexpr(M):
        acc[m] = cutlass.Float32(0.0)

    if k_main > 0:
        kb0 = tidx * VPT
        bp0 = (gB.iterator + (n_idx * K_pairs + kb0)).align(16)
        bt0 = cute.make_tensor(bp0, cute.make_layout((VPT,)))
        br0 = cute.make_rmem_tensor((VPT,), cutlass.Int16)
        cute.autovec_copy(bt0, br0)

        cute.arch.griddepcontrol_wait()
        
        bf0 = cute.make_rmem_tensor((F_PER_T,), cutlass.Float32)
        for p in cutlass.range_constexpr(VPT):
            bf0[p * 2], bf0[p * 2 + 1] = fp8x2_cvt(br0[p])
        for m in cutlass.range_constexpr(M):
            ap0 = (gA.iterator + (m * K_pairs + kb0)).align(16)
            at0 = cute.make_tensor(ap0, cute.make_layout((VPT,)))
            ar0 = cute.make_rmem_tensor((VPT,), cutlass.Int16)
            cute.autovec_copy(at0, ar0)
            for p in cutlass.range_constexpr(VPT):
                a00, a10 = fp8x2_cvt(ar0[p])
                acc[m] = acc[m] + a00 * bf0[p * 2] + a10 * bf0[p * 2 + 1]
        
        for ki in cutlass.range(k_main - 1, unroll=4):
            kb = (ki + 1) * KPI + tidx * VPT
            bp = (gB.iterator + (n_idx * K_pairs + kb)).align(16)
            bt = cute.make_tensor(bp, cute.make_layout((VPT,)))
            br = cute.make_rmem_tensor((VPT,), cutlass.Int16)
            cute.autovec_copy(bt, br)
            bf = cute.make_rmem_tensor((F_PER_T,), cutlass.Float32)
            for p in cutlass.range_constexpr(VPT):
                bf[p * 2], bf[p * 2 + 1] = fp8x2_cvt(br[p])
            for m in cutlass.range_constexpr(M):
                ap = (gA.iterator + (m * K_pairs + kb)).align(16)
                at = cute.make_tensor(ap, cute.make_layout((VPT,)))
                ar = cute.make_rmem_tensor((VPT,), cutlass.Int16)
                cute.autovec_copy(at, ar)
                for p in cutlass.range_constexpr(VPT):
                    a0, a1 = fp8x2_cvt(ar[p])
                    acc[m] = acc[m] + a0 * bf[p * 2] + a1 * bf[p * 2 + 1]
    else:
        cute.arch.griddepcontrol_wait()

    # Tail: 64-bit loads
    for ti in cutlass.range(k_tail):
        kb = k_main * KPI + ti * KPT + tidx * VPT_T
        bp = (gB.iterator + (n_idx * K_pairs + kb)).align(8)
        bt = cute.make_tensor(bp, cute.make_layout((VPT_T,)))
        br = cute.make_rmem_tensor((VPT_T,), cutlass.Int16)
        cute.autovec_copy(bt, br)
        bf = cute.make_rmem_tensor((F_PER_T_T,), cutlass.Float32)
        for p in cutlass.range_constexpr(VPT_T):
            bf[p * 2], bf[p * 2 + 1] = fp8x2_cvt(br[p])
        for m in cutlass.range_constexpr(M):
            ap = (gA.iterator + (m * K_pairs + kb)).align(8)
            at = cute.make_tensor(ap, cute.make_layout((VPT_T,)))
            ar = cute.make_rmem_tensor((VPT_T,), cutlass.Int16)
            cute.autovec_copy(at, ar)
            for p in cutlass.range_constexpr(VPT_T):
                a0, a1 = fp8x2_cvt(ar[p])
                acc[m] = acc[m] + a0 * bf[p * 2] + a1 * bf[p * 2 + 1]

    # Scalar tail
    kp = k_main * KPI + k_tail * KPT
    kr = K_pairs - kp
    ks_full = kr // BS
    ks_part = kr - ks_full * BS
    for si in cutlass.range(ks_full):
        ko = kp + si * BS + tidx
        bp = (gB.iterator + (n_idx * K_pairs + ko)).align(2)
        bt = cute.make_tensor(bp, cute.make_layout((1,)))
        br = cute.make_rmem_tensor((1,), cutlass.Int16)
        cute.autovec_copy(bt, br)
        b0, b1 = fp8x2_cvt(br[0])
        for m in cutlass.range_constexpr(M):
            ap = (gA.iterator + (m * K_pairs + ko)).align(2)
            at = cute.make_tensor(ap, cute.make_layout((1,)))
            ar = cute.make_rmem_tensor((1,), cutlass.Int16)
            cute.autovec_copy(at, ar)
            a0, a1 = fp8x2_cvt(ar[0])
            acc[m] = acc[m] + a0 * b0 + a1 * b1
    if tidx < ks_part:
        ko = kp + ks_full * BS + tidx
        bp = (gB.iterator + (n_idx * K_pairs + ko)).align(2)
        bt = cute.make_tensor(bp, cute.make_layout((1,)))
        br = cute.make_rmem_tensor((1,), cutlass.Int16)
        cute.autovec_copy(bt, br)
        b0, b1 = fp8x2_cvt(br[0])
        for m in cutlass.range_constexpr(M):
            ap = (gA.iterator + (m * K_pairs + ko)).align(2)
            at = cute.make_tensor(ap, cute.make_layout((1,)))
            ar = cute.make_rmem_tensor((1,), cutlass.Int16)
            cute.autovec_copy(at, ar)
            a0, a1 = fp8x2_cvt(ar[0])
            acc[m] = acc[m] + a0 * b0 + a1 * b1

    # Reduction
    WS: cutlass.Constexpr = 32
    NW: cutlass.Constexpr = BS // WS
    for m in cutlass.range_constexpr(M):
        acc[m] = cute.arch.warp_reduction(acc[m], operator.add)
    wid = tidx // WS
    lid = tidx % WS
    sp = cute.arch.alloc_smem(cutlass.Float32, M * NW, alignment=16)
    sm = cute.make_tensor(sp, cute.make_layout((M, NW)))
    for m in cutlass.range_constexpr(M):
        if lid == 0:
            sm[m, wid] = acc[m]
    cute.arch.sync_threads()
    if tidx == 0:
        for m in cutlass.range_constexpr(M):
            t = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(NW):
                t = t + sm[m, w]
            gC[m * N_dim + n_idx] = t.to(cutlass.Float32)
    cute.arch.griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------
# Host-side JIT wrappers
# ---------------------------------------------------------------------------


@cute.jit
def host_bf16(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    M: cutlass.Constexpr,
    K_dim: cutlass.Int32,
    N_dim: cutlass.Int32,
    stream: CUstream,
):
    dotprod_bf16(gA, gB, gC, M, K_dim, N_dim).launch(
        grid=[N_dim, 1, 1],
        block=[128, 1, 1],
        smem=M * 4 * 4,
        stream=stream,
        use_pdl=True,
    )


@cute.jit
def host_fp8(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    M: cutlass.Constexpr,
    K_pairs: cutlass.Int32,
    N_dim: cutlass.Int32,
    stream: CUstream,
):
    dotprod_fp8(gA, gB, gC, M, K_pairs, N_dim).launch(
        grid=[N_dim, 1, 1],
        block=[128, 1, 1],
        smem=M * 4 * 4,
        stream=stream,
        use_pdl=True,
    )
