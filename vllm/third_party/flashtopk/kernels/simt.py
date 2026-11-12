# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""PTX helpers for the SIMT kernels (calibration, selection, recovery): integer and float conversions with explicit
rounding, shared and global atomics, and block-wide reductions. Every float operation names its rounding, so no
compiler choice (contraction, fast math) can change a bit."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

THREADS = 256
WARPS = THREADS // 32


def _asm(ret_type, values, text, constraints, side_effects=False, *, loc=None, ip=None):
    return llvm.inline_asm(
        ret_type,
        [v.ir_value(loc=loc, ip=ip) for v in values],
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def f32_to_i32_rz(x, *, loc=None, ip=None) -> Int32:
    """static_cast<int>(float): round toward zero, saturating (NaN to 0)."""
    return Int32(_asm(Int32.mlir_type, [Float32(x)], "cvt.rzi.s32.f32 $0, $1;", "=r,f", loc=loc, ip=ip))


@dsl_user_op
def ceil_f32(x, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(Float32.mlir_type, [Float32(x)], "cvt.rpi.f32.f32 $0, $1;", "=f,f", loc=loc, ip=ip))


@dsl_user_op
def bit_length_u32(x, *, loc=None, ip=None) -> Int32:
    """32 - clz(x): the bits x needs (0 for 0)."""
    clz = Int32(_asm(Int32.mlir_type, [cutlass.Uint32(x)], "clz.b32 $0, $1;", "=r,r", loc=loc, ip=ip))
    return Int32(32) - clz


@dsl_user_op
def atomic_min_shared_u32(smem_addr, value, *, loc=None, ip=None) -> None:
    _asm(Int32.mlir_type, [Int32(smem_addr), cutlass.Uint32(value)], "atom.shared.min.u32 $0, [$1], $2;", "=r,r,r",
         True, loc=loc, ip=ip)


@dsl_user_op
def atomic_max_shared_u32(smem_addr, value, *, loc=None, ip=None) -> None:
    _asm(Int32.mlir_type, [Int32(smem_addr), cutlass.Uint32(value)], "atom.shared.max.u32 $0, [$1], $2;", "=r,r,r",
         True, loc=loc, ip=ip)


@dsl_user_op
def red_add_shared_u32(smem_addr, value, *, loc=None, ip=None) -> None:
    """A shared-memory add whose old value is not needed."""
    llvm.inline_asm(None, [Int32(smem_addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
                    "red.shared.add.u32 [$0], $1;", "r,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def atomic_add_shared_u32(smem_addr, value, *, loc=None, ip=None) -> Int32:
    """Returns the old value."""
    return Int32(_asm(Int32.mlir_type, [Int32(smem_addr), Int32(value)], "atom.shared.add.u32 $0, [$1], $2;",
                      "=r,r,r", True, loc=loc, ip=ip))


@dsl_user_op
def atomic_min_shared_s32(smem_addr, value, *, loc=None, ip=None) -> None:
    _asm(Int32.mlir_type, [Int32(smem_addr), Int32(value)], "atom.shared.min.s32 $0, [$1], $2;", "=r,r,r", True,
         loc=loc, ip=ip)


@dsl_user_op
def atomic_add_global_u64(address, value, *, loc=None, ip=None) -> cutlass.Int64:
    """Returns the old value."""
    return cutlass.Int64(_asm(cutlass.Int64.mlir_type, [cutlass.Int64(address), cutlass.Int64(value)],
                              "atom.global.add.u64 $0, [$1], $2;", "=l,l,l", True, loc=loc, ip=ip))


@dsl_user_op
def atomic_max_global_u64(address, value, *, loc=None, ip=None) -> None:
    _asm(cutlass.Int64.mlir_type, [cutlass.Int64(address), cutlass.Int64(value)],
         "atom.global.max.u64 $0, [$1], $2;", "=l,l,l", True, loc=loc, ip=ip)


@dsl_user_op
def atomic_add_global_u32(address, value, *, loc=None, ip=None) -> Int32:
    """Returns the old value."""
    return Int32(_asm(Int32.mlir_type, [cutlass.Int64(address), Int32(value)], "atom.global.add.u32 $0, [$1], $2;",
                      "=r,l,r", True, loc=loc, ip=ip))


@dsl_user_op
def atomic_exch_global_u32(address, value, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(Int32.mlir_type, [cutlass.Int64(address), Int32(value)], "atom.global.exch.b32 $0, [$1], $2;",
                      "=r,l,r", True, loc=loc, ip=ip))


@dsl_user_op
def load_global_cg_u32(address, *, loc=None, ip=None) -> Int32:
    """A 32-bit load from L2 (no L1), for values other CTAs wrote."""
    return Int32(_asm(Int32.mlir_type, [cutlass.Int64(address)], "ld.global.cg.u32 $0, [$1];", "=r,l", True,
                      loc=loc, ip=ip))


@dsl_user_op
def fence_gpu(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "fence.acq_rel.gpu;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def f32_bits(x, *, loc=None, ip=None) -> cutlass.Uint32:
    return cutlass.Uint32(llvm.bitcast(cutlass.Uint32.mlir_type, Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def bits_f32(x, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(Float32.mlir_type, cutlass.Uint32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@cute.jit
def finite(x) -> cutlass.Boolean:
    """isfinite(x): the exponent is not all ones."""
    return (f32_bits(x) & cutlass.Uint32(0x7F800000)) != cutlass.Uint32(0x7F800000)


@cute.jit
def ascending_key(v) -> cutlass.Uint32:
    """The float's bits as an unsigned key in ascending float order (-0.0 below +0.0; NaN at the ends)."""
    bits = f32_bits(v)
    flip = cutlass.Uint32(0x80000000)
    if (bits & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
        flip = cutlass.Uint32(0xFFFFFFFF)
    return bits ^ flip


@cute.jit
def score_key(score) -> cutlass.Uint32:
    """Recovery's and canonical order's key, ascending in the score: -0.0 and +0.0 equal, NaN above everything."""
    bits = f32_bits(score)
    key = cutlass.Uint32(0xFFFFFFFF)
    if score == score:
        if (bits & cutlass.Uint32(0x7FFFFFFF)) == cutlass.Uint32(0):
            bits = cutlass.Uint32(0)
        flip = cutlass.Uint32(0x80000000)
        if (bits & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
            flip = cutlass.Uint32(0xFFFFFFFF)
        key = bits ^ flip
    return key


@cute.jit
def block_sum_i32(value, sScratch, tid) -> Int32:
    """The sum over the block's THREADS threads (every thread gets it); sScratch holds WARPS ints."""
    v = cute.arch.warp_reduction_sum(value)
    lane = tid % 32
    if lane == 0:
        sScratch[tid // 32] = v
    cute.arch.sync_threads()
    total = Int32(0)
    for w in cutlass.range_constexpr(WARPS):
        total = total + sScratch[w]
    cute.arch.sync_threads()
    return total


@cute.jit
def block_max_i32(value, sScratch, tid) -> Int32:
    """The maximum over the block's THREADS threads (every thread gets it); sScratch holds WARPS ints."""
    v = value
    for off in cutlass.range_constexpr(5):
        v = max(v, cute.arch.shuffle_sync_bfly(v, 1 << off))
    if tid % 32 == 0:
        sScratch[tid // 32] = v
    cute.arch.sync_threads()
    m = sScratch[0]
    for w in cutlass.range_constexpr(1, WARPS):
        m = max(m, sScratch[w])
    cute.arch.sync_threads()
    return m


@cute.jit
def block_inclusive_scan_i32(value, sScratch, tid):
    """(inclusive prefix over the block in thread order, block total); sScratch holds WARPS ints."""
    lane = tid % 32
    incl = value
    for off in cutlass.range_constexpr(5):
        up = cute.arch.shuffle_sync_up(incl, 1 << off, mask_and_clamp=0)
        if lane >= (1 << off):
            incl = incl + up
    if lane == 31:
        sScratch[tid // 32] = incl
    cute.arch.sync_threads()
    prefix = Int32(0)
    total = Int32(0)
    warp = tid // 32
    for w in cutlass.range_constexpr(WARPS):
        s = sScratch[w]
        if w < warp:
            prefix = prefix + s
        total = total + s
    cute.arch.sync_threads()
    return prefix + incl, total
