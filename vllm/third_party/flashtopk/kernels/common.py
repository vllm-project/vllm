# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""What the kernels of every architecture share: the indexer's shape, the tiling the host side plans with, and PTX
helpers. Every float operation names its rounding, so no compiler choice (contraction, fast math) can change a bit."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import arith, llvm, math
from cutlass.cutlass_dsl import dsl_user_op

NUM_HEADS = 32
HEAD_DIM = 128
NEXT_N = 8  # query rows of a scan tile
BLOCK_KV = 128  # keys of a KV block: KV splits and the pruning mask are counted in these
PRUNE_BLOCK_Q = 16  # query rows of a pruning tile
PRUNE_COLS = 16  # columns of the pruning operand
REFINE_BLOCK_Q = 8  # query rows of a refine tile


@dsl_user_op
def store_global_b32(address, value, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [cutlass.Int64(address).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_global_i32(address, value, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [cutlass.Int64(address).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
            "atom.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def select_i32(predicate, true_value, false_value, *, loc=None, ip=None):
    return Int32(
        arith.select(
            cutlass.Boolean(predicate).ir_value(loc=loc, ip=ip),
            Int32(true_value).ir_value(loc=loc, ip=ip),
            Int32(false_value).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fp32_to_i32_bits(value, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.bitcast(Int32.mlir_type, Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def abs_f32(value, *, loc=None, ip=None) -> Float32:
    return Float32(math.absf(Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def _f32_asm(op, *values, loc=None, ip=None) -> Float32:
    operands = ",".join(["=f"] + ["f"] * len(values))
    args = ", ".join(f"${i + 1}" for i in range(len(values)))
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [Float32(v).ir_value(loc=loc, ip=ip) for v in values],
            f"{op} $0, {args};",
            operands,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("mul.rn.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def sub_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("sub.rn.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def add_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("add.rn.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def div_rn(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("div.rn.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def fma_rn(a, b, c, *, loc=None, ip=None) -> Float32:
    return _f32_asm("fma.rn.f32", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def fmax(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("max.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def fmin(a, b, *, loc=None, ip=None) -> Float32:
    return _f32_asm("min.f32", a, b, loc=loc, ip=ip)


@dsl_user_op
def i32_to_f32_rn(value, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [Int32(value).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f32.s32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def select_f32(predicate, true_value, false_value, *, loc=None, ip=None):
    return Float32(
        arith.select(
            cutlass.Boolean(predicate).ir_value(loc=loc, ip=ip),
            Float32(true_value).ir_value(loc=loc, ip=ip),
            Float32(false_value).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def loose_raw_gate(inv, origin, th_bucket) -> Float32:
    """A raw score below which no key passes the bucket gate
    fma(-inv, raw, c0) < th + 1 (c0 = -origin * inv), a little below the exact edge so rounding
    never excludes one; -inf where the bucket map is degenerate, at most FLT_MAX."""
    c0 = mul_rn(-origin, inv)
    o = i32_to_f32_rn(th_bucket + 1)
    valid = (inv > Float32(0.0)) & (inv <= Float32(3.4e38)) & (abs_f32(c0) <= Float32(3.4e38))
    slack = mul_rn(Float32(2.0**-16), fmax(fmax(abs_f32(c0), abs_f32(o)), Float32(1.0)))
    gate = fmin(div_rn(sub_rn(sub_rn(c0, o), slack), inv), Float32(3.4028234663852886e38))
    return select_f32(valid, gate, Float32(float("-inf")))


@dsl_user_op
def find_nth_set(word, n, *, loc=None, ip=None) -> Int32:
    """The position of the n-th (1-based) set bit of word, from bit 0."""
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [Int32(word).ir_value(loc=loc, ip=ip), Int32(n).ir_value(loc=loc, ip=ip)],
            "fns.b32 $0, $1, 0, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cp_async_16(smem_addr, gmem_addr, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip), cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_4(smem_addr, gmem_addr, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip), cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "cp.async.ca.shared.global [$0], [$1], 4;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_global_v4_b32(gmem_addr, *, loc=None, ip=None):
    """A 128-key block's four mask words with one 16-byte load."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 4),
        [cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(Int32.mlir_type, result, [i], loc=loc, ip=ip)) for i in range(4))
