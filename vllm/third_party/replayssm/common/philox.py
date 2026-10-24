# SPDX-License-Identifier: Apache-2.0
"""Shared CuTe device operations."""

from __future__ import annotations

from cutlass import Float32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

PHILOX_ROUNDS_DEFAULT = 7
_ROUND_A = 3528531795
_ROUND_B = 3449720151
_KEY_A = 2654435769
_KEY_B = 3144134277


@dsl_user_op
def _mul_wide_u32(a: Uint32, b: Uint32, *, loc=None, ip=None):
    result_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    result = llvm.inline_asm(
        result_type,
        [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
        (
            "{\n"
            "\t.reg .u64 product;\n"
            "\tmul.wide.u32 product, $2, $3;\n"
            "\tmov.b64 {$1, $0}, product;\n"
            "\t}"
        ),
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    word_type = T.i32()
    hi = Uint32(llvm.extractvalue(word_type, result, [0], loc=loc, ip=ip))
    lo = Uint32(llvm.extractvalue(word_type, result, [1], loc=loc, ip=ip))
    return (hi, lo)


@dsl_user_op
def philox4x32(
    counter0: Uint32,
    counter1: Uint32,
    counter2: Uint32,
    counter3: Uint32,
    key0: Uint32,
    key1: Uint32,
    n_rounds: int = PHILOX_ROUNDS_DEFAULT,
    *,
    loc=None,
    ip=None,
):
    c0, c1, c2, c3 = (
        Uint32(counter0),
        Uint32(counter1),
        Uint32(counter2),
        Uint32(counter3),
    )
    k0, k1 = (Uint32(key0), Uint32(key1))
    round_a, round_b = (Uint32(_ROUND_A), Uint32(_ROUND_B))
    key_a, key_b = (Uint32(_KEY_A), Uint32(_KEY_B))
    for _ in range(n_rounds):
        hi_b, lo_b = _mul_wide_u32(c2, round_b, loc=loc, ip=ip)
        hi_a, lo_a = _mul_wide_u32(c0, round_a, loc=loc, ip=ip)
        c0 = hi_b ^ c1 ^ k0
        c2 = hi_a ^ c3 ^ k1
        c1 = lo_b
        c3 = lo_a
        k0 = k0 + key_a
        k1 = k1 + key_b
    return (c0, c1, c2, c3)


@dsl_user_op
def cvt_f32x2_f16x2_rs(
    a: Float32, b: Float32, rand_bits: Uint32, *, loc=None, ip=None
) -> Uint32:
    result = llvm.inline_asm(
        T.i32(),
        [
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Uint32(rand_bits).ir_value(loc=loc, ip=ip),
        ],
        "cvt.rs.satfinite.f16x2.f32 $0, $2, $1, $3;",
        "=r,f,f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)
