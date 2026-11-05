# SPDX-License-Identifier: Apache-2.0
"""Shared CuTe device operations."""

from __future__ import annotations

from cutlass import Float32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def f32x8_to_e2m1x8(
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    a4: Float32,
    a5: Float32,
    a6: Float32,
    a7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, a4, a5, a6, a7)],
        (
            "{\n"
            "\t.reg .b8 x0, x1, x2, x3;\n"
            "\tcvt.rn.satfinite.e2m1x2.f32 x0, $2, $1;\n"
            "\tcvt.rn.satfinite.e2m1x2.f32 x1, $4, $3;\n"
            "\tcvt.rn.satfinite.e2m1x2.f32 x2, $6, $5;\n"
            "\tcvt.rn.satfinite.e2m1x2.f32 x3, $8, $7;\n"
            "\tmov.b32 $0, {x0, x1, x2, x3};\n"
            "\t}\n"
        ),
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


@dsl_user_op
def e2m1_bytes8_fma_f16x2_to_f32(
    bytes0: Uint32,
    bytes1: Uint32,
    q_word0: Uint32,
    q_word1: Uint32,
    q_word2: Uint32,
    q_word3: Uint32,
    q_word4: Uint32,
    q_word5: Uint32,
    q_word6: Uint32,
    q_word7: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [
            Uint32(value).ir_value(loc=loc, ip=ip)
            for value in (
                bytes0,
                bytes1,
                q_word0,
                q_word1,
                q_word2,
                q_word3,
                q_word4,
                q_word5,
                q_word6,
                q_word7,
            )
        ],
        (
            "{\n"
            "\t.reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n"
            "\t.reg .b32 h, acc;\n"
            "\t.reg .b16 lo, hi;\n"
            "\t.reg .f32 flo, fhi;\n"
            "\tmov.b32 {b0, b1, b2, b3}, $1;\n"
            "\tmov.b32 {b4, b5, b6, b7}, $2;\n"
            "\tmov.b32 acc, 0;\n"
            "\tcvt.rn.f16x2.e2m1x2 h, b0;\n"
            "\tfma.rn.f16x2 acc, h, $3, acc;\n"
            "\tcvt.rn.f16x2.e2m1x2 h, b1;\n"
            "\tfma.rn.f16x2 acc, h, $4, acc;\n"
            "\tcvt.rn.f16x2.e2m1x2 h, b2;\n"
            "\tfma.rn.f16x2 acc, h, $5, acc;\n"
            "\tcvt.rn.f16x2.e2m1x2 h, b3;\n"
            "\tfma.rn.f16x2 acc, h, $6, acc;\n"
            "\tcvt.rn.f16x2.e2m1x2 h, b4;\n"
            "\tfma.rn.f16x2 acc, h, $7, acc;\n"
            "\tcvt.rn.f16x2.e2m1x2 h, b5;\n"
            "\tfma.rn.f16x2 acc, h, $8, acc;\n"
            "\tcvt.rn.f16x2.e2m1x2 h, b6;\n"
            "\tfma.rn.f16x2 acc, h, $9, acc;\n"
            "\tcvt.rn.f16x2.e2m1x2 h, b7;\n"
            "\tfma.rn.f16x2 acc, h, $10, acc;\n"
            "\tmov.b32 {lo, hi}, acc;\n"
            "\tcvt.f32.f16 flo, lo;\n"
            "\tcvt.f32.f16 fhi, hi;\n"
            "\tadd.f32 $0, flo, fhi;\n"
            "\t}\n"
        ),
        "=f,r,r,r,r,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


@dsl_user_op
def f32_to_e4m3_byte(v: Float32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [Float32(v).ir_value(loc=loc, ip=ip)],
        (
            "{\n"
            "\t.reg .b16 t0;\n"
            "\tcvt.rn.satfinite.e4m3x2.f32 t0, $1, $1;\n"
            "\tcvt.u32.u16 $0, t0;\n"
            "\tand.b32 $0, $0, 0xFF;\n"
            "\t}\n"
        ),
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


@dsl_user_op
def e4m3_byte_to_f32(byte: Uint32, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [Uint32(byte).ir_value(loc=loc, ip=ip)],
        (
            "{\n"
            "\t.reg .b16 in16, flo, fhi;\n"
            "\t.reg .b32 packed;\n"
            "\tcvt.u16.u32 in16, $1;\n"
            "\tcvt.rn.f16x2.e4m3x2 packed, in16;\n"
            "\tmov.b32 {flo, fhi}, packed;\n"
            "\tcvt.f32.f16 $0, flo;\n"
            "\t}\n"
        ),
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


@dsl_user_op
def f32x2_to_f16x2_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "cvt.rn.f16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)
