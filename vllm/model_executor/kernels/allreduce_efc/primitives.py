# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predicated PTX loads/stores and scalar conversions for the AR epilogues."""

import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(result, operands, text, constraints, *, side_effects, loc, ip):
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _pred(predicate, loc, ip):
    return Int32(predicate).ir_value(loc=loc, ip=ip)


@dsl_user_op
def load_u32x4_pred(address: Int64, predicate, *, loc=None, ip=None):
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip), _pred(predicate, loc, ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $5, 0;\n\t"
        "@!p mov.u32 $0, 0;\n\t@!p mov.u32 $1, 0;\n\t"
        "@!p mov.u32 $2, 0;\n\t@!p mov.u32 $3, 0;\n\t"
        "@p ld.global.v4.u32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), loaded, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def store_u32x4_pred(address: Int64, packed, predicate, *, loc=None, ip=None):
    words = [packed[i].ir_value(loc=loc, ip=ip) for i in range(4)]
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words, _pred(predicate, loc, ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $5, 0;\n\t"
        "@p st.global.v4.u32 [$0], {$1, $2, $3, $4};\n\t}",
        "l,r,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_u32x2_pred(address: Int64, w0, w1, predicate, *, loc=None, ip=None):
    _asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
            _pred(predicate, loc, ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $3, 0;\n\t"
        "@p st.global.v2.u32 [$0], {$1, $2};\n\t}",
        "l,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_u32_pred(address: Int64, word, predicate, *, loc=None, ip=None):
    _asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            Uint32(word).ir_value(loc=loc, ip=ip),
            _pred(predicate, loc, ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t"
        "@p st.global.u32 [$0], $1;\n\t}",
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_u8_pred(address: Int64, byte, predicate, *, loc=None, ip=None):
    _asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            Uint32(byte).ir_value(loc=loc, ip=ip),
            _pred(predicate, loc, ip),
        ],
        "{\n\t.reg .pred p;\n\t.reg .b16 b;\n\tsetp.ne.s32 p, $2, 0;\n\t"
        "cvt.u16.u32 b, $1;\n\t@p st.global.u8 [$0], b;\n\t}",
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def f32_to_e4m3_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 h;\n\t.reg .f32 z;\n\tmov.f32 z, 0f00000000;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 h, z, $1;\n\t"
            "cvt.u32.u16 $0, h;\n\tand.b32 $0, $0, 255;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def round_f32_to_e4m3(value: Float32, *, loc=None, ip=None) -> Float32:
    """FP32 -> E4M3 (RN, satfinite) -> FP32, like ``float(__nv_fp8_e4m3(x))``."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 h, lo, hi;\n\t.reg .b32 w;\n\t.reg .f32 z;\n\t"
            "mov.f32 z, 0f00000000;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 h, z, $1;\n\t"
            "cvt.rn.f16x2.e4m3x2 w, h;\n\t"
            "mov.b32 {lo, hi}, w;\n\t"
            "cvt.f32.f16 $0, lo;\n\t}",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f32_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ue8m0_ceil(value: Float32, *, loc=None, ip=None) -> Float32:
    """``exp2(ceil(log2(max(|x|, 1e-10))))`` by exponent bump, as vLLM's
    per-token-group quant computes UE8M0 scales."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 b, e, m;\n\t.reg .f32 a;\n\t.reg .pred nz;\n\t"
            "abs.f32 a, $1;\n\tmax.f32 a, a, 0f2EDBE6FF;\n\t"
            "mov.b32 b, a;\n\tshr.u32 e, b, 23;\n\tand.b32 m, b, 8388607;\n\t"
            "setp.ne.u32 nz, m, 0;\n\t@nz add.u32 e, e, 1;\n\t"
            "shl.b32 b, e, 23;\n\tmov.b32 $0, b;\n\t}",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def abs_f32(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )
