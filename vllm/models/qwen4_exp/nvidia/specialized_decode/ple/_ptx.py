# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the PLE row gather (`.gather`), beside the boundary kernel's
and the logits gather's own.

The dequantization is spelled out instruction by instruction, so that its bits
match the stock path's `rows.to(torch.bfloat16) * scale.to(torch.bfloat16)`:
an E4M3 value is exact in FP16 and FP32, the product of an E4M3 value and a
BF16 scale is exact in FP32 (4 and 8 significant bits), and PyTorch's BF16
product rounds that FP32 product to nearest BF16 once, as here.
"""

from cutlass import Float32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..hc._ptx import _asm


@dsl_user_op
def load_x2(address: Int64, *, loc=None, ip=None) -> tuple[Uint32, Uint32]:
    """8 bytes at an 8-byte aligned global address that an earlier kernel
    wrote, as two words (the first holds the lower address)."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), loaded, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), loaded, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def dequant_e4m3x4(
    word: Uint32, scale: Float32, *, loc=None, ip=None
) -> tuple[Uint32, Uint32]:
    """Four E4M3 values (the bytes of `word`, the lowest first) times
    `scale`, each product rounded to nearest BF16: two words of two BF16
    values, the first holding values 0 and 1 (value 0 in its lower half)."""
    converted = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [
            Uint32(word).ir_value(loc=loc, ip=ip),
            Float32(scale).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b16 lo, hi, a0, a1, a2, a3;\n\t.reg .b32 h0, h1;\n\t"
        ".reg .f32 f0, f1, f2, f3;\n\t"
        "mov.b32 {lo, hi}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 h0, lo;\n\tcvt.rn.f16x2.e4m3x2 h1, hi;\n\t"
        "mov.b32 {a0, a1}, h0;\n\tmov.b32 {a2, a3}, h1;\n\t"
        "cvt.f32.f16 f0, a0;\n\tcvt.f32.f16 f1, a1;\n\t"
        "cvt.f32.f16 f2, a2;\n\tcvt.f32.f16 f3, a3;\n\t"
        "mul.rn.f32 f0, f0, $3;\n\tmul.rn.f32 f1, f1, $3;\n\t"
        "mul.rn.f32 f2, f2, $3;\n\tmul.rn.f32 f3, f3, $3;\n\t"
        "cvt.rn.bf16x2.f32 $0, f1, f0;\n\tcvt.rn.bf16x2.f32 $1, f3, f2;\n\t}",
        "=r,=r,r,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), converted, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), converted, [1], loc=loc, ip=ip)),
    )
