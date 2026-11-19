# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the MTP draft front norm.

The arithmetic helpers name the instruction each step of the stock
`GemmaRMSNorm` (PyTorch's elementwise and reduction kernels, built without
fast math) compiles to, so the fused kernel rounds every step as stock does:
`mul.rn.f32` and `add.rn.f32` (never contracted into a multiply-add) and
`rsqrt.approx.f32` without flush-to-zero.

Memory accesses are side-effecting assembly, so the compiler keeps them where
the kernel issues them: the immutable norm weights before the grid-dependency
wait, the activations an earlier kernel wrote after it. A 16-byte access holds
eight BF16 values; the lower half of each 32-bit word is the lower address.
"""

from cutlass import Float32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..hc._ptx import _asm

Floats8 = tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32]

_WIDEN8 = (
    "{\n\t.reg .b32 w0, w1, w2, w3;\n\t"
    "{load} {w0, w1, w2, w3}, [$8];\n\t"
    "shl.b32 $0, w0, 16;\n\tand.b32 $1, w0, 0xffff0000;\n\t"
    "shl.b32 $2, w1, 16;\n\tand.b32 $3, w1, 0xffff0000;\n\t"
    "shl.b32 $4, w2, 16;\n\tand.b32 $5, w2, 0xffff0000;\n\t"
    "shl.b32 $6, w3, 16;\n\tand.b32 $7, w3, 0xffff0000;\n\t}"
)


def _load8(load: str, address: Int64, loc, ip) -> Floats8:
    """Eight consecutive BF16 values loaded by `load`, widened to FP32."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        _WIDEN8.replace("{load}", load),
        "=f,=f,=f,=f,=f,=f,=f,=f,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return tuple(  # type: ignore[return-value]
        Float32(llvm.extractvalue(T.f32(), loaded, [index], loc=loc, ip=ip))
        for index in range(8)
    )


@dsl_user_op
def load_weight_bf16x8(address: Int64, *, loc=None, ip=None) -> Floats8:
    """Eight BF16 values of an immutable weight (non-coherent), widened.
    Issued where it is written, so a weight load placed before the
    grid-dependency wait overlaps the previous kernel."""
    return _load8("ld.global.nc.v4.b32", address, loc, ip)


@dsl_user_op
def load_bf16x8(address: Int64, *, loc=None, ip=None) -> Floats8:
    """Eight BF16 values an earlier kernel wrote (a coherent load, kept after
    the grid-dependency wait), widened."""
    return _load8("ld.global.v4.b32", address, loc, ip)


@dsl_user_op
def store_bf16x8(address: Int64, values: Floats8, *, loc=None, ip=None) -> None:
    """Store eight values rounded to nearest BF16 (16 bytes)."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)]
        + [Float32(value).ir_value(loc=loc, ip=ip) for value in values],
        "{\n\t.reg .b32 w0, w1, w2, w3;\n\t"
        "cvt.rn.bf16x2.f32 w0, $2, $1;\n\tcvt.rn.bf16x2.f32 w1, $4, $3;\n\t"
        "cvt.rn.bf16x2.f32 w2, $6, $5;\n\tcvt.rn.bf16x2.f32 w3, $8, $7;\n\t"
        "st.global.v4.b32 [$0], {w0, w1, w2, w3};\n\t}",
        "l,f,f,f,f,f,f,f,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


def _binary(text: str, a: Float32, b: Float32, loc, ip) -> Float32:
    """One FP32 instruction of two operands."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            text,
            "=f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a * b` rounded to nearest (never fused with an add)."""
    return _binary("mul.rn.f32 $0, $1, $2;", a, b, loc, ip)


@dsl_user_op
def add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a + b` rounded to nearest (never fused with a multiply)."""
    return _binary("add.rn.f32 $0, $1, $2;", a, b, loc, ip)


@dsl_user_op
def rsqrt_approx(a: Float32, *, loc=None, ip=None) -> Float32:
    """`rsqrtf(a)` as nvcc compiles it without fast math."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )
