# SPDX-License-Identifier: Apache-2.0
"""Shared CuTe device operations."""

from __future__ import annotations

from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def ldg_nc_v4_u32(gaddr, *, loc=None, ip=None):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(gaddr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (Uint32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4))
    )


def _exact_dot_asm() -> str:
    lines = [
        "{",
        ".reg .b16 h0, h1, q0, q1;",
        ".reg .b32 p;",
        ".reg .f32 a, b, acc0, acc1;",
        "mov.f32 acc0, 0f00000000;",
        "mov.f32 acc1, 0f00000000;",
    ]
    for k in range(4):
        c = 5 + 4 * k
        lines += [
            f"mov.b32 {{h0, h1}}, ${1 + k};",
            "cvt.rn.f16x2.e4m3x2 p, h0;",
            "mov.b32 {q0, q1}, p;",
            "cvt.f32.f16 a, q0;",
            "cvt.f32.f16 b, q1;",
            f"fma.rn.f32 acc0, a, ${c}, acc0;",
            f"fma.rn.f32 acc1, b, ${c + 1}, acc1;",
            "cvt.rn.f16x2.e4m3x2 p, h1;",
            "mov.b32 {q0, q1}, p;",
            "cvt.f32.f16 a, q0;",
            "cvt.f32.f16 b, q1;",
            f"fma.rn.f32 acc0, a, ${c + 2}, acc0;",
            f"fma.rn.f32 acc1, b, ${c + 3}, acc1;",
        ]
    lines += ["add.f32 $0, acc0, acc1;", "}"]
    return "\n\t".join(lines) + "\n"


_EXACT_DOT_ASM = _exact_dot_asm()


@dsl_user_op
def e4m3x16_dot_f32(
    w0,
    w1,
    w2,
    w3,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
    *,
    loc=None,
    ip=None,
) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)]
        + [
            Float32(c).ir_value(loc=loc, ip=ip)
            for c in (
                c0,
                c1,
                c2,
                c3,
                c4,
                c5,
                c6,
                c7,
                c8,
                c9,
                c10,
                c11,
                c12,
                c13,
                c14,
                c15,
            )
        ],
        _EXACT_DOT_ASM,
        "=f,r,r,r,r," + ",".join(["f"] * 16),
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


@dsl_user_op
def cp_async_bulk_g2s(dst_smem, src_gmem, nbytes, mbar_smem, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [
            Int32(dst_smem).ir_value(loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem).ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n"
            "\tcp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2], $3, [$4];\n"
            "\tmov.u32 $0, 0;\n"
            "\t}\n"
        ),
        "=r,r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_v4_u32(smem_addr, *, loc=None, ip=None):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (Uint32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4))
    )


@dsl_user_op
def lds_f32(smem_addr, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.f32 $0, [$1];",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


@dsl_user_op
def lds_bf16_f32(smem_addr, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        (
            "{\n"
            "\t.reg .b32 v;\n"
            "\tld.shared.u16 v, [$1];\n"
            "\tshl.b32 v, v, 16;\n"
            "\tmov.b32 $0, v;\n"
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
def bf16x4_bits_to_f32x4(bits: Uint64, *, loc=None, ip=None):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint64(bits).ir_value(loc=loc, ip=ip)],
        (
            "{\n"
            "\t.reg .b32 lo, hi;\n"
            "\t.reg .b16 b0, b1, b2, b3;\n"
            "\tmov.b64 {lo, hi}, $4;\n"
            "\tmov.b32 {b0, b1}, lo;\n"
            "\tmov.b32 {b2, b3}, hi;\n"
            "\tcvt.f32.bf16 $0, b0;\n"
            "\tcvt.f32.bf16 $1, b1;\n"
            "\tcvt.f32.bf16 $2, b2;\n"
            "\tcvt.f32.bf16 $3, b3;\n"
            "\t}\n"
        ),
        "=f,=f,=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (
            Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip))
            for i in range(4)
        )
    )


@dsl_user_op
def lds_e4m3x2_to_f32x2(smem_addr, *, loc=None, ip=None):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        (
            "{\n"
            "\t.reg .b16 v, lo, hi;\n"
            "\t.reg .b32 p;\n"
            "\tld.shared.u16 v, [$2];\n"
            "\tcvt.rn.f16x2.e4m3x2 p, v;\n"
            "\tmov.b32 {lo, hi}, p;\n"
            "\tcvt.f32.f16 $0, lo;\n"
            "\tcvt.f32.f16 $1, hi;\n"
            "\t}\n"
        ),
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    lo = Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip))
    hi = Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip))
    return (lo, hi)
