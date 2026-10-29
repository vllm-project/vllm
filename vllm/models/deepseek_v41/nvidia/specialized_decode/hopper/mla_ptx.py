# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PTX building blocks of the Hopper small-batch sparse-MLA kernels
(``mla_cute``)."""

import cutlass
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(ret, args, s, cons, side=False):
    return llvm.inline_asm(
        ret,
        args,
        s,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _ex(v, i):
    return Uint32(llvm.extractvalue(T.i32(), v, [i]))


@dsl_user_op
def ld_nc_v4(addr: Int64, *, loc=None, ip=None):
    """128-bit non-coherent global load (KV records, read-only)."""
    v = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
    )
    return _ex(v, 0), _ex(v, 1), _ex(v, 2), _ex(v, 3)


@dsl_user_op
def ld_nc_v2(addr: Int64, *, loc=None, ip=None):
    v = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
    )
    return _ex(v, 0), _ex(v, 1)


@dsl_user_op
def ld_nc_b32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.b32 $0, [$1];",
            "=r,l",
        )
    )


@dsl_user_op
def ld_nc_u8(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
        )
    )


@dsl_user_op
def ld_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
        )
    )


@dsl_user_op
def st_shared_v4(
    addr: Int32, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None
):
    _asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Uint32(b).ir_value(loc=loc, ip=ip),
            Uint32(c).ir_value(loc=loc, ip=ip),
            Uint32(d).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        side=True,
    )


@dsl_user_op
def st_shared_u8(addr: Int32, a: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(a).ir_value(loc=loc, ip=ip)],
        "st.shared.u8 [$0], $1;",
        "r,r",
        side=True,
    )


@dsl_user_op
def st_global_b32(addr: Int64, a: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(a).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        side=True,
    )


@dsl_user_op
def st_global_v4(
    addr: Int64, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None
):
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Uint32(b).ir_value(loc=loc, ip=ip),
            Uint32(c).ir_value(loc=loc, ip=ip),
            Uint32(d).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side=True,
    )


@dsl_user_op
def cvt_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Float32(b).ir_value(loc=loc, ip=ip), Float32(a).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
        )
    )


@dsl_user_op
def mapa(addr: Int32, rank: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Int32(rank).ir_value(loc=loc, ip=ip),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
        )
    )


@dsl_user_op
def st_global_v2(addr: Int64, a: Uint32, b: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Uint32(b).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        side=True,
    )


@dsl_user_op
def st_shared_b32(addr: Int32, a: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(a).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        side=True,
    )


@dsl_user_op
def dq_e4m3x4_f16(w: Uint32, s1: Uint32, *, loc=None, ip=None):
    """4 x e4m3 -> 2 x f16x2 (hardware cvt), times f16x2 scale s1."""
    v = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Uint32(w).ir_value(loc=loc, ip=ip), Uint32(s1).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 l, h;\n\t.reg .b32 a, b;\n\t"
        "mov.b32 {l, h}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 a, l;\n\t"
        "cvt.rn.f16x2.e4m3x2 b, h;\n\t"
        "mul.rn.f16x2 $0, a, $3;\n\t"
        "mul.rn.f16x2 $1, b, $3;\n\t}",
        "=r,=r,r,r",
    )
    return _ex(v, 0), _ex(v, 1)


@dsl_user_op
def dq_e4m3x4_f16_2(w: Uint32, s1: Uint32, s2: Uint32, *, loc=None, ip=None):
    """Same with a two-step scale (for scales below the f16 normal range)."""
    v = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [
            Uint32(w).ir_value(loc=loc, ip=ip),
            Uint32(s1).ir_value(loc=loc, ip=ip),
            Uint32(s2).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b16 l, h;\n\t.reg .b32 a, b;\n\t"
        "mov.b32 {l, h}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 a, l;\n\t"
        "cvt.rn.f16x2.e4m3x2 b, h;\n\t"
        "mul.rn.f16x2 a, a, $3;\n\t"
        "mul.rn.f16x2 b, b, $3;\n\t"
        "mul.rn.f16x2 $0, a, $4;\n\t"
        "mul.rn.f16x2 $1, b, $4;\n\t}",
        "=r,=r,r,r,r",
    )
    return _ex(v, 0), _ex(v, 1)


@dsl_user_op
def bf16x2_to_f16x2(w: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(w).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 lo, hi;\n\t"
            "shl.b32 lo, $1, 16;\n\t"
            "and.b32 hi, $1, 0xFFFF0000;\n\t"
            "cvt.rn.f16x2.f32 $0, hi, lo;\n\t}",
            "=r,r",
        )
    )


@dsl_user_op
def cvt_f16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Float32(b).ir_value(loc=loc, ip=ip), Float32(a).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.f32 $0, $1, $2;",
            "=r,f,f",
        )
    )


@dsl_user_op
def st_global_u64(addr: Int64, v: Int64, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(v).ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        side=True,
    )


@dsl_user_op
def ld_shared_v2(addr: Int32, *, loc=None, ip=None):
    v = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r",
        side=True,
    )
    return _ex(v, 0), _ex(v, 1)


@dsl_user_op
def bf16x2_unpack(w: Uint32, *, loc=None, ip=None):
    v = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 a, b;\n\tshl.b32 a, $2, 16;\n\tand.b32 b, $2, 0xFFFF0000;\n\tmov.b32 $0, a;\n\tmov.b32 $1, b;\n\t}",  # noqa: E501
        "=f,=f,r",
    )
    return Float32(llvm.extractvalue(T.f32(), v, [0])), Float32(
        llvm.extractvalue(T.f32(), v, [1])
    )


@dsl_user_op
def bulk_g2s(dst: Int32, src: Int64, nbytes: Int32, mbar: Int32, *, loc=None, ip=None):
    """cp.async.bulk global -> shared (this CTA), completion as tx bytes on mbar."""
    _asm(
        None,
        [
            Int32(dst).ir_value(loc=loc, ip=ip),
            Int64(src).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",  # noqa: E501
        "r,l,r,r",
        side=True,
    )


@dsl_user_op
def st_global_f32(addr: Int64, v: Float32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        side=True,
    )


@dsl_user_op
def ld_cg_v4(addr: Int64, *, loc=None, ip=None):
    v = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side=True,
    )
    return _ex(v, 0), _ex(v, 1), _ex(v, 2), _ex(v, 3)


@dsl_user_op
def ld_cg_b32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.b32 $0, [$1];",
            "=r,l",
            side=True,
        )
    )


@dsl_user_op
def bulk_s2g(dst: Int64, src: Int32, nbytes: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [
            Int64(dst).ir_value(loc=loc, ip=ip),
            Int32(src).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;\n\tcp.async.bulk.commit_group;",  # noqa: E501
        "l,r,r",
        side=True,
    )


@dsl_user_op
def bulk_wait_read0(*, loc=None, ip=None):
    _asm(None, [], "cp.async.bulk.wait_group.read 0;", "", side=True)


@dsl_user_op
def ld_cg_v2(addr: Int64, *, loc=None, ip=None):
    v = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        side=True,
    )
    return _ex(v, 0), _ex(v, 1)


@dsl_user_op
def u32_as_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [Uint32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r")
    )


@dsl_user_op
def st_global_u8(addr: Int64, v: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        side=True,
    )


@cutlass.cutlass_dsl.dsl_user_op
def st_i32(addr: Int64, v: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(), Int32(v).ir_value()],
        "st.global.b32 [$0], $1;",
        "l,r",
        side=True,
    )


@cutlass.cutlass_dsl.dsl_user_op
def st_u8(addr: Int64, v: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(), Int32(v).ir_value()],
        "st.global.u8 [$0], $1;",
        "l,r",
        side=True,
    )
