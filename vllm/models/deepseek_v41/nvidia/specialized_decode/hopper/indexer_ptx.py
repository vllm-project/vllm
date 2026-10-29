# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PTX helpers of the fused decode indexer kernels (``indexer_cute``)."""

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
def st_global_b32(addr: Int64, a: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(a).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        side=True,
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
def st_global_f32(addr: Int64, v: Float32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        side=True,
    )


@dsl_user_op
def cp_async16(dst: Int32, src: Int64, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        side=True,
    )


@dsl_user_op
def cp_async4(dst: Int32, src: Int64, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip)],
        "cp.async.ca.shared.global [$0], [$1], 4;",
        "r,l",
        side=True,
    )


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None):
    _asm(None, [], "cp.async.commit_group;", "", side=True)


@dsl_user_op
def cp_async_wait(n: int, *, loc=None, ip=None):
    _asm(None, [], f"cp.async.wait_group {n};", "", side=True)


@dsl_user_op
def ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            side=True,
        )
    )


@dsl_user_op
def red_or_b32(addr: Int64, v: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(v).ir_value(loc=loc, ip=ip)],
        "red.global.or.b32 [$0], $1;",
        "l,r",
        side=True,
    )
