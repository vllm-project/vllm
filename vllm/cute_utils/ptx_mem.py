# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""16-byte global memory accesses at raw (16-byte aligned) addresses."""

from cutlass import Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def ld_global_v4(
    addr: Int64, nc: bool = False, *, loc=None, ip=None
) -> tuple[Uint32, ...]:
    """Load four 32-bit words. `nc` reads through the non-coherent path
    without L1 allocation, for data nothing writes while the kernel runs;
    otherwise the load is coherent and keeps its program order (volatile)."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        f"ld.global{'.nc.L1::no_allocate' if nc else ''}.v4.u32 "
        "{$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=not nc,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def st_global_v4(addr: Int64, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
    """Store four 32-bit words (any 32-bit integer DSL values)."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [w.ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
