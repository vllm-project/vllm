# SPDX-License-Identifier: Apache-2.0
"""Shared CuTe device operations."""

from __future__ import annotations

from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def _exit_if_nonzero(flag: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(flag).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $0, 0;\n\t@p exit;\n\t}",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
