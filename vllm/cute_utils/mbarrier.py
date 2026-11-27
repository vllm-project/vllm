# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

_VALID_SPACES = ("cta", "cluster")
_VALID_ORDERS = ("relaxed", "release")


def _validate(space: str, order: str) -> None:
    if space not in _VALID_SPACES:
        raise ValueError(f"Unsupported mbarrier space: {space!r}")
    if order not in _VALID_ORDERS:
        raise ValueError(f"Unsupported mbarrier memory order: {order!r}")


@dsl_user_op
def arrive(mbar, space: str = "cta", order: str = "relaxed", *, loc=None, ip=None):
    _validate(space, order)
    clobber = ",~{memory}" if order != "relaxed" else ""
    llvm.inline_asm(
        None,
        [Int32(mbar.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip)],
        f"mbarrier.arrive.{order}.{space}.shared::{space}.b64 _, [$0];",
        f"r{clobber}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def arrive_expect_tx(
    mbar, size, space: str = "cta", order: str = "relaxed", *, loc=None, ip=None
):
    _validate(space, order)
    clobber = ",~{memory}" if order != "relaxed" else ""
    llvm.inline_asm(
        None,
        [
            Int32(mbar.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        f"mbarrier.arrive.expect_tx.{order}.{space}.shared::{space}.b64 _, [$0], $1;",
        f"r,r{clobber}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
