# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from cutlass import cute
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op

_SCOPE_MAP = {
    "cta": nvvm.MemScopeKind.CTA,
    "cluster": nvvm.MemScopeKind.CLUSTER,
}

_RELAXED_MAP = {
    "relaxed": True,
    "release": False,
}


def _resolve_scope(space: str):
    if space not in _SCOPE_MAP:
        raise ValueError(f"Unsupported mbarrier space: {space!r}")
    return _SCOPE_MAP[space]


def _resolve_relaxed(order: str) -> bool:
    if order not in _RELAXED_MAP:
        raise ValueError(f"Unsupported mbarrier memory order: {order!r}")
    return _RELAXED_MAP[order]


def _to_mbar_ptr(mbar, space: str, *, loc=None, ip=None):
    # Upstream NVVM mbarrier ops select .shared::cta vs .shared::cluster from
    # the pointer address space (smem vs dsmem). Preserve cluster addressing
    # when callers pass remote CTA pointers constructed in AddressSpace.smem.
    if space == "cluster" and mbar.memspace != cute.AddressSpace.dsmem:
        return cute.make_ptr(
            mbar.dtype,
            mbar.toint(loc=loc, ip=ip),
            cute.AddressSpace.dsmem,
            assumed_align=8,
            loc=loc,
            ip=ip,
        )
    return mbar


@dsl_user_op
def arrive(mbar, space: str = "cta", order: str = "relaxed", *, loc=None, ip=None):
    cute.arch.mbarrier_arrive(
        _to_mbar_ptr(mbar, space, loc=loc, ip=ip),
        relaxed=_resolve_relaxed(order),
        scope=_resolve_scope(space),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def arrive_expect_tx(
    mbar, size, space: str = "cta", order: str = "relaxed", *, loc=None, ip=None
):
    cute.arch.mbarrier_arrive_and_expect_tx(
        _to_mbar_ptr(mbar, space, loc=loc, ip=ip),
        size,
        relaxed=_resolve_relaxed(order),
        scope=_resolve_scope(space),
        loc=loc,
        ip=ip,
    )
