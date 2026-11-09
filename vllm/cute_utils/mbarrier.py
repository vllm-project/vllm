# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

_SPACE_MAP = {
    "cta": nvvm.MBarrierSpaceKind.CTA,
    "cluster": nvvm.MBarrierSpaceKind.CLUSTER,
}

_MEMORY_ORDER_MAP = {
    "weak": nvvm.MemOrderKind.WEAK,
    "relaxed": nvvm.MemOrderKind.RELAXED,
    "acquire": nvvm.MemOrderKind.ACQUIRE,
    "release": nvvm.MemOrderKind.RELEASE,
    "acq_rel": nvvm.MemOrderKind.ACQ_REL,
}


@dsl_user_op
def arrive(mbar, space: str = "cta", order: str = "relaxed", *, loc=None, ip=None):
    nvvm.mbarrier_txn(
        mbar.to_llvm_ptr(loc=loc, ip=ip),
        Int32(1).ir_value(loc=loc, ip=ip),
        kind=nvvm.MBarrierTxnKind.ARRIVE,
        space=_SPACE_MAP[space],
        order=_MEMORY_ORDER_MAP[order],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def arrive_expect_tx(
    mbar, size, space: str = "cta", order: str = "relaxed", *, loc=None, ip=None
):
    nvvm.mbarrier_txn(
        mbar.to_llvm_ptr(loc=loc, ip=ip),
        Int32(size).ir_value(loc=loc, ip=ip),
        kind=nvvm.MBarrierTxnKind.ARRIVE_EXPECT_TX,
        space=_SPACE_MAP[space],
        order=_MEMORY_ORDER_MAP[order],
        loc=loc,
        ip=ip,
    )


@cute.jit
def wait_parity_acquire_cluster(mbar, phase: Int32) -> None:
    """``cute.arch.mbarrier_wait`` with ``.acquire.cluster`` semantics.

    Use it when the barrier is completed by a peer CTA's ``st.async`` or
    ``cp.async.bulk`` complete-tx, a release at cluster scope: the default
    ``.acquire.cta`` wait does not synchronize with other CTAs' threads, so the
    peer-written DSMEM could be read before the peer's bytes are visible.
    ``cute.arch.mbarrier_wait`` has no scope argument, so this uses the
    Primitives ``mbarrier_try_wait_parity`` with ``MBarrierScope.CLUSTER``, which
    lowers to ``mbarrier.try_wait.parity.acquire.cluster``.
    """
    while not prims.mbarrier_try_wait_parity(
        mbar, phase, scope=prims.MBarrierScope.CLUSTER
    ):
        pass
