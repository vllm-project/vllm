# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

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


@cute.jit
def wait_parity_acquire_cluster(mbar, phase: Int32) -> None:
    """``cute.arch.mbarrier_wait`` with acquire semantics at cluster scope.

    Use it when the barrier is completed by a peer CTA's ``st.async`` or
    ``cp.async.bulk`` complete-tx, a release at cluster scope: the default
    ``.acquire.cta`` wait does not synchronize with other CTAs' threads, so the
    peer-written DSMEM could be read before the peer's bytes are visible.
    ``cute.arch.mbarrier_wait`` has no scope argument, so this spins on the
    Primitives ``mbarrier_try_wait_parity`` with ``MBarrierScope.CLUSTER`` and
    ``MemOrder.RELAXED`` (``mbarrier.try_wait.parity.relaxed.cluster``) and then
    acquires with ``fence_sync_restrict(MemOrder.ACQUIRE)``
    (``fence.acquire.sync_restrict::shared::cluster.cluster``). The acquire is
    restricted to shared memory, which is all these mailboxes need, whereas a
    ``.acquire.cluster`` try_wait also invalidates L1 after every attempt.
    """
    while not prims.mbarrier_try_wait_parity(
        mbar,
        phase,
        scope=prims.MBarrierScope.CLUSTER,
        order=prims.MemOrder.RELAXED,
    ):
        pass
    prims.fence_sync_restrict(prims.MemOrder.ACQUIRE)
