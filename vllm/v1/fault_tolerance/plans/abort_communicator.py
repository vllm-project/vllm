# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AbortCommunicatorPlan — interrupt a worker stuck in NCCL.

Used when ``executor.collective_rpc`` cannot reach the worker because its
main thread is blocked in NCCL. The supervisor's interrupt PUB socket
(see ``DefaultFaultSupervisor._interrupt_pub``) is the only way to reach
the worker's auxiliary thread, which calls ``ncclCommAbort`` from outside
the blocked main thread.

After the abort, the worker's main thread errors out of NCCL and becomes
responsive again — at which point a subsequent ``pause`` or ``retry``
instruction will reach it via ``collective_rpc`` normally.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.v1.fault_tolerance.registry import register_recovery_plan
from vllm.v1.fault_tolerance.types import (
    BaseRecoveryPlan,
    FaultToleranceResult,
    KvAction,
)

logger = init_logger(__name__)


@register_recovery_plan("abort_communicator")
class AbortCommunicatorPlan(BaseRecoveryPlan):
    """Fire a one-way interrupt to a worker's aux thread.

    Unlike most plans, this one calls into the supervisor (rather than the
    executor) because it uses the supervisor's interrupt PUB socket, not
    the executor's RPC channel. The plan's ``execute`` therefore takes a
    supervisor instance.

    KV cache: ``KvAction.NONE`` — the abort just unblocks the worker so
    subsequent plans can run; no KV state is touched.
    """

    instruction = "abort_communicator"
    kv_action = KvAction.NONE

    def execute(  # type: ignore[override]
        self, supervisor: Any, params: dict[str, Any]
    ) -> FaultToleranceResult:
        target = (params or {}).get("worker_index", "all")
        send_interrupt = getattr(supervisor, "send_interrupt", None)
        if send_interrupt is None:
            return FaultToleranceResult(
                success=False,
                reason=(
                    "Supervisor does not expose send_interrupt; cannot "
                    "deliver abort_communicator. Use the default supervisor "
                    "or implement send_interrupt in your custom supervisor."
                ),
            )
        try:
            send_interrupt(target, "abort_communicator")
        except Exception as e:
            logger.exception("send_interrupt failed: %s", e)
            return FaultToleranceResult(
                success=False, reason=f"send_interrupt raised: {e}"
            )
        # Fire-and-forget; aux thread does the abort and the caller is
        # expected to follow with `pause` or `retry` after a grace period.
        return FaultToleranceResult(success=True)
