# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AbortRequestsPlan — fail in-flight requests on a target engine.

Cluster-scope. Different from :class:`AbortCommunicatorPlan` (low-level
``ncclCommAbort`` interrupt). This plan walks the client's in-flight
request map for the target engine and routes each one through
``OutputProcessor.abort_requests`` (or equivalent) so clients receive
``FinishReason.ABORT`` instead of hanging.

Typical orchestrator flow:

1. Engine death detected → client publishes ``FaultInfo(DEAD)`` on bus.
2. Orchestrator decides "fail any in-flight on the dead rank, but don't
   yet scale down" (e.g. while diagnosing whether the death is permanent).
3. ``POST /fault_tolerance/apply {"instruction": "abort", "params":
   {"engine_index": N}}`` runs this plan.
4. Later, orchestrator may follow with ``scale_down`` if recovery is
   warranted, or ``resume`` if the engine recovers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.fault_tolerance.registry import register_recovery_plan
from vllm.v1.fault_tolerance.types import (
    ClusterRecoveryPlan,
    FaultToleranceResult,
    KvAction,
)

if TYPE_CHECKING:
    from vllm.v1.fault_tolerance.cluster import ClusterCoordinator

logger = init_logger(__name__)


@register_recovery_plan("abort")
class AbortRequestsPlan(ClusterRecoveryPlan):
    """Abort all in-flight requests on a target engine.

    KV cache: ``KvAction.NONE`` — aborted requests' KV is freed by the
    scheduler when the abort lands on the engine, but no surviving engine's
    KV is touched.

    Required params:
        ``engine_index`` (int): The DP rank to abort.
    """

    instruction = "abort"
    kv_action = KvAction.NONE

    def execute(  # type: ignore[override]
        self, coord: ClusterCoordinator, params: dict[str, Any]
    ) -> FaultToleranceResult:
        engine_index = (params or {}).get("engine_index")
        if not isinstance(engine_index, int):
            return FaultToleranceResult(
                success=False,
                reason="abort requires params.engine_index (int)",
            )

        try:
            num_aborted = coord.abort_inflight_on(engine_index)
        except Exception as e:
            logger.exception("abort_inflight_on(%d) failed: %s", engine_index, e)
            return FaultToleranceResult(
                success=False, reason=f"abort_inflight_on failed: {e}"
            )

        return FaultToleranceResult(
            success=True,
            reason=(
                f"Aborted {num_aborted} in-flight requests on engine {engine_index}"
                if isinstance(num_aborted, int)
                else None
            ),
        )
