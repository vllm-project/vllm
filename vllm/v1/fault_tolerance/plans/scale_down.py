# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ScaleDownRecoveryPlan — cluster-scope recovery for a dead DP engine.

Delegates to the existing elastic EP machinery in
``vllm/distributed/elastic_ep`` (which is upstream from PR #38862). The
plan itself is a thin façade so ``/fault_tolerance/apply`` becomes the
single instruction surface for ``pause``, ``retry``, ``scale_down``, etc.

Nothing in ``vllm/distributed/elastic_ep`` is rewritten — only the trigger
path changes from "detector calls scale-down directly" to
"orchestrator POSTs scale_down to the FT API".
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
    from vllm.config import VllmConfig
    from vllm.v1.fault_tolerance.cluster import ClusterCoordinator

logger = init_logger(__name__)


@register_recovery_plan("scale_down")
class ScaleDownRecoveryPlan(ClusterRecoveryPlan):
    """Drop a dead DP engine and rebuild the cluster topology.

    Backend constraints (Ray + EP fault tolerance + TP=1, etc.) are the
    same as the existing PR #38862 stack documents. ``is_applicable`` checks
    them so the registry refuses the plan on incompatible deployments.

    KV cache: ``KvAction.LOSE_DEAD_ENGINE_BLOCKS`` — the dead engine's KV
    is unrecoverable. In-flight requests on the dead engine are aborted
    with ``FinishReason.ABORT``; surviving engines' KV is untouched.
    CUDA-graph recapture across survivors does not affect KV memory.
    """

    instruction = "scale_down"
    kv_action = KvAction.LOSE_DEAD_ENGINE_BLOCKS

    def is_applicable(self, vllm_config: VllmConfig) -> bool:
        pc = vllm_config.parallel_config
        # `enable_ep_fault_tolerance` and `data_parallel_backend` are added
        # by #38862; tolerate older configs that don't carry them.
        if not getattr(pc, "enable_ep_fault_tolerance", False):
            return False
        if pc.tensor_parallel_size != 1:
            return False
        return getattr(pc, "data_parallel_backend", "") == "ray"

    def execute(  # type: ignore[override]
        self, coord: ClusterCoordinator, params: dict[str, Any]
    ) -> FaultToleranceResult:
        dead_index = (params or {}).get("dead_engine_index")
        if not isinstance(dead_index, int):
            return FaultToleranceResult(
                success=False,
                reason="scale_down requires params.dead_engine_index (int)",
            )

        try:
            coord.scale_down(dead_index)
        except Exception as e:
            logger.exception("scale_down(%d) failed: %s", dead_index, e)
            return FaultToleranceResult(success=False, reason=f"scale_down failed: {e}")
        return FaultToleranceResult(success=True)
