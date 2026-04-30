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
        pc = getattr(vllm_config, "parallel_config", None)
        if pc is None:
            return False
        if not getattr(pc, "enable_ep_fault_tolerance", False):
            return False
        if getattr(pc, "tensor_parallel_size", 1) != 1:
            return False
        # Ray DP backend is required by the existing elastic EP machinery.
        backend = getattr(pc, "data_parallel_backend", "")
        return backend in {"ray"}

    def execute(  # type: ignore[override]
        self, coord: ClusterCoordinator, params: dict[str, Any]
    ) -> FaultToleranceResult:
        dead_index = (params or {}).get("dead_engine_index")
        if dead_index is None or not isinstance(dead_index, int):
            return FaultToleranceResult(
                success=False,
                reason="scale_down requires params.dead_engine_index (int)",
            )

        # Delegate to the existing elastic_ep implementation. The exact
        # symbol is whatever the existing fault-scale-down entry point is
        # in `vllm.distributed.elastic_ep`; resolve at call time so we
        # don't fail at import if the user is on a build that hasn't
        # landed it yet.
        try:
            from vllm.distributed.elastic_ep import elastic_execute
        except ImportError as e:
            return FaultToleranceResult(
                success=False,
                reason=(
                    "vllm.distributed.elastic_ep.elastic_execute is "
                    f"unavailable: {e}. scale_down requires the elastic EP "
                    "machinery (PR #38862 et al.)."
                ),
            )

        entry = getattr(elastic_execute, "execute_fault_scale_down", None)
        if entry is None:
            entry = getattr(elastic_execute, "fault_scale_down", None)
        if entry is None:
            return FaultToleranceResult(
                success=False,
                reason=(
                    "vllm.distributed.elastic_ep.elastic_execute does not "
                    "expose execute_fault_scale_down or fault_scale_down. "
                    "Check the elastic EP API surface for the current "
                    "entry-point name."
                ),
            )

        try:
            entry(coord, dead_index)
        except Exception as e:
            logger.exception("scale_down execute failed: %s", e)
            return FaultToleranceResult(
                success=False, reason=f"elastic_execute raised: {e}"
            )
        return FaultToleranceResult(success=True)
