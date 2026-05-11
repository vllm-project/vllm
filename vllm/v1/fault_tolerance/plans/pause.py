# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PauseRecoveryPlan — stop healthy workers cleanly via collective_rpc."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.v1.fault_tolerance.plans._common import run_collective_plan
from vllm.v1.fault_tolerance.registry import register_recovery_plan
from vllm.v1.fault_tolerance.types import (
    EngineLocalRecoveryPlan,
    FaultToleranceResult,
    KvAction,
)

if TYPE_CHECKING:
    from vllm.v1.executor.abstract import Executor


@register_recovery_plan("pause")
class PauseRecoveryPlan(EngineLocalRecoveryPlan):
    """Pause an engine's workers via ``executor.collective_rpc("ft_pause")``.

    Used when the worker main threads are responsive (either between steps
    on a healthy engine, or after an FT-capable kernel has timed out and
    returned control to the main thread). For the case where the worker is
    blocked in NCCL and unresponsive, see ``AbortCommunicatorPlan``.

    KV cache: ``KvAction.NONE`` — running requests' allocated blocks remain;
    on resume work continues.
    """

    instruction = "pause"
    kv_action = KvAction.NONE

    def execute(  # type: ignore[override]
        self, executor: Executor, params: dict[str, Any]
    ) -> FaultToleranceResult:
        return run_collective_plan(executor, "ft_pause", params)
