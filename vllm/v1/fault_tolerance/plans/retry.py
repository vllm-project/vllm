# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RetryRecoveryPlan — clean up worker state and resume after a transient
fault.

Worker-side cleanup (clear input batch, clean FT mask, rebuild DP gloo
group) lives in ``Worker.ft_resume_after_retry`` and is invoked here via
``executor.collective_rpc``. Result aggregation is automatic — no separate
dispatcher to maintain.
"""

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


@register_recovery_plan("retry")
class RetryRecoveryPlan(EngineLocalRecoveryPlan):
    """Cleanup-and-retry for transient errors.

    KV cache: ``KvAction.PREEMPT_LOGICAL_FREE`` — the scheduler preempts
    running requests back to the waiting queue and logically frees their KV
    blocks. The blocks themselves are not zeroed, so prefix-cache hits on
    re-add cover most already-computed prefixes; only the tail tokens
    recompute (similar to a chunked prefill).
    """

    instruction = "retry"
    kv_action = KvAction.PREEMPT_LOGICAL_FREE

    def execute(  # type: ignore[override]
        self, executor: Executor, params: dict[str, Any]
    ) -> FaultToleranceResult:
        return run_collective_plan(executor, "ft_resume_after_retry", params)
