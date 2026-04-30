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

from vllm.logger import init_logger
from vllm.v1.fault_tolerance.registry import register_recovery_plan
from vllm.v1.fault_tolerance.types import (
    EngineLocalRecoveryPlan,
    FaultToleranceResult,
    KvAction,
)

if TYPE_CHECKING:
    from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)


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
        try:
            results = executor.collective_rpc(
                "ft_resume_after_retry", kwargs=params or {}
            )
        except AttributeError:
            return FaultToleranceResult(
                success=False,
                reason=(
                    "Workers do not implement ft_resume_after_retry. "
                    "Pass --worker-cls or ensure the worker class includes "
                    "the ft_* methods."
                ),
            )
        except Exception as e:
            logger.exception("ft_resume_after_retry collective_rpc failed: %s", e)
            return FaultToleranceResult(
                success=False, reason=f"collective_rpc raised: {e}"
            )

        ok = all(_extract_ok(r) for r in (results or []))
        reasons = [
            f"worker {i}: {_extract_reason(r)}"
            for i, r in enumerate(results or [])
            if not _extract_ok(r) and _extract_reason(r) is not None
        ]
        return FaultToleranceResult(
            success=ok,
            reason="\n".join(reasons) if reasons else None,
        )


def _extract_ok(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, bool):
        return result
    return bool(getattr(result, "success", True))


def _extract_reason(result: Any) -> str | None:
    if result is None:
        return None
    return getattr(result, "reason", None)
