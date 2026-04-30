# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PauseRecoveryPlan — stop healthy workers cleanly via collective_rpc."""

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
        try:
            results = executor.collective_rpc("ft_pause", kwargs=params or {})
        except AttributeError:
            return FaultToleranceResult(
                success=False,
                reason=(
                    "Workers do not implement ft_pause. Pass --worker-cls or "
                    "ensure the worker class includes the ft_* methods."
                ),
            )
        except Exception as e:
            logger.exception("ft_pause collective_rpc failed: %s", e)
            return FaultToleranceResult(
                success=False, reason=f"collective_rpc raised: {e}"
            )

        # results is a list of FaultToleranceResult-shaped objects (one per
        # worker). Aggregate.
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
