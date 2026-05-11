# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for engine-local recovery plans.

Plans like ``pause`` and ``retry`` follow the same shape: invoke a worker
method via ``executor.collective_rpc``, aggregate per-worker results, and
return a single ``FaultToleranceResult`` to the supervisor. The aggregation
logic lives here so adding a new ``EngineLocalRecoveryPlan`` is one line in
``execute`` rather than a copy-paste of the boilerplate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.fault_tolerance.types import FaultToleranceResult

if TYPE_CHECKING:
    from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)


def run_collective_plan(
    executor: Executor,
    method: str,
    params: dict[str, Any] | None,
) -> FaultToleranceResult:
    """Invoke ``method`` on every worker via ``collective_rpc`` and aggregate.

    Each worker is contracted to return a ``FaultToleranceResult``; this
    helper trusts that contract and lets ``AttributeError`` propagate as a
    real bug if a worker returns the wrong shape. The supervisor's caller
    converts any uncaught exception into an HTTP-friendly result, so we do
    not need an extra defensive layer here.
    """
    try:
        results: list[FaultToleranceResult] = executor.collective_rpc(
            method, kwargs=params or {}
        )
    except Exception as e:
        logger.exception("%s collective_rpc failed: %s", method, e)
        return FaultToleranceResult(success=False, reason=f"collective_rpc raised: {e}")

    failed = [(i, r) for i, r in enumerate(results) if not r.success]
    return FaultToleranceResult(
        success=not failed,
        reason=(
            "\n".join(
                f"worker {i}: {r.reason}" for i, r in failed if r.reason is not None
            )
            or None
        ),
    )
