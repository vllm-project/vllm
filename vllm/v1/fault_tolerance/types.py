# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core types for the fault tolerance framework.

This module defines the primitives that cross hot-path code (typed signals
on the existing output object) and the extension points that orchestrators
plug into. Everything else in `vllm.v1.fault_tolerance` builds on these.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.outputs import ModelRunnerOutput


class FaultStatus(enum.IntEnum):
    """Per-engine health status published on the fault state bus.

    Integer values are the wire format and must remain stable across renames
    or insertions; new values must be APPENDED, never inserted.
    """

    HEALTHY = 0
    UNHEALTHY = 1
    PAUSED = 2
    RECOVERING = 3
    DEAD = 4


class Disposition(enum.Enum):
    """How the engine busy loop should react to a fault."""

    CONTINUE = "continue"
    PAUSE_LOOP = "pause_loop"
    PROPAGATE = "propagate"
    SCHEDULE_RETRY = "schedule_retry"


class InterruptCommand(str, enum.Enum):
    """Worker-side aux-thread interrupt commands.

    Sent by ``DefaultFaultSupervisor.send_interrupt`` over the supervisor's
    interrupt PUB; received by ``Worker._ft_interrupt_loop`` on the SUB.
    The wire payload is the enum's ``.value`` so msgpack/JSON encodings
    don't need a custom resolver. Use the enum everywhere in code.

    Inherits from ``str`` so an enum member is interchangeable with its
    string value at the wire boundary, and ``InterruptCommand(cmd_str)``
    raises ``ValueError`` for unknown commands rather than dispatching
    them to the wrong handler.
    """

    ABORT_COMMUNICATOR = "abort_communicator"


class KvAction(enum.Enum):
    """KV cache lifecycle effect of a recovery action.

    Each `BaseRecoveryPlan` declares one of these so the KV impact is part of
    the plan's contract, not a side effect.
    """

    NONE = "none"
    """KV unchanged (e.g., pause)."""

    PREEMPT_LOGICAL_FREE = "preempt_logical_free"
    """Preempt running requests; logically free blocks; rely on prefix cache
    on re-add. Used by `retry`."""

    LOSE_DEAD_ENGINE_BLOCKS = "lose_dead_engine_blocks"
    """Dead engine's KV is unrecoverable. In-flight on dead engine is aborted;
    survivors keep their KV. Used by `scale_down`."""

    ADD_NEW_ENGINE_BLOCKS = "add_new_engine_blocks"
    """New engine starts with empty KV pool; no migration. Used by `scale_up`."""


@dataclass
class FaultSignal:
    """Typed signal carried on the success path of `ModelRunnerOutput`.

    The worker writes this when execution surfaces a fault that the engine
    needs to know about (e.g., DP all-reduce timeout, mask change, paused).
    The engine reads it via the supervisor's `after_step` hook.

    This replaces the existing pattern of raising `EngineLoopPausedError` and
    pattern-matching exception text in the executor — typed field on success
    path instead of string matching on error path.
    """

    kind: str
    """Discriminator. Conventional values: 'paused', 'dp_allreduce_failed',
    'ep_mask_changed'. Plans may add their own."""

    detail: Any = None
    """Optional payload. Kept untyped so callers can attach context without
    forcing a schema change."""


@dataclass
class FaultInfo:
    """Status update published on `FaultStateBus`.

    Wire-compatible with the `engine_status_dict` payload established by the
    fault-reporting work (see vllm-project/vllm#34833): each engine reports
    its current `FaultStatus` plus optional discriminator and detail.
    """

    engine_index: int
    status: FaultStatus
    kind: str | None = None
    detail: Any = None

    @classmethod
    def from_signal(cls, engine_index: int, signal: FaultSignal) -> FaultInfo:
        # Default mapping; supervisors can override based on signal.kind.
        status = (
            FaultStatus.PAUSED if signal.kind == "paused" else FaultStatus.UNHEALTHY
        )
        return cls(
            engine_index=engine_index,
            status=status,
            kind=signal.kind,
            detail=signal.detail,
        )

    @classmethod
    def from_exception(cls, engine_index: int, exc: BaseException) -> FaultInfo:
        return cls(
            engine_index=engine_index,
            status=FaultStatus.UNHEALTHY,
            kind=type(exc).__name__,
            detail=str(exc),
        )


@dataclass
class FaultToleranceRequest:
    """Request body for `/fault_tolerance/apply`."""

    instruction: str
    params: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None


@dataclass
class FaultToleranceResult:
    """Result returned from a `BaseRecoveryPlan.execute` call."""

    success: bool
    reason: str | None = None
    request_id: str | None = None


@runtime_checkable
class FaultToleranceHooks(Protocol):
    """Contract that engine hot-path code sees.

    Default implementation (`NoOpFaultHooks`) is a no-op singleton installed
    when fault tolerance is disabled, so the hook calls in `run_busy_loop`
    pay only the cost of empty method dispatch when FT is off.
    """

    def before_step(self, engine: Any) -> None: ...

    def after_step(self, engine: Any, output: ModelRunnerOutput | None) -> None: ...

    def on_step_error(self, engine: Any, exc: BaseException) -> Disposition: ...


class NoOpFaultHooks:
    """No-op hooks installed when fault tolerance is disabled."""

    def before_step(self, engine: Any) -> None:
        return

    def after_step(self, engine: Any, output: ModelRunnerOutput | None) -> None:
        return

    def on_step_error(self, engine: Any, exc: BaseException) -> Disposition:
        return Disposition.PROPAGATE


# Module-level singleton, used by `get_fault_hooks` when FT is disabled.
GLOBAL_NOOP_HOOKS = NoOpFaultHooks()


class BaseRecoveryPlan(ABC):
    """Strategy for one fault tolerance instruction.

    Subclasses declare their `instruction` name (registered via
    `@register_recovery_plan`), their `kv_action` (the KV-cache lifecycle
    effect, see `KvAction`), and their `execute` method.

    Two scopes: engine-local plans take `(executor, params)`; cluster plans
    take `(coord, params)` where `coord` is a `ClusterCoordinator`. Subclasses
    pick which by inheriting from `EngineLocalRecoveryPlan` or
    `ClusterRecoveryPlan` (defined alongside the registry).
    """

    instruction: str
    kv_action: KvAction = KvAction.NONE

    def is_applicable(self, vllm_config: VllmConfig) -> bool:
        """Backend / config preconditions. Default: always applicable.

        Plans with backend constraints (e.g., `ScaleDownRecoveryPlan` requires
        Ray + EP fault tolerance + TP=1) override this so the registry can
        refuse to run them on incompatible configs.
        """
        return True

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> FaultToleranceResult:
        """Run the plan. Subclasses should narrow the signature."""


class EngineLocalRecoveryPlan(BaseRecoveryPlan):
    """Engine-local plan: takes the engine's executor for `collective_rpc`."""

    @abstractmethod
    def execute(  # type: ignore[override]
        self, executor: Executor, params: dict[str, Any]
    ) -> FaultToleranceResult: ...


class ClusterRecoveryPlan(BaseRecoveryPlan):
    """Cluster-scope plan: takes a `ClusterCoordinator` (defined in cluster.py)
    for cross-engine RPC. Used by `scale_down` (PR #38862) and future
    `scale_up`.
    """

    @abstractmethod
    def execute(  # type: ignore[override]
        self, coord: Any, params: dict[str, Any]
    ) -> FaultToleranceResult: ...
