# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry for fault tolerance supervisors and recovery plans.

Mirrors the shape of `vllm.model_executor.model_loader.register_model_loader`
so orchestrators (Dynamo, k8s controllers, custom platforms) can plug their
own supervisor in via a decorator without forking vLLM.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from vllm.v1.fault_tolerance.types import (
    GLOBAL_NOOP_HOOKS,
    BaseRecoveryPlan,
    FaultToleranceHooks,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_S = TypeVar("_S", bound=type)
_P = TypeVar("_P", bound=type[BaseRecoveryPlan])


# Registries are module-globals: small, write-only at import time, read-only
# at runtime. Mirrors how `register_model_loader` works for GMS et al.
_SUPERVISORS: dict[str, type[FaultToleranceHooks]] = {}
_PLANS: dict[str, BaseRecoveryPlan] = {}


def register_fault_supervisor(name: str) -> Callable[[_S], _S]:
    """Register a fault-tolerance supervisor implementation.

    Usage::

        @register_fault_supervisor("dynamo")
        class DynamoFaultSupervisor(FaultToleranceHooks): ...

    User selects with ``--fault-supervisor dynamo`` (mirrors GMS's
    ``--load-format gms`` pattern).
    """

    def deco(cls: _S) -> _S:
        if name in _SUPERVISORS:
            raise ValueError(
                f"Fault supervisor '{name}' is already registered "
                f"by {_SUPERVISORS[name].__module__}.{_SUPERVISORS[name].__name__}"
            )
        _SUPERVISORS[name] = cls  # type: ignore[assignment]
        return cls

    return deco


def register_recovery_plan(instruction: str) -> Callable[[_P], _P]:
    """Register a recovery plan for a `/fault_tolerance/apply` instruction.

    Usage::

        @register_recovery_plan("pause")
        class PauseRecoveryPlan(EngineLocalRecoveryPlan): ...
    """

    def deco(cls: _P) -> _P:
        if instruction in _PLANS:
            raise ValueError(
                f"Recovery plan '{instruction}' is already registered "
                f"by {type(_PLANS[instruction]).__name__}"
            )
        # Plans are stateless; instantiate once at registration time.
        _PLANS[instruction] = cls()  # type: ignore[call-arg]
        return cls

    return deco


def get_supervisor_class(name: str) -> type[FaultToleranceHooks]:
    """Look up a registered supervisor class by name."""
    if name not in _SUPERVISORS:
        raise KeyError(
            f"Unknown fault supervisor '{name}'. Registered: {sorted(_SUPERVISORS)}"
        )
    return _SUPERVISORS[name]


def get_plan(instruction: str) -> BaseRecoveryPlan:
    """Look up a registered recovery plan by instruction name."""
    if instruction not in _PLANS:
        raise KeyError(
            f"Unknown recovery instruction '{instruction}'. "
            f"Registered: {sorted(_PLANS)}"
        )
    return _PLANS[instruction]


def list_plans() -> list[str]:
    """List currently registered recovery instructions."""
    return sorted(_PLANS)


def get_fault_hooks(
    vllm_config: VllmConfig | None,
    engine: Any | None = None,
) -> FaultToleranceHooks:
    """Get the active hooks implementation.

    Returns the no-op singleton when FT is disabled or when no engine instance
    is available yet (e.g., during early init). Otherwise returns the engine's
    supervisor instance.
    """
    if vllm_config is None or not getattr(
        vllm_config.parallel_config, "enable_fault_tolerance", False
    ):
        return GLOBAL_NOOP_HOOKS
    if engine is None:
        return GLOBAL_NOOP_HOOKS
    supervisor = getattr(engine, "fault_supervisor", None)
    if supervisor is None:
        return GLOBAL_NOOP_HOOKS
    return supervisor


def reset_for_testing() -> None:
    """Clear all registries. Tests only."""
    _SUPERVISORS.clear()
    _PLANS.clear()
