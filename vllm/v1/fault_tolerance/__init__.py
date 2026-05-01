# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM fault tolerance framework.

The public surface intentionally mirrors `register_model_loader` so
orchestrators (Dynamo, k8s controllers, custom platforms) can extend FT
without forking vLLM. Three things are exposed:

1. `FaultToleranceHooks` (Protocol) — what `EngineCoreProc.run_busy_loop`
   calls on every iteration. Default is `NoOpFaultHooks` when FT is off.
2. `register_fault_supervisor(name)` — register a hooks implementation.
3. `register_recovery_plan(instruction)` — register a plan for one
   `/fault_tolerance/apply` instruction.

See `vllm/v1/fault_tolerance/types.py` for the full type surface.
"""

from vllm.v1.fault_tolerance.registry import (
    get_fault_hooks,
    get_plan,
    get_supervisor_class,
    list_plans,
    register_fault_supervisor,
    register_recovery_plan,
)
from vllm.v1.fault_tolerance.types import (
    GLOBAL_NOOP_HOOKS,
    BaseRecoveryPlan,
    ClusterRecoveryPlan,
    Disposition,
    EngineLocalRecoveryPlan,
    FaultInfo,
    FaultSignal,
    FaultStatus,
    FaultToleranceHooks,
    FaultToleranceRequest,
    FaultToleranceResult,
    KvAction,
    NoOpFaultHooks,
)

__all__ = [
    # Types
    "BaseRecoveryPlan",
    "ClusterRecoveryPlan",
    "Disposition",
    "EngineLocalRecoveryPlan",
    "FaultInfo",
    "FaultSignal",
    "FaultStatus",
    "FaultToleranceHooks",
    "FaultToleranceRequest",
    "FaultToleranceResult",
    "GLOBAL_NOOP_HOOKS",
    "KvAction",
    "NoOpFaultHooks",
    # Registry
    "get_fault_hooks",
    "get_plan",
    "get_supervisor_class",
    "list_plans",
    "register_fault_supervisor",
    "register_recovery_plan",
]
