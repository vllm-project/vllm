# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BYOG execution contracts (design doc C0): `spec.yaml`, catalog, and `ResolvedJob`.

Reading map (follows design doc pipeline order):
  1. Shared enums & helpers  -- vocabulary reused across execution & result schemas.
  2. User input (`spec.yaml`) -- `RunSpec`, `ServeSpec`, `HardwareSpec`, selections.
  3. Catalog (C2)            -- `WorkloadDefinition`, `SuiteDefinition`.
  4. Resolved job (C3 -> C4) -- `ResolvedJob` (`spec.resolved.yaml`),
                                `ResolvedWorkload`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# 1. Shared enums & ID validation
# ---------------------------------------------------------------------------

SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
_ID_FIRST = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_ID_REST = _ID_FIRST | frozenset("._-")


def is_safe_id(value: str) -> bool:
    """Return True iff `value` is a safe path/slug identifier (`SAFE_ID_PATTERN`)."""
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and value[0] in _ID_FIRST
        and all(ch in _ID_REST for ch in value[1:])
    )


class WorkloadKind(StrEnum):
    """Whether a catalog workload is a performance benchmark or accuracy eval."""

    BENCHMARK = "benchmark"
    EVAL = "eval"


class ExecutionMode(StrEnum):
    """Whether a workload talks to `vllm serve` (server) or runs directly."""

    SERVER = "server"
    DIRECT = "direct"


class LifecyclePolicy(StrEnum):
    """Server-state isolation enforced by the coordinator (C4) between points."""

    NONE = "none"
    RESET_CACHE = "reset_cache"
    RESTART_SERVER = "restart_server"


class ExecutionStatus(StrEnum):
    """Process-level status of a job or measurement point."""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class ErrorKind(StrEnum):
    """Per-request failure classification (`model` error vs `infra` failure)."""

    NONE = "none"
    MODEL = "model"
    INFRA = "infra"


# ---------------------------------------------------------------------------
# 2. User input: `spec.yaml` (design doc section 1.2)
# ---------------------------------------------------------------------------


class BenchmarkSelection(BaseModel):
    """One benchmark entry in `spec.yaml` or a suite (extra params checked in C3)."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(..., min_length=1)


class EvalSelection(BaseModel):
    """One eval entry in `spec.yaml` or a suite YAML (extra params validated in C3).

    `repeats` and `concurrency` are execution controls every eval accepts, so they
    are typed here rather than listed in each workload's `allowed_params`. Any
    other key is a workload param and must appear in `allowed_params`.
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., min_length=1)
    repeats: int | None = Field(default=None, ge=1)
    concurrency: list[int] | None = None

    @model_validator(mode="after")
    def _validate(self) -> EvalSelection:
        if self.concurrency is not None and (
            not self.concurrency or any(c < 1 for c in self.concurrency)
        ):
            raise ValueError("concurrency must be a non-empty list of positive ints.")
        return self


class ServeSpec(BaseModel):
    """vLLM serving flags (`config`) and process environment (`env`)."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


class HardwareSpec(BaseModel):
    """Requested GPU families and count in `spec.yaml` (single-node in Phase 0)."""

    model_config = ConfigDict(extra="forbid")

    gpu: list[str] = Field(default_factory=list)
    count: int = Field(default=1, ge=1)
    nodes: Literal[1] = 1


class RunSpec(BaseModel):
    """Top-level user input `spec.yaml` (`schema_version: 1`)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    name: str | None = None
    serve: ServeSpec
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    benchmark_suite: str | None = None
    benchmarks: list[BenchmarkSelection] | None = None
    eval_suite: str | None = None
    evals: list[EvalSelection] | None = None
    timeout_s: int = Field(default=3600, ge=1)


# ---------------------------------------------------------------------------
# 3. Catalog definitions: workloads and suites (C2)
# ---------------------------------------------------------------------------


class WorkloadDefinition(BaseModel):
    """Schema for one auto-discovered `catalog/workloads/<name>.yaml` file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]*$")
    kind: WorkloadKind
    description: str = ""
    execution_mode: ExecutionMode
    lifecycle: LifecyclePolicy = LifecyclePolicy.NONE
    platforms: list[Literal["nvidia", "rocm"]] = Field(
        default_factory=lambda: ["nvidia"]
    )
    entrypoint: str = Field(..., pattern=r"^[a-zA-Z0-9_.]+:[a-zA-Z0-9_]+$")
    protocol_version: str = Field(default="1.0", min_length=1)
    # Catalog-pinned defaults; specs may only override keys listed in `allowed_params`.
    defaults: dict[str, Any] = Field(default_factory=dict)
    allowed_params: list[str] = Field(default_factory=list)

    def valid_param_keys(self) -> frozenset[str]:
        """Return parameter keys that `spec.yaml` or suites are allowed to set."""
        return frozenset(self.allowed_params)

    @model_validator(mode="after")
    def _validate(self) -> WorkloadDefinition:
        if (
            self.execution_mode == ExecutionMode.DIRECT
            and self.lifecycle != LifecyclePolicy.NONE
        ):
            raise ValueError("Direct workloads must use lifecycle='none'.")
        if (
            self.kind == WorkloadKind.EVAL
            and self.execution_mode == ExecutionMode.DIRECT
        ):
            raise ValueError("Eval workloads must use execution_mode='server'.")
        if not self.platforms:
            raise ValueError("Workload must declare at least one platform.")
        return self


class SuiteDefinition(BaseModel):
    """Named group of benchmarks and evals (`catalog/suites/<name>.yaml`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    benchmarks: list[BenchmarkSelection] = Field(default_factory=list)
    evals: list[EvalSelection] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4. Resolved job: `spec.resolved.yaml` (C3 Spec resolver -> C4 Executor)
# ---------------------------------------------------------------------------


class ResolvedWorkload(BaseModel):
    """Immutable catalog workload paired with merged `spec.yaml` parameters."""

    model_config = ConfigDict(extra="forbid")

    definition: WorkloadDefinition
    params: dict[str, Any] = Field(default_factory=dict)


class ResolvedJob(BaseModel):
    """Concrete, self-contained job handed from Spec resolver (C3) to Executor (C4)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(default="default", pattern=SAFE_ID_PATTERN)
    spec_name: str | None = None
    serve_config: dict[str, Any] = Field(default_factory=dict)
    serve_env: dict[str, str] = Field(default_factory=dict)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    workloads: list[ResolvedWorkload] = Field(default_factory=list)
    timeout_s: int = Field(default=3600, ge=1)
