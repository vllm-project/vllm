# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned schemas for auto-discovered BYOG catalog workloads and suites."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"

_SAFE_ID_FIRST = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
_SAFE_ID_REST = _SAFE_ID_FIRST | frozenset("._-")


def is_safe_id(value: str) -> bool:
    """Return True iff `value` matches `SAFE_ID_PATTERN` (pure Python)."""
    if not isinstance(value, str) or not (1 <= len(value) <= 128):
        return False
    if value[0] not in _SAFE_ID_FIRST:
        return False
    return all(ch in _SAFE_ID_REST for ch in value[1:])


class WorkloadKind(StrEnum):
    BENCHMARK = "benchmark"
    EVAL = "eval"


class ExecutionMode(StrEnum):
    SERVER = "server"
    DIRECT = "direct"


class LifecyclePolicy(StrEnum):
    """Declarative server-state isolation between workload points.

    Owned and enforced by the coordinator, never by individual plugins.
    Names align 1:1 with `ProtocolVariant`.
    """

    NONE = "none"
    RESET_CACHE = "reset_cache"
    RESTART_SERVER = "restart_server"


class PlatformKind(StrEnum):
    NVIDIA = "nvidia"
    ROCM = "rocm"


class EvidenceSource(StrEnum):
    """Trust label declared by the catalog, not self-chosen by plugins."""

    HARNESS = "harness"
    CANDIDATE = "candidate"


class AssetTier(StrEnum):
    SMALL_PERMISSIVE = "small_permissive"
    LARGE_PERMISSIVE = "large_permissive"
    GATED = "gated"


class PinnedAsset(BaseModel):
    """Pinned dataset or trace reference owned by the trusted catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., pattern=SAFE_ID_PATTERN)
    source: str = Field(
        ...,
        min_length=1,
        description="Upstream URI, HF repo ID, or local path.",
    )
    revision: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=SHA256_PATTERN)
    tier: AssetTier = AssetTier.SMALL_PERMISSIVE
    mirror_url: str | None = None


class PublicationPolicy(BaseModel):
    """Immutable catalog policy controlling per-field eval publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewed: bool = False
    allow_question_text: bool = False
    allow_reference_answer: bool = False
    allow_model_response: bool = False
    note: str | None = None

    def can_publish_question(self) -> bool:
        return self.reviewed and self.allow_question_text

    def can_publish_reference(self) -> bool:
        return self.reviewed and self.allow_reference_answer

    def can_publish_response(self) -> bool:
        return self.reviewed and self.allow_model_response


class QualityThreshold(BaseModel):
    """Explicit quality gate evaluated on a measurement metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(..., min_length=1)
    min_value: float | None = None
    max_value: float | None = None
    source: Literal["catalog", "spec"] = "catalog"

    @model_validator(mode="after")
    def _validate_bounds(self) -> QualityThreshold:
        if self.min_value is None and self.max_value is None:
            raise ValueError(
                "QualityThreshold must specify at least one of min_value or max_value."
            )
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value must be <= max_value.")
        return self


class WorkloadDefinition(BaseModel):
    """Schema for a single auto-discovered catalog/workloads/<name>.yaml file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]*$")
    kind: WorkloadKind
    description: str = ""
    execution_mode: ExecutionMode
    lifecycle: LifecyclePolicy = LifecyclePolicy.NONE
    allow_lifecycle_fallback: bool = False
    evidence_source: EvidenceSource = EvidenceSource.HARNESS
    platforms: list[PlatformKind] = Field(default_factory=lambda: [PlatformKind.NVIDIA])
    entrypoint: str = Field(
        ...,
        description="Import path in 'module.path:ClassName' format.",
        pattern=r"^[a-zA-Z0-9_.]+:[a-zA-Z0-9_]+$",
    )
    protocol_version: str = Field(default="1.0", min_length=1)
    assets: dict[str, PinnedAsset] = Field(default_factory=dict)
    publication_policy: PublicationPolicy = Field(default_factory=PublicationPolicy)
    defaults: dict[str, Any] = Field(default_factory=dict)
    allowed_params: list[str] | None = None
    quality_thresholds: list[QualityThreshold] = Field(default_factory=list)
    required_for_quality: bool = False

    def valid_param_keys(self) -> frozenset[str]:
        keys = set(self.defaults.keys())
        if self.allowed_params is not None:
            keys.update(self.allowed_params)
        return frozenset(keys)

    @model_validator(mode="after")
    def _validate_mode_kind_and_lifecycle(self) -> WorkloadDefinition:
        if (
            self.execution_mode == ExecutionMode.DIRECT
            and self.lifecycle != LifecyclePolicy.NONE
        ):
            raise ValueError(
                "Direct workloads must use lifecycle='none' (no server is managed)."
            )
        if (
            self.kind == WorkloadKind.EVAL
            and self.execution_mode == ExecutionMode.DIRECT
        ):
            raise ValueError(
                "Eval workloads must use execution_mode='server' in Phase 0."
            )
        if not self.platforms:
            raise ValueError("Workload must declare at least one supported platform.")
        if any(t.source != "catalog" for t in self.quality_thresholds):
            raise ValueError(
                "Catalog WorkloadDefinition quality_thresholds must have "
                "source='catalog'."
            )
        if self.required_for_quality and not self.quality_thresholds:
            raise ValueError(
                "required_for_quality=True requires at least one catalog "
                "quality_threshold."
            )
        return self


class SuiteDefinition(BaseModel):
    """Named collection of benchmarks and/or evals (smoke, standard, extended)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    benchmarks: list[dict[str, Any]] = Field(default_factory=list)
    evals: list[dict[str, Any]] = Field(default_factory=list)
