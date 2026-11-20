# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned schemas for BYOG user run specs and resolved executable jobs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tools.byog.schemas.catalog import (
    SAFE_ID_PATTERN,
    EvidenceSource,
    PinnedAsset,
    PlatformKind,
    PublicationPolicy,
    QualityThreshold,
)


class RecipeSelector(BaseModel):
    """Selector against vllm/recipes/models/<org>/<repo>.yaml (resolved in P1.3)."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1)
    revision: str | None = None
    variant: str = "default"
    strategy: str = "single_node_tp"
    preset: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)


class ServeSpec(BaseModel):
    """Native vLLM serving configuration and process environment variables."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


class HardwareSpec(BaseModel):
    """Declared hardware target using canonical taxonomy.yaml GPU IDs."""

    model_config = ConfigDict(extra="forbid")

    gpu: list[str] = Field(default_factory=list)
    count: int = Field(default=1, ge=1)
    nodes: int = Field(default=1, ge=1)


class BenchmarkSelection(BaseModel):
    """Single benchmark entry inside spec.yaml.

    Workload-specific keys are validated strictly against the catalog entry's
    `valid_param_keys()` during spec resolution (C3).
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(..., min_length=1)


class EvalSelection(BaseModel):
    """Single eval entry inside spec.yaml.

    Workload-specific keys are validated strictly against the catalog entry's
    `valid_param_keys()` during spec resolution (C3).
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., min_length=1)
    repeats: int | None = Field(default=None, ge=1)
    concurrency: list[int] | None = None
    min_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_concurrency(self) -> EvalSelection:
        if self.concurrency is not None and (
            not self.concurrency or any(c < 1 for c in self.concurrency)
        ):
            raise ValueError("evals[].concurrency must contain positive integers")
        return self


class RunSpec(BaseModel):
    """Top-level user input spec (schema_version: 1)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    name: str | None = None
    recipe: RecipeSelector | None = None
    serve: ServeSpec | None = None
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    matrix: dict[str, list[Any]] | None = None
    matrix_include: list[dict[str, Any]] | None = None
    matrix_exclude: list[dict[str, Any]] | None = None
    benchmark_suite: str | None = None
    benchmarks: list[BenchmarkSelection] | None = None
    eval_suite: str | None = None
    evals: list[EvalSelection] | None = None
    compare_baseline: bool = False
    timeout_s: int = Field(default=3600, ge=1)

    @model_validator(mode="after")
    def _validate_serving_source(self) -> RunSpec:
        if self.recipe is None and self.serve is None:
            raise ValueError("Spec must specify at least one of 'recipe' or 'serve'.")
        return self


class ResolvedWorkload(BaseModel):
    """Self-contained workload snapshot bound into a ResolvedJob.

    Captures catalog provenance, assets, policy, and quality thresholds so the
    executor and bundler never depend on mutable live catalog state at run time.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["benchmark", "eval"]
    execution_mode: Literal["server", "direct"]
    lifecycle: Literal["none", "reset_cache", "restart_server"]
    allow_lifecycle_fallback: bool = False
    evidence_source: EvidenceSource = EvidenceSource.HARNESS
    platforms: list[PlatformKind] = Field(default_factory=lambda: [PlatformKind.NVIDIA])
    entrypoint: str
    protocol_version: str
    assets: dict[str, PinnedAsset] = Field(default_factory=dict)
    publication_policy: PublicationPolicy = Field(default_factory=PublicationPolicy)
    quality_thresholds: list[QualityThreshold] = Field(default_factory=list)
    required_for_quality: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class ResolvedJob(BaseModel):
    """Concrete single-job specification consumed by C4 Executor and C5 Bundler."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(
        default="default",
        pattern=SAFE_ID_PATTERN,
        description="'default' for single jobs; '<slug>-<hash>' for matrix jobs.",
    )
    attempt_id: str = Field(
        ...,
        pattern=SAFE_ID_PATTERN,
        description="Unique execution attempt ID generated by the harness.",
    )
    spec_name: str | None = None
    matrix_coords: dict[str, Any] = Field(default_factory=dict)
    serve_config: dict[str, Any] = Field(default_factory=dict)
    serve_env: dict[str, str] = Field(default_factory=dict)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    workloads: list[ResolvedWorkload] = Field(default_factory=list)
    timeout_s: int = Field(default=3600, ge=1)
    source_tier: Literal["ci_nightly", "ci_pr", "community"] = "community"
    resolved_config_hash: str = ""
