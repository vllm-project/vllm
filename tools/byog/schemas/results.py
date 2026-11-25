# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BYOG result contracts (design doc C0): `result.json`, eval JSONL, `manifest.json`.

Reading map:
  1. Per-question eval JSONL -- `EvalSampleRecord`, `EvalRenderedInputRecord`,
                                `EvalResponseRecord` (text fields nullable so P1.13
                                dataset withholding is non-breaking).
  2. Job summary (`result.json`) -- `ObservedHardware`, `MeasurementRecord`,
                                    `RunResult`.
  3. Bundle integrity (`manifest.json`) -- `ArtifactEntry`, `BundleManifest`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tools.byog.schemas.execution import (
    SAFE_ID_PATTERN,
    SHA256_PATTERN,
    ErrorKind,
    ExecutionMode,
    ExecutionStatus,
    HardwareSpec,
    LifecyclePolicy,
    WorkloadKind,
)


def _validate_rel_path(value: str) -> str:
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or "//" in value
        or value.endswith("/")
    ):
        raise ValueError(f"Unsafe relative artifact path: {value!r}.")
    parts = PurePosixPath(value).parts
    if not parts or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"Unsafe relative artifact path: {value!r}.")
    return value


# ---------------------------------------------------------------------------
# 1. Per-question eval JSONL records (`evals/<name>/*.jsonl`)
# ---------------------------------------------------------------------------


class EvalSampleRecord(BaseModel):
    """Deduplicated question and reference answer (`samples.jsonl`)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    dataset_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_revision: str = Field(..., min_length=1)
    sample_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    question: str | None = None
    reference_answer: str | None = None


class EvalRenderedInputRecord(BaseModel):
    """Deduplicated rendered prompt/input (`inputs.jsonl`)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    dataset_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_revision: str = Field(..., min_length=1)
    sample_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    input_hash: str = Field(..., min_length=8)
    rendered_input: Any = None


class EvalResponseRecord(BaseModel):
    """One scored response per `(sample_id, point_id, repeat)` (`responses.jsonl`)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    point_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_revision: str = Field(..., min_length=1)
    sample_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    input_hash: str = Field(..., min_length=8)
    repeat: int = Field(ge=0)
    concurrency: int = Field(ge=1)
    model_response: str | None = None
    extracted_answer: Any = None
    score: float | None = None
    correct: bool | None = None
    error_kind: ErrorKind = ErrorKind.NONE
    error_message: str | None = None
    latency_s: float | None = Field(default=None, ge=0.0)


# ---------------------------------------------------------------------------
# 2. Job summary (`result.json`)
# ---------------------------------------------------------------------------


class ObservedHardware(BaseModel):
    """Typed hardware telemetry recorded in `result.json`."""

    model_config = ConfigDict(extra="forbid")

    platform: Literal["nvidia", "rocm"]
    gpu_id: str = Field(..., min_length=1)
    gpu_count: int = Field(ge=0)
    gpu_name: str | None = None
    driver_version: str | None = None
    device_memory_used_mib: list[int] | None = None


class MeasurementRecord(BaseModel):
    """One benchmark or eval execution point inside `result.json`."""

    model_config = ConfigDict(extra="forbid")

    workload: str = Field(..., min_length=1)
    kind: WorkloadKind
    point_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    execution_mode: ExecutionMode
    protocol_version: str = Field(..., min_length=1)
    requested_lifecycle: LifecyclePolicy
    actual_lifecycle: LifecyclePolicy
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    attempted: int | None = Field(default=None, ge=0)
    completed: int | None = Field(default=None, ge=0)
    scoring_denominator: int | None = Field(default=None, ge=0)
    model_errors: int = Field(default=0, ge=0)
    infra_errors: int = Field(default=0, ge=0)
    error_message: str | None = None
    artifacts: list[str] = Field(default_factory=list)

    @field_validator("artifacts")
    @classmethod
    def _validate_artifacts(cls, value: list[str]) -> list[str]:
        return [_validate_rel_path(p) for p in value]

    @model_validator(mode="after")
    def _validate(self) -> MeasurementRecord:
        if (
            self.kind == WorkloadKind.EVAL
            and self.execution_mode == ExecutionMode.DIRECT
        ):
            raise ValueError("Eval measurements cannot use execution_mode='direct'.")
        if self.attempted is not None:
            if self.completed is not None and self.completed > self.attempted:
                raise ValueError("completed cannot exceed attempted.")
            if (
                self.scoring_denominator is not None
                and self.scoring_denominator > self.attempted
            ):
                raise ValueError("scoring_denominator cannot exceed attempted.")
            if self.model_errors + self.infra_errors > self.attempted:
                raise ValueError("errors cannot exceed attempted.")
        return self


class RunResult(BaseModel):
    """Top-level `result.json` sealed inside `bundle.tar.gz`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    spec_name: str | None = None
    execution_status: ExecutionStatus
    started_at: datetime
    finished_at: datetime
    duration_s: float = Field(ge=0.0)
    harness_version: str
    vllm_commit: str | None = None
    image_digest: str | None = None
    requested_hardware: HardwareSpec
    observed_hardware: ObservedHardware
    measurements: list[MeasurementRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> RunResult:
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("Timestamps must be timezone-aware.")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at.")
        return self


# ---------------------------------------------------------------------------
# 3. Bundle manifest (`manifest.json`)
# ---------------------------------------------------------------------------


class ArtifactEntry(BaseModel):
    """File size and SHA256 digest recorded in `manifest.json`."""

    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(..., pattern=SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _validate_rel_path(value)


class BundleManifest(BaseModel):
    """Top-level `manifest.json` inside `bundle.tar.gz`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    files: list[ArtifactEntry] = Field(default_factory=list)
