# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned schemas for result.json, manifest.json, eval records, and Upload API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tools.byog.schemas.catalog import (
    SAFE_ID_PATTERN,
    SHA256_PATTERN,
    EvidenceSource,
    LifecyclePolicy,
)


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class QualityState(StrEnum):
    """Quality status tracked separately from ExecutionStatus."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"
    INCONCLUSIVE = "inconclusive"


class ProtocolVariant(StrEnum):
    """Actual server-state handling executed by the coordinator."""

    NONE = "none"
    DIRECT = "direct"
    RESET_CACHE = "reset_cache"
    RESTART_SERVER = "restart_server"


class ErrorKind(StrEnum):
    NONE = "none"
    MODEL = "model"
    INFRA = "infra"


class EvalSampleRecord(BaseModel):
    """Deduplicated source question/reference keyed by (dataset_revision, sample_id)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    dataset_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_revision: str = Field(..., min_length=1)
    sample_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    question: str | None = None
    reference_answer: str | None = None
    media_refs: list[str] = Field(default_factory=list)
    withheld_reason: str | None = None


class EvalRenderedInputRecord(BaseModel):
    """Exact rendered prompt/messages sent to the model, keyed by input_hash."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    dataset_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    input_hash: str = Field(..., min_length=8)
    dataset_revision: str = Field(..., min_length=1)
    sample_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    rendered_input: Any = None
    withheld_reason: str | None = None


class EvalResponseRecord(BaseModel):
    """Per-(point, sample, repeat, concurrency) model response and score."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    point_id: str = Field(..., min_length=1)
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
    output_tokens: int | None = Field(default=None, ge=0)
    withheld_reason: str | None = None


class ThresholdCheckResult(BaseModel):
    """Outcome of checking one configured quality threshold on a measurement."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    observed: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    passed: bool


class MeasurementRecord(BaseModel):
    """Per-workload / per-point measurement with named compatibility components."""

    model_config = ConfigDict(extra="forbid")

    workload: str
    kind: Literal["benchmark", "eval"]
    point_id: str
    execution_mode: Literal["server", "direct"]
    protocol_version: str
    requested_lifecycle: LifecyclePolicy
    protocol_variant: ProtocolVariant
    lifecycle_fallback_reason: str | None = None
    evidence_source: EvidenceSource = EvidenceSource.HARNESS
    compatibility_rule_version: str = "1"
    compatibility_components: dict[str, Any] = Field(default_factory=dict)
    compatibility_key: str = Field(
        ...,
        pattern=SHA256_PATTERN,
        description="SHA256 over compatibility_components.",
    )
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    quality_state: QualityState = QualityState.NOT_EVALUATED
    threshold_checks: list[ThresholdCheckResult] = Field(default_factory=list)
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    attempted: int | None = Field(default=None, ge=0)
    completed: int | None = Field(default=None, ge=0)
    scoring_denominator: int | None = Field(default=None, ge=0)
    model_errors: int = Field(default=0, ge=0)
    infra_errors: int = Field(default=0, ge=0)
    error_message: str | None = None
    artifacts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_invariants(self) -> MeasurementRecord:
        if (
            self.attempted is not None
            and self.completed is not None
            and self.completed > self.attempted
        ):
            raise ValueError("completed cannot exceed attempted.")

        if self.execution_mode == "direct":
            if self.requested_lifecycle != LifecyclePolicy.NONE:
                raise ValueError(
                    "direct measurements must have requested_lifecycle=none."
                )
            if self.protocol_variant != ProtocolVariant.DIRECT:
                raise ValueError(
                    "direct measurements must have protocol_variant=direct."
                )
        else:
            if self.protocol_variant == ProtocolVariant.DIRECT:
                raise ValueError(
                    "server measurements cannot use protocol_variant=direct."
                )
            if (
                self.requested_lifecycle.value != self.protocol_variant.value
                and not self.lifecycle_fallback_reason
            ):
                raise ValueError(
                    f"requested_lifecycle={self.requested_lifecycle.value} differs "
                    f"from protocol_variant={self.protocol_variant.value} without "
                    "lifecycle_fallback_reason."
                )

        if self.quality_state == QualityState.PASSED:
            if self.status != ExecutionStatus.COMPLETED:
                raise ValueError("quality_state=passed requires status=completed.")
            if not self.threshold_checks or not all(
                c.passed for c in self.threshold_checks
            ):
                raise ValueError(
                    "quality_state=passed requires at least one passing threshold."
                )
        return self


class RunResult(BaseModel):
    """Top-level result.json sealed inside bundle.tar.gz."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    spec_name: str | None = None
    source_tier: Literal["ci_nightly", "ci_pr", "community"] = "community"
    execution_status: ExecutionStatus
    quality_state: QualityState = QualityState.NOT_EVALUATED
    started_at: datetime
    finished_at: datetime
    duration_s: float = Field(ge=0.0)
    harness_version: str
    vllm_commit: str | None = None
    image_digest: str | None = None
    requested_hardware: dict[str, Any] = Field(default_factory=dict)
    observed_hardware: dict[str, Any] = Field(default_factory=dict)
    combo_key: dict[str, Any] = Field(default_factory=dict)
    resolved_config_hash: str = ""
    measurements: list[MeasurementRecord] = Field(default_factory=list)
    withheld_artifacts: list[dict[str, str]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_run_quality(self) -> RunResult:
        if self.quality_state == QualityState.PASSED:
            if self.execution_status != ExecutionStatus.COMPLETED:
                raise ValueError(
                    "RunResult cannot have quality_state=passed when execution failed."
                )
            evals = [m for m in self.measurements if m.kind == "eval"]
            if not evals or not all(
                m.quality_state == QualityState.PASSED for m in evals
            ):
                raise ValueError(
                    "RunResult quality_state=passed requires passing thresholded evals."
                )
        return self


class ArtifactEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(..., pattern=SHA256_PATTERN)


class BundleManifest(BaseModel):
    """Top-level manifest.json listing every file inside bundle.tar.gz."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    files: list[ArtifactEntry] = Field(default_factory=list)


# --- Upload HTTP API Contracts (co-reviewed with §3 Results Service) ---


class UploadRequestMeta(BaseModel):
    """Metadata sent alongside bundle.tar.gz to POST /api/v1/bundles.

    Trust tier (`ci_nightly`/`ci_pr`/`community`) is derived by the service from
    the upload credential/arrival path rather than self-claimed by the client.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    bundle_sha256: str = Field(..., pattern=SHA256_PATTERN)
    manifest_sha256: str = Field(..., pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=1)
    ci_run_id: str | None = None


class UploadSuccessResponse(BaseModel):
    """HTTP 201 Created (new) or 200 OK (idempotent retry with identical SHA256)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["created", "duplicate_identical"]
    job_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    bundle_sha256: str = Field(..., pattern=SHA256_PATTERN)
    result_url: str
    uploaded_by: str
    derived_source_tier: Literal["ci_nightly", "ci_pr", "community"]
    uploaded_at: datetime


class UploadConflictResponse(BaseModel):
    """HTTP 409 when attempt_id already exists with a different bundle_sha256."""

    model_config = ConfigDict(extra="forbid")

    error: Literal["attempt_conflict"] = "attempt_conflict"
    job_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    existing_bundle_sha256: str = Field(..., pattern=SHA256_PATTERN)
    submitted_bundle_sha256: str = Field(..., pattern=SHA256_PATTERN)
    message: str
