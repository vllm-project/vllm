# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned wire schemas for rollout trajectory artifacts."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import torch

from vllm.distributed.artifact_transfer.artifact_connector.v1.base import (
    ArtifactHandle,
)

TRAJECTORY_SCHEMA_NAME = "vllm.trajectory"
TRAJECTORY_SCHEMA_VERSION_V1ALPHA1 = "v1alpha1"
TRAJECTORY_STATUS_COMPLETE = "complete"

_REQUIRED_FIELDS = (
    "prompt_token_ids",
    "response_token_ids",
    "response_logprobs",
)
_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
_FLOAT_DTYPES = {
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
}


def _require_nonempty(name: str, value: str) -> str:
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _key_component(value: str | int) -> str:
    text = str(value)
    _require_nonempty("trajectory key component", text)
    return quote(text, safe="-_.")


def build_trajectory_partition_id(
    run_id: str,
    policy_version: str | int,
) -> str:
    """Build the stable TransferQueue partition for a rollout policy."""

    return f"rollout-{_key_component(run_id)}-{_key_component(policy_version)}"


def build_trajectory_artifact_id(
    run_id: str,
    policy_version: str | int,
    request_id: str,
    sample_index: int,
) -> str:
    """Build an idempotent key for one trajectory sample."""

    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    return ":".join(
        (
            _key_component(run_id),
            _key_component(policy_version),
            _key_component(request_id),
            str(sample_index),
        )
    )


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_nested:
            raise ValueError(f"{name} must be a regular tensor")
        return value.detach()
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be tensor-like") from exc


def _normalize_integer_vector(value: Any, name: str) -> torch.Tensor:
    tensor = _as_tensor(value, name)
    if tensor.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must use an integer dtype, got {tensor.dtype}")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tensor.shape}")
    return tensor.to(device="cpu", dtype=torch.int64).contiguous()


def _normalize_float_vector(
    value: Any,
    name: str,
    *,
    allow_scalar: bool = False,
) -> torch.Tensor:
    tensor = _as_tensor(value, name)
    if tensor.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"{name} must use a floating dtype, got {tensor.dtype}")
    allowed_dims = (0, 1) if allow_scalar else (1,)
    if tensor.ndim not in allowed_dims:
        raise ValueError(
            f"{name} must have dimension {allowed_dims}, got shape {tensor.shape}"
        )
    return tensor.to(device="cpu", dtype=torch.float32).contiguous()


def _normalize_optional_tensor(
    value: Any,
    name: str,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = _as_tensor(value, name)
    if tensor.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device="cpu").contiguous()


def unwrap_transfer_queue_sample(value: Any, field_name: str) -> Any:
    """Remove TransferQueue's batch wrapper for a single retrieved sample."""

    if isinstance(value, torch.Tensor):
        if value.is_nested:
            samples = list(value.unbind())
            if len(samples) != 1:
                raise ValueError(
                    f"{field_name} contains {len(samples)} samples; expected one"
                )
            return samples[0]
        if value.ndim > 1 and value.shape[0] == 1:
            return value[0]
        return value

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        samples = tolist()
        if isinstance(samples, list) and len(samples) == 1:
            return samples[0]
    return value


@dataclass
class TransferQueueTrajectoryRecord:
    """TransferQueue-ready representation of one trajectory."""

    key: str
    partition_id: str
    fields: dict[str, torch.Tensor]
    tag: dict[str, Any]

    def to_handle(self) -> ArtifactHandle:
        return ArtifactHandle(
            backend="transfer_queue",
            artifact_id=self.key,
            location={
                "partition_id": self.partition_id,
                "key": self.key,
            },
            fields=list(self.fields),
            metadata={
                "schema_name": self.tag["schema_name"],
                "schema_version": self.tag["schema_version"],
                "run_id": self.tag["run_id"],
                "policy_version": self.tag["policy_version"],
                "status": self.tag["status"],
            },
        )


@dataclass
class TrajectoryArtifactV1Alpha1:
    """Consumer-neutral rollout trajectory contract."""

    run_id: str
    request_id: str
    engine_id: str
    model_id: str
    policy_version: str | int
    prompt_token_ids: torch.Tensor | Sequence[int]
    response_token_ids: torch.Tensor | Sequence[int]
    response_logprobs: torch.Tensor | Sequence[float]
    sample_index: int = 0
    group_id: str | None = None
    finish_reason: str | None = None
    created_at_ns: int = field(default_factory=time.time_ns)
    prompt_logprobs: torch.Tensor | Sequence[float] | None = None
    rewards: torch.Tensor | Sequence[float] | float | None = None
    values: torch.Tensor | Sequence[float] | None = None
    loss_mask: torch.Tensor | Sequence[float] | None = None
    routed_experts: torch.Tensor | None = None

    @property
    def key(self) -> str:
        return build_trajectory_artifact_id(
            self.run_id,
            self.policy_version,
            self.request_id,
            self.sample_index,
        )

    @property
    def partition_id(self) -> str:
        return build_trajectory_partition_id(self.run_id, self.policy_version)

    def _normalized_fields(self) -> dict[str, torch.Tensor]:
        prompt_token_ids = _normalize_integer_vector(
            self.prompt_token_ids, "prompt_token_ids"
        )
        response_token_ids = _normalize_integer_vector(
            self.response_token_ids, "response_token_ids"
        )
        response_logprobs = _normalize_float_vector(
            self.response_logprobs, "response_logprobs"
        )
        if len(response_token_ids) != len(response_logprobs):
            raise ValueError(
                "response_token_ids and response_logprobs must have equal length"
            )
        if len(prompt_token_ids) == 0:
            raise ValueError("prompt_token_ids must not be empty")

        fields = {
            "prompt_token_ids": prompt_token_ids,
            "response_token_ids": response_token_ids,
            "response_logprobs": response_logprobs,
        }
        if self.prompt_logprobs is not None:
            fields["prompt_logprobs"] = _normalize_float_vector(
                self.prompt_logprobs, "prompt_logprobs"
            )
        if self.rewards is not None:
            fields["rewards"] = _normalize_float_vector(
                self.rewards, "rewards", allow_scalar=True
            )
        if self.values is not None:
            fields["values"] = _normalize_float_vector(self.values, "values")
        if self.loss_mask is not None:
            loss_mask = _normalize_float_vector(self.loss_mask, "loss_mask")
            if len(loss_mask) != len(response_token_ids):
                raise ValueError(
                    "loss_mask and response_token_ids must have equal length"
                )
            fields["loss_mask"] = loss_mask
        if self.routed_experts is not None:
            fields["routed_experts"] = _normalize_optional_tensor(
                self.routed_experts, "routed_experts"
            )
        return fields

    def _tag(self) -> dict[str, Any]:
        _require_nonempty("run_id", self.run_id)
        _require_nonempty("request_id", self.request_id)
        _require_nonempty("engine_id", self.engine_id)
        _require_nonempty("model_id", self.model_id)
        _require_nonempty("policy_version", str(self.policy_version))
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.created_at_ns <= 0:
            raise ValueError("created_at_ns must be positive")

        tag: dict[str, Any] = {
            "schema_name": TRAJECTORY_SCHEMA_NAME,
            "schema_version": TRAJECTORY_SCHEMA_VERSION_V1ALPHA1,
            "status": TRAJECTORY_STATUS_COMPLETE,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "group_id": self.group_id or self.request_id,
            "sample_index": self.sample_index,
            "engine_id": self.engine_id,
            "model_id": self.model_id,
            "policy_version": self.policy_version,
            "created_at_ns": self.created_at_ns,
        }
        if self.finish_reason is not None:
            tag["finish_reason"] = self.finish_reason
        return tag

    def validate(self) -> None:
        self._normalized_fields()
        self._tag()

    def to_transfer_queue_record(self) -> TransferQueueTrajectoryRecord:
        return TransferQueueTrajectoryRecord(
            key=self.key,
            partition_id=self.partition_id,
            fields=self._normalized_fields(),
            tag=self._tag(),
        )

    @classmethod
    def from_transfer_queue(
        cls,
        fields: Mapping[str, Any],
        tag: Mapping[str, Any],
    ) -> TrajectoryArtifactV1Alpha1:
        schema_name = tag.get("schema_name")
        schema_version = tag.get("schema_version")
        if schema_name != TRAJECTORY_SCHEMA_NAME:
            raise ValueError(f"Unsupported trajectory schema: {schema_name}")
        if schema_version != TRAJECTORY_SCHEMA_VERSION_V1ALPHA1:
            raise ValueError(f"Unsupported trajectory schema version: {schema_version}")

        status = tag.get("status")
        if status != TRAJECTORY_STATUS_COMPLETE:
            raise ValueError(f"Trajectory is not complete: status={status}")

        missing_fields = [name for name in _REQUIRED_FIELDS if name not in fields]
        if missing_fields:
            raise ValueError(f"Missing required trajectory fields: {missing_fields}")

        required_tags = (
            "run_id",
            "request_id",
            "engine_id",
            "model_id",
            "policy_version",
            "created_at_ns",
        )
        missing_tags = [name for name in required_tags if name not in tag]
        if missing_tags:
            raise ValueError(f"Missing required trajectory tags: {missing_tags}")

        unwrapped = {
            name: unwrap_transfer_queue_sample(value, name)
            for name, value in fields.items()
        }
        artifact = cls(
            run_id=str(tag["run_id"]),
            request_id=str(tag["request_id"]),
            engine_id=str(tag["engine_id"]),
            model_id=str(tag["model_id"]),
            policy_version=tag["policy_version"],
            sample_index=int(tag.get("sample_index", 0)),
            group_id=tag.get("group_id"),
            finish_reason=tag.get("finish_reason"),
            created_at_ns=int(tag["created_at_ns"]),
            prompt_token_ids=unwrapped["prompt_token_ids"],
            response_token_ids=unwrapped["response_token_ids"],
            response_logprobs=unwrapped["response_logprobs"],
            prompt_logprobs=unwrapped.get("prompt_logprobs"),
            rewards=unwrapped.get("rewards"),
            values=unwrapped.get("values"),
            loss_mask=unwrapped.get("loss_mask"),
            routed_experts=unwrapped.get("routed_experts"),
        )
        artifact.validate()
        return artifact
