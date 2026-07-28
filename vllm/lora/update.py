# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Completion and staging helpers for explicit LoRA replacement scopes."""

import hashlib
import json
from pathlib import Path

import torch

from vllm.lora.request import LoRARequest, TensorLoRARequest
from vllm.model_executor.model_loader.reload.scope import (
    LoRAAdapterScope,
    normalize_update_scope,
)


def config_digest(config: dict) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def artifact_digest(path: str) -> str:
    """Hash the adapter files which determine one path-backed replacement."""
    root = Path(path)
    candidates = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_model.pt",
    )
    digest = hashlib.sha256()
    found = False
    for name in candidates:
        file_path = root / name
        if not file_path.is_file():
            continue
        found = True
        digest.update(name.encode())
        with file_path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    if not found:
        raise ValueError(f"{path} does not contain a LoRA adapter artifact")
    return digest.hexdigest()


def validate_lora_update_scope(
    request: LoRARequest | TensorLoRARequest,
) -> LoRAAdapterScope | None:
    """Validate an optional scope before loading or replacing an adapter."""
    if request.update_scope is None:
        return None
    scope = normalize_update_scope(request.update_scope)
    if not isinstance(scope, LoRAAdapterScope):
        raise ValueError("A LoRA request requires a lora_adapter update scope")
    if scope.operation != "replace":
        raise ValueError("add_lora only accepts a LoRA replace scope")
    if not request.load_inplace:
        raise ValueError("A scoped LoRA replacement requires load_inplace=True")
    if (scope.adapter_id, scope.adapter_name) != (
        request.lora_int_id,
        request.lora_name,
    ):
        raise ValueError(
            "LoRA request identity does not match its update scope: "
            f"request=({request.lora_int_id}, {request.lora_name!r}), "
            f"scope=({scope.adapter_id}, {scope.adapter_name!r})"
        )

    if isinstance(request, TensorLoRARequest):
        received = set(request.lora_tensors)
        if scope.tensor_names is None:
            raise ValueError(
                "An in-memory LoRA replacement scope requires tensor_names"
            )
        expected = set(scope.tensor_names)
        missing = expected - received
        unexpected = received - expected
        if missing or unexpected:
            raise ValueError(
                "LoRA tensor manifest mismatch: "
                f"missing={sorted(missing)[:20]}, "
                f"unexpected={sorted(unexpected)[:20]}"
            )
        if (
            scope.config_digest is not None
            and config_digest(request.peft_config) != scope.config_digest
        ):
            raise ValueError("LoRA PEFT configuration digest mismatch")
    else:
        if scope.tensor_names is not None:
            raise ValueError(
                "A path-backed LoRA replacement cannot declare tensor_names"
            )
        if (
            scope.artifact_digest is not None
            and artifact_digest(request.lora_path) != scope.artifact_digest
        ):
            raise ValueError("LoRA adapter artifact digest mismatch")
        if scope.config_digest is not None:
            config_path = Path(request.lora_path) / "adapter_config.json"
            if not config_path.is_file():
                raise ValueError("LoRA adapter configuration is missing")
            with config_path.open() as file:
                path_config = json.load(file)
            if config_digest(path_config) != scope.config_digest:
                raise ValueError("LoRA PEFT configuration digest mismatch")
    return scope


def validate_complete_lora_weights(loras: dict[str, object]) -> None:
    """Require every staged module to contain both low-rank matrices."""
    incomplete = sorted(
        name
        for name, weights in loras.items()
        if getattr(weights, "lora_a", None) is None
        or getattr(weights, "lora_b", None) is None
    )
    if incomplete:
        raise ValueError(
            "LoRA replacement contains incomplete A/B pairs for modules: "
            f"{incomplete[:20]}"
        )


class TensorLoRAUpdateSession:
    """Accumulate transport buckets and construct one complete replacement."""

    def __init__(
        self,
        scope: LoRAAdapterScope | dict,
        peft_config: dict,
    ) -> None:
        normalized = normalize_update_scope(scope)
        if not isinstance(normalized, LoRAAdapterScope):
            raise ValueError("TensorLoRAUpdateSession requires a LoRA scope")
        if normalized.operation != "replace" or normalized.tensor_names is None:
            raise ValueError(
                "Tensor LoRA updates require replace and an exact tensor manifest"
            )
        self.scope = normalized
        self.peft_config = peft_config
        self._tensors: dict[str, torch.Tensor] = {}

    def add_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        expected = set(self.scope.tensor_names or ())
        unexpected = set(tensors) - expected
        duplicates = set(tensors) & set(self._tensors)
        if unexpected or duplicates:
            raise ValueError(
                "Invalid LoRA tensor bucket: "
                f"unexpected={sorted(unexpected)[:20]}, "
                f"duplicates={sorted(duplicates)[:20]}"
            )
        self._tensors.update(tensors)

    def finish(self) -> TensorLoRARequest:
        request = TensorLoRARequest(
            lora_name=self.scope.adapter_name,
            lora_int_id=self.scope.adapter_id,
            lora_tensors=self._tensors,
            peft_config=self.peft_config,
            update_scope={
                "kind": self.scope.kind.value,
                "operation": self.scope.operation,
                "adapter_id": self.scope.adapter_id,
                "adapter_name": self.scope.adapter_name,
                "tensor_names": list(self.scope.tensor_names or ()),
                "config_digest": self.scope.config_digest,
                "artifact_digest": self.scope.artifact_digest,
            },
        )
        validate_lora_update_scope(request)
        return request


__all__ = [
    "TensorLoRAUpdateSession",
    "artifact_digest",
    "config_digest",
    "validate_complete_lora_weights",
    "validate_lora_update_scope",
]
