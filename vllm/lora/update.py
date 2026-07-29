# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Completion and staging helpers for explicit LoRA update scopes."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from vllm.lora.lora_model import LoRAModel
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
    if scope.operation not in ("replace", "patch"):
        raise ValueError("add_lora only accepts a LoRA replace or patch scope")
    if not request.load_inplace:
        raise ValueError("A scoped LoRA update requires load_inplace=True")
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


def validate_complete_lora_weights(loras: Mapping[str, object]) -> None:
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


def merge_lora_patch(
    current: LoRAModel,
    patch: LoRAModel,
    module_names: set[str],
) -> LoRAModel:
    """Build a copy-on-write adapter by replacing complete runtime modules."""
    if current.rank != patch.rank:
        raise ValueError(
            f"LoRA patch rank mismatch: current={current.rank}, patch={patch.rank}"
        )
    if current.is_3d_lora_weight != patch.is_3d_lora_weight:
        raise ValueError("LoRA patch storage format does not match the adapter")
    missing_current = module_names - set(current.loras)
    missing_patch = module_names - set(patch.loras)
    unexpected = set(patch.loras) - module_names
    if missing_current or missing_patch or unexpected:
        raise ValueError(
            "LoRA patch module manifest mismatch: "
            f"unknown={sorted(missing_current)[:20]}, "
            f"missing={sorted(missing_patch)[:20]}, "
            f"unexpected={sorted(unexpected)[:20]}"
        )
    for name in module_names:
        current_weights = current.loras[name]
        patch_weights = patch.loras[name]
        _validate_patch_tensor_layout(
            current_weights.lora_a, patch_weights.lora_a, f"{name}.lora_a"
        )
        _validate_patch_tensor_layout(
            current_weights.lora_b, patch_weights.lora_b, f"{name}.lora_b"
        )
    merged = current.clone(current.id)
    merged.loras.update({name: patch.loras[name] for name in module_names})
    return merged


def _validate_patch_tensor_layout(
    current: torch.Tensor | list[torch.Tensor | None],
    patch: torch.Tensor | list[torch.Tensor | None],
    name: str,
) -> None:
    if isinstance(current, list) or isinstance(patch, list):
        if not isinstance(current, list) or not isinstance(patch, list):
            raise ValueError(f"LoRA patch layout mismatch for {name}")
        if len(current) != len(patch):
            raise ValueError(f"LoRA patch fragment count mismatch for {name}")
        for index, (current_item, patch_item) in enumerate(zip(current, patch)):
            if current_item is None or patch_item is None:
                if current_item is not patch_item:
                    raise ValueError(
                        f"LoRA patch fragment presence mismatch for {name}[{index}]"
                    )
                continue
            _validate_patch_tensor_layout(current_item, patch_item, f"{name}[{index}]")
        return
    if current.shape != patch.shape or current.dtype != patch.dtype:
        raise ValueError(
            f"LoRA patch tensor mismatch for {name}: "
            f"current=({tuple(current.shape)}, {current.dtype}), "
            f"patch=({tuple(patch.shape)}, {patch.dtype})"
        )


class TensorLoRAUpdateSession:
    """Accumulate transport buckets and construct one LoRA update."""

    def __init__(
        self,
        scope: LoRAAdapterScope | dict,
        peft_config: dict,
    ) -> None:
        normalized = normalize_update_scope(scope)
        if not isinstance(normalized, LoRAAdapterScope):
            raise ValueError("TensorLoRAUpdateSession requires a LoRA scope")
        if (
            normalized.operation not in ("replace", "patch")
            or normalized.tensor_names is None
        ):
            raise ValueError(
                "Tensor LoRA updates require replace or patch and an exact "
                "tensor manifest"
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
                "base_generation": self.scope.base_generation,
                "module_names": (
                    None
                    if self.scope.module_names is None
                    else list(self.scope.module_names)
                ),
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
    "merge_lora_patch",
    "validate_complete_lora_weights",
    "validate_lora_update_scope",
]
