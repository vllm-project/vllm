# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serializable declarations for one model-state update."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypeAlias


class UpdateKind(str, Enum):
    BASE_CHECKPOINT = "base_checkpoint"
    BASE_KERNEL = "base_kernel"
    LORA_ADAPTER = "lora_adapter"


@dataclass(frozen=True)
class FullBaseWeightScope:
    kind: Literal[UpdateKind.BASE_CHECKPOINT] = UpdateKind.BASE_CHECKPOINT
    mode: Literal["full"] = "full"


@dataclass(frozen=True)
class PartialBaseWeightScope:
    source_names: tuple[str, ...]
    kind: Literal[UpdateKind.BASE_CHECKPOINT] = UpdateKind.BASE_CHECKPOINT
    mode: Literal["partial"] = "partial"

    def __post_init__(self) -> None:
        _validate_names("source_names", self.source_names)


@dataclass(frozen=True)
class KernelWeightScope:
    target_names: tuple[str, ...]
    kind: Literal[UpdateKind.BASE_KERNEL] = UpdateKind.BASE_KERNEL

    def __post_init__(self) -> None:
        _validate_names("target_names", self.target_names)


@dataclass(frozen=True)
class LoRAAdapterScope:
    adapter_id: int
    adapter_name: str
    operation: Literal["replace", "patch", "remove"] = "replace"
    base_generation: int | None = None
    module_names: tuple[str, ...] | None = None
    tensor_names: tuple[str, ...] | None = None
    config_digest: str | None = None
    artifact_digest: str | None = None
    kind: Literal[UpdateKind.LORA_ADAPTER] = UpdateKind.LORA_ADAPTER

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_id, int) or self.adapter_id < 1:
            raise ValueError("adapter_id must be greater than zero")
        if not isinstance(self.adapter_name, str) or not self.adapter_name:
            raise ValueError("adapter_name must not be empty")
        if self.operation not in ("replace", "patch", "remove"):
            raise ValueError(f"Unsupported LoRA operation: {self.operation}")
        if self.module_names is not None:
            _validate_names("module_names", self.module_names)
        if self.tensor_names is not None:
            _validate_names("tensor_names", self.tensor_names)
        if self.operation == "patch" and self.module_names is None:
            raise ValueError("A LoRA patch scope requires module_names")
        if self.operation == "patch" and (
            not isinstance(self.base_generation, int) or self.base_generation < 1
        ):
            raise ValueError("A LoRA patch scope requires a positive base_generation")
        if self.operation != "patch" and self.base_generation is not None:
            raise ValueError("base_generation is only valid for a LoRA patch")
        if self.operation != "patch" and self.module_names is not None:
            raise ValueError("module_names is only valid for a LoRA patch")
        if self.operation == "remove" and any(
            value is not None
            for value in (
                self.module_names,
                self.tensor_names,
                self.config_digest,
                self.artifact_digest,
                self.base_generation,
            )
        ):
            raise ValueError("A LoRA remove scope cannot declare replacement data")


UpdateScope: TypeAlias = (
    FullBaseWeightScope | PartialBaseWeightScope | KernelWeightScope | LoRAAdapterScope
)


def _validate_names(field: str, names: tuple[str, ...]) -> None:
    if not names:
        raise ValueError(f"{field} must not be empty")
    if any(not name for name in names):
        raise ValueError(f"{field} must not contain empty names")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} must not contain duplicate names")


def normalize_update_scope(scope: UpdateScope | dict[str, Any] | None) -> UpdateScope:
    """Normalize an API declaration into one immutable scope."""
    if scope is None:
        return FullBaseWeightScope()
    if isinstance(
        scope,
        (
            FullBaseWeightScope,
            PartialBaseWeightScope,
            KernelWeightScope,
            LoRAAdapterScope,
        ),
    ):
        return scope
    if not isinstance(scope, dict):
        raise TypeError("update scope must be a scope object, dict, or None")

    values = dict(scope)
    try:
        kind = UpdateKind(values.pop("kind"))
    except (KeyError, ValueError) as error:
        raise ValueError("update scope requires a valid `kind`") from error

    if kind is UpdateKind.BASE_CHECKPOINT:
        mode = values.pop("mode", "full")
        if mode == "full":
            if values:
                raise ValueError(
                    f"Unexpected full checkpoint scope fields: {sorted(values)}"
                )
            return FullBaseWeightScope()
        if mode == "partial":
            names = values.pop("source_names", None)
            if values:
                raise ValueError(
                    f"Unexpected partial checkpoint scope fields: {sorted(values)}"
                )
            if names is None:
                raise ValueError("partial checkpoint scope requires source_names")
            return PartialBaseWeightScope(tuple(names))
        raise ValueError(f"Unsupported checkpoint scope mode: {mode}")

    if kind is UpdateKind.BASE_KERNEL:
        names = values.pop("target_names", None)
        if values:
            raise ValueError(f"Unexpected kernel scope fields: {sorted(values)}")
        if names is None:
            raise ValueError("kernel scope requires target_names")
        return KernelWeightScope(tuple(names))

    adapter_id = values.pop("adapter_id", None)
    adapter_name = values.pop("adapter_name", None)
    module_names = values.pop("module_names", None)
    tensor_names = values.pop("tensor_names", None)
    result = LoRAAdapterScope(
        adapter_id=adapter_id,
        adapter_name=adapter_name,
        operation=values.pop("operation", "replace"),
        base_generation=values.pop("base_generation", None),
        module_names=None if module_names is None else tuple(module_names),
        tensor_names=None if tensor_names is None else tuple(tensor_names),
        config_digest=values.pop("config_digest", None),
        artifact_digest=values.pop("artifact_digest", None),
    )
    if values:
        raise ValueError(f"Unexpected LoRA scope fields: {sorted(values)}")
    return result


def scope_source_names(scope: UpdateScope) -> set[str] | None:
    """Return an exact transport manifest declared by a scope, if any."""
    if isinstance(scope, PartialBaseWeightScope):
        return set(scope.source_names)
    if isinstance(scope, LoRAAdapterScope) and scope.tensor_names is not None:
        return set(scope.tensor_names)
    return None


__all__ = [
    "FullBaseWeightScope",
    "KernelWeightScope",
    "LoRAAdapterScope",
    "PartialBaseWeightScope",
    "UpdateKind",
    "UpdateScope",
    "normalize_update_scope",
    "scope_source_names",
]
