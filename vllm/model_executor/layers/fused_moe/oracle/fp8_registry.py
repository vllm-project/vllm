# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry for out-of-tree FP8 MoE expert implementations."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from vllm.config.kernel import BUILTIN_MOE_BACKENDS
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEExperts


@dataclass(frozen=True)
class RegisteredFp8MoeBackend:
    """Description of an out-of-tree FP8 MoE backend."""

    name: str
    expert_class_paths: tuple[str, ...]
    auto_select: bool = False


_REGISTERED_FP8_MOE_BACKENDS: dict[str, RegisteredFp8MoeBackend] = {}


def _normalize_backend_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    if not normalized:
        raise ValueError("FP8 MoE backend name must not be empty.")
    return normalized


def register_fp8_moe_backend(
    name: str,
    expert_class_paths: str | Sequence[str],
    *,
    auto_select: bool = False,
) -> None:
    """Register an out-of-tree FP8 MoE backend.

    Registration is idempotent when the normalized name, class paths, and
    automatic-selection setting are identical. Registered implementations
    consume canonical vLLM FP8 MoE weights.

    Args:
        name: User-facing backend name accepted by ``--moe-backend``.
        expert_class_paths: One or more qualified ``FusedMoEExperts`` class
            names. Classes are imported lazily when the oracle considers the
            backend.
        auto_select: Whether the backend may be considered after built-in
            backends when ``--moe-backend auto`` is used.

    Raises:
        ValueError: If the registration is empty, conflicts with a built-in
            backend, or conflicts with an existing registration.
    """
    normalized_name = _normalize_backend_name(name)
    if normalized_name in BUILTIN_MOE_BACKENDS:
        raise ValueError(
            f"Cannot register FP8 MoE backend '{normalized_name}': "
            "the name is reserved by a built-in MoE backend."
        )

    paths = (
        (expert_class_paths,)
        if isinstance(expert_class_paths, str)
        else tuple(expert_class_paths)
    )
    if not paths or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("At least one expert class path is required.")

    registration = RegisteredFp8MoeBackend(
        name=normalized_name,
        expert_class_paths=paths,
        auto_select=auto_select,
    )
    existing = _REGISTERED_FP8_MOE_BACKENDS.get(normalized_name)
    if existing is not None and existing != registration:
        raise ValueError(
            f"FP8 MoE backend '{normalized_name}' is already registered "
            "with a different configuration."
        )
    _REGISTERED_FP8_MOE_BACKENDS[normalized_name] = registration


def get_registered_fp8_moe_backend(name: str) -> RegisteredFp8MoeBackend | None:
    """Return a registered backend by normalized name."""
    return _REGISTERED_FP8_MOE_BACKENDS.get(_normalize_backend_name(name))


def iter_auto_fp8_moe_backends() -> Iterable[RegisteredFp8MoeBackend]:
    """Iterate registered backends that opted into automatic selection."""
    return (
        backend
        for backend in _REGISTERED_FP8_MOE_BACKENDS.values()
        if backend.auto_select
    )


def registered_fp8_moe_backend_names() -> tuple[str, ...]:
    """Return registered backend names in registration order."""
    return tuple(_REGISTERED_FP8_MOE_BACKENDS)


@cache
def resolve_fp8_moe_experts(
    backend: RegisteredFp8MoeBackend,
) -> tuple[type["FusedMoEExperts"], ...]:
    """Resolve and validate the expert classes for a registered backend."""
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEExperts

    classes: list[type[FusedMoEExperts]] = []
    for class_path in backend.expert_class_paths:
        obj = resolve_obj_by_qualname(class_path)
        if not isinstance(obj, type) or not issubclass(obj, FusedMoEExperts):
            raise TypeError(
                f"Registered FP8 MoE expert '{class_path}' must be a subclass "
                "of FusedMoEExperts."
            )
        classes.append(obj)
    return tuple(classes)
