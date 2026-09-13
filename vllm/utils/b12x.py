# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accessors for the optional ``b12x`` package."""

import importlib
import importlib.util
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, fields, is_dataclass
from types import ModuleType
from typing import Any

import torch


@dataclass(frozen=True)
class B12xWarmupUnit:
    name: str
    key: Hashable
    compile: Callable[[], None]


_HAS_B12X = importlib.util.find_spec("b12x") is not None


def _import_submodule(module_name: str) -> ModuleType | None:
    if not _HAS_B12X:
        return None
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


_B12X_SUBMODULES = {
    module_name: _import_submodule(module_name)
    for module_name in (
        "b12x.attention.paged",
        "b12x.gemm.blockscaled",
        # TODO: Remove once B12X exposes the scale-swizzle API publicly.
        "b12x._lib.intrinsics",
        "b12x.gemm.mxfp8_linear",
        "b12x.gemm.tensor_fp8_linear",
        "b12x.moe.fused_moe",
    )
}


def has_b12x() -> bool:
    """Return whether the B12X package is installed."""
    return _HAS_B12X


def _get_submodule(module_name: str) -> ModuleType | None:
    return _B12X_SUBMODULES.get(module_name)


def get_b12x_blockscaled() -> ModuleType | None:
    return _get_submodule("b12x.gemm.blockscaled")


def get_b12x_intrinsics() -> ModuleType | None:
    return _get_submodule("b12x._lib.intrinsics")


def get_b12x_mxfp8_linear() -> ModuleType | None:
    return _get_submodule("b12x.gemm.mxfp8_linear")


def get_b12x_tensor_fp8_linear() -> ModuleType | None:
    return _get_submodule("b12x.gemm.tensor_fp8_linear")


def get_b12x_fused_moe() -> ModuleType | None:
    return _get_submodule("b12x.moe.fused_moe")


def get_b12x_paged_attention() -> ModuleType | None:
    return _get_submodule("b12x.attention.paged")


def b12x_warmup_token_counts(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: Iterable[int] = (),
) -> tuple[int, ...]:
    # B12X deduplicates shapes that select the same internal kernel policy.
    # Keep the complete serving shape set here rather than duplicating its
    # policy-selection heuristics in vLLM.
    counts = {1}
    counts.update(int(size) for size in cudagraph_capture_sizes if int(size) > 0)
    if int(max_tokens) > 0:
        counts.add(int(max_tokens))
    return tuple(sorted(counts))


def _same_packed_layout(current: Any, replacement: Any) -> bool:
    if type(current) is not type(replacement):
        return False
    if isinstance(current, torch.Tensor):
        return (
            current.shape == replacement.shape
            and current.stride() == replacement.stride()
            and current.dtype == replacement.dtype
            and current.device == replacement.device
        )
    if is_dataclass(current):
        return all(
            _same_packed_layout(
                getattr(current, field.name),
                getattr(replacement, field.name),
            )
            for field in fields(current)
        )
    return bool(current == replacement)


def _copy_packed_tensors(current: Any, replacement: Any) -> None:
    if isinstance(current, torch.Tensor):
        current.copy_(replacement)
    elif is_dataclass(current):
        for field in fields(current):
            _copy_packed_tensors(
                getattr(current, field.name),
                getattr(replacement, field.name),
            )


@torch.no_grad()
def reuse_packed_weight_storage(current: Any, replacement: Any) -> Any:
    """Reuse packed tensor addresses when a compatible weight is reloaded."""
    if current is None or not _same_packed_layout(current, replacement):
        return replacement
    _copy_packed_tensors(current, replacement)
    return current
