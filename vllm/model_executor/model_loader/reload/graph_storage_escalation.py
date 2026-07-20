# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Walk-discovered tensor escalation for graph storage.

Production mode (default): WARNING for all walk-discovered unregistered
tensors, with a suggestion to register them for O(1) copy-back.

Strict mode (VLLM_GRAPH_STORAGE_STRICT=1): raises for tensors whose
storage address appears in the CUDA graph address set (i.e., tensors
that are actively referenced by captured graphs).

Rate-limited: same (layer_type, path) warned at most once per process.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# Env var to enable strict mode (raises instead of warning)
_STRICT_MODE: bool | None = None

# Rate limiter: set of (layer_type, path) already warned
_warned_paths: set[tuple[str, str]] = set()


def is_strict_mode() -> bool:
    """Check if strict escalation mode is enabled."""
    global _STRICT_MODE
    if _STRICT_MODE is None:
        _STRICT_MODE = os.environ.get(
            "VLLM_GRAPH_STORAGE_STRICT", "0") == "1"
    return _STRICT_MODE


def reset_warned_paths() -> None:
    """Clear rate-limiting state. Useful for tests."""
    _warned_paths.clear()


def _is_graph_relevant(
    tensor: torch.Tensor | None,
    graph_address_set: set[int] | None,
    layer_is_graph_captured: bool,
) -> bool:
    """Determine if a walk-discovered tensor is graph-relevant.

    A tensor is graph-relevant when the containing layer is graph-captured
    AND its storage address is in the CUDA graph address set.
    """
    if (layer_is_graph_captured
            and graph_address_set is not None
            and tensor is not None
            and isinstance(tensor, torch.Tensor)):
        storage_ptr = tensor.untyped_storage().data_ptr()
        if storage_ptr in graph_address_set:
            return True
    return False


class GraphStorageStrictError(RuntimeError):
    """Raised in strict mode when graph-relevant unregistered tensors
    are discovered by the walk fallback."""
    pass


def escalate_walk_discoveries(
    layer: nn.Module,
    extra_tensor_slots: list[tuple[str, Any]],
    graph_address_set: set[int] | None = None,
    graph_captured_layers: set[nn.Module] | None = None,
) -> None:
    """Escalate walk-discovered unregistered tensors.

    In production mode: WARNING with layer type, path, and migration
    suggestion. Rate-limited by (layer_type, path).

    In strict mode: raises GraphStorageStrictError for graph-relevant
    unregistered tensors (those whose storage address is in the CUDA
    graph address set on a graph-captured layer).

    Args:
        layer: The layer being reloaded.
        extra_tensor_slots: Walk-discovered (path, tensor) pairs that
            are NOT in the registry (already filtered by exclude_paths).
        graph_address_set: Optional set of storage data_ptrs from CUDA
            graph captures.
        graph_captured_layers: Optional set of layers that participate in
            CUDA graph captures.
    """
    if not extra_tensor_slots:
        return

    layer_type = type(layer).__name__
    strict = is_strict_mode()
    strict_violations: list[str] = []

    layer_is_graph_captured = (
        graph_captured_layers is not None and layer in graph_captured_layers
    )

    for path, tensor in extra_tensor_slots:
        graph_relevant = _is_graph_relevant(
            tensor, graph_address_set, layer_is_graph_captured)

        # Strict mode: always collect violations (no rate limiting)
        if graph_relevant and strict:
            strict_violations.append(path)
            continue

        # Rate limiting applies only to warning emission
        key = (layer_type, path)
        if key in _warned_paths:
            continue
        _warned_paths.add(key)

        logger.warning(
            "Unregistered graph-storage tensor on %s at '%s'. "
            "Consider calling register_graph_storage(layer, '%s', "
            "tensor) for O(1) registry copy-back.",
            layer_type, path, path,
        )

    if strict_violations:
        paths_str = ", ".join(f"'{p}'" for p in strict_violations)
        raise GraphStorageStrictError(
            f"VLLM_GRAPH_STORAGE_STRICT=1: {len(strict_violations)} "
            f"graph-relevant unregistered tensor(s) found on "
            f"{layer_type}: {paths_str}. "
            f"Register these tensors with register_graph_storage()."
        )
