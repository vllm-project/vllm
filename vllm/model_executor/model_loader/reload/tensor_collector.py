# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Collect and restore unmanaged CUDA tensors reachable from an nn.Module.

"Unmanaged" = not registered as nn.Parameter or nn.Buffer.  These tensors
are invisible to the layerwise-reload copy-back and their device addresses
drift when process_weights_after_loading recreates them, which breaks
CUDA-graph replay.

Typical locations:
  - Plain attribute on the layer   (layer.g_idx_sort_indices)
  - Attribute on a nested object   (layer.quant_method.kernel.workspace)
  - Deep nesting                   (layer.quant_method.moe_kernel
                                          .fused_experts.ab_strides1)
"""
from __future__ import annotations

import functools
import types
from typing import Any

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# Maximum recursion depth when walking nested Python objects.
# Real-world deepest path is ~5 (quant_method.moe_kernel.fused_experts.xxx).
_MAX_DEPTH = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_extra_tensors(
    layer: nn.Module,
) -> list[tuple[str, torch.Tensor]]:
    """Snapshot every CUDA tensor reachable from *layer* that is **not**
    registered as a parameter or buffer.

    Returns a list of ``(dotted_path, tensor)`` pairs.  Paths that share the
    same underlying storage are **all** recorded (needed so that
    ``set_by_path`` can fix every alias).  Tensor objects are **not** added
    to the visited set so that the same tensor found via two attribute names
    (e.g. ``ab_strides1`` and ``c_strides2``) produces two entries.
    """
    # Storage ids of tensors that are already managed by the reload system.
    managed_storages: set[int] = set()
    for p in layer._parameters.values():
        if p is not None:
            managed_storages.add(p.data.untyped_storage().data_ptr())
    for b in layer._buffers.values():
        if b is not None:
            managed_storages.add(b.untyped_storage().data_ptr())

    # Names to skip at the layer level.
    skip_names: set[str] = set()
    skip_names.update(layer._parameters.keys())
    skip_names.update(layer._buffers.keys())
    skip_names.update(layer._modules.keys())

    visited: set[int] = set()       # ids of non-tensor objects already walked
    results: list[tuple[str, torch.Tensor]] = []

    for attr_name in list(vars(layer)):
        # Skip PyTorch internal attrs and managed names.
        if attr_name.startswith("_") or attr_name in skip_names:
            continue
        val = getattr(layer, attr_name, None)
        if val is None:
            continue
        _walk(attr_name, val, managed_storages, visited, results, 0)

    return results


def resolve_path(root: Any, path: str) -> torch.Tensor | None:
    """Navigate *root* along the dot-separated *path* and return the tensor
    found there, or ``None`` if the path is broken."""
    cur = root
    for seg in path.split("."):
        cur = getattr(cur, seg, None)
        if cur is None:
            return None
    return cur if isinstance(cur, torch.Tensor) else None


def set_by_path(root: Any, path: str, value: torch.Tensor) -> bool:
    """Set the attribute at the end of *path* to *value*.
    Returns ``True`` on success."""
    parts = path.split(".")
    cur = root
    for seg in parts[:-1]:
        cur = getattr(cur, seg, None)
        if cur is None:
            return False
    try:
        setattr(cur, parts[-1], value)
        return True
    except (AttributeError, TypeError):
        return False


def copy_back_extra_tensors(
    layer: nn.Module,
    slots: list[tuple[str, torch.Tensor]],
) -> None:
    """For every recorded slot, copy the freshly-computed value into the old
    tensor's storage (preserving the CUDA-graph address) and point the
    attribute back at the old tensor.

    Shared-storage tensors are copied only once but ``set_by_path`` is called
    for every recorded path so that all aliases are restored.
    """
    if not slots:
        return

    copied_storages: set[int] = set()
    n_copied = 0
    n_skipped = 0

    for path, old_tensor in slots:
        new_tensor = resolve_path(layer, path)

        if not isinstance(new_tensor, torch.Tensor):
            logger.debug("extra-tensor path gone after reload: %s", path)
            n_skipped += 1
            continue

        if old_tensor.shape != new_tensor.shape:
            logger.warning(
                "extra-tensor shape mismatch at %s: %s vs %s, skipping",
                path, old_tensor.shape, new_tensor.shape,
            )
            n_skipped += 1
            continue

        if old_tensor.dtype != new_tensor.dtype:
            logger.warning(
                "extra-tensor dtype mismatch at %s: %s vs %s, skipping",
                path, old_tensor.dtype, new_tensor.dtype,
            )
            n_skipped += 1
            continue

        # Copy new value into old address (once per unique storage).
        storage_ptr = old_tensor.untyped_storage().data_ptr()
        if storage_ptr not in copied_storages:
            old_tensor.data.copy_(new_tensor)
            copied_storages.add(storage_ptr)
            n_copied += 1

        # Point the attribute back at the old tensor (every alias path).
        set_by_path(layer, path, old_tensor)

    if n_copied or n_skipped:
        logger.debug(
            "%s: extra-tensor copy-back: %d copied, %d skipped",
            layer.__class__.__name__, n_copied, n_skipped,
        )


# ---------------------------------------------------------------------------
# Internal walk
# ---------------------------------------------------------------------------

def _walk(
    path: str,
    obj: Any,
    managed_storages: set[int],
    visited: set[int],
    results: list[tuple[str, torch.Tensor]],
    depth: int,
) -> None:
    if depth > _MAX_DEPTH:
        return

    # -- Tensor leaf --------------------------------------------------------
    if isinstance(obj, torch.Tensor):
        if not obj.is_cuda or obj.numel() == 0:
            return
        if obj.untyped_storage().data_ptr() in managed_storages:
            return
        # Intentionally do NOT add tensor id to `visited` so that the same
        # tensor reachable from two attribute names is recorded twice (needed
        # for alias restoration in copy_back_extra_tensors).
        results.append((path, obj))
        return

    # -- nn.Module: stop (has its own reload cycle) -------------------------
    if isinstance(obj, nn.Module):
        return

    # -- Cycle guard for non-tensor objects ---------------------------------
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    # -- Containers ---------------------------------------------------------
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v is not None:
                _walk(f"{path}[{k!r}]", v, managed_storages, visited,
                      results, depth + 1)
        return

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if v is not None:
                _walk(f"{path}[{i}]", v, managed_storages, visited,
                      results, depth + 1)
        return

    # -- functools.partial --------------------------------------------------
    if isinstance(obj, functools.partial):
        for i, arg in enumerate(obj.args):
            _walk(f"{path}.args[{i}]", arg, managed_storages, visited,
                  results, depth + 1)
        for k, v in obj.keywords.items():
            _walk(f"{path}.keywords[{k!r}]", v, managed_storages, visited,
                  results, depth + 1)
        return

    # -- Closures -----------------------------------------------------------
    if isinstance(obj, types.FunctionType) and obj.__closure__:
        for i, cell in enumerate(obj.__closure__):
            try:
                cell_val = cell.cell_contents
            except ValueError:
                continue
            _walk(f"{path}.__closure__[{i}]", cell_val, managed_storages,
                  visited, results, depth + 1)
        return

    # -- Generic Python object (kernel instances, quant methods, …) ---------
    obj_dict = getattr(obj, "__dict__", None)
    if obj_dict is None or isinstance(obj, type):
        return
    for attr_name in list(obj_dict):
        # Skip dunder but keep single-underscore (e.g. _fc2_input_scale).
        if attr_name.startswith("__"):
            continue
        val = obj_dict.get(attr_name)
        if val is not None:
            _walk(f"{path}.{attr_name}", val, managed_storages, visited,
                  results, depth + 1)
