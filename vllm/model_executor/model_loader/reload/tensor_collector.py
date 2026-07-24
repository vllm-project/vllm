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

    if results:
        logger.info(
            "%s: found %d unmanaged CUDA tensor(s) to preserve:",
            layer.__class__.__name__, len(results),
        )
        for path, t in results:
            logger.info(
                "  %s  shape=%s dtype=%s ptr=%s",
                path, list(t.shape), t.dtype, hex(t.data_ptr()),
            )

    return results


def _parse_segments(path: str) -> list[tuple[str, str | int | None]]:
    """Split a recorded path into (action, key) segments.

    Handles mixed dot-access and bracket-access paths produced by ``_walk``:
      ``"quant_method.cache['scale']"``  →  [("attr","quant_method"),
                                              ("attr","cache"),
                                              ("item","scale")]
      ``"fn.args[0]"``                   →  [("attr","fn"), ("attr","args"),
                                              ("index",0)]
      ``"fn.__closure__[2]"``            →  [("attr","fn"),
                                              ("attr","__closure__"),
                                              ("index",2)]
    """
    import re
    segs: list[tuple[str, str | int | None]] = []
    for tok in re.split(r"(?<!\[)\.", path):  # split on dots not inside []
        # tok might be  "args[0]"  or  "cache['scale']"  or plain "name"
        m = re.match(r"^([^\[]+)\[(.+)\]$", tok)
        if m:
            segs.append(("attr", m.group(1)))
            inner = m.group(2)
            # Detect int index vs string key (strip quotes)
            stripped = inner.strip("'\"")
            if inner.isdigit():
                segs.append(("index", int(inner)))
            elif inner != stripped:
                segs.append(("item", stripped))
            else:
                segs.append(("item", inner))
        else:
            segs.append(("attr", tok))
    return segs


def _navigate(root: Any, segments: list[tuple[str, str | int | None]],
              ) -> Any | None:
    """Walk *root* along parsed segments, returning the final object."""
    cur = root
    for action, key in segments:
        try:
            if action == "attr":
                cur = getattr(cur, key)
            elif action == "index":
                cur = cur[key]
            elif action == "item":
                cur = cur[key]
            else:
                return None
        except (AttributeError, TypeError, IndexError, KeyError):
            return None
    return cur


def _set_final(root: Any,
               segments: list[tuple[str, str | int | None]],
               value: torch.Tensor) -> bool:
    """Navigate to the parent and set the final segment to *value*."""
    parent = _navigate(root, segments[:-1]) if len(segments) > 1 else root
    if parent is None:
        return False
    action, key = segments[-1]
    try:
        if action == "attr":
            setattr(parent, key, value)
        elif action in ("index", "item"):
            parent[key] = value
        else:
            return False
        return True
    except (AttributeError, TypeError, IndexError, KeyError):
        return False


def resolve_path(root: Any, path: str) -> torch.Tensor | None:
    """Navigate *root* along *path* and return the tensor found there,
    or ``None`` if the path is broken.  Supports both dot-access and
    bracket-access segments (dicts, lists, closures, partials)."""
    result = _navigate(root, _parse_segments(path))
    return result if isinstance(result, torch.Tensor) else None


def set_by_path(root: Any, path: str, value: torch.Tensor) -> bool:
    """Set the value at the end of *path*.
    Returns ``True`` on success.  Supports bracket-access segments."""
    return _set_final(root, _parse_segments(path), value)


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
            logger.warning(
                "extra-tensor path unresolvable after reload: %s "
                "(tensor was recorded but cannot be restored)", path)
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
        if not set_by_path(layer, path, old_tensor):
            logger.warning(
                "extra-tensor set_by_path failed for: %s "
                "(value was copied but attribute not restored)", path)

    if n_copied or n_skipped:
        logger.warning(
            "%s: repaired %d unmanaged tensor(s) during reload "
            "(%d skipped). These tensors are not registered as "
            "parameters or buffers and should be migrated to the "
            "tensor registry (#48478).",
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
