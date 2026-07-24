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
import re
import types
from typing import Any

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# Regex for parsing path segments emitted by _walk().
# Matches: .attr_name | [integer] | ['string'] | ["string"]
_PATH_TOKEN_RE = re.compile(
    r"""
    \.([A-Za-z_][A-Za-z0-9_]*)    |  # .attr_name
    \[(\d+)\]                       |  # [integer_index]
    \["([^"\\]*(?:\\.[^"\\]*)*)"\]  |  # ["string_key"]
    \['([^'\\]*(?:\\.[^'\\]*)*)'\]     # ['string_key']
    """,
    re.VERBOSE,
)

# Maximum recursion depth when walking nested Python objects.
# Real-world deepest path is ~5 (quant_method.moe_kernel.fused_experts.xxx).
_MAX_DEPTH = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_extra_tensors(
    layer: nn.Module,
    exclude_paths: set[str] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Snapshot every CUDA tensor reachable from *layer* that is **not**
    registered as a parameter or buffer.

    Returns a list of ``(dotted_path, tensor)`` pairs.  Paths that share the
    same underlying storage are **all** recorded (needed so that
    ``set_by_path`` can fix every alias).  Tensor objects are **not** added
    to the visited set so that the same tensor found via two attribute names
    (e.g. ``ab_strides1`` and ``c_strides2``) produces two entries.

    Args:
        layer: Module to walk.
        exclude_paths: Set of paths already managed by the registry.
            These paths are excluded from the results to avoid double
            copy-back (registry handles them with fail-closed semantics).
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

    # Filter out paths managed by the registry
    if exclude_paths:
        results = [(p, t) for p, t in results if p not in exclude_paths]

    return results


def _parse_path(path: str) -> list[tuple[str, Any]]:
    """Parse a path string into (kind, value) tokens.

    Handles formats emitted by _walk():
      - Leading attribute: ``attr_name``
      - Dot attribute: ``.attr_name``
      - Integer index: ``[N]``
      - String key: ``['key']`` or ``["key"]``

    Returns list of ("attr", str) | ("index", int) | ("key", str).
    """
    tokens: list[tuple[str, Any]] = []
    pos = 0

    # Leading attribute (no dot prefix)
    leading = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", path)
    if leading:
        tokens.append(("attr", leading.group(1)))
        pos = leading.end()

    while pos < len(path):
        m = _PATH_TOKEN_RE.match(path, pos)
        if not m:
            # Fallback: treat remainder as unparseable
            return []

        if m.group(1) is not None:
            tokens.append(("attr", m.group(1)))
        elif m.group(2) is not None:
            tokens.append(("index", int(m.group(2))))
        elif m.group(3) is not None:
            tokens.append(("key", m.group(3)))
        elif m.group(4) is not None:
            tokens.append(("key", m.group(4)))

        pos = m.end()

    return tokens


def _navigate_token(cur: Any, kind: str, value: Any) -> tuple[bool, Any]:
    """Navigate one token. Returns (success, result)."""
    if kind == "attr":
        if isinstance(cur, functools.partial) and value == "keywords":
            return (True, cur.keywords)
        if isinstance(cur, functools.partial) and value == "args":
            return (True, cur.args)
        if isinstance(cur, types.FunctionType) and value == "__closure__":
            closure = cur.__closure__
            return (True, closure) if closure else (False, None)
        result = getattr(cur, value, None)
        if result is None and not hasattr(cur, value):
            return (False, None)
        return (True, result)
    elif kind == "index":
        try:
            if hasattr(cur, "cell_contents"):
                # cur is a cell — reading cell_contents already happened
                # via __closure__ access; this is a list/tuple index
                pass
            result = cur[value]
            return (True, result)
        except (IndexError, KeyError, TypeError):
            return (False, None)
    elif kind == "key":
        try:
            result = cur[value]
            return (True, result)
        except (KeyError, IndexError, TypeError):
            return (False, None)
    return (False, None)


def resolve_path(root: Any, path: str) -> torch.Tensor | None:
    """Navigate *root* along *path* and return the tensor found there,
    or ``None`` if the path is broken.

    Supports the full path grammar emitted by _walk():
    dot attrs, [N] indices, ['key'] dict keys, .__closure__[N] cells,
    .args[N], .keywords['key'].
    """
    tokens = _parse_path(path)
    if not tokens:
        return None

    cur = root
    for kind, value in tokens:
        if cur is None:
            return None

        # Special: closure cell — index into __closure__ tuple returns cell,
        # then we need cell_contents
        if (isinstance(cur, tuple) and kind == "index"
                and len(cur) > value
                and hasattr(cur[value], "cell_contents")):
            try:
                cur = cur[value].cell_contents
            except ValueError:
                return None
            continue

        ok, cur = _navigate_token(cur, kind, value)
        if not ok:
            return None

    return cur if isinstance(cur, torch.Tensor) else None


def set_by_path(root: Any, path: str, value: torch.Tensor) -> bool:
    """Set the value at the end of *path* to *value*.
    Returns ``True`` on success.

    Supports:
    - Dot attrs: setattr
    - [N] on mutable sequences: __setitem__
    - ['key'] on dicts: __setitem__
    - .__closure__[N]: cell mutation via ctypes
    - .keywords['key']: partial.keywords mutation
    - .args[N]: immutable (warns and returns False)
    """
    tokens = _parse_path(path)
    if not tokens:
        return False

    # Navigate to parent (all tokens except last)
    cur = root
    parent_tokens = tokens[:-1]
    for i, (kind, tok_value) in enumerate(parent_tokens):
        if cur is None:
            return False

        # Special: closure cell navigation
        if (isinstance(cur, tuple) and kind == "index"
                and len(cur) > tok_value
                and hasattr(cur[tok_value], "cell_contents")):
            try:
                cur = cur[tok_value].cell_contents
            except ValueError:
                return False
            continue

        ok, cur = _navigate_token(cur, kind, tok_value)
        if not ok:
            return False

    if cur is None:
        return False

    # Set on parent using last token
    last_kind, last_val = tokens[-1]

    # Special case: setting into closure cell (__closure__[N])
    # Parent is a tuple of cells; we need to mutate the cell
    if (isinstance(cur, tuple) and last_kind == "index"
            and len(cur) > last_val
            and hasattr(cur[last_val], "cell_contents")):
        import ctypes
        cell = cur[last_val]
        try:
            # Use CPython internals to mutate cell contents
            # PyCell_Set equivalent
            ctypes.pythonapi.PyCell_Set(
                ctypes.py_object(cell), ctypes.py_object(value))
            return True
        except (SystemError, OSError):
            return False

    # Special case: partial.args[N] — immutable tuple
    if (isinstance(cur, tuple) and last_kind == "index"
            and _is_partial_args(root, tokens)):
        logger.warning(
            "Cannot restore tensor at '%s': partial.args is an immutable "
            "tuple. Consider using partial.keywords instead.", path)
        return False

    if last_kind == "attr":
        try:
            setattr(cur, last_val, value)
            return True
        except (AttributeError, TypeError):
            return False
    elif last_kind == "index":
        try:
            cur[last_val] = value
            return True
        except (IndexError, TypeError):
            return False
    elif last_kind == "key":
        try:
            cur[last_val] = value
            return True
        except (KeyError, TypeError):
            return False

    return False


def _is_partial_args(root: Any, tokens: list[tuple[str, Any]]) -> bool:
    """Check if the path navigates through partial.args.

    Handles both direct (fn.args[0]) and closure-nested
    (fn.__closure__[0].args[0]) partial ownership.
    """
    for i, (kind, val) in enumerate(tokens):
        if kind == "attr" and val == "args" and i > 0:
            # Navigate to the object before "args", handling closure cells
            cur = root
            for k, v in tokens[:i]:
                if cur is None:
                    break
                # Closure cell dereference (same logic as resolve_path)
                if (isinstance(cur, tuple) and k == "index"
                        and len(cur) > v
                        and hasattr(cur[v], "cell_contents")):
                    try:
                        cur = cur[v].cell_contents
                    except ValueError:
                        cur = None
                        break
                    continue
                ok, cur = _navigate_token(cur, k, v)
                if not ok:
                    cur = None
                    break
            if cur is not None and isinstance(cur, functools.partial):
                return True
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
            logger.warning(
                "extra-tensor path broken after reload: %s "
                "(object is gone or replaced by non-tensor)", path)
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
        # Best-effort: if copy fails, warn and skip (walk tensors are
        # unregistered, so we must not abort the reload path).
        storage_ptr = old_tensor.untyped_storage().data_ptr()
        if storage_ptr not in copied_storages:
            try:
                old_tensor.data.copy_(new_tensor)
            except RuntimeError as e:
                logger.warning(
                    "extra-tensor copy failed at %s: %s, skipping",
                    path, e,
                )
                n_skipped += 1
                continue
            copied_storages.add(storage_ptr)
            n_copied += 1

        # Point the attribute back at the old tensor (every alias path).
        if not set_by_path(layer, path, old_tensor):
            logger.warning(
                "extra-tensor restore failed at %s: "
                "set_by_path returned False (path may be immutable)", path)
            n_skipped += 1

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
