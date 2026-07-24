"""
Dual-Mode Graph Storage Registry for vLLM layerwise reload.

Provides O(1) explicit tensor registration and O(N) walk-based audit
for CUDA graph tensor identity preservation across model reloads.
"""

import functools
import inspect

import re
import types
import weakref
from collections.abc import Mapping
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import torch
import torch.nn as nn

from vllm.logger import init_logger

# Metadata tuple: (data_ptr, storage_offset, shape, stride, dtype, device)
_MetadataTuple = Tuple[int, int, torch.Size, Tuple[int, ...], torch.dtype, torch.device]


class SnapshotEntry(NamedTuple):
    """Immutable snapshot of a registered tensor captured before PWAL."""
    tensor: torch.Tensor
    metadata: _MetadataTuple

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PathResolutionError(Exception):
    """Raised when a registered path cannot be resolved on the layer."""

    def __init__(self, layer_type: str, path: str, detail: str = ""):
        self.layer_type = layer_type
        self.path = path
        msg = f"Path resolution failed on {layer_type}: '{path}'"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


class DriftError(Exception):
    """Raised when a registered tensor has drifted (identity/metadata mismatch)."""

    def __init__(
        self,
        layer_type: str,
        path: str,
        expected_ptr: int,
        actual_ptr: int,
        detail: str = "",
    ):
        self.layer_type = layer_type
        self.path = path
        self.expected_ptr = expected_ptr
        self.actual_ptr = actual_ptr
        msg = (
            f"Drift detected on {layer_type} at '{path}': "
            f"expected ptr=0x{expected_ptr:x}, actual ptr=0x{actual_ptr:x}"
        )
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Path Grammar: parse and resolve
# ---------------------------------------------------------------------------

# Token types: attr (.name), index ([N]), key (["str"] or ['str'])
_PATH_TOKEN_RE = re.compile(
    r"""
    \.([A-Za-z_][A-Za-z0-9_]*)    |  # .attr_name
    \[(\d+)\]                       |  # [integer_index]
    \["([^"\\]*(?:\\.[^"\\]*)*)"\]  |  # ["string_key"]
    \['([^'\\]*(?:\\.[^'\\]*)*)'\]     # ['string_key']
    """,
    re.VERBOSE,
)


def parse_path(path: str) -> List[Tuple[str, Any]]:
    """Parse a path string into a list of (kind, value) tokens.

    Returns list of:
      ("attr", name)
      ("index", int)
      ("key", str)
    """
    tokens = []
    pos = 0

    # Handle leading attribute (no dot prefix)
    leading_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", path)
    if leading_match:
        tokens.append(("attr", leading_match.group(1)))
        pos = leading_match.end()

    while pos < len(path):
        m = _PATH_TOKEN_RE.match(path, pos)
        if not m:
            raise ValueError(f"Invalid path syntax at position {pos}: '{path[pos:]}'")

        if m.group(1) is not None:
            tokens.append(("attr", m.group(1)))
        elif m.group(2) is not None:
            tokens.append(("index", int(m.group(2))))
        elif m.group(3) is not None:
            tokens.append(("key", m.group(3).replace('\\"', '"')))
        elif m.group(4) is not None:
            tokens.append(("key", m.group(4).replace("\\'", "'")))

        pos = m.end()

    if not tokens:
        raise ValueError(f"Empty path: '{path}'")

    return tokens


def get_by_path(root: Any, path: str) -> Any:
    """Resolve a path on an object tree. Raises on failure."""
    tokens = parse_path(path)
    current = root

    for i, (kind, value) in enumerate(tokens):
        if current is None:
            partial = _reconstruct_path(tokens[:i])
            raise PathResolutionError(
                type(root).__name__, path, f"None at '{partial}'"
            )

        if kind == "attr":
            # Check in __dict__ first, then PyTorch registries
            if isinstance(current, nn.Module):
                if value in current._modules:
                    current = current._modules[value]
                elif value in current._parameters:
                    current = current._parameters[value]
                elif value in current._buffers:
                    current = current._buffers[value]
                elif hasattr(current, value):
                    current = getattr(current, value)
                else:
                    raise PathResolutionError(
                        type(root).__name__, path, f"attribute '{value}' not found"
                    )
            elif hasattr(current, value):
                current = getattr(current, value)
            else:
                raise PathResolutionError(
                    type(root).__name__, path, f"attribute '{value}' not found"
                )

        elif kind == "index":
            if not hasattr(current, "__getitem__"):
                raise PathResolutionError(
                    type(root).__name__,
                    path,
                    f"not indexable at [{value}], got {type(current).__name__}",
                )
            try:
                current = current[value]
            except (IndexError, KeyError) as e:
                raise PathResolutionError(
                    type(root).__name__,
                    path,
                    f"index [{value}] failed: {e}",
                )

        elif kind == "key":
            if not hasattr(current, "__getitem__"):
                raise PathResolutionError(
                    type(root).__name__,
                    path,
                    f"not a mapping at key [\"{value}\"], got {type(current).__name__}",
                )
            if hasattr(current, "__contains__") and value not in current:
                raise PathResolutionError(
                    type(root).__name__,
                    path,
                    f"key \"{value}\" not found",
                )
            try:
                current = current[value]
            except (KeyError, IndexError) as e:
                raise PathResolutionError(
                    type(root).__name__,
                    path,
                    f"key \"{value}\" access failed: {e}",
                )

    return current


def set_by_path(root: Any, path: str, value: Any) -> None:
    """Set a value at the given path. Raises on failure."""
    tokens = parse_path(path)
    if len(tokens) < 1:
        raise ValueError("Cannot set on empty path")

    # Navigate to parent
    parent = root
    for i, (kind, tok_val) in enumerate(tokens[:-1]):
        parent = _navigate_one(parent, kind, tok_val, root, path, tokens[:i + 1])

    # Set on parent
    last_kind, last_val = tokens[-1]
    if last_kind == "attr":
        setattr(parent, last_val, value)
    elif last_kind == "index":
        if not hasattr(parent, "__setitem__"):
            raise PathResolutionError(
                type(root).__name__, path, "cannot set index on non-indexable type"
            )
        parent[last_val] = value
    elif last_kind == "key":
        if not hasattr(parent, "__setitem__"):
            raise PathResolutionError(
                type(root).__name__, path, "cannot set key on non-mapping type"
            )
        parent[last_val] = value


def _navigate_one(
    current: Any, kind: str, value: Any, root: Any, full_path: str, tokens_so_far
) -> Any:
    """Navigate one step in a path."""
    if current is None:
        raise PathResolutionError(
            type(root).__name__, full_path, "None encountered during navigation"
        )
    if kind == "attr":
        if not hasattr(current, value):
            raise PathResolutionError(
                type(root).__name__, full_path, f"attribute '{value}' not found"
            )
        return getattr(current, value)
    elif kind == "index":
        if not hasattr(current, "__getitem__"):
            raise PathResolutionError(
                type(root).__name__, full_path, f"not indexable at [{value}]"
            )
        try:
            return current[value]
        except (IndexError, KeyError) as e:
            raise PathResolutionError(
                type(root).__name__, full_path, f"index [{value}] failed: {e}"
            )
    elif kind == "key":
        if not hasattr(current, "__getitem__"):
            raise PathResolutionError(
                type(root).__name__, full_path, f"not a mapping at [\"{value}\"]"
            )
        if hasattr(current, "__contains__") and value not in current:
            raise PathResolutionError(
                type(root).__name__, full_path, f"key \"{value}\" not found"
            )
        try:
            return current[value]
        except (KeyError, IndexError) as e:
            raise PathResolutionError(
                type(root).__name__, full_path, f"key \"{value}\" failed: {e}"
            )
    raise PathResolutionError(type(root).__name__, full_path, f"unknown token kind: {kind}")


def _reconstruct_path(tokens: List[Tuple[str, Any]]) -> str:
    """Reconstruct a path string from tokens."""
    parts = []
    for i, (kind, value) in enumerate(tokens):
        if kind == "attr":
            if i == 0:
                parts.append(str(value))
            else:
                parts.append(f".{value}")
        elif kind == "index":
            parts.append(f"[{value}]")
        elif kind == "key":
            parts.append(f'["{value}"]')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tensor Metadata
# ---------------------------------------------------------------------------


def _tensor_metadata(t: torch.Tensor) -> Tuple:
    """Extract identity-relevant metadata from a tensor."""
    return (
        t.data_ptr(),
        t.storage_offset(),
        tuple(t.shape),
        tuple(t.stride()),
        t.dtype,
        t.device,
    )


def _view_aware_key(t: torch.Tensor) -> Tuple:
    """View-aware copy key for deduplicating disjoint slices of same storage."""
    return (
        t.untyped_storage().data_ptr(),
        t.storage_offset(),
        tuple(t.shape),
        tuple(t.stride()),
    )


# ---------------------------------------------------------------------------
# Walk Audit
# ---------------------------------------------------------------------------

# Attributes to skip during walk
_SKIP_ATTRS = frozenset({
    "__class__", "__dict__", "__module__", "__spec__", "__weakref__",
    "__slots__", "__getattribute__", "__subclasshook__",
    "_backward_hooks", "_forward_hooks", "_forward_pre_hooks",
    "_state_dict_hooks", "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "training",
})

# Types that are terminal leaves (not traversed into)
_TERMINAL_TYPES = (
    torch.Tensor,
    torch.nn.Parameter,
    torch.device,
    torch.dtype,
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    weakref.ref,
)

DEFAULT_MAX_DEPTH = 8
MAX_OBJECTS = 50_000


def _is_discoverable_attr(name: str) -> bool:
    """Check if an attribute should be traversed during walk audit."""
    if name in _SKIP_ATTRS:
        return False
    if name.startswith("__") and name.endswith("__"):
        return False
    if name.startswith("_"):
        return False
    return True


def _iter_children(obj: Any) -> List[Tuple[str, Any]]:
    """Enumerate traversable children of an object."""
    children = []

    # Special-case sequential/indexed containers BEFORE generic nn.Module
    # so they emit bracket indices [0], [1], ... instead of .0, .1, ...
    if isinstance(obj, (nn.ModuleList, nn.ParameterList, nn.Sequential)):
        for idx, item in enumerate(obj):
            if item is not None:
                children.append((f"[{idx}]", item))
        return children

    # Dict-like containers emit key paths
    if isinstance(obj, (nn.ModuleDict, nn.ParameterDict)):
        for key in sorted(obj.keys()):
            value = obj[key]
            if value is not None:
                children.append((f'["{key}"]', value))
        return children

    if isinstance(obj, nn.Module):
        # PyTorch registries first
        for name, child in obj._modules.items():
            if child is not None:
                children.append((f".{name}", child))
        for name, param in obj._parameters.items():
            if param is not None:
                children.append((f".{name}", param))
        for name, buf in obj._buffers.items():
            if buf is not None:
                children.append((f".{name}", buf))
        # Instance attrs not in registries
        for name in sorted(vars(obj).keys()):
            if name in ("_modules", "_parameters", "_buffers"):
                continue
            if not _is_discoverable_attr(name):
                continue
            value = vars(obj)[name]
            if value is None:
                continue
            children.append((f".{name}", value))
        return children

    if isinstance(obj, (list, tuple, nn.ModuleList, nn.ParameterList)):
        for idx, item in enumerate(obj):
            if item is not None:
                children.append((f"[{idx}]", item))
        return children

    if isinstance(obj, (dict, nn.ModuleDict, nn.ParameterDict)):
        for key in sorted(obj.keys()):
            value = obj[key]
            if value is not None:
                children.append((f'["{key}"]', value))
        return children

    # Generic object with __dict__
    if hasattr(obj, "__dict__"):
        for name in sorted(vars(obj).keys()):
            if not _is_discoverable_attr(name):
                continue
            value = vars(obj)[name]
            if value is None:
                continue
            children.append((f".{name}", value))

    return children


def walk_audit(
    layer: nn.Module,
    registered_paths: Optional[Set[str]] = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> List[Tuple[str, torch.Tensor]]:
    """Discover unregistered tensors in a layer via cycle-safe recursive traversal.

    Returns a list of (path, tensor) for tensors NOT in registered_paths.
    Deduplicates by path (not storage pointer) to preserve aliases.
    Uses deterministic walk order (sorted keys, stable attributes).
    """
    if registered_paths is None:
        registered_paths = set()

    discovered: List[Tuple[str, torch.Tensor]] = []
    seen_paths: Set[str] = set()
    seen_objects: Set[int] = set()
    object_count = 0

    # Get registered parameters and buffers to exclude
    registered_params = set()
    for _, p in layer.named_parameters():
        registered_params.add(id(p))
    for _, b in layer.named_buffers():
        registered_params.add(id(b))

    def _discover_closure_in_walk(obj: Any, path_prefix: str):
        """Discover tensors inside closure cells and partial keywords."""
        # For bound methods, check __func__.__closure__
        func = getattr(obj, "__func__", obj)
        closure = getattr(func, "__closure__", None)
        if closure is not None:
            for i, cell in enumerate(closure):
                try:
                    value = cell.cell_contents
                except ValueError:
                    continue
                if isinstance(value, torch.Tensor):
                    path = f"{path_prefix}.__closure__[{i}]".lstrip(".")
                    if path not in registered_paths and path not in seen_paths:
                        seen_paths.add(path)
                        discovered.append((path, value))
        # Handle functools.partial
        if isinstance(obj, functools.partial):
            for key, value in obj.keywords.items():
                if isinstance(value, torch.Tensor):
                    path = f'{path_prefix}.keywords["{key}"]'.lstrip(".")
                    if path not in registered_paths and path not in seen_paths:
                        seen_paths.add(path)
                        discovered.append((path, value))
            # Check underlying func
            if hasattr(obj.func, "__closure__") and obj.func.__closure__:
                _discover_closure_in_walk(obj.func, f"{path_prefix}.func")

    def _walk(obj: Any, path_prefix: str, depth: int):
        nonlocal object_count

        if depth > max_depth:
            return
        if object_count >= MAX_OBJECTS:
            return

        # Check if this object is a tensor leaf first (before cycle detection)
        # Tensors are NOT added to seen_objects so the same tensor at multiple
        # paths is reported for each path (deduplicate by path, not pointer).
        if isinstance(obj, (torch.Tensor, nn.Parameter)):
            # Skip registered params/buffers
            if id(obj) in registered_params:
                return
            # Build the path (strip leading dot if present)
            path = path_prefix.lstrip(".")
            if path and path not in registered_paths and path not in seen_paths:
                seen_paths.add(path)
                discovered.append((path, obj))
            return

        obj_id = id(obj)
        if obj_id in seen_objects:
            return
        seen_objects.add(obj_id)
        object_count += 1

        # Terminal types - don't traverse (but callables are handled below for closure discovery)
        if isinstance(obj, _TERMINAL_TYPES):
            # For functions/methods, discover closure tensors
            if isinstance(obj, (types.FunctionType, types.MethodType)):
                _discover_closure_in_walk(obj, path_prefix)
            return

        # functools.partial - discover closure tensors in keywords and underlying func
        if isinstance(obj, functools.partial):
            _discover_closure_in_walk(obj, path_prefix)
            return

        # Skip CUDA-related objects
        if hasattr(torch, "cuda"):
            cuda_types = []
            for tname in ("Stream", "Event", "CUDAGraph"):
                t = getattr(torch.cuda, tname, None)
                if t is not None:
                    cuda_types.append(t)
            if cuda_types and isinstance(obj, tuple(cuda_types)):
                return

        # Traverse children
        for child_suffix, child in _iter_children(obj):
            child_path = path_prefix + child_suffix
            _walk(child, child_path, depth + 1)

    # Add root layer to seen to prevent self-reference cycles
    seen_objects.add(id(layer))
    object_count += 1

    # Start walk from layer's children (not the layer itself as a tensor)
    for child_suffix, child in _iter_children(layer):
        _walk(child, child_suffix, 1)

    return discovered


# ---------------------------------------------------------------------------
# GraphStorageRegistry
# ---------------------------------------------------------------------------


class GraphStorageRegistry:
    """Registry for graph-visible tensors that need identity preservation across reloads.

    Stores paths (not tensor objects) keyed by layer, enabling correct lifecycle
    management across multiple reloads.
    """

    def __init__(self):
        # WeakKeyDictionary: layer -> {path: metadata_tuple}
        self._registry: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def register_graph_storage(
        self, layer: nn.Module, path: str, tensor: torch.Tensor
    ) -> None:
        """Register a graph-visible tensor by path.

        Args:
            layer: The owning nn.Module
            path: Path string following the grammar (e.g., "quant_method.moe_kernel.fused_experts.b_strides1")
            tensor: The tensor to register (metadata is captured, not the object)
        """
        # Validate path grammar
        tokens = parse_path(path)

        # Reject closure/partial paths — these are handled by walk + closure copy-back
        for kind, val in tokens:
            if kind == "attr" and val == "__closure__":
                raise ValueError(
                    f"Cannot register closure path '{path}': closure tensors "
                    f"are handled by the walk + closure copy-back mechanism, "
                    f"not the explicit registry"
                )

        # Validate that the path resolves to the passed tensor with matching metadata
        layer_type = type(layer).__name__
        resolved = get_by_path(layer, path)
        if not isinstance(resolved, (torch.Tensor, nn.Parameter)):
            raise TypeError(
                f"Registered path '{path}' on {layer_type} resolved to "
                f"{type(resolved).__name__}, expected Tensor"
            )
        resolved_meta = _tensor_metadata(resolved)
        tensor_meta = _tensor_metadata(tensor)
        if resolved_meta[0] != tensor_meta[0]:
            raise DriftError(
                layer_type, path, tensor_meta[0], resolved_meta[0],
                "path resolves to a different tensor than the one being registered"
            )
        if resolved_meta[1:] != tensor_meta[1:]:
            raise DriftError(
                layer_type, path, tensor_meta[0], resolved_meta[0],
                f"path resolves to same storage but different metadata: "
                f"registered={tensor_meta[1:]}, resolved={resolved_meta[1:]}"
            )

        if layer not in self._registry:
            self._registry[layer] = {}

        # Store path -> metadata (not tensor ref)
        self._registry[layer][path] = _tensor_metadata(tensor)

    def get_registered_paths(self, layer: nn.Module) -> Set[str]:
        """Get all registered paths for a layer."""
        if layer not in self._registry:
            return set()
        return set(self._registry[layer].keys())

    def snapshot(self, layer: nn.Module) -> Dict[str, SnapshotEntry]:
        """Create a transient snapshot of registered tensors for a reload cycle.

        Returns a dict of {path: SnapshotEntry(tensor, metadata)} for use
        during copy-back. The metadata is captured immutably at snapshot time
        so that copy_back_registered() can validate against it even after
        re-registration during PWAL overwrites registry state.

        Fail-closed: raises if any registered path cannot be resolved.
        """
        snapshot: Dict[str, SnapshotEntry] = {}
        if layer not in self._registry:
            return snapshot

        layer_type = type(layer).__name__
        for path, reg_meta in self._registry[layer].items():
            tensor = get_by_path(layer, path)
            if not isinstance(tensor, (torch.Tensor, nn.Parameter)):
                raise TypeError(
                    f"Registered path '{path}' on {layer_type} resolved to "
                    f"{type(tensor).__name__}, expected Tensor"
                )
            # Validate that the tensor matches registration metadata
            current_meta = _tensor_metadata(tensor)
            if current_meta[0] != reg_meta[0]:
                raise DriftError(
                    layer_type, path, reg_meta[0], current_meta[0],
                    "tensor pointer drifted since registration"
                )
            # Check storage_offset, shape, stride, dtype, device
            if current_meta[1:] != reg_meta[1:]:
                raise DriftError(
                    layer_type, path, reg_meta[0], current_meta[0],
                    f"metadata drifted since registration: "
                    f"registered={reg_meta[1:]}, current={current_meta[1:]}"
                )
            # Capture both tensor and immutable metadata at snapshot time
            snapshot[path] = SnapshotEntry(tensor=tensor, metadata=current_meta)

        return snapshot

    def copy_back_registered(
        self,
        layer: nn.Module,
        old_snapshot: Dict[str, SnapshotEntry],
    ) -> int:
        """Restore tensor identity for all registered paths after PWAL.

        Iterates the snapshot entries (not the registry), so that re-registration
        during PWAL does not corrupt the active copy-back cycle. Validates the
        current post-PWAL tensor's metadata against the immutable snapshot
        metadata — only data_ptr is allowed to differ.

        Fail-closed: raises on any resolution or compatibility error.

        Args:
            layer: The layer after PWAL has replaced weights
            old_snapshot: Snapshot from before PWAL with SnapshotEntry references

        Returns:
            Number of tensors successfully copied back.
        """
        if not old_snapshot:
            return 0

        copied = 0
        layer_type = type(layer).__name__

        for path, entry in old_snapshot.items():
            old_tensor = entry.tensor
            snap_meta = entry.metadata

            # Resolve current tensor at path
            try:
                current = get_by_path(layer, path)
            except PathResolutionError:
                raise PathResolutionError(
                    layer_type, path, "path not resolvable after PWAL"
                )

            if not isinstance(current, (torch.Tensor, nn.Parameter)):
                raise TypeError(
                    f"Path '{path}' on {layer_type} resolved to "
                    f"{type(current).__name__}, expected Tensor"
                )

            # Validate current tensor metadata against snapshot metadata.
            # All fields except data_ptr must match the snapshot.
            current_meta = _tensor_metadata(current)
            if current_meta[1:] != snap_meta[1:]:
                detail = (
                    f"metadata mismatch after PWAL: "
                    f"current=(offset={current_meta[1]}, shape={current_meta[2]}, "
                    f"stride={current_meta[3]}, dtype={current_meta[4]}, "
                    f"device={current_meta[5]}), "
                    f"snapshot=(offset={snap_meta[1]}, shape={snap_meta[2]}, "
                    f"stride={snap_meta[3]}, dtype={snap_meta[4]}, "
                    f"device={snap_meta[5]})"
                )
                raise DriftError(
                    layer_type, path, snap_meta[0], current_meta[0], detail
                )

            # Skip copy if pointer already matches (no PWAL replacement occurred)
            if current.data_ptr() == old_tensor.data_ptr():
                copied += 1
                continue

            # Copy new (post-PWAL) data into old tensor's storage
            # This preserves the CUDA graph's pointer to old_tensor's storage
            # while updating it with fresh checkpoint values
            old_tensor.data.copy_(current.data)

            # Restore the old tensor at the path (CUDA graph references this storage)
            set_by_path(layer, path, old_tensor)
            copied += 1

        # Update registry metadata to reflect restored tensors (for next cycle)
        if layer in self._registry:
            for path in old_snapshot:
                if path in self._registry[layer]:
                    restored = get_by_path(layer, path)
                    self._registry[layer][path] = _tensor_metadata(restored)

        return copied

    def walk_audit(
        self, layer: nn.Module, max_depth: int = DEFAULT_MAX_DEPTH
    ) -> List[Tuple[str, torch.Tensor]]:
        """Discover unregistered graph-visible tensors via cycle-safe walk.

        Returns list of (path, tensor) for tensors not in this layer's registry.
        """
        registered = self.get_registered_paths(layer)
        return walk_audit(layer, registered, max_depth)

    def clear(self, layer: Optional[nn.Module] = None) -> None:
        """Clear registry entries. If layer is None, clear all."""
        if layer is None:
            self._registry.clear()
        elif layer in self._registry:
            del self._registry[layer]

    def __contains__(self, layer: nn.Module) -> bool:
        return layer in self._registry

    def __len__(self) -> int:
        return len(self._registry)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_registry = GraphStorageRegistry()


def get_global_registry() -> GraphStorageRegistry:
    """Get the global GraphStorageRegistry singleton."""
    return _global_registry


def register_graph_storage(
    layer: nn.Module, path: str, tensor: torch.Tensor
) -> None:
    """Register a graph-visible tensor in the global registry."""
    _global_registry.register_graph_storage(layer, path, tensor)
