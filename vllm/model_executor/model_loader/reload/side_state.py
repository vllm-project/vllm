# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep tensors held outside a layer's parameters and buffers on their
startup storage across a reload.

`process_weights_after_loading` often stores tensors on objects hanging off
the layer rather than registering them: `quant_method.moe_quant_config`,
`quant_method.moe_kernel`, experts objects, plain layer attributes.  A reload
reruns it on freshly materialized tensors, so those objects end up holding the
reload's temporaries, views of them, or freshly derived tensors, while CUDA
graphs captured at startup keep reading the startup storage.

Instead of rewriting the references, the tensors they hold are moved onto
startup storage in place (`Tensor.set_`), which every holder sees:
- each temporary is aliased onto its kernel tensor once the values are copied;
- any other tensor found where a different startup tensor used to be (same
  attribute path, shape, dtype and device) is copied into it and aliased.
"""

import torch

__all__ = ["alias_onto", "collect_side_tensors", "pin_side_tensors"]

Path = tuple[str, ...]

_MAX_DEPTH = 6
_SKIP_MODULE_PREFIXES = ("vllm.config", "transformers", "torch", "threading", "logging")
_SKIP_LAYER_ATTRS = {"_parameters", "_buffers", "_modules", "training"}


def _children(o: object) -> list[tuple[str, object]]:
    if isinstance(o, (list, tuple)):
        return [(f"[{i}]", v) for i, v in enumerate(o)]
    if isinstance(o, dict):
        return [(f"[{k!r}]", v) for k, v in o.items()]
    if (type(o).__module__ or "").startswith(_SKIP_MODULE_PREFIXES):
        return []
    d = getattr(o, "__dict__", None)
    out = [(f".{k}", v) for k, v in d.items()] if isinstance(d, dict) else []
    for k in getattr(type(o), "__slots__", ()):
        if isinstance(k, str) and hasattr(o, k):
            out.append((f".{k}", getattr(o, k)))
    return out


def collect_side_tensors(layer: torch.nn.Module) -> dict[Path, torch.Tensor]:
    """Every tensor reachable from the layer's non-module attributes, by path."""
    found: dict[Path, torch.Tensor] = {}
    seen: set[int] = set()

    def walk(o: object, path: Path) -> None:
        if isinstance(o, torch.Tensor):
            found[path] = o
            return
        if (
            len(path) > _MAX_DEPTH
            or o is None
            or isinstance(o, (str, bytes, int, float, bool, torch.nn.Module, type))
            or id(o) in seen
        ):
            return
        seen.add(id(o))
        for key, child in _children(o):
            walk(child, path + (key,))

    for name, value in vars(layer).items():
        if name in _SKIP_LAYER_ATTRS or name.startswith("_forward") or "hooks" in name:
            continue
        walk(value, (f".{name}",))
    return found


def alias_onto(t: torch.Tensor, startup: torch.Tensor) -> bool:
    """Point `t` at `startup`'s storage; every reference to `t` follows.
    Returns False where torch refuses (tensor subclasses, inference tensors)."""
    try:
        t.set_(
            startup.untyped_storage(),
            startup.storage_offset(),
            startup.size(),
            startup.stride(),
        )
    except (RuntimeError, NotImplementedError):
        return False
    return True


@torch.no_grad()
def pin_side_tensors(
    layer: torch.nn.Module, before: dict[Path, torch.Tensor]
) -> list[str]:
    """Move side tensors back onto startup storage (see module docstring).

    Args:
        layer: the reloaded layer, its temporaries already aliased onto kernels
        before: `collect_side_tensors(layer)` taken before the rerun

    Returns:
        paths whose tensor moved but could not be pinned

    """
    unpinned = []
    for path, t in collect_side_tensors(layer).items():
        old = before.get(path)
        if old is None or old is t or t.device.type == "meta":
            continue
        if old.data_ptr() == t.data_ptr() and old.stride() == t.stride():
            continue
        if (old.shape, old.dtype, old.device) == (t.shape, t.dtype, t.device):
            try:
                old.copy_(t)  # raises if `old` overlaps itself (expanded view)
            except RuntimeError:
                pass
            else:
                if alias_onto(t, old):
                    continue
        unpinned.append("".join(path))
    return unpinned
