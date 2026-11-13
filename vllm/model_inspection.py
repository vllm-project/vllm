# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model inspection utilities for vLLM."""

from collections import defaultdict

import torch.nn as nn


def _get_module_info(module: nn.Module) -> str:
    """Get info string for a module."""
    class_name = type(module).__name__
    parts = []

    # Add quant_method if present
    quant_method = getattr(module, "quant_method", None)
    if quant_method is not None:
        quant_name = type(quant_method).__name__
        # For CompressedTensors, show the underlying scheme instead
        scheme = getattr(module, "scheme", None)
        if scheme is not None:
            quant_name = type(scheme).__name__
        # Skip unquantized methods
        if "Unquantized" not in quant_name:
            parts.append(f"quant={quant_name}")

    # If module has extra_repr, use it
    if hasattr(module, "extra_repr"):
        parts.append(module.extra_repr().replace("\n", ""))

    if parts:
        return f"{class_name}({', '.join(parts)})"

    # For unknown modules, use the default PyTorch repr
    return str(module)


def _get_child_signature(child: nn.Module) -> str:
    """Get a signature for a child module to detect duplicates."""
    lines = []
    for name, submodule in child.named_modules():
        lines.append(f"{name}:{_get_module_info(submodule)}")
    return "\n".join(lines)


def _format_index_ranges(indices: list[int]) -> str:
    """Format indices into range notation (e.g., [0,1,2,4,5,6] -> '0-2, 4-6')."""
    indices = sorted(indices)
    ranges = []
    start = end = indices[0]

    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append(str(start) if start == end else f"{start}-{end}")
            start = end = idx

    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ", ".join(ranges)


def _format_module_tree(
    module: nn.Module,
    name: str = "",
    indent: int = 0,
) -> list[str]:
    """Format a module tree with indentation, grouping identical layers.

    Produces output like:
        (layers): ModuleList(
          (0-27, 29-47): 47 x LlamaDecoderLayer(
            ...
          )
          (28, 48): 2 x DifferentDecoderLayer(
            ...
          )
        )
    """
    lines = []
    prefix = "  " * indent
    children = list(module.named_children())

    # Leaf node - just output the module info
    if not children:
        info = _get_module_info(module)
        lines.append(f"{prefix}({name}): {info}" if name else f"{prefix}{info}")
        return lines

    # Non-leaf node - output opening line and recurse into children
    info = _get_module_info(module)
    lines.append(f"{prefix}({name}): {info}(" if name else f"{prefix}{info}(")

    # Separate numbered children (e.g., "0", "1") from named ones (e.g., "norm")
    numbered: list[tuple[int, nn.Module]] = []
    non_numbered: list[tuple[str, nn.Module]] = []
    for child_name, child_module in children:
        try:
            numbered.append((int(child_name), child_module))
        except ValueError:
            non_numbered.append((child_name, child_module))

    # Group numbered children by structure signature to collapse identical layers
    # e.g., layers 0-27 and 29-47 with same structure become "(0-27, 29-47): 47 x"
    if numbered:
        sig_to_group: dict[str, list[tuple[int, nn.Module]]] = {}
        for idx, child_module in numbered:
            sig = _get_child_signature(child_module)
            sig_to_group.setdefault(sig, []).append((idx, child_module))

        # Output groups sorted by first index
        for group in sorted(sig_to_group.values(), key=lambda g: g[0][0]):
            indices = [idx for idx, _ in group]
            representative = group[0][1]
            child_lines = _format_module_tree(representative, "", indent + 1)
            first_line = child_lines[0].lstrip()
            child_prefix = "  " * (indent + 1)

            if len(indices) > 1:
                range_str = _format_index_ranges(indices)
                child_lines[0] = (
                    f"{child_prefix}({range_str}): {len(indices)} x {first_line}"
                )
            else:
                child_lines[0] = f"{child_prefix}({indices[0]}): {first_line}"
            lines.extend(child_lines)

    # Output non-numbered children (e.g., "embed_tokens", "norm")
    for child_name, child_module in non_numbered:
        lines.extend(_format_module_tree(child_module, child_name, indent + 1))

    lines.append(f"{prefix})")
    return lines


def format_model_inspection(model: nn.Module) -> str:
    """Format a model into a transformers-style hierarchical string."""
    return "\n".join(_format_module_tree(model))


def _dtype_name(dtype: object) -> str:
    return str(dtype).removeprefix("torch.") if dtype is not None else "none"


def _describe_kernel(method: object) -> str | None:
    """The kernel a quant method (or compressed-tensors scheme) selected,
    whatever attribute it keeps it in, with the precision it runs if known."""
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel

    for value in vars(method).values():
        if isinstance(value, FusedMoEKernel):
            experts = value.fused_experts
            quant = experts.quant_config
            if quant.weight_quant_dtype is None and quant.quant_dtype is None:
                return type(experts).__name__
            return (
                f"{type(experts).__name__} "
                f"[w={_dtype_name(quant.weight_quant_dtype)}, "
                f"a={_dtype_name(quant.quant_dtype)}]"
            )
        if type(value).__module__.startswith("vllm.model_executor.kernels.linear"):
            return type(value).__name__
    return None


def _describe_backend(module: nn.Module) -> tuple[str, str] | None:
    """`(kind, backend)` for a layer that selects a backend, else None."""
    from vllm.model_executor.layers.attention import MMEncoderAttention
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    if isinstance(module, AttentionLayerBase):
        backend = module.get_attn_backend().get_name()
        kv_cache_dtype = getattr(module, "kv_cache_dtype", "auto")
        if kv_cache_dtype != "auto":
            backend += f" [kv_cache={kv_cache_dtype}]"
        return "attention", backend
    if isinstance(module, MMEncoderAttention):
        return "attention", module.attn_backend.name

    quant_method = getattr(module, "quant_method", None)
    if quant_method is None:
        return None
    kind = "moe" if isinstance(quant_method, FusedMoEMethodBase) else "linear"
    # Compressed-tensors keeps the per-layer scheme, and its kernel, on the layer.
    method = getattr(module, "scheme", None) or quant_method
    if (kernel := _describe_kernel(method)) is not None:
        return kind, kernel
    # No kernel to name: an unquantized layer, or a method that runs its own op.
    scheme = type(method).__name__
    if "Unquantized" in scheme:
        weight = getattr(module, "weight", None)
        scheme = "unquantized"
        if weight is not None:
            scheme += f" [{_dtype_name(weight.dtype)}]"
    return kind, scheme


def _format_layer_indices(indices: set[int], max_ranges: int = 3) -> str:
    ranges = _format_index_ranges(list(indices))
    if ranges.count(",") < max_ranges:
        return f"[{ranges}]"
    return f"[{min(indices)}-{max(indices)} ({len(indices)})]"


def _format_layer_names(names: list[str], max_patterns: int = 6) -> str:
    """Collapse layer names: `layers.[0-47].attn.{q_proj,k_proj}`."""
    # Layers that differ only in their numeric indices share a pattern.
    indices: dict[tuple[str, ...], list[set[int]]] = {}
    for name in names:
        parts = name.split(".")
        pattern = tuple("#" if part.isdigit() else part for part in parts)
        numbers = [int(part) for part in parts if part.isdigit()]
        seen = indices.setdefault(pattern, [set() for _ in numbers])
        for position, number in enumerate(numbers):
            seen[position].add(number)

    # Patterns that differ only in the leaf name share a prefix.
    leaves: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for pattern, seen in indices.items():
        ranges = iter(_format_layer_indices(numbers) for numbers in seen)
        parts = [next(ranges) if part == "#" else part for part in pattern]
        leaves[tuple(parts[:-1])].append(parts[-1])

    formatted = []
    for prefix, leaf_names in leaves.items():
        leaf = leaf_names[0]
        if len(leaf_names) > 1:
            leaf = "{" + ",".join(leaf_names) + "}"
        formatted.append(".".join((*prefix, leaf)))
    if len(formatted) > max_patterns:
        hidden = len(formatted) - max_patterns + 1
        formatted = [*formatted[: max_patterns - 1], f"... {hidden} more"]
    return ", ".join(formatted)


def format_backend_summary(model: nn.Module) -> str | None:
    """A table with one row per (layer kind, backend) and the layers that run
    it, or None if the model has no layer that selects a backend."""
    layers: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name, module in model.named_modules():
        if (described := _describe_backend(module)) is not None:
            layers[described].append(name)
    if not layers:
        return None

    kinds = ("attention", "linear", "moe")
    rows = [("layer", "count", "backend", "modules")]
    rows += [
        (kind, str(len(names)), backend, _format_layer_names(names))
        for (kind, backend), names in sorted(
            layers.items(), key=lambda item: kinds.index(item[0][0])
        )
    ]
    widths = [max(len(row[col]) for row in rows) for col in range(3)]
    return "\n".join(
        f"  {kind:<{widths[0]}} | {count:>{widths[1]}} | "
        f"{backend:<{widths[2]}} | {names}"
        for kind, count, backend, names in rows
    )
