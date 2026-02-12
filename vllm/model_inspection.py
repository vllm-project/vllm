# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model inspection utilities for vLLM."""

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
