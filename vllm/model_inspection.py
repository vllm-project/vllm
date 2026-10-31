# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model inspection utilities for vLLM."""

import enum

import torch.nn as nn

# Attributes on a quant method or scheme that hold the kernel resolved for a
# layer. The name varies by method, and the values are selected in either
# `create_weights` or `process_weights_after_loading`. Kept in the order they
# are reported: a concrete kernel object is preferred over the backend enum
# below, which is recorded earlier and can go stale (e.g. the MXFP4 MoE method
# defaults the enum to MARLIN, then selects Cutlass experts without updating
# it).
_KERNEL_ATTRIBUTES = (
    "moe_kernel",
    "fp8_linear",
    "linear_kernel",
    "ocp_mx_linear",
    "kernel",
    # MoE methods that keep only the selected experts implementation.
    "experts_cls",
)

# MoE methods that record the resolved backend as an enum. Only consulted when
# no kernel object was found above, and only when the value really is an enum,
# so string-valued `backend` style config fields are ignored.
_BACKEND_ATTRIBUTES = (
    "mxfp4_backend",
    "mxfp8_backend",
    "nvfp4_backend",
    "int8_backend",
    "fp8_backend",
    "w4a8_backend",
    "wna16_backend",
    "wna16_moe_backend",
    "unquantized_backend",
    "backend",
)

# `FusedMoEKernel` carries no name of its own; the experts implementation it
# dispatches to is what identifies the backend that was chosen.
_MOE_EXPERTS_ATTRIBUTE = "fused_experts"


def _describe_kernel(kernel: object) -> str:
    """Name a resolved kernel, which may be stored as a class or an instance."""
    experts = getattr(kernel, _MOE_EXPERTS_ATTRIBUTE, None)
    kernel_name = kernel.__name__ if isinstance(kernel, type) else type(kernel).__name__
    if experts is not None:
        experts_name = (
            experts.__name__ if isinstance(experts, type) else type(experts).__name__
        )
        return f"{kernel_name}(experts={experts_name})"
    return kernel_name


def _get_selected_kernel(quant_method: object, scheme: object | None) -> str | None:
    """Name the quantization kernel/backend resolved for a layer.

    Returns None when no kernel has been selected yet, which is the case for
    unquantized layers and for inspection that runs before the weights have
    been processed.
    """
    for source in (scheme, quant_method):
        if source is None:
            continue
        for attribute in _KERNEL_ATTRIBUTES:
            kernel = getattr(source, attribute, None)
            if kernel is not None:
                return _describe_kernel(kernel)
        for attribute in _BACKEND_ATTRIBUTES:
            backend = getattr(source, attribute, None)
            if isinstance(backend, enum.Enum):
                return backend.name

    return None


def _get_quant_info(module: nn.Module) -> str | None:
    """Get the quantization description for a module, if it is quantized."""
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None:
        return None

    # For CompressedTensors, the scheme holds both the name and the kernel.
    scheme = getattr(module, "scheme", None)
    quant_name = type(scheme if scheme is not None else quant_method).__name__
    # Skip unquantized methods
    if "Unquantized" in quant_name:
        return None

    kernel_name = _get_selected_kernel(quant_method, scheme)
    if kernel_name is None:
        return f"quant={quant_name}"
    return f"quant={quant_name}(kernel={kernel_name})"


def _get_module_info(module: nn.Module) -> str:
    """Get info string for a module."""
    class_name = type(module).__name__
    parts = []

    quant_info = _get_quant_info(module)
    if quant_info is not None:
        parts.append(quant_info)

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
