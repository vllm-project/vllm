# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Model inspection utilities for vLLM.

Provides a transformers-style hierarchical view of vLLM models with
quantization method and attention backend information.

Usage:
    from vllm.model_inspection import print_model_inspection

    # Print verbose hierarchical view
    print_model_inspection(model)
"""

from typing import Any

import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_module_info(module: nn.Module) -> str:
    """Get info string for a module, including quant_method if present."""
    # Import here to avoid circular imports
    from vllm.attention.layer import Attention, MultiHeadAttention
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    class_name = type(module).__name__
    parts = []

    # Add attention backend
    if isinstance(module, (Attention, MultiHeadAttention)):
        backend = getattr(module, "backend", None)
        if backend is not None:
            parts.append(f"backend={backend.name}")

    # Add size info for linear layers
    if isinstance(module, LinearBase):
        in_size = getattr(module, "input_size", None)
        out_size = getattr(module, "output_size", None)
        if in_size and out_size:
            parts.append(f"in={in_size}, out={out_size}")

    # Add info for embeddings
    if isinstance(module, (VocabParallelEmbedding, ParallelLMHead)):
        vocab = getattr(module, "num_embeddings", None)
        dim = getattr(module, "embedding_dim", None)
        if vocab and dim:
            parts.append(f"{vocab}, {dim}")

    # Add info for MoE
    if isinstance(module, FusedMoE):
        num_experts = getattr(module, "num_experts", None)
        top_k = getattr(module, "top_k", None)
        if num_experts and top_k:
            parts.append(f"experts={num_experts}, top_k={top_k}")

    # Add info for attention
    if isinstance(module, (Attention, MultiHeadAttention)):
        num_heads = getattr(module, "num_heads", None)
        head_size = getattr(module, "head_size", None)
        if num_heads and head_size:
            parts.append(f"num_heads={num_heads}, head_size={head_size}")

    # Add quant_method for relevant layers
    if isinstance(
        module, (LinearBase, FusedMoE, VocabParallelEmbedding, ParallelLMHead)
    ):
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            parts.append(f"quant={type(quant_method).__name__}")

    if parts:
        return f"{class_name}({', '.join(parts)})"
    return class_name


def _get_child_signature(child: nn.Module) -> str:
    """Get a signature for a child module to detect duplicates."""
    # For detecting identical layers, we use the repr of the module structure
    # This captures the class and all nested children
    lines = []
    for name, submodule in child.named_modules():
        info = _get_module_info(submodule)
        lines.append(f"{name}:{info}")
    return "\n".join(lines)


def _format_module_tree(
    module: nn.Module,
    name: str = "",
    indent: int = 0,
) -> list[str]:
    """Format a module tree with indentation, grouping identical layers."""
    lines = []
    prefix = "  " * indent

    # Get direct children
    children = list(module.named_children())

    if not children:
        # Leaf node
        info = _get_module_info(module)
        if name:
            lines.append(f"{prefix}({name}): {info}")
        else:
            lines.append(f"{prefix}{info}")
        return lines

    # Non-leaf node - show this node and recurse
    info = _get_module_info(module)
    if name:
        lines.append(f"{prefix}({name}): {info}(")
    else:
        lines.append(f"{prefix}{info}(")

    # Group consecutive identical children
    i = 0
    while i < len(children):
        child_name, child_module = children[i]

        # Check if this is a numbered layer that might repeat
        try:
            idx = int(child_name)
            # Find all consecutive identical layers
            sig = _get_child_signature(child_module)
            j = i + 1
            while j < len(children):
                _, next_module = children[j]
                try:
                    if _get_child_signature(next_module) == sig:
                        j += 1
                    else:
                        break
                except ValueError:
                    break

            count = j - i
            if count > 1:
                # Multiple identical layers
                end_idx = int(children[j - 1][0])
                child_lines = _format_module_tree(child_module, "", indent + 1)
                # Replace the first line to add the range prefix
                first_line = child_lines[0].lstrip()
                range_prefix = "  " * (indent + 1)
                child_lines[0] = (
                    f"{range_prefix}({idx}-{end_idx}): {count} x {first_line}"
                )
                lines.extend(child_lines)
                i = j
                continue
        except ValueError:
            pass

        # Single child or non-numbered
        child_lines = _format_module_tree(child_module, child_name, indent + 1)
        lines.extend(child_lines)
        i += 1

    lines.append(f"{prefix})")
    return lines


def format_model_inspection(model: nn.Module) -> str:
    """
    Format a model into a transformers-style hierarchical string.

    Args:
        model: The nn.Module model to inspect

    Returns:
        Formatted string representation with indentation
    """
    lines = _format_module_tree(model)
    return "\n".join(lines)


def print_model_inspection(model: nn.Module) -> None:
    """
    Print a transformers-style hierarchical view of the model.

    Args:
        model: The nn.Module model to inspect
    """
    print(format_model_inspection(model))


def get_model_summary(model: nn.Module) -> dict[str, Any]:
    """
    Get a dictionary summary of the model's quantization and backend choices.

    Args:
        model: The nn.Module model to inspect

    Returns:
        Dictionary with summary information
    """
    from collections import defaultdict

    from vllm.attention.layer import Attention, MultiHeadAttention
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    counts: dict[str, int] = defaultdict(int)

    for _, module in model.named_modules():
        class_name = type(module).__name__

        if isinstance(
            module, (LinearBase, FusedMoE, VocabParallelEmbedding, ParallelLMHead)
        ):
            quant_method = getattr(module, "quant_method", None)
            method_name = type(quant_method).__name__ if quant_method else "None"
            counts[f"{class_name}[{method_name}]"] += 1
        elif isinstance(module, (Attention, MultiHeadAttention)):
            backend = getattr(module, "backend", None)
            backend_name = backend.name if backend else "None"
            counts[f"{class_name}[{backend_name}]"] += 1
        else:
            counts[class_name] += 1

    return {
        "model_class": type(model).__name__,
        "layer_counts": dict(counts),
    }
