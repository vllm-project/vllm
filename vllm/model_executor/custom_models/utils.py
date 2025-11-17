# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utility functions for TorchTitan model integration with vLLM.

This module provides helper functions for common TorchTitan integration tasks:
- Converting RoPE frequency tensors from complex to real format
- Managing forward context for position indices
- Creating KV cache specifications for MLA
- Loading weights from HuggingFace checkpoints with name mapping

Example usage:
    ```python
    from vllm.model_executor.utils.torchtitan_utils import (
        convert_freqs_cis_to_real,
        create_mla_kv_cache_spec,
    )

    # Convert TorchTitan's complex freqs_cis to vLLM-compatible format
    model.freqs_cis = convert_freqs_cis_to_real(model.freqs_cis)

    # Create KV cache spec for MLA attention
    kv_cache_spec = create_mla_kv_cache_spec(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        vllm_config=vllm_config,
    )
    ```
"""

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn


def convert_freqs_cis_to_real(freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Convert complex RoPE frequencies to real format (cos, sin concatenated).

    TorchTitan uses complex exponentials e^(i*theta) for RoPE, but vLLM and
    dtype conversion to bfloat16 require real tensors. This converts:
        complex[max_seq_len, dim//2] -> real[max_seq_len, dim]

    Args:
        freqs_cis: Complex frequency tensor [max_seq_len, dim//2]

    Returns:
        Real tensor [max_seq_len, dim] with cos and sin concatenated
    """
    if not freqs_cis.is_complex():
        # Already in real format
        return freqs_cis

    # Extract cos and sin from complex exponentials
    # e^(i*theta) = cos(theta) + i*sin(theta)
    freqs_cos = freqs_cis.real  # [max_seq_len, dim//2]
    freqs_sin = freqs_cis.imag  # [max_seq_len, dim//2]

    # Concatenate: [max_seq_len, dim]
    freqs_real = torch.cat([freqs_cos, freqs_sin], dim=-1)

    return freqs_real


def store_positions_in_context(positions: torch.Tensor | None) -> None:
    """
    Store position indices in vLLM's forward context.

    This allows attention layers to access per-token positions for RoPE indexing
    during inference without explicitly passing them through every layer.

    Args:
        positions: Position indices from vLLM [total_tokens] or None
    """
    if positions is None:
        return

    try:
        from vllm.forward_context import get_forward_context

        forward_ctx = get_forward_context()
        # Store positions in a custom attribute accessible to attention layers
        forward_ctx._torchtitan_positions = positions
    except (ImportError, RuntimeError, AttributeError):
        # Not in vLLM context - this is fine (e.g., during testing)
        pass


def create_mla_kv_cache_spec(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    dtype: torch.dtype,
) -> Any:
    """
    Create KV cache specification for Multi-Head Latent Attention (MLA).

    MLA uses compressed KV cache with layout:
        [kv_lora_rank + qk_rope_head_dim] per token

    Args:
        kv_lora_rank: LoRA rank for compressed KV (e.g., 512)
        qk_rope_head_dim: Dimension of RoPE-encoded keys (e.g., 64)
        block_size: KV cache block size from vLLM config
        dtype: Data type for KV cache

    Returns:
        MLAAttentionSpec instance
    """
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    # MLA cache layout: compressed KV + shared K_PE
    head_size = kv_lora_rank + qk_rope_head_dim

    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,  # MLA shares K_PE across all heads
        head_size=head_size,
        dtype=dtype,
    )


def load_external_weights(
    model: nn.Module,
    weights_iter: Iterator[tuple[str, torch.Tensor]],
    name_mapping: dict[str, str],
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Load weights from HuggingFace checkpoint into external model.

    Maps HuggingFace parameter names to model parameter names and loads
    them into the model. Supports layer-specific patterns with {} placeholders.

    Args:
        model: Model instance to load weights into
        weights_iter: Iterator yielding (name, tensor) from HF checkpoint
        name_mapping: Dict mapping HF names to model parameter names.
            Use {} as placeholder for layer numbers, e.g.:
            {"model.layers.{}.attn.weight": "layers.{}.attention.weight"}
        verbose: Whether to print detailed loading progress

    Returns:
        Tuple of (loaded_count, skipped_count)

    Example:
        ```python
        name_mapping = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "lm_head.weight": "output.weight",
        }
        loaded, skipped = load_external_weights(model, weights_iter, name_mapping)
        ```
    """
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    # Get all parameter names in the model
    params_dict = dict(model.named_parameters())

    loaded_count = 0
    skipped_count = 0

    # Convert iterator to list to check if empty
    weights_list = list(weights_iter)
    if len(weights_list) == 0:
        if verbose:
            print("  ⚠️  No weight files found - using random initialization")
        return 0, 0

    for hf_name, loaded_weight in weights_list:
        # Try to find matching pattern in name_mapping
        target_name = None

        # Check if it's a layer-specific weight
        if "layers" in hf_name:
            # Extract layer number
            import regex as re

            layer_match = re.search(r"layers\.(\d+)\.", hf_name)
            if layer_match:
                layer_num = layer_match.group(1)

                # Try to find matching pattern
                for hf_pattern, target_pattern in name_mapping.items():
                    if "{}" in hf_pattern:
                        hf_concrete = hf_pattern.format(layer_num)
                        if hf_name == hf_concrete:
                            target_name = target_pattern.format(layer_num)
                            break
        else:
            # Non-layer weight (embeddings, norms, output)
            target_name = name_mapping.get(hf_name)

        if target_name is None:
            # Skip MoE weights and other unmapped weights
            if (
                "mlp.experts" in hf_name
                or "mlp.gate" in hf_name
                or "mlp.shared_experts" in hf_name
            ):
                # MoE weights - skip silently
                skipped_count += 1
                continue
            else:
                if verbose:
                    print(f"  ⚠️  No mapping for: {hf_name}")
                skipped_count += 1
            continue

        # Check if parameter exists in model
        if target_name not in params_dict:
            if verbose:
                print(f"  ⚠️  Parameter not found in model: {target_name}")
            skipped_count += 1
            continue

        # Load the weight
        param = params_dict[target_name]

        # Verify shapes match
        if param.shape != loaded_weight.shape:
            if verbose:
                print(f"  ⚠️  Shape mismatch for {target_name}:")
                print(f"      Model: {param.shape}, Checkpoint: {loaded_weight.shape}")
            skipped_count += 1
            continue

        # Load the weight
        default_weight_loader(param, loaded_weight)
        loaded_count += 1

        # Log first few loads for verification
        if verbose and loaded_count <= 5:
            print(f"  ✓ Loaded {target_name}: {loaded_weight.shape}")

    return loaded_count, skipped_count
