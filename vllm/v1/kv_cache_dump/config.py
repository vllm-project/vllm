# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for KV cache dumping."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig, ParallelConfig

logger = init_logger(__name__)

# Global singleton config
_kv_dump_config: KVCacheDumpConfig | None = None


@dataclass
class KVCacheDumpConfig:
    """Configuration for KV cache dumping during decode phase."""

    enabled: bool = False
    output_dir: str = "/tmp/vllm_kv_dump"
    layer_indices: set[int] = field(default_factory=set)
    num_layers: int = 0
    model_name: str = ""
    num_kv_heads: int = 0
    head_size: int = 0
    num_q_heads: int = 0
    dtype: torch.dtype = torch.float16
    max_decode_tokens: int = 0  # 0 = unlimited

    def should_dump_layer(self, layer_idx: int) -> bool:
        """Check if a specific layer should be dumped."""
        return self.enabled and layer_idx in self.layer_indices


def _parse_layer_indices(layers_str: str, num_layers: int) -> set[int]:
    """
    Parse layer indices from environment variable.

    Args:
        layers_str: Comma-separated layer indices or "early,mid,late"
        num_layers: Total number of layers in the model

    Returns:
        Set of layer indices to dump
    """
    if not layers_str:
        # Default to early, mid, late
        if num_layers >= 3:
            return {0, num_layers // 2, num_layers - 1}
        elif num_layers == 2:
            return {0, 1}
        elif num_layers == 1:
            return {0}
        return set()

    layers_str = layers_str.strip().lower()

    # Handle "early,mid,late" format
    if "early" in layers_str or "mid" in layers_str or "late" in layers_str:
        indices = set()
        parts = [p.strip() for p in layers_str.split(",")]
        for part in parts:
            if part == "early":
                indices.add(0)
            elif part == "mid":
                indices.add(num_layers // 2)
            elif part == "late":
                indices.add(num_layers - 1)
        return indices

    # Handle numeric format "0,15,31"
    try:
        indices = set()
        for part in layers_str.split(","):
            idx = int(part.strip())
            if 0 <= idx < num_layers:
                indices.add(idx)
            else:
                logger.warning(
                    f"Layer index {idx} out of range [0, {num_layers}), skipping"
                )
        return indices
    except ValueError as e:
        logger.error(f"Failed to parse VLLM_KV_CACHE_DUMP_LAYERS: {e}")
        return set()


def init_kv_dump_config(
    num_layers: int,
    model_config: ModelConfig,
    parallel_config: ParallelConfig | None = None,
) -> KVCacheDumpConfig:
    """
    Initialize the KV cache dump configuration.

    Args:
        num_layers: Total number of transformer layers
        model_config: Model configuration
        parallel_config: Optional parallel configuration for TP/PP

    Returns:
        Initialized KVCacheDumpConfig
    """
    global _kv_dump_config

    enabled = envs.VLLM_KV_CACHE_DUMP_ENABLED
    output_dir = envs.VLLM_KV_CACHE_DUMP_OUTPUT_DIR
    layers_str = envs.VLLM_KV_CACHE_DUMP_LAYERS
    max_decode_tokens = envs.VLLM_KV_CACHE_DUMP_MAX_TOKENS

    if not enabled:
        _kv_dump_config = KVCacheDumpConfig(enabled=False)
        return _kv_dump_config

    # Parse layer indices
    layer_indices = _parse_layer_indices(layers_str, num_layers)

    if not layer_indices:
        logger.warning("No valid layer indices for KV cache dumping, disabling")
        _kv_dump_config = KVCacheDumpConfig(enabled=False)
        return _kv_dump_config

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get model metadata
    num_kv_heads = model_config.get_total_num_kv_heads()
    if parallel_config is not None:
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)

    head_size = model_config.get_head_size()
    num_q_heads = (
        model_config.get_num_attention_heads(parallel_config)
        if parallel_config
        else model_config.hf_config.num_attention_heads
    )
    dtype = model_config.dtype

    _kv_dump_config = KVCacheDumpConfig(
        enabled=True,
        output_dir=output_dir,
        layer_indices=layer_indices,
        num_layers=num_layers,
        model_name=model_config.model,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_q_heads=num_q_heads,
        dtype=dtype,
        max_decode_tokens=max_decode_tokens,
    )

    logger.info(
        f"KV cache dumping enabled: "
        f"layers={sorted(layer_indices)}, "
        f"output_dir={output_dir}, "
        f"num_kv_heads={num_kv_heads}, "
        f"head_size={head_size}, "
        f"max_decode_tokens={max_decode_tokens or 'unlimited'}"
    )

    return _kv_dump_config


def get_kv_dump_config() -> KVCacheDumpConfig:
    """Get the global KV dump configuration singleton."""
    global _kv_dump_config
    if _kv_dump_config is None:
        # Return a disabled config if not initialized
        return KVCacheDumpConfig(enabled=False)
    return _kv_dump_config
