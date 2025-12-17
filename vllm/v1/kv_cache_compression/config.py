# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for experimental KV cache key rank compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig, ParallelConfig

logger = init_logger(__name__)

_kv_compression_config: KVCacheCompressionConfig | None = None


@dataclass
class KVCacheCompressionConfig:
    """Controls optional rank-based K projection before caching."""

    enabled: bool = False
    energy_threshold: float = 0.995
    max_rank: int | None = None
    min_tokens_before_svd: int = 16
    recompute_every: int = 8
    log_every: int = 64
    layer_indices: set[int] = field(default_factory=set)
    num_layers: int = 0
    num_kv_heads: int = 0
    head_size: int = 0

    def should_compress_layer(self, layer_idx: int) -> bool:
        """Check if a specific layer should be compressed."""
        return self.enabled and layer_idx in self.layer_indices


def _parse_layer_indices(layers_str: str, num_layers: int) -> set[int]:
    """Parse layer indices or shorthand from env."""
    if not layers_str:
        if num_layers >= 3:
            return {0, num_layers // 2, num_layers - 1}
        if num_layers == 2:
            return {0, 1}
        if num_layers == 1:
            return {0}
        return set()

    layers_str = layers_str.strip().lower()

    if "early" in layers_str or "mid" in layers_str or "late" in layers_str:
        indices: set[int] = set()
        for part in (p.strip() for p in layers_str.split(",")):
            if part == "early":
                indices.add(0)
            elif part == "mid":
                indices.add(num_layers // 2)
            elif part == "late":
                indices.add(num_layers - 1)
        return indices

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
    except ValueError as exc:
        logger.error(f"Failed to parse VLLM_KV_KEY_COMPRESS_LAYERS: {exc}")
        return set()


def init_kv_compression_config(
    num_layers: int,
    model_config: ModelConfig,
    parallel_config: ParallelConfig | None = None,
) -> KVCacheCompressionConfig:
    """
    Initialize the KV cache compression configuration.

    Args:
        num_layers: Total transformer layers
        model_config: Model configuration
        parallel_config: Optional parallel configuration
    """
    global _kv_compression_config

    enabled = envs.VLLM_KV_KEY_COMPRESS_ENABLED
    if not enabled:
        _kv_compression_config = KVCacheCompressionConfig(enabled=False)
        return _kv_compression_config

    layer_indices = _parse_layer_indices(
        envs.VLLM_KV_KEY_COMPRESS_LAYERS, num_layers
    )
    if not layer_indices:
        logger.warning("KV key compression enabled but no valid layers selected")
        _kv_compression_config = KVCacheCompressionConfig(enabled=False)
        return _kv_compression_config

    max_rank = envs.VLLM_KV_KEY_COMPRESS_MAX_RANK
    if max_rank is not None and max_rank <= 0:
        max_rank = None

    num_kv_heads = model_config.get_total_num_kv_heads()
    if parallel_config is not None:
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)

    _kv_compression_config = KVCacheCompressionConfig(
        enabled=True,
        energy_threshold=envs.VLLM_KV_KEY_COMPRESS_ENERGY,
        max_rank=max_rank,
        min_tokens_before_svd=max(envs.VLLM_KV_KEY_COMPRESS_MIN_TOKENS, 1),
        recompute_every=max(envs.VLLM_KV_KEY_COMPRESS_RECOMPUTE_EVERY, 1),
        log_every=max(envs.VLLM_KV_KEY_COMPRESS_LOG_EVERY, 1),
        layer_indices=layer_indices,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_size=model_config.get_head_size(),
    )

    logger.info(
        "KV key compression enabled: layers=%s, energy=%.3f, "
        "max_rank=%s, warmup_tokens=%d, recompute_every=%d",
        sorted(layer_indices),
        _kv_compression_config.energy_threshold,
        _kv_compression_config.max_rank or "auto",
        _kv_compression_config.min_tokens_before_svd,
        _kv_compression_config.recompute_every,
    )

    return _kv_compression_config


def get_kv_compression_config() -> KVCacheCompressionConfig:
    """Return the KV compression config (disabled if uninitialized)."""
    global _kv_compression_config
    if _kv_compression_config is None:
        return KVCacheCompressionConfig(enabled=False)
    return _kv_compression_config
