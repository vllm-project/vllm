# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Legacy DeepSeek-V4 fields derived from `transformers.DeepseekV4Config`.

Upstream folds the checkpoint's `compress_ratios` and `num_hash_layers` into
`layer_types` and `mlp_layer_types`. DeepSeek-V4.1 configs (and test stubs)
still carry the legacy fields, which take precedence.
"""

from typing import Any

from transformers import DeepseekV4Config


def get_compress_ratios(config: Any) -> list[int]:
    """Per-layer compress ratios (0 = sliding window), MTP layers included.

    Returns an empty list if `config` is not a DeepSeek-V4 style config.
    """
    if (compress_ratios := getattr(config, "compress_ratios", None)) is not None:
        return compress_ratios
    compress_rates = getattr(config, "compress_rates", None)
    if compress_rates is None:
        return []
    rates = {"sliding_attention": 0, **compress_rates}
    compress_ratios = [rates[layer_type] for layer_type in config.layer_types]
    # Upstream drops the MTP layers, which are always sliding window
    return compress_ratios + [0] * config.num_nextn_predict_layers


def get_num_hash_layers(config: Any) -> int:
    """Number of leading hash-routed MoE layers."""
    if (num_hash_layers := getattr(config, "num_hash_layers", None)) is not None:
        return num_hash_layers
    return config.mlp_layer_types.count("hash_moe")


def get_mm_prefix_clamp_sliding_window(config: Any) -> bool:
    """Whether image spans widen the sliding window in-kernel (V4 vision)."""
    if (clamp := getattr(config, "mm_prefix_clamp_sliding_window", None)) is not None:
        return clamp
    vision_n_layers = getattr(config, "vision_n_layers", 0)
    return isinstance(config, DeepseekV4Config) and vision_n_layers > 0
