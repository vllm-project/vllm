# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV4 rotary embedding initialization."""

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


def build_deepseek_v4_rope(
    config,
    *,
    head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    compress_ratio: int,
) -> RotaryEmbedding:
    rope_parameters = config.rope_parameters
    # Newer checkpoints nest per-layer-type rope dicts ({"main", "compress"});
    # older ones ship a single flat dict shared by all layer types.
    if isinstance(rope_parameters.get("main"), dict) and isinstance(
        rope_parameters.get("compress"), dict
    ):
        key = "compress" if compress_ratio > 1 else "main"
        rope_parameters = dict(rope_parameters[key])
    else:
        rope_parameters = dict(rope_parameters)

    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    )
    if compress_ratio > 1 and rope_parameters["rope_type"] != "default":
        # YaRN applies only to compressor (CSA/HCA) layers.
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
    else:
        # Sliding-window layers use plain RoPE (theta=rope_theta, no YaRN).
        rope_parameters["rope_type"] = "deepseek_yarn"
        rope_parameters["factor"] = 1.0
        rope_parameters["original_max_position_embeddings"] = max_position_embeddings
    rope_parameters["mscale"] = 0  # Disable mscale
    rope_parameters["mscale_all_dim"] = 0  # Disable mscale
    rope_parameters["is_deepseek_v4"] = True
    rope_parameters["rope_dim"] = rope_head_dim
    return get_rope(
        head_dim,
        max_position=max_position_embeddings,
        rope_parameters=rope_parameters,
        is_neox_style=False,
    )
