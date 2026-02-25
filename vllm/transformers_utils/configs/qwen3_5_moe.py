# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5-MoE model configuration

Qwen3.5-MoE uses the same hybrid GDN (GatedDeltaNet) architecture as Qwen3-Next.
The outer config.json wraps a text_config for the language model and a
vision_config for the vision encoder. For text-only serving we extract the
text_config fields and delegate to Qwen3NextConfig.
"""
from __future__ import annotations

from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig


class Qwen3_5MoeConfig(Qwen3NextConfig):
    """Config for Qwen3.5-MoE models (model_type: qwen3_5_moe).

    Qwen3.5-MoE shares its LLM architecture with Qwen3-Next (hybrid
    full-attention + GatedDeltaNet layers, shared MoE experts). This config
    handles the multimodal outer wrapper by flattening text_config fields
    into the base Qwen3NextConfig.
    """

    model_type = "qwen3_5_moe"

    def __init__(self, text_config: dict | None = None, **kwargs):
        if text_config is not None and isinstance(text_config, dict):
            # Flatten text_config fields into kwargs.
            # text_config fields take precedence over outer kwargs so that
            # the LLM-specific values (hidden_size, layer_types, etc.) win.
            text_kw = {k: v for k, v in text_config.items()
                       if k not in ("model_type", "transformers_version")}
            merged = {**kwargs, **text_kw}
        else:
            merged = kwargs

        # Strip multimodal-only keys that Qwen3NextConfig doesn't expect.
        for key in ("vision_config", "image_token_id", "video_token_id",
                    "vision_start_token_id", "vision_end_token_id",
                    "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings",
                    "dtype", "full_attention_interval", "mlp_only_layers",
                    "mamba_ssm_dtype"):
            merged.pop(key, None)

        super().__init__(**merged)
