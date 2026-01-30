# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import LlamaConfig
from transformers.configuration_utils import PretrainedConfig


class OpenVLAConfig(PretrainedConfig):
    """Configuration class for OpenVLA model."""

    model_type = "openvla"

    def __init__(
        self,
        timm_model_ids: list[str] | None = None,
        image_sizes: list[int] | None = None,
        use_fused_vision_backbone: bool = True,
        image_token_index: int = 32000,
        n_action_bins: int = 256,
        text_config: dict[str, Any] | LlamaConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.timm_model_ids = timm_model_ids or [
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ]
        self.image_sizes = image_sizes or [224, 224]
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_token_index = image_token_index
        self.n_action_bins = n_action_bins

        # Handle text_config for the Llama backbone
        if text_config is None:
            # Default Llama-2-7b config matching OpenVLA
            text_config = LlamaConfig(
                vocab_size=32064,
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                hidden_act="silu",
                max_position_embeddings=2048,
                rms_norm_eps=1e-6,
                architectures=["LlamaForCausalLM"],
            )
        elif isinstance(text_config, dict):
            if "architectures" not in text_config:
                text_config["architectures"] = ["LlamaForCausalLM"]
            text_config = LlamaConfig(**text_config)
        self.text_config = text_config

        # Also add pad_token_id for image token placeholder
        self.pad_token_id = 32000
