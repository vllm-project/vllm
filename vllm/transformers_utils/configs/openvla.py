# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenVLA configuration support.

OpenVLA checkpoints use a custom ``model_type`` and nest the language model
configuration under ``text_config``. This shim lets vLLM load the checkpoint
configuration without executing Hugging Face remote code.
"""

from typing import Any

from transformers import LlamaConfig, PretrainedConfig


class OpenVLAConfig(PretrainedConfig):
    """Configuration class for OpenVLA models."""

    model_type = "openvla"

    def __init__(
        self,
        timm_model_ids: list[str] | None = None,
        timm_override_act_layers: list[str | None] | None = None,
        image_sizes: list[int] | None = None,
        use_fused_vision_backbone: bool = True,
        image_token_index: int = 32000,
        n_action_bins: int = 256,
        text_config: dict[str, Any] | LlamaConfig | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("architectures", ["OpenVLAForActionPrediction"])
        super().__init__(**kwargs)

        self.timm_model_ids = timm_model_ids or [
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ]
        self.timm_override_act_layers = timm_override_act_layers or [None, None]
        self.image_sizes = image_sizes or [224, 224]
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_token_index = image_token_index
        self.n_action_bins = n_action_bins

        if text_config is None:
            text_config = LlamaConfig(architectures=["LlamaForCausalLM"])
        elif isinstance(text_config, dict):
            text_config = text_config.copy()
            text_config.setdefault("architectures", ["LlamaForCausalLM"])
            text_config = LlamaConfig(**text_config)
        self.text_config = text_config
