# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers.configuration_utils import PreTrainedConfig

from vllm.transformers_utils.configs.lfm3 import Lfm3Config


class Lfm3VlVisionConfig(PreTrainedConfig):
    model_type = "lfm3_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2560,
        num_hidden_layers: int = 14,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        layer_types: list[str] | None = None,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        spatial_merge_size: int = 2,
        conv_kernel_size: int = 3,
        norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        rope_parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.layer_types = layer_types or [
            "conv" if i < 2 or i % 2 else "full_attention"
            for i in range(num_hidden_layers)
        ]
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.spatial_merge_size = spatial_merge_size
        self.conv_kernel_size = conv_kernel_size
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range
        self.rope_parameters = rope_parameters or {
            "rope_type": "axial",
            "rope_theta": 1_000_000.0,
        }
        super().__init__(**kwargs)


class Lfm3VlConfig(PreTrainedConfig):
    model_type = "lfm3_vl"
    sub_configs = {"text_config": Lfm3Config, "vision_config": Lfm3VlVisionConfig}

    def __init__(
        self,
        vision_config: dict | Lfm3VlVisionConfig | None = None,
        text_config: dict | Lfm3Config | None = None,
        image_token_id: int = 125_030,
        video_token_id: int = 125_029,
        vision_start_token_id: int = 125_009,
        vision_end_token_id: int = 125_010,
        projector_hidden_act: str = "gelu",
        projector_hidden_size: int = 4096,
        projector_bias: bool = True,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        if vision_config is None:
            vision_config = Lfm3VlVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Lfm3VlVisionConfig(**vision_config)
        self.vision_config: Lfm3VlVisionConfig = vision_config

        if text_config is None:
            text_config = Lfm3Config()
        elif isinstance(text_config, dict):
            text_config = Lfm3Config(**text_config)
        self.text_config: Lfm3Config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.projector_hidden_act = projector_hidden_act
        self.projector_hidden_size = projector_hidden_size
        self.projector_bias = projector_bias
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
