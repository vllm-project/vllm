# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class GlmOcrVisionConfig(PretrainedConfig):
    model_type = "glm_ocr_vision"

    def __init__(
        self,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        attention_bias: bool = True,
        intermediate_size: int = 4096,
        hidden_act: str = "silu",
        hidden_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        image_size: int = 336,
        in_channels: int = 3,
        patch_size: int = 14,
        out_hidden_size: int = 1536,
        rms_norm_eps: float = 1e-5,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.attention_bias = attention_bias
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.out_hidden_size = out_hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size


class GlmOcrConfig(PretrainedConfig):
    model_type = "glm_ocr"

    def __init__(
        self,
        text_config: dict | None = None,
        vision_config: dict | None = None,
        image_start_token_id: int = 59256,
        image_end_token_id: int = 59257,
        video_start_token_id: int = 59258,
        video_end_token_id: int = 59259,
        image_token_id: int = 59280,
        video_token_id: int = 59281,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_config = GlmOcrVisionConfig(**(vision_config or {}))

        if isinstance(text_config, dict):
            from transformers import AutoConfig

            model_type = text_config.get("model_type", "chatglm")
            self.text_config = AutoConfig.for_model(model_type, **text_config)
        elif text_config is None:
            from transformers import AutoConfig

            self.text_config = AutoConfig.for_model("chatglm")
        else:
            self.text_config = text_config

    def get_text_config(self) -> PretrainedConfig:
        return self.text_config

    def save_pretrained(self, save_directory, **kwargs):
        self._auto_class = None
        super().save_pretrained(save_directory, **kwargs)
