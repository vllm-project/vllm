# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 XLANG Lab, The University of Hong Kong

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2 import Qwen2Config
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)


class OpenCUAConfig(PretrainedConfig):
    model_type = "opencua"

    def __init__(
        self,
        vision_config: dict | Qwen2_5_VLVisionConfig | None = None,
        text_config: dict | Qwen2Config | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 151664,
        pad_token_id: int = 0,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = Qwen2_5_VLVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Qwen2_5_VLVisionConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Qwen2Config()
        elif isinstance(text_config, dict):
            text_config = Qwen2Config(**text_config)
        self.text_config = text_config

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id

        super().__init__(pad_token_id=pad_token_id, **kwargs)

        if not hasattr(self, "image_token_id"):
            self.image_token_id = media_placeholder_token_id
        if not hasattr(self, "video_token_id"):
            self.video_token_id = media_placeholder_token_id
        if not hasattr(self, "vision_start_token_id"):
            self.vision_start_token_id = 151661

