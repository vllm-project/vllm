# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://huggingface.co/Skywork/Skywork-R1V-38B/blob/main/configuration_skywork_chat.py
# --------------------------------------------------------
# SkyworkR1V
# Copyright (c) 2025 Skywork
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from transformers.configuration_utils import PretrainedConfig


class SkyworkR1VChatConfig(PretrainedConfig):
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(self,
                 vision_config=None,
                 llm_config=None,
                 use_backbone_lora=0,
                 use_llm_lora=0,
                 select_layer=-1,
                 force_image_size=None,
                 downsample_ratio=0.5,
                 template=None,
                 dynamic_image_size=False,
                 use_thumbnail=False,
                 ps_version='v1',
                 min_dynamic_patch=1,
                 max_dynamic_patch=6,
                 **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}

        if llm_config is None:
            llm_config = {}

        self.vision_config = PretrainedConfig(**vision_config)
        self.text_config = PretrainedConfig(**llm_config)

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
