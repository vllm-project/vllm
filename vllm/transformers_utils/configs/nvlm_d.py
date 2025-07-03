# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/nvidia/NVLM-D-72B/blob/main/configuration_nvlm_d.py
# --------------------------------------------------------
# NVLM-D
# Copyright (c) 2024 NVIDIA
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
from transformers import Qwen2Config
from transformers.configuration_utils import PretrainedConfig


class NVLM_D_Config(PretrainedConfig):
    model_type = 'NVLM_D'
    is_composition = True

    def __init__(self, vision_config=None, llm_config=None, **kwargs):
        super().__init__(**kwargs)

        # Handle vision_config initialization
        if vision_config is None:
            vision_config = {}

        # Handle llm_config initialization
        if llm_config is None:
            llm_config = {}

        self.vision_config = PretrainedConfig(**vision_config)
        self.text_config = Qwen2Config(**llm_config)
