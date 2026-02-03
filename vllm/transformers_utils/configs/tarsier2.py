# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import Qwen2VLConfig


class Tarsier2Config(Qwen2VLConfig):
    """
    Tarsier2's config.json is written such that AutoConfig.from_pretrained will create
    a deeply nested config consisting of:

    - LlavaConfig
      - Qwen2VLConfig
        - Qwen2VLTextConfig
        - Qwen2VLVisionConfig
      - Qwen2VLConfig
        - Qwen2VLTextConfig
        - Qwen2VLVisionConfig

    When it should really just be a single Qwen2VLConfig.

    This class is a hack to stop AutoConfig from creating the nested config structure.
    """

    model_type = "tarsier2"
