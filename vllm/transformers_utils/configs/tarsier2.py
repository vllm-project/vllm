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
    In Transformers v5, Qwen2VLConfig creates a proper Qwen2VLTextConfig sub-object
    inside __post_init__.  When the malformed config.json is loaded, self.text_config
    is a Qwen2VLConfig-like dict that itself contains a nested text_config sub-dict
    with the actual text parameters.  Without unwrapping, __post_init__ would call
    Qwen2VLTextConfig(**outer_dict), which silently loses the real field values
    (num_attention_heads, hidden_size, etc.) because they are one level too deep.
    """

    model_type = "tarsier2"

    def __post_init__(self, **kwargs):
        # Unwrap the doubly-nested text_config before the parent processes it.
        # Tarsier2's config stores the real Qwen2VLTextConfig parameters inside
        # text_config["text_config"], not at the top level of text_config.
        if isinstance(self.text_config, dict):
            nested = self.text_config.get("text_config")
            if isinstance(nested, dict):
                self.text_config = nested
        super().__post_init__(**kwargs)
