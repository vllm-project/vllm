# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig, Qwen2VLConfig, Qwen2VLTextConfig


def resolve_text_config(text_config):
    # If it is a dictionary
    if isinstance(text_config, dict):
        if text_config.get("text_config") is not None:
            return resolve_text_config(text_config["text_config"])
        return Qwen2VLTextConfig.from_dict(text_config)

    # If it is a PretrainedConfig instance
    if isinstance(text_config, PretrainedConfig):
        if hasattr(text_config, "num_attention_heads"):
            return text_config
        if hasattr(text_config, "text_config") and text_config.text_config is not None:
            return resolve_text_config(text_config.text_config)

    raise ValueError(
        f"Cannot resolve text_config: {type(text_config)} has no "
        "'num_attention_heads' or nested 'text_config' to unwrap"
    )


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

    def get_text_config(self, *args, **kwargs):
        text_config = super().get_text_config(*args, **kwargs)
        return resolve_text_config(text_config)
