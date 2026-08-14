# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer config resolution for Gemma4 and its variants.

Also provides thin config classes for model types (gemma4_assistant,
gemma4_text) not yet in the installed Transformers version.
"""

from copy import copy

from transformers import PretrainedConfig


def gemma4_layer_config(
    text_config: PretrainedConfig, layer_idx: int
) -> PretrainedConfig:
    """The Gemma4 text config as it applies to one layer.

    Gemma4 uses a larger head dimension on its full attention layers than on its
    sliding ones, and with `attention_k_eq_v` it uses more KV heads there too.
    Transformers >= 5.15.0 says so in the config itself; before that the values
    are flat attributes that `layer_types` picks between, so resolve them here.
    The result is homogeneous either way, so callers read `head_dim` and
    `num_key_value_heads` off it without caring which version they are on.
    """
    if getattr(text_config, "is_heterogeneous", False):
        return text_config.per_layer_config[layer_idx]

    layer = copy(text_config)
    if text_config.layer_types[layer_idx] == "full_attention":
        global_head_dim = getattr(text_config, "global_head_dim", None)
        layer.head_dim = global_head_dim or text_config.head_dim
        if getattr(text_config, "attention_k_eq_v", False):
            global_kv_heads = getattr(
                text_config, "num_global_key_value_heads", None
            )
            layer.num_key_value_heads = (
                global_kv_heads or text_config.num_key_value_heads
            )
    return layer


class Gemma4TextConfig(PretrainedConfig):
    model_type = "gemma4_text"


class Gemma4AssistantConfig(PretrainedConfig):
    model_type = "gemma4_assistant"

    def __init__(self, text_config=None, **kwargs):
        if text_config is not None and isinstance(text_config, dict):
            self.text_config = Gemma4TextConfig(**text_config)
        elif text_config is not None:
            self.text_config = text_config
        super().__init__(**kwargs)
