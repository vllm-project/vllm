# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer config resolution for Gemma4 and its variants."""

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
        try:
            layer.head_dim = global_head_dim or text_config.head_dim
        except Exception:
            layer.head_dim = global_head_dim or 256
            
        if getattr(text_config, "attention_k_eq_v", False):
            global_kv_heads = getattr(text_config, "num_global_key_value_heads", None)
            try:
                layer.num_key_value_heads = (
                    global_kv_heads or text_config.num_key_value_heads
                )
            except Exception:
                layer.num_key_value_heads = global_kv_heads or 1
                
    # Fallback to per_layer_config if they raise when accessed
    plc = getattr(text_config, "per_layer_config", None)
    if plc:
        try:
            layer_cfg = plc[layer_idx]
            layer.head_dim = getattr(layer_cfg, "head_dim", getattr(layer, "head_dim", None))
            layer.num_key_value_heads = getattr(layer_cfg, "num_key_value_heads", getattr(layer, "num_key_value_heads", None))
        except Exception:
            pass
            
    return layer
