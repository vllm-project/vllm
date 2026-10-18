# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agnes 3.0 model configuration"""

from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

AGNES_GLOBAL_ATTENTION = "agnes_global_attention"


class AgnesTextConfig(Qwen3_5TextConfig):
    """Agnes's text tower, which is Qwen3.5's plus a parallel FFN branch."""

    model_type = "qwen3_5_text"

    def __init__(self, layer_types=None, parallel_ffn_intermediate_size=0, **kwargs):
        kwargs.pop("model_type", None)
        kwargs.pop("auto_map", None)

        interval = kwargs.pop("global_attention_interval", None)
        if interval is None and layer_types:
            interval = next(
                (
                    i + 1
                    for i, t in enumerate(layer_types)
                    if t == AGNES_GLOBAL_ATTENTION
                ),
                None,
            )
        if interval is not None:
            kwargs["full_attention_interval"] = int(interval)

        # Agnes spells Qwen3.5's two layer types differently; let the parent
        # regenerate the plan from the interval instead of translating names.
        super().__init__(layer_types=None, **kwargs)

        self.parallel_ffn_intermediate_size = int(parallel_ffn_intermediate_size or 0)


class AgnesConfig(Qwen3_5Config):
    model_type = "agnes"
    sub_configs = {
        "vision_config": Qwen3_5VisionConfig,
        "text_config": AgnesTextConfig,
    }

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        kwargs.pop("auto_map", None)  # vLLM implements the model itself

        if isinstance(vision_config, dict):
            vision_config = {
                k: v for k, v in vision_config.items() if k != "model_type"
            }

        super().__init__(text_config=text_config, vision_config=vision_config, **kwargs)


__all__ = ["AgnesConfig", "AgnesTextConfig"]
