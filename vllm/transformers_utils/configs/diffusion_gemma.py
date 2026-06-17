# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig


def _init_text_config(self: PretrainedConfig, **kwargs: Any) -> None:
    PretrainedConfig.__init__(self, **kwargs)
    # DiffusionGemma always uses MoE and K=V sharing for full_attention
    # layers. The HF reference removed these config fields entirely.
    if getattr(self, "num_experts", None):
        self.enable_moe_block = True
    self.attention_k_eq_v = True


class DiffusionGemmaTextConfig(PretrainedConfig):
    model_type = "diffusion_gemma_text"

    def __init__(self, **kwargs: Any):
        _init_text_config(self, **kwargs)


class DiffusionGemmaConfig(PretrainedConfig):
    model_type = "diffusion_gemma"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        canvas_length: int = 256,
        self_conditioning_size: int | None = None,
        **kwargs: Any,
    ):
        self.text_config = DiffusionGemmaTextConfig(**(text_config or {}))
        self.canvas_length = canvas_length
        self.self_conditioning_size = self_conditioning_size
        vision_config = kwargs.pop("vision_config", None)
        if isinstance(vision_config, dict):
            self.vision_config = Gemma4VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config
        self.audio_config = None
        PretrainedConfig.__init__(self, **kwargs)
