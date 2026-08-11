# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import GlmMoeDsaConfig
from transformers.configuration_utils import PretrainedConfig

from vllm.transformers_utils.configs.kimi_k25 import KimiK25VisionConfig


class Glm5vConfig(PretrainedConfig):
    """MoonViT vision tower and GLM MoE-DSA text model configuration."""

    model_type = "glm5v"

    def __init__(
        self,
        vision_config: dict | KimiK25VisionConfig | None = None,
        text_config: dict | GlmMoeDsaConfig | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 154854,
        pad_token_id: int = 154820,
        use_unified_vision_chunk: bool = True,
        video_placeholder: str = "<|glm5v_video_placeholder|>",
        encoder_only: bool = False,
        language_only: bool = False,
        **kwargs,
    ) -> None:
        outer_quantization_config = kwargs.pop("quantization_config", None)
        raw_text_config = dict(text_config) if isinstance(text_config, dict) else None

        if vision_config is None:
            self.vision_config = KimiK25VisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = KimiK25VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if text_config is None:
            self.text_config = GlmMoeDsaConfig()
        elif isinstance(text_config, dict):
            self.text_config = GlmMoeDsaConfig(**text_config)
        else:
            self.text_config = text_config

        # Transformers 5.5.x can overwrite these GLM DSA values while applying
        # its config attribute map. The generic vLLM repair reads top-level
        # fields, but GLM-5.2 Vision stores them in the nested text config.
        if raw_text_config is not None:
            for key in ("qk_rope_head_dim", "index_topk_freq"):
                if key in raw_text_config:
                    setattr(self.text_config, key, raw_text_config[key])
            if hasattr(self.text_config, "qk_nope_head_dim"):
                self.text_config.qk_head_dim = (
                    self.text_config.qk_nope_head_dim
                    + self.text_config.qk_rope_head_dim
                )

        if self.vision_config.mm_hidden_size == self.vision_config.hidden_size:
            self.vision_config.mm_hidden_size = self.text_config.hidden_size

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.use_unified_vision_chunk = use_unified_vision_chunk
        self.video_placeholder = video_placeholder
        self.encoder_only = encoder_only
        self.language_only = language_only

        super().__init__(pad_token_id=pad_token_id, **kwargs)

        text_quantization_config = getattr(
            self.text_config, "quantization_config", None
        )
        if text_quantization_config is not None:
            self.quantization_config = text_quantization_config
        elif outer_quantization_config is not None:
            self.quantization_config = outer_quantization_config

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def num_nextn_predict_layers(self) -> int:
        return self.text_config.num_nextn_predict_layers

    @property
    def index_topk(self) -> int:
        return self.text_config.index_topk

    @property
    def index_topk_pattern(self) -> str | None:
        return getattr(self.text_config, "index_topk_pattern", None)

    @index_topk_pattern.setter
    def index_topk_pattern(self, value: str | None) -> None:
        self.text_config.index_topk_pattern = value

    @property
    def max_position_embeddings(self) -> int:
        return self.text_config.max_position_embeddings
