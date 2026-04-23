# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for ColModernVBERT visual document retrieval model.

ColModernVBERT combines SigLIP vision encoder + ModernBERT text encoder
with a pixel shuffle connector and ColBERT-style 128-dim per-token embeddings.

Reference: https://huggingface.co/ModernVBERT/colmodernvbert-merged
"""

from transformers import ModernBertConfig, PretrainedConfig, SiglipVisionConfig


class ColModernVBertConfig(PretrainedConfig):
    model_type = "colmodernvbert"

    def __init__(
        self,
        embedding_dim: int = 128,
        image_token_id: int = 50407,
        pixel_shuffle_factor: int = 4,
        text_config: dict | None = None,
        vision_config: dict | None = None,
        **kwargs,
    ):
        self.embedding_dim = embedding_dim
        self.image_token_id = image_token_id
        self.pixel_shuffle_factor = pixel_shuffle_factor

        text_config = text_config or {}
        self.hidden_size = text_config.get("hidden_size", 768)

        self.text_config = ModernBertConfig(
            vocab_size=text_config.get("vocab_size", 50408),
            hidden_size=text_config.get("hidden_size", 768),
            intermediate_size=text_config.get("intermediate_size", 1152),
            num_hidden_layers=text_config.get("num_hidden_layers", 22),
            num_attention_heads=text_config.get("num_attention_heads", 12),
            mlp_bias=text_config.get("mlp_bias", False),
            max_position_embeddings=text_config.get("max_position_embeddings", 8192),
        )

        vision_config = vision_config or {}
        self.vision_config = SiglipVisionConfig(
            hidden_size=vision_config.get("hidden_size", 768),
            image_size=vision_config.get("image_size", 512),
            patch_size=vision_config.get("patch_size", 16),
            num_hidden_layers=vision_config.get("num_hidden_layers", 12),
            intermediate_size=vision_config.get("intermediate_size", 3072),
            num_attention_heads=vision_config.get("num_attention_heads", 12),
        )

        # Ensure architectures is set so vLLM routes to our model class
        kwargs.setdefault("architectures", ["ColModernVBertForRetrieval"])
        super().__init__(**kwargs)

    @property
    def image_seq_len(self) -> int:
        ps = self.vision_config.image_size // self.vision_config.patch_size
        return (ps * ps) // (self.pixel_shuffle_factor**2)

    def get_text_config(self, **kwargs):
        return self.text_config
