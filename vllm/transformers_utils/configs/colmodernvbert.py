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
        vlm_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim

        if vlm_config is None:
            vlm_config = {}

        # Top-level VLM fields
        self.image_token_id = vlm_config.get("image_token_id", 50407)
        self.pixel_shuffle_factor = vlm_config.get("pixel_shuffle_factor", 4)
        self.hidden_size = vlm_config.get("hidden_size", 768)
        additional_vocab_size = vlm_config.get("additional_vocab_size", 40)

        # Text config (ModernBERT)
        text_cfg = vlm_config.get("text_config", {})
        base_vocab = text_cfg.get("vocab_size", 50368)
        self.text_config = ModernBertConfig(
            vocab_size=base_vocab + additional_vocab_size,
            hidden_size=text_cfg.get("hidden_size", 768),
            intermediate_size=text_cfg.get("intermediate_size", 1152),
            num_hidden_layers=text_cfg.get("num_hidden_layers", 22),
            num_attention_heads=text_cfg.get("num_attention_heads", 12),
            mlp_bias=text_cfg.get("mlp_bias", False),
            max_position_embeddings=vlm_config.get("max_position_embeddings", 8192),
        )

        # Vision config (SigLIP)
        vis_cfg = vlm_config.get("vision_config", {})
        self.vision_config = SiglipVisionConfig(
            hidden_size=vis_cfg.get("embed_dim", 768),
            image_size=vis_cfg.get("image_size", 512),
            patch_size=vis_cfg.get("patch_size", 16),
            num_hidden_layers=vis_cfg.get("num_hidden_layers", 12),
            intermediate_size=vis_cfg.get("intermediate_size", 3072),
            num_attention_heads=vis_cfg.get("num_attention_heads", 12),
        )

    @property
    def image_seq_len(self) -> int:
        ps = self.vision_config.image_size // self.vision_config.patch_size
        return (ps * ps) // (self.pixel_shuffle_factor**2)

    def get_text_config(self, **kwargs):
        return self.text_config
