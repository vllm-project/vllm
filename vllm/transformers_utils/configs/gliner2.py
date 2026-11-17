# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PreTrainedConfig


class GLiNER2Config(PreTrainedConfig):
    model_type = "gliner2"

    def __init__(self, encoder_config: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder_config = encoder_config or {}
        # The DeBERTa encoder is bidirectional and does not use a KV cache.
        self.num_hidden_layers = 0
        self.num_attention_heads = 1
        self.hidden_size = self.encoder_config.get("hidden_size", 1024)
        self.max_position_embeddings = self.encoder_config.get(
            "max_position_embeddings", 512
        )
        self.vocab_size = self.encoder_config.get("vocab_size", 128011)
