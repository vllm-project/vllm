# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig, SiglipVisionConfig
from transformers.modeling_rope_utils import rope_config_validation


class CheersTextConfig(PretrainedConfig):
    """Qwen2-based text config with Cheers-specific defaults."""

    model_type = "umm"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=131072,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class CheersConfig(PretrainedConfig):
    """Configuration class for Cheers (UMM) model."""

    model_type = "umm"

    def __init__(
        self,
        text_config: dict | CheersTextConfig | None = None,
        vision_representation_config: dict | SiglipVisionConfig | None = None,
        vae_encoder_config: dict | None = None,
        vae_decoder_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            self.text_config = CheersTextConfig(**text_config)
        else:
            self.text_config = text_config or CheersTextConfig()

        if isinstance(vision_representation_config, dict):
            self.vision_representation_config = SiglipVisionConfig(
                **vision_representation_config
            )
        else:
            self.vision_representation_config = (
                vision_representation_config or SiglipVisionConfig()
            )

        self.vae_encoder_config = vae_encoder_config or {"resolution": 512}
        self.vae_decoder_config = vae_decoder_config or {"resolution": 512}

    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the language model."""
        return self.text_config.hidden_size
