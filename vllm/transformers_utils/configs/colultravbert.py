# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from transformers import PreTrainedConfig, SiglipVisionConfig


class UltraBertConfig(PreTrainedConfig):
    model_type = "ultrabert"

    def __init__(
        self,
        vocab_size: int = 49280,
        hidden_size: int = 768,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 6,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        sliding_window: int = 128,
        layer_types: list[str] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        **kwargs,
    ):
        if layer_types is None:
            layer_types = [
                "sliding_attention" if (i + 1) % 3 else "full_attention"
                for i in range(num_hidden_layers)
            ]
        if rope_parameters is None:
            rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 2_560_000.0},
            }

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        super().__init__(**kwargs)
        # Set after the base init: transformers < 5.19 adds a shared
        # `rope_theta` key next to the per-layer-type dicts, which 5.17 and
        # 5.18 then reject.
        self.rope_parameters = rope_parameters


class ColUltraVBertConfig(PreTrainedConfig):
    model_type = "ultravbert"

    def __init__(
        self,
        text_config: dict | UltraBertConfig | None = None,
        vision_config: dict | SiglipVisionConfig | None = None,
        image_token_id: int = 49191,
        pixel_shuffle_factor: int = 4,
        connector_norm: bool = True,
        **kwargs,
    ):
        if not isinstance(text_config, UltraBertConfig):
            text_config = {
                k: v
                for k, v in (text_config or {}).items()
                if k not in ("model_type", "auto_map")
            }
            text_config = UltraBertConfig(**text_config)
        if not isinstance(vision_config, SiglipVisionConfig):
            vision_config = {
                k: v for k, v in (vision_config or {}).items() if k != "model_type"
            }
            vision_config = SiglipVisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config: SiglipVisionConfig = vision_config
        self.image_token_id = image_token_id
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.connector_norm = connector_norm
        self.hidden_size = text_config.hidden_size

        kwargs.setdefault("architectures", ["ColUltraVBertForRetrieval"])
        super().__init__(**kwargs)

    @property
    def image_seq_len(self) -> int:
        grid = self.vision_config.image_size // self.vision_config.patch_size
        return grid * grid // self.pixel_shuffle_factor**2

    def get_text_config(self, **kwargs):
        return self.text_config
