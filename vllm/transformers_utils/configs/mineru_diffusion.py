# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig


class SDARConfig(PretrainedConfig):
    """Text backbone config used by MinerU-Diffusion checkpoints."""

    model_type = "sdar"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: dict[str, Any] | None = None,
        attention_bias: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int | None = None,
        max_window_layers: int = 28,
        attention_dropout: float = 0.0,
        **kwargs: Any,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class MinerUDiffusionConfig(PretrainedConfig):
    """Config shim for MinerU-Diffusion checkpoints.

    MinerU-Diffusion combines a Qwen2-VL vision encoder with an SDAR diffusion
    language model.  vLLM uses this class to load checkpoints without
    trust_remote_code and to derive native diffusion defaults.
    """

    model_type = "mineru_diffusion"
    sub_configs = {"vision_config": Qwen2VLVisionConfig, "text_config": SDARConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    architectures = ["MinerUDiffusionForConditionalGeneration"]

    def __init__(
        self,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        vision_config: dict[str, Any] | PretrainedConfig | None = None,
        language_model_config: dict[str, Any] | PretrainedConfig | None = None,
        vision_model_config: dict[str, Any] | PretrainedConfig | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        mask_token_id: int = 151669,
        image_size: int = 512,
        patch_size: int = 16,
        downsample_ratio: float = 0.5,
        vision_projector_type: str = "patch_merger2x",
        vision_select_layer: int = -2,
        canvas_length: int = 32,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ):
        kwargs.pop("rm_vit_merger", None)
        top_level_torch_dtype = kwargs.pop("torch_dtype", None)
        text_config = text_config if text_config is not None else language_model_config
        vision_config = (
            vision_config if vision_config is not None else vision_model_config
        )

        if isinstance(text_config, dict):
            self.text_config = SDARConfig(**text_config)
        elif text_config is None:
            self.text_config = SDARConfig()
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            vision_model_type = vision_config.get("model_type", "qwen2_vl")
            if vision_model_type not in {"qwen2_vl", "qwen2_vl_vision"}:
                raise ValueError(
                    f"Unsupported MinerU-Diffusion vision config: "
                    f"{vision_model_type!r}"
                )
            self.vision_config = Qwen2VLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen2VLVisionConfig()
        else:
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.mask_token_id = mask_token_id
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.vision_projector_type = vision_projector_type
        self.vision_select_layer = vision_select_layer
        self.canvas_length = canvas_length
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=getattr(self.text_config, "bos_token_id", None),
            eos_token_id=getattr(self.text_config, "eos_token_id", None),
            pad_token_id=getattr(self.text_config, "pad_token_id", None),
            **kwargs,
        )
        self.torch_dtype = getattr(
            self.text_config, "torch_dtype", top_level_torch_dtype
        )

    @property
    def language_model_config(self):
        return self.text_config

    @property
    def vision_model_config(self):
        return self.vision_config

    @property
    def vision_model_type(self):
        return getattr(self.vision_config, "model_type", None)

    @property
    def hidden_size(self):
        return self.text_config.hidden_size


__all__ = ["MinerUDiffusionConfig", "SDARConfig"]
