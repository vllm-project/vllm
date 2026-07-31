# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers.configuration_utils import PretrainedConfig
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig

from .nemotron_h import NemotronHConfig


def _normalize_mrope_parameters(config: NemotronHConfig) -> None:
    """Normalize the checkpoint's mRoPE configuration for vLLM."""
    rope_parameters = dict(getattr(config, "rope_parameters", None) or {})
    mrope_section = rope_parameters.get("mrope_section")
    if mrope_section is None:
        return

    rope_parameters["mrope_section"] = list(mrope_section)
    rope_parameters.setdefault("mrope_interleaved", True)
    config.rope_parameters = rope_parameters


class Cosmos3EdgeTextConfig(NemotronHConfig):
    """Dense Cosmos3 Edge text config backed by Nemotron-H layers."""

    model_type = "cosmos3_edge_text"
    has_no_defaults_at_init = True
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __init__(
        self,
        num_hidden_layers: int = 28,
        hidden_act: str = "relu2",
        rms_norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(
            num_hidden_layers=2 * num_hidden_layers,
            hybrid_override_pattern="*-" * num_hidden_layers,
            mlp_hidden_act=hidden_act,
            layer_norm_epsilon=rms_norm_eps,
            **kwargs,
        )

    def to_dict(self) -> dict:
        config_dict = super().to_dict()
        config_dict["num_hidden_layers"] = self.num_hidden_layers // 2
        config_dict["hidden_act"] = config_dict.pop("mlp_hidden_act")
        config_dict["rms_norm_eps"] = config_dict.pop("layer_norm_epsilon")
        config_dict.pop("hybrid_override_pattern", None)
        return config_dict


class Cosmos3EdgeProjectorConfig(PretrainedConfig):
    model_type = "cosmos3_edge_projector"

    def __init__(
        self,
        input_hidden_size: int = 1024,
        use_postshuffle_norm: bool = False,
        spatial_merge_size: int = 2,
        merger_intermediate_size: int = 11520,
        out_hidden_size: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_hidden_size = input_hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.spatial_merge_size = spatial_merge_size
        self.merger_intermediate_size = merger_intermediate_size
        self.out_hidden_size = out_hidden_size


class Cosmos3EdgeVisionConfig(Siglip2VisionConfig):
    model_type = "cosmos3_edge_vision"


class Cosmos3EdgeConfig(PretrainedConfig):
    model_type = "cosmos3_edge"
    keys_to_ignore_at_inference = ["past_key_values"]
    has_no_defaults_at_init = True
    sub_configs = {
        "vision_config": Cosmos3EdgeVisionConfig,
        "projector_config": Cosmos3EdgeProjectorConfig,
        "text_config": Cosmos3EdgeTextConfig,
    }

    def __init__(
        self,
        text_config: Cosmos3EdgeTextConfig | dict | None = None,
        vision_config: Cosmos3EdgeVisionConfig | dict | None = None,
        projector_config: Cosmos3EdgeProjectorConfig | dict | None = None,
        image_token_id: int = 19,
        video_token_id: int = 18,
        vision_start_token_id: int = 20,
        vision_end_token_id: int = 21,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        if text_config is None:
            text_config = {}
        if vision_config is None:
            vision_config = {}
        if projector_config is None:
            projector_config = {}

        self.text_config = (
            Cosmos3EdgeTextConfig(**text_config)
            if isinstance(text_config, dict)
            else text_config
        )
        _normalize_mrope_parameters(self.text_config)

        self.vision_config = (
            Cosmos3EdgeVisionConfig(**vision_config)
            if isinstance(vision_config, dict)
            else vision_config
        )
        self.projector_config = (
            Cosmos3EdgeProjectorConfig(**projector_config)
            if isinstance(projector_config, dict)
            else projector_config
        )

        # Qwen3-VL processing and M-RoPE read these attributes from the
        # vision config, while this checkpoint stores them in projector config.
        self.vision_config.spatial_merge_size = self.projector_config.spatial_merge_size
        self.vision_config.temporal_patch_size = 1
        self.vision_config.out_hidden_size = self.projector_config.out_hidden_size

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = [
    "Cosmos3EdgeConfig",
    "Cosmos3EdgeProjectorConfig",
    "Cosmos3EdgeTextConfig",
    "Cosmos3EdgeVisionConfig",
]
