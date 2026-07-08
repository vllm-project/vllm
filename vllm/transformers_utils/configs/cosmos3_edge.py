# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers.configuration_utils import PretrainedConfig
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig

from .nemotron_h import NemotronHConfig


def _normalize_mrope_parameters(config: NemotronHConfig) -> None:
    """Expose legacy mRoPE fields in vLLM's canonical configuration.

    Transformers 4.x can leave ``rope_parameters`` unset, while 5.x creates a
    partial dictionary. vLLM's model runner detects 3-axis position tracking
    from ``mrope_section`` in this dictionary, so normalize it during loading.
    """
    if not getattr(config, "enable_mrope", False):
        return

    rope_parameters = dict(getattr(config, "rope_parameters", None) or {})
    rope_parameters.setdefault("rope_type", "default")
    rope_parameters.setdefault("rope_theta", config.rope_theta)
    rope_parameters.setdefault("mrope_section", list(config.mrope_section))
    rope_parameters.setdefault("mrope_interleaved", True)
    config.rope_parameters = rope_parameters


class Cosmos3EdgeProjectorConfig(PretrainedConfig):
    model_type = "qwen3_style_projector"

    def __init__(
        self,
        input_hidden_size: int = 1024,
        use_postshuffle_norm: bool = False,
        spatial_merge_size: int = 2,
        merger_intermedia: int = 11520,
        out_hidden_size: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_hidden_size = input_hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.spatial_merge_size = spatial_merge_size
        self.merger_intermedia = merger_intermedia
        self.out_hidden_size = out_hidden_size


class Cosmos3EdgeConfig(PretrainedConfig):
    model_type = "cosmos3_edge"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "vision_config": Siglip2VisionConfig,
        "projector_config": Cosmos3EdgeProjectorConfig,
        "text_config": NemotronHConfig,
    }

    def __init__(
        self,
        text_config: NemotronHConfig | dict | None = None,
        vision_config: Siglip2VisionConfig | dict | None = None,
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
            NemotronHConfig(**text_config)
            if isinstance(text_config, dict)
            else text_config
        )
        _normalize_mrope_parameters(self.text_config)

        self.vision_config = (
            Siglip2VisionConfig(**vision_config)
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
        self.vision_config.spatial_merge_size = (
            self.projector_config.spatial_merge_size
        )
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
]
