# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import transformers


class Granite4VisionConfig(transformers.PretrainedConfig):
    """Configuration for Granite 4 Vision model.

    This config is needed because the granite4_vision model type is not yet
    in the transformers version pinned by vLLM.  Once transformers adds native
    support, this file can be removed and the _CONFIG_REGISTRY entry dropped.
    """

    model_type = "granite4_vision"
    is_composition = False

    def __init__(
        self,
        vision_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | None = None,
        image_token_index: int = 100352,
        image_seq_length: int = 576,
        image_grid_pinpoints: list[list[int]] | None = None,
        vision_feature_select_strategy: str = "full",
        vision_feature_layer: int | list[int] = -2,
        projector_hidden_act: str = "gelu",
        projector_dropout: float = 0.1,
        downsample_rate: str | None = None,
        use_image_newline_parameter: bool = True,
        deepstack_layer_map: list[list[int]] | None = None,
        use_spatial_sampling: bool = False,
        spatial_stride: int = 2,
        spatial_vision_layer: int = -1,
        spatial_target_layers: list[int] | None = None,
        # Hub aliases — base model config uses different field names
        vision_layer_to_llm_layer: list[list[int]] | None = None,
        use_checkerboard_sampling: bool | None = None,
        checkerboard_stride: int | None = None,
        checkerboard_vision_layer: int | None = None,
        checkerboard_llm_layers: list[int] | None = None,
        **kwargs: Any,
    ):
        self.image_token_index = image_token_index
        self.image_seq_length = image_seq_length
        self.image_grid_pinpoints = image_grid_pinpoints or []
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.projector_hidden_act = projector_hidden_act
        self.projector_dropout = projector_dropout
        self.downsample_rate = downsample_rate
        self.use_image_newline_parameter = use_image_newline_parameter
        self.deepstack_layer_map = deepstack_layer_map or vision_layer_to_llm_layer
        self.use_spatial_sampling = (
            use_spatial_sampling
            if use_checkerboard_sampling is None
            else use_checkerboard_sampling
        )
        self.spatial_stride = (
            spatial_stride if checkerboard_stride is None else checkerboard_stride
        )
        self.spatial_vision_layer = (
            spatial_vision_layer
            if checkerboard_vision_layer is None
            else checkerboard_vision_layer
        )
        self.spatial_target_layers = (
            spatial_target_layers or checkerboard_llm_layers or [0, 10, 20, 30]
        )

        if vision_config is None:
            vision_config = {}
        if text_config is None:
            text_config = {}

        vision_model_type = vision_config.get("model_type", "siglip_vision_model")
        if vision_model_type in transformers.CONFIG_MAPPING:
            self.vision_config = transformers.CONFIG_MAPPING[vision_model_type](
                **vision_config
            )
        else:
            self.vision_config = transformers.PretrainedConfig(**vision_config)

        text_model_type = text_config.get("model_type", "granite")
        if text_model_type in transformers.CONFIG_MAPPING:
            self.text_config = transformers.CONFIG_MAPPING[text_model_type](
                **text_config
            )
        else:
            self.text_config = transformers.PretrainedConfig(**text_config)

        super().__init__(**kwargs)
