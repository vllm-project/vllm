# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Radio vision model configuration"""

from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

VIT_TIMM_DIM_BY_NAME: dict[str, tuple[int, int, int, int]] = {
    "vit_small_patch16_224": (384, 12, 6, 1536),
    "vit_base_patch16_224": (768, 12, 12, 3072),
    "vit_large_patch16_224": (1024, 24, 16, 4096),
    "vit_huge_patch16_224": (1280, 32, 16, 5120),
}

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class RadioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a Radio
    vision model. It is used to instantiate a Radio model according to the
    specified arguments, defining the model architecture.

    Args:
        model_name: Name of the vision transformer model
            (e.g., "vit_base_patch16_224"). Used to determine architecture
            dimensions from `VIT_TIMM_DIM_BY_NAME`.
        image_size: The size (resolution) of each image.
        patch_size: The size (resolution) of each patch.
        qkv_bias: Whether to add a bias to the queries, keys and values.
        qk_normalization: Whether to apply normalization to queries and keys.
        norm_type: The normalization type to use.
        layer_norm_eps: The epsilon used by the layer normalization layers.
        initializer_factor: A factor for initializing all weight matrices.
        hidden_act: The non-linear activation function in the encoder.
        cpe_max_size: Maximum image size for position embeddings.
        norm_mean: Mean values for image normalization (RGB channels).
            Defaults to (0.48145466, 0.4578275, 0.40821073)).
        norm_std: Standard deviation values for image normalization
            (RGB channels). Defaults to (0.26862954, 0.26130258, 0.27577711)).
        register_multiple: Number of register tokens to use.
        teachers: A list of teacher model configurations. Each teacher configuration is
            a dict with keys like "name" and some may have "use_summary".
        cls_token_per_teacher: Whether to use a separate CLS token for each teacher.
    """

    model_type = "radio"

    def __init__(
        self,
        model_name: str,
        image_size: int = 224,
        patch_size: int = 16,
        qkv_bias: bool = True,
        qk_normalization: bool = False,
        norm_type: str = "layer_norm",
        layer_norm_eps: float = 1e-6,
        initializer_factor: float = 1.0,
        hidden_act: str = "gelu",
        cpe_max_size: int = 2048,
        norm_mean: tuple[float, float, float] | list = OPENAI_CLIP_MEAN,
        norm_std: tuple[float, float, float] | list = OPENAI_CLIP_STD,
        register_multiple: int | None = None,
        teachers: list[dict[str, Any]] | None = None,
        cls_token_per_teacher: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        (
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.intermediate_size,
        ) = VIT_TIMM_DIM_BY_NAME[model_name]
        self.image_size = image_size
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.qk_normalization = qk_normalization
        self.norm_type = norm_type
        self.layer_norm_eps = layer_norm_eps
        self.initializer_factor = initializer_factor
        self.hidden_act = hidden_act
        self.cpe_max_size = cpe_max_size
        self.norm_mean = (
            list(norm_mean) if isinstance(norm_mean, (tuple, list)) else norm_mean
        )
        self.norm_std = (
            list(norm_std) if isinstance(norm_std, (tuple, list)) else norm_std
        )
        self.register_multiple = register_multiple
        self.teachers = teachers if teachers is not None else []
        self.cls_token_per_teacher = cls_token_per_teacher
        super().__init__(**kwargs)
