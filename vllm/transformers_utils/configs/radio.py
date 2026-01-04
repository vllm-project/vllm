# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Radio vision model configuration"""

from typing import Optional, Union

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
        model_name (`str`, *optional*, defaults to "vit_base_patch16_224"):
            Name of the vision transformer model (e.g., "vit_base_patch16_224").
            Used to determine architecture dimensions from
            `VIT_TIMM_DIM_BY_NAME`.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        qkv_bias (`bool`, *optional*, defaults to True):
            Whether to add a bias to the queries, keys and values.
        qk_normalization (`bool`, *optional*, defaults to False):
            Whether to apply normalization to queries and keys.
        norm_type (`str`, *optional*, defaults to "layer_norm"):
            The normalization type to use.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices.
        hidden_act (`str`, *optional*, defaults to "gelu"):
            The non-linear activation function in the encoder.
        max_img_size (`int`, *optional*, defaults to 2048):
            Maximum image size for position embeddings.
        norm_mean (`tuple` or `list`, *optional*,
            defaults to (0.48145466, 0.4578275, 0.40821073)):
            Mean values for image normalization (RGB channels).
        norm_std (`tuple` or `list`, *optional*,
            defaults to (0.26862954, 0.26130258, 0.27577711)):
            Standard deviation values for image normalization (RGB channels).
        reg_tokens (`int`, *optional*):
            Number of register tokens to use.
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
        max_img_size: int = 2048,
        norm_mean: Union[tuple[float, float, float], list] = OPENAI_CLIP_MEAN,
        norm_std: Union[tuple[float, float, float], list] = OPENAI_CLIP_STD,
        reg_tokens: Optional[int] = None,
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
        self.max_img_size = max_img_size
        self.norm_mean = list(norm_mean) if isinstance(norm_mean,
                                                       (tuple,
                                                        list)) else norm_mean
        self.norm_std = list(norm_std) if isinstance(norm_std,
                                                     (tuple,
                                                      list)) else norm_std
        self.reg_tokens = reg_tokens
        super().__init__(**kwargs)
