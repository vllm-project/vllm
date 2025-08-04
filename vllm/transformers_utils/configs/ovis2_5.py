# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional, Union

from transformers import AutoConfig, PretrainedConfig

# Model Constants
IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"
INDICATOR_IDS = [-301, -302, -303, -304]


class Siglip2NavitConfig(PretrainedConfig):
    """This is the configuration class to store the configuration of an [`AIMv2Model`].
    Instantiating a configuration with the defaults will yield a similar configuration
    to that of the [apple/aimv2-large-patch14-224](https://huggingface.co/apple/aimv2-large-patch14-224).
    Args:
        hidden_size: Dimension of the hidden representations.
        intermediate_size: Dimension of the SwiGLU representations.
        num_hidden_layers: Number of hidden layers in the Transformer.
        num_attention_heads: Number of attention heads for each attention layer
            in the Transformer.
        num_channels: Number of input channels.
        image_size: Image size.
        patch_size: Patch size.
        rms_norm_eps: Epsilon value used for the RMS normalization layer.
        attention_dropout: Dropout ratio for attention probabilities.
        projection_dropout: Dropout ratio for the projection layer after the attention.
        qkv_bias: Whether to add a bias to the queries, keys and values.
        use_bias: Whether to add a bias in the feed-forward and projection layers.
        kwargs: Keyword arguments for the [`PretrainedConfig`].
    """
    model_type: str = "siglip2_navit"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        num_patches: int = -1,
        image_size: int = 512,
        patch_size: int = 16,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_stride: int = 2,
        window_size: int = 112,
        fullatt_block_indexes: Optional[list] = None,
        temporal_patch_size: int = 1,
        preserve_original_pe: bool = True,
        use_rope: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_stride = hidden_stride
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.temporal_patch_size = temporal_patch_size
        self.preserve_original_pe = preserve_original_pe
        self.use_rope = use_rope


AutoConfig.register("siglip2_navit", Siglip2NavitConfig)

# ----------------------------------------------------------------------
#                           Ovis Configuration
# ----------------------------------------------------------------------


class Ovis2_5Config(PretrainedConfig):
    model_type = "ovis2_5"

    # sub_configs = dict(llm_config=Qwen3Config, vit_config=Siglip2NavitConfig)
    def __init__(self,
                 llm_config: Optional[Union[PretrainedConfig, dict]] = None,
                 vit_config: Optional[Union[PretrainedConfig, dict]] = None,
                 visual_vocab_size=65536,
                 hidden_size=None,
                 conversation_formatter_class=None,
                 **kwargs):
        super().__init__(**kwargs)
        if llm_config is not None:
            assert isinstance(llm_config, (PretrainedConfig, dict)), \
                f"expect `llm_config` to be instance of PretrainedConfig or dict, but got {type(llm_config)} type"
            if not isinstance(llm_config, PretrainedConfig):
                model_type = llm_config['model_type']
                llm_config.pop('model_type')
                llm_config = AutoConfig.for_model(model_type, **llm_config)
        self.llm_config = llm_config
        if vit_config is not None:
            assert isinstance(vit_config, (PretrainedConfig, dict)), \
                f"expect `vit_config` to be instance of PretrainedConfig or dict, but got {type(vit_config)} type"
            if not isinstance(vit_config, PretrainedConfig):
                model_type = vit_config['model_type']
                vit_config.pop('model_type')
                vit_config = AutoConfig.for_model(model_type, **vit_config)
        self.vit_config = vit_config
        self.visual_vocab_size = visual_vocab_size
        self.hidden_size = hidden_size
        self.conversation_formatter_class = conversation_formatter_class
        if kwargs.get('attn_implementation'):
            self.llm_config._attn_implementation = kwargs[
                'attn_implementation']
            self.vit_config._attn_implementation = kwargs[
                'attn_implementation']
