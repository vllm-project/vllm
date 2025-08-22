# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# yapf: disable
# ruff: noqa: E501
# adapted from https://huggingface.co/AIDC-AI/Ovis2-1B/blob/main/configuration_aimv2.py
# and https://huggingface.co/AIDC-AI/Ovis2-1B/blob/main/configuration_ovis.py
# Ovis Config with AimV2 config registration removed for Transformers compatibility
from typing import Any, Optional, Union

from transformers import AutoConfig, PretrainedConfig


class AIMv2Config(PretrainedConfig):
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

    model_type: str = "aimv2"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = False,
        use_bias: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps

        self.projection_dropout = projection_dropout
        self.qkv_bias = qkv_bias
        self.use_bias = use_bias


# ----------------------------------------------------------------------
#                     Visual Tokenizer Configuration
# ----------------------------------------------------------------------
class BaseVisualTokenizerConfig(PretrainedConfig):

    def __init__(self,
                 vocab_size=16384,
                 tokenize_function="softmax",
                 tau=1.0,
                 depths=None,
                 drop_cls_token=False,
                 backbone_config: Optional[Union[PretrainedConfig,
                                                 dict]] = None,
                 hidden_stride: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.tokenize_function = tokenize_function
        self.tau = tau
        if isinstance(depths, str):
            depths = [int(x) for x in depths.split('|')]
        self.depths = depths
        self.backbone_kwargs = dict[str, Any]()
        self.drop_cls_token = drop_cls_token
        if backbone_config is not None:
            assert isinstance(backbone_config, (PretrainedConfig, dict)), \
                f"expect `backbone_config` to be instance of PretrainedConfig or dict, but got {type(backbone_config)} type"
            if not isinstance(backbone_config, PretrainedConfig):
                model_type = backbone_config['model_type']
                if model_type != "aimv2":
                    backbone_config.pop('model_type')
                    backbone_config = AutoConfig.for_model(model_type, **backbone_config)
                else:
                    backbone_config = AIMv2Config(**backbone_config)
        self.backbone_config = backbone_config
        self.hidden_stride = hidden_stride


class Aimv2VisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = "aimv2_visual_tokenizer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.drop_cls_token:
            self.drop_cls_token = False
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs['num_hidden_layers'] = self.depths[0]


class SiglipVisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = "siglip_visual_tokenizer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.drop_cls_token:
            self.drop_cls_token = False
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs['num_hidden_layers'] = self.depths[0]


AutoConfig.register("siglip_visual_tokenizer", SiglipVisualTokenizerConfig)
AutoConfig.register("aimv2_visual_tokenizer", Aimv2VisualTokenizerConfig)


# ----------------------------------------------------------------------
#                           Ovis Configuration
# ----------------------------------------------------------------------
class OvisConfig(PretrainedConfig):
    model_type = "ovis"

    def __init__(self,
                 llm_config: Optional[Union[PretrainedConfig, dict]] = None,
                 visual_tokenizer_config: Optional[Union[PretrainedConfig,
                                                         dict]] = None,
                 multimodal_max_length=8192,
                 hidden_size=None,
                 conversation_formatter_class=None,
                 llm_attn_implementation=None,
                 disable_tie_weight=False,
                 **kwargs):
        super().__init__(**kwargs)
        if llm_config is not None:
            assert isinstance(llm_config, (PretrainedConfig, dict)), \
                f"expect `llm_config` to be instance of PretrainedConfig or dict, but got {type(llm_config)} type"
            if not isinstance(llm_config, PretrainedConfig):
                model_type = llm_config['model_type']
                llm_config.pop('model_type')
                llm_config = AutoConfig.for_model(model_type, **llm_config)

        # map llm_config to text_config
        self.text_config = llm_config
        if visual_tokenizer_config is not None:
            assert isinstance(visual_tokenizer_config, (PretrainedConfig, dict)), \
                f"expect `visual_tokenizer_config` to be instance of PretrainedConfig or dict, but got {type(visual_tokenizer_config)} type"
            if not isinstance(visual_tokenizer_config, PretrainedConfig):
                model_type = visual_tokenizer_config['model_type']
                visual_tokenizer_config.pop('model_type')
                visual_tokenizer_config = AutoConfig.for_model(
                    model_type, **visual_tokenizer_config)

        self.visual_tokenizer_config = visual_tokenizer_config
        self.multimodal_max_length = multimodal_max_length
        self.hidden_size = hidden_size
        self.conversation_formatter_class = conversation_formatter_class
        self.llm_attn_implementation = llm_attn_implementation
        self.disable_tie_weight = disable_tie_weight
