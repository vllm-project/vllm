# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: E501
# adapted from https://huggingface.co/AIDC-AI/Ovis2-1B/blob/main/configuration_ovis.py
from typing import Any

from transformers import Aimv2VisionConfig, AutoConfig, PreTrainedConfig


# ----------------------------------------------------------------------
#                     Visual Tokenizer Configuration
# ----------------------------------------------------------------------
class BaseVisualTokenizerConfig(PreTrainedConfig):
    def __init__(
        self,
        vocab_size=16384,
        tokenize_function="softmax",
        tau=1.0,
        depths=None,
        drop_cls_token=False,
        backbone_config: PreTrainedConfig | dict | None = None,
        hidden_stride: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.tokenize_function = tokenize_function
        self.tau = tau
        if isinstance(depths, str):
            depths = [int(x) for x in depths.split("|")]
        self.depths = depths
        self.backbone_kwargs = dict[str, Any]()
        self.drop_cls_token = drop_cls_token
        if backbone_config is not None:
            assert isinstance(backbone_config, (PreTrainedConfig, dict)), (
                f"expect `backbone_config` to be instance of PreTrainedConfig or dict, but got {type(backbone_config)} type"
            )
            if not isinstance(backbone_config, PreTrainedConfig):
                model_type = backbone_config.pop("model_type")
                if model_type == "aimv2":
                    backbone_config = Aimv2VisionConfig(**backbone_config)
                else:
                    backbone_config = AutoConfig.for_model(
                        model_type, **backbone_config
                    )
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
            self.backbone_kwargs["num_hidden_layers"] = self.depths[0]


class SiglipVisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = "siglip_visual_tokenizer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.drop_cls_token:
            self.drop_cls_token = False
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs["num_hidden_layers"] = self.depths[0]


AutoConfig.register("siglip_visual_tokenizer", SiglipVisualTokenizerConfig)
AutoConfig.register("aimv2_visual_tokenizer", Aimv2VisualTokenizerConfig)


# ----------------------------------------------------------------------
#                           Ovis Configuration
# ----------------------------------------------------------------------
class OvisConfig(PreTrainedConfig):
    model_type = "ovis"

    def __init__(
        self,
        llm_config: PreTrainedConfig | dict | None = None,
        visual_tokenizer_config: PreTrainedConfig | dict | None = None,
        multimodal_max_length=8192,
        hidden_size=None,
        conversation_formatter_class=None,
        llm_attn_implementation=None,
        disable_tie_weight=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if llm_config is not None:
            assert isinstance(llm_config, (PreTrainedConfig, dict)), (
                f"expect `llm_config` to be instance of PreTrainedConfig or dict, but got {type(llm_config)} type"
            )
            if not isinstance(llm_config, PreTrainedConfig):
                model_type = llm_config["model_type"]
                llm_config.pop("model_type")
                llm_config = AutoConfig.for_model(model_type, **llm_config)

        # map llm_config to text_config
        self.text_config = llm_config
        if visual_tokenizer_config is not None:
            assert isinstance(visual_tokenizer_config, (PreTrainedConfig, dict)), (
                f"expect `visual_tokenizer_config` to be instance of PreTrainedConfig or dict, but got {type(visual_tokenizer_config)} type"
            )
            if not isinstance(visual_tokenizer_config, PreTrainedConfig):
                model_type = visual_tokenizer_config["model_type"]
                visual_tokenizer_config.pop("model_type")
                visual_tokenizer_config = AutoConfig.for_model(
                    model_type, **visual_tokenizer_config
                )

        self.visual_tokenizer_config = visual_tokenizer_config
        self.multimodal_max_length = multimodal_max_length
        self.hidden_size = hidden_size
        self.conversation_formatter_class = conversation_formatter_class
        self.llm_attn_implementation = llm_attn_implementation
        self.disable_tie_weight = disable_tie_weight
