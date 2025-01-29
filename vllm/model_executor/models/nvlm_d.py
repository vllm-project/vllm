# adapted from https://huggingface.co/nvidia/NVLM-D-72B/blob/main/modeling_nvlm_d.py
# --------------------------------------------------------
# NVLM-D
# Copyright (c) 2024 NVIDIA
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional

import torch.nn as nn
from transformers import PretrainedConfig

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from .intern_vit import InternVisionModel
from .internvl import (InternVLChatModel, InternVLDummyInputsBuilder,
                       InternVLMultiModalProcessor, InternVLProcessingInfo,
                       InternVLProcessor)

IMG_PAD = "<|vision_pad|>"


class NVLMProcessor(InternVLProcessor):

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_PAD]

    def get_image_repl_features(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> str:
        if num_patches is None:
            raise NotImplementedError("Embedding inputs are not supported")

        tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_patches)]
        if self.use_thumbnail:
            tile_pos_identifiers += ["<tile_global_thumbnail>"]

        context_size = feature_size // num_patches
        return "".join(identifier + IMG_PAD * context_size
                       for identifier in tile_pos_identifiers)

    def get_image_repl_full(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> str:
        features = self.get_image_repl_features(feature_size, num_patches)
        return "<Image>" + features + "</Image>"


class NVLMProcessingInfo(InternVLProcessingInfo):

    def get_hf_processor(
        self,
        *,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> NVLMProcessor:
        return NVLMProcessor(
            self.get_hf_config(),
            self.get_tokenizer(),
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )


@MULTIMODAL_REGISTRY.register_processor(
    InternVLMultiModalProcessor,
    info=NVLMProcessingInfo,
    dummy_inputs=InternVLDummyInputsBuilder)
class NVLM_D_Model(InternVLChatModel):

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Sequential:
        vit_hidden_size = config.vision_config.hidden_size
        llm_intermediate_size = config.text_config.intermediate_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      llm_intermediate_size,
                      bias=False),
            nn.GELU(),
            nn.Linear(llm_intermediate_size, llm_hidden_size, bias=False),
        )

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        is_mono: bool,
        prefix: str,
    ):
        if not is_mono:
            vision_feature_layer = config.select_layer
            if vision_feature_layer < 0:
                num_hidden_layers = config.vision_config.num_hidden_layers \
                    + vision_feature_layer + 1
            else:
                num_hidden_layers = vision_feature_layer + 1

            # We added additional dummy heads to the original num of heads to
            # make the number of heads divisible by 8.
            return InternVisionModel(
                config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                num_dummy_heads=7,
                prefix=prefix,
            )
        else:
            msg = "Monolith mode is not applicable to NVLM_D"
            raise NotImplementedError(msg)
