# SPDX-License-Identifier: Apache-2.0

# adapted from https://huggingface.co/nvidia/NVLM-D-72B/blob/main/modeling_nvlm_d.py
# --------------------------------------------------------
# NVLM-D
# Copyright (c) 2024 NVIDIA
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
from collections.abc import Mapping, Sequence
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalKwargs
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)

from .intern_vit import InternVisionModel
from .internvl import (BaseInternVLProcessingInfo, BaseInternVLProcessor,
                       InternVLChatModel, InternVLDummyInputsBuilder,
                       InternVLMultiModalProcessor)

IMG_PAD = "<|vision_pad|>"


class NVLMProcessor(BaseInternVLProcessor):

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_PAD]

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> PromptUpdateDetails[str]:
        if num_patches is None:
            raise NotImplementedError("Embedding inputs are not supported")

        tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_patches)]
        if self.use_thumbnail:
            tile_pos_identifiers += ["<tile_global_thumbnail>"]

        context_size = feature_size // num_patches
        features = "".join(identifier + IMG_PAD * context_size
                           for identifier in tile_pos_identifiers)

        # We include the start and end as well because "<Image><tile" is
        # tokenized as ["<Image", "><", "tile"], resulting in assertion error
        # when trying to find "<tile" as a subsequence of "<Image><tile"
        repl = "<Image>" + features + "</Image>"

        return PromptUpdateDetails.select_text(repl, IMG_PAD)


class NVLMProcessingInfo(BaseInternVLProcessingInfo):

    def get_hf_processor(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        **kwargs: object,
    ) -> NVLMProcessor:
        if min_dynamic_patch is not None:
            kwargs["min_dynamic_patch"] = min_dynamic_patch
        if max_dynamic_patch is not None:
            kwargs["max_dynamic_patch"] = max_dynamic_patch
        if dynamic_image_size is not None:
            kwargs["dynamic_image_size"] = dynamic_image_size

        return self.ctx.init_processor(
            NVLMProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )


class NVLMDummyInputsBuilder(InternVLDummyInputsBuilder[NVLMProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        # The newline is necessary to separate ">" of the current item
        # and "<" of the next item
        return "<image>\n" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        num_images = mm_counts.get("image", 0)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class NVLMMultiModalProcessor(InternVLMultiModalProcessor[NVLMProcessingInfo]):

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        if "image_num_patches" in out_mm_kwargs:
            image_num_patches = out_mm_kwargs["image_num_patches"]
            assert isinstance(image_num_patches, torch.Tensor)
            image_num_patches = image_num_patches.tolist()
        elif "image_embeds" in out_mm_kwargs:
            # TODO: Use image size information in dictionary embedding inputs
            # to compute num_patches (similar to Qwen2-VL)
            image_num_patches = [None] * len(out_mm_kwargs["image_embeds"])
        else:
            image_num_patches = []

        def get_replacement_nvlm(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                feature_size = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )

            num_patches = image_num_patches[item_idx]
            if num_patches is not None:
                assert isinstance(num_patches, int)

            repl = hf_processor.get_image_repl(feature_size, num_patches)

            return PromptUpdateDetails.select_text(repl.full + "\n", IMG_PAD)

        # See note in dummy data regarding why we have the extra newline
        return [
            PromptReplacement(
                modality="image",
                target="<image>\n",
                replacement=get_replacement_nvlm,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(NVLMMultiModalProcessor,
                                        info=NVLMProcessingInfo,
                                        dummy_inputs=NVLMDummyInputsBuilder)
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
