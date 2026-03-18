# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/h2oai/h2ovl-mississippi-2b/blob/main/modeling_h2ovl_chat.py
# https://huggingface.co/h2oai/h2ovl-mississippi-2b/blob/main/image_process.py
# --------------------------------------------------------
# H2OVL-Mississippi
# Copyright (c) 2024 H2O.AI
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
from collections.abc import Mapping, Sequence

import torch
from transformers import PretrainedConfig

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    MultiModalDataItems,
)
from vllm.multimodal.processing.processor import (
    MultiModalProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    TimingContext,
)
from vllm.transformers_utils.processors.h2ovl import H2OVLImageProcessor, H2OVLProcessor

from .intern_vit import InternVisionModel
from .internvl import (
    BaseInternVLDummyInputsBuilder,
    BaseInternVLMultiModalProcessor,
    BaseInternVLProcessingInfo,
    InternVLChatModel,
)


class H2OVLProcessingInfo(BaseInternVLProcessingInfo):
    def get_image_processor(self, **kwargs):
        config = self.get_hf_config()
        vision_config = config.vision_config

        kwargs = self.ctx.get_merged_mm_kwargs(kwargs)
        kwargs.setdefault("image_size", vision_config.image_size)
        kwargs.setdefault("min_dynamic_patch", config.min_dynamic_patch)
        kwargs.setdefault("max_dynamic_patch", config.max_dynamic_patch)
        kwargs.setdefault("dynamic_image_size", config.dynamic_image_size)
        kwargs.setdefault("use_thumbnail", config.use_thumbnail)
        kwargs.setdefault("use_msac", config.use_msac)

        return H2OVLImageProcessor(**kwargs)

    def get_hf_processor(self, **kwargs: object) -> H2OVLProcessor:
        config = self.get_hf_config()
        vision_config = config.vision_config

        image_processor = self.get_image_processor(**kwargs)
        image_size = image_processor.image_size
        patch_size = vision_config.patch_size
        downsample_ratio = config.downsample_ratio
        image_seq_length = int((image_size // patch_size) ** 2 * (downsample_ratio**2))

        return H2OVLProcessor(
            tokenizer=self.get_tokenizer(),
            image_processor=image_processor,
            image_seq_length=image_seq_length,
        )

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: H2OVLProcessor,
        use_msac: bool | None = None,
    ) -> int:
        return processor.get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
            use_msac=use_msac,
        )


class H2OVLMultiModalProcessor(BaseInternVLMultiModalProcessor[H2OVLProcessingInfo]):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        out_mm_data = out_mm_kwargs.get_data()
        if "image_num_patches" in out_mm_data:
            image_num_patches = out_mm_data["image_num_patches"]
            assert isinstance(image_num_patches, torch.Tensor)
            image_num_patches = image_num_patches.tolist()
        elif "image_embeds" in out_mm_data:
            # TODO: Use image size information in dictionary embedding inputs
            # to compute num_patches (similar to Qwen2-VL)
            image_num_patches = [None] * len(out_mm_data["image_embeds"])
        else:
            image_num_patches = []

        num_images = len(image_num_patches)

        def get_replacement_internvl(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                feature_size = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                    use_msac=None if num_images == 1 else False,
                )

            num_patches = image_num_patches[item_idx]
            if num_patches is not None:
                assert isinstance(num_patches, int)

            return hf_processor.get_image_repl(num_patches, num_features=feature_size)

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement_internvl,
            )
        ]

    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        # The processor logic is different for len(images) <= 1 vs > 1
        # Since the processing cache assumes that the processor output is
        # invariant of how many images are passed per prompt, we only
        # perform caching for the most common case
        if inputs.mm_data_items.get_count("image", strict=False) > 1:
            return self._apply_hf_processor(inputs, timing_ctx)

        return super()._cached_apply_hf_processor(inputs, timing_ctx)


@MULTIMODAL_REGISTRY.register_processor(
    H2OVLMultiModalProcessor,
    info=H2OVLProcessingInfo,
    dummy_inputs=BaseInternVLDummyInputsBuilder,
)
class H2OVLChatModel(InternVLChatModel):
    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        *,
        is_mono: bool,
        prefix: str,
    ):
        if not is_mono:
            vision_feature_layer = config.select_layer
            if vision_feature_layer < 0:
                num_hidden_layers = (
                    config.vision_config.num_hidden_layers + vision_feature_layer + 1
                )
            else:
                num_hidden_layers = vision_feature_layer + 1

            return InternVisionModel(
                config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                prefix=prefix,
            )
        else:
            msg = "Monolith mode is not applicable to H2OVL"
            raise NotImplementedError(msg)
