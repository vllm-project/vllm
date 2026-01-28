# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from Qwen2.5-VL implementation
# Copyright 2025 The vLLM team.
# Copyright 2025 XLANG Lab, The University of Hong Kong

"""Inference-only OpenCUA-7B model compatible with HuggingFace weights."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.qwen2_vl import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.tokenizers import TokenizerLike

from .qwen2_5_vl import (
    Qwen2_5_VisionTransformer as OpenCUAVisionTransformer,
)
from .qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from .qwen2_vl import (
    Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalDataParser,
    Qwen2VLProcessingInfo,
    _create_qwen2vl_field_factory,
)
from .utils import (
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)


class OpenCUAProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_hf_processor(self, **kwargs: object):
        """Load OpenCUA processor."""
        tokenizer = self.get_tokenizer()
        vision_config = self.ctx.get_hf_image_processor_config()
        return OpenCUAProcessor(
            vision_config=vision_config,
            tokenizer=tokenizer,
            **kwargs,
        )


class OpenCUAProcessor(Qwen2VLProcessor):
    def check_argument_for_proper_class(self, attribute_name: str, arg: object) -> None:
        if attribute_name == "tokenizer":
            return
        return super().check_argument_for_proper_class(attribute_name, arg)

    def __init__(
        self,
        vision_config: dict,
        tokenizer: TokenizerLike,
        **kwargs,
    ):
        image_processor = Qwen2VLImageProcessor(**vision_config)
        video_processor = Qwen2VLVideoProcessor(**vision_config)
        chat_template = kwargs.pop("chat_template", None)

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )

        self.image_token = "<|media_placeholder|>"

    def __call__(
        self,
        text=None,
        images=None,
        return_tensors=None,
        **kwargs,
    ):
        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(text, **kwargs)
        else:
            text_inputs = {}

        image_inputs = {}
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            if len(images) > 0:
                image_inputs = self.image_processor(
                    images, return_tensors=return_tensors or "pt"
                )

        combined_inputs = {**text_inputs, **image_inputs}

        return BatchFeature(combined_inputs, tensor_type=return_tensors)


class OpenCUAMultiModalProcessor(BaseMultiModalProcessor[OpenCUAProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return Qwen2VLMultiModalDataParser(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_qwen2vl_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )(hf_inputs)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """vLLM이 prompt 업데이트를 처리하도록 False 반환."""
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        hf_config = self.info.get_hf_config()

        image_token_str = getattr(hf_processor, "image_token", "<|media_placeholder|>")
        image_token_id = vocab.get(
            image_token_str,
            getattr(hf_config, "media_placeholder_token_id", 151664),
        )

        merge_length = image_processor.merge_size**2

        def get_replacement_opencua(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_opencua,
            )
        ]


class OpenCUADummyInputsBuilder(Qwen2VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        image_token = "<|media_placeholder|>"

        return image_token * num_images


@MULTIMODAL_REGISTRY.register_processor(
    OpenCUAMultiModalProcessor,
    info=OpenCUAProcessingInfo,
    dummy_inputs=OpenCUADummyInputsBuilder,
)
class OpenCUAForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "vision_tower.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|media_placeholder|>"
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, "image"):
            self.visual = OpenCUAVisionTransformer(
                vision_config=config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
