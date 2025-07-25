# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Optional, TypedDict, TypeVar, Union

import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import BatchEncoding, PretrainedConfig, TensorType
from transformers import AutoProcessor
from transformers.models.internvl import InternVLProcessor
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.interns1_vit import InternS1VisionModel
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.processor import (
    cached_image_processor_from_config)

from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

from transformers.models.got_ocr2.image_processing_got_ocr2_fast import GotOcr2ImageProcessorFast 

IMG_START = '<img>'
IMG_END = '</img>'
IMG_CONTEXT = '<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values_flat: torch.Tensor
    """
    Shape:
    `(batch_size * num_images * (1 + num_patches), num_channels, height, width)`
    """

    num_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""


class InternVLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Union[torch.Tensor, list[torch.Tensor]]
    """ 
    A tensor of shape `(num_images, total_image_feature_size, hidden_size)`
    or a list of tensors of shape `(total_image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


InternVLImageInputs = Union[InternVLImagePixelInputs,
                            InternVLImageEmbeddingInputs]


class InternVLVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    pixel_values_flat: torch.Tensor
    """
    Shape:
    `(batch_size * num_video * num_frames, num_channels, height, width)`
    """

    num_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""


class InternVLVideoEmbeddingInputs(TypedDict):
    type: Literal["video_embeds"]
    data: Union[torch.Tensor, list[torch.Tensor]]
    """ 
    A tensor of shape `(num_videos, total_video_feature_size, hidden_size)`
    or a list of tensors of shape `(total_video_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


InternVLVideoInputs = Union[InternVLVideoPixelInputs,
                            InternVLVideoEmbeddingInputs]


def resolve_internvl_min_max_num(
    *,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]:
    min_dynamic_patch = min_dynamic_patch if dynamic_image_size else 1
    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1

    if use_thumbnail and max_dynamic_patch != 1:
        max_dynamic_patch += 1

    return min_dynamic_patch, max_dynamic_patch


def get_internvl_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}
    return sorted(target_ratios, key=lambda x: x[0] * x[1])

def resolve_min_max_num(
        self,
        min_dynamic_patch: int,
        max_dynamic_patch: int,
        dynamic_image_size: bool,
        use_thumbnail: bool,
    ) -> tuple[int, int]:
        return resolve_internvl_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

class BaseInternS1ProcessingInfo(BaseProcessingInfo):
    """Basic image-only ProcessingInfo for InternS1-style models."""

    @abstractmethod
    def get_hf_processor(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        **kwargs: object,
    ):
        raise NotImplementedError

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional['GotOcr2ImageProcessorFast'] = None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor().image_processor

        if not isinstance(processor, GotOcr2ImageProcessorFast):
            raise ValueError(f'GotOcr2ImageProcessorFast is expected but got {type(processor)}')
        num_image_patches = processor.get_number_of_image_tokens(image_height, image_width)
        num_image_tokens = 2 + self.get_hf_processor().image_seq_length * num_image_patches
        return num_image_tokens

    def resolve_target_ratios(self, use_thumbnail: Optional[bool] = None):
        image_processor = self.get_hf_processor().image_processor
        min_dynamic_patch = image_processor.min_patches
        max_dynamic_patch = image_processor.max_patches
        dynamic_image_size = True
        if use_thumbnail is None:
            use_thumbnail = True
        min_num, max_num = resolve_min_max_num(min_dynamic_patch,
                                                    max_dynamic_patch,
                                                    dynamic_image_size,
                                                    use_thumbnail=use_thumbnail)

        return get_internvl_target_ratios(min_num, max_num)
    
    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()

        hf_config = self.ctx.get_hf_config()
        base_size = hf_config.vision_config.image_size
        target_ratios = self.resolve_target_ratios()

        largest_feature_size, largest_feature_pinpoint = 0, None
        for wr, hr in target_ratios:
            width, height = base_size * wr, base_size * hr

            feat_size = self.get_num_image_tokens(
                image_width=width,
                image_height=height,
                processor=processor.image_processor,
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width,
                                                     height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint

    def get_max_image_tokens(self) -> int:
        processor = self.get_hf_processor()
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            processor=processor.image_processor,
        )


_I = TypeVar("_I", bound=BaseInternS1ProcessingInfo)


class BaseInternVLDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    """Basic image-only DummyInputsBuilder for InternVL-style models."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        return "<image>" * num_images

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


class BaseInternVLMultiModalProcessor(BaseMultiModalProcessor[_I]):
    """ Basic image-only MultiModalProcessor for InternVL-style models."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        image_token_id = hf_processor.image_token_id

        # Since there may be extra tokens in the feature placeholders,
        # we need to pass the image token ID to the model to select the
        # tokens to merge from the vision encoder outputs
        processed_outputs["image_token_id"] = torch.tensor(image_token_id)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_num_patches = hf_inputs.get("image_num_patches", torch.empty(0))
        num_images = len(image_num_patches)

        return dict(
            pixel_values_flat=MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_patches),
            image_num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            image_token_id=MultiModalFieldConfig.shared("image", num_images),
        )

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

        def get_replacement_internvl(item_idx: int):
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

            return hf_processor.get_image_repl(feature_size, num_patches)

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement_internvl,
            )
        ]


class InternS1ProcessingInfo(BaseInternS1ProcessingInfo):
    """InternVL ProcessingInfo extended for video processing"""

    @property
    def image_token_id(self) -> int:
        processor = self.get_hf_processor()
        return processor.image_token_id
    
    @property
    def video_token_id(self) -> Optional[int]:
        processor = self.get_hf_processor()
        if processor.video_token is None:
            return None
        return processor.tokenizer.get_vocab().get(processor.video_token, None)
    
    @property
    def supports_video(self) -> bool:
        return self.video_token_id is not None

    def get_supported_mm_limits(self):
        video_limit = {"video": None} if self.supports_video else {}
        return {**super().get_supported_mm_limits(), **video_limit}

    def get_video_token(self) -> Optional[str]:
        return self.get_hf_processor().video_token

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int]
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        processor = self.get_hf_processor()

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = (seq_len -
                            max_image_tokens) // processor.num_image_token
        max_frames_per_video = max_total_frames // max(max_videos, 1)

        return max(max_frames_per_video, 1)

    def get_image_processor(
        self,
        **kwargs: object,
    ) -> GotOcr2ImageProcessorFast:
        return cached_image_processor_from_config(self.ctx.model_config)
        
    def get_video_processor(
        self,
        **kwargs: object):
        from transformers import AutoVideoProcessor
        return AutoVideoProcessor.from_pretrained(self.ctx.model_config.model, trust_remote_code=True)
    
    def get_hf_processor(
        self,
        **kwargs: object,
    ) -> InternVLProcessor:
        return self.ctx.get_hf_processor(
            InternVLProcessor,
            image_processor=self.get_image_processor(),
            video_processor=self.get_video_processor(),
        )

class InternS1DummyInputsBuilder(
        BaseInternVLDummyInputsBuilder[InternS1ProcessingInfo]):
    """InternVL DummyInputsBuilder extended for video support"""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_videos = mm_counts.get("video", 0)

        return super().get_dummy_text(mm_counts) + "<video>" * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        dummy_image = super().get_dummy_mm_data(seq_len=seq_len,
                                                mm_counts=mm_counts)
        if self.info.supports_video:
            config = self.info.get_hf_config()
            image_size: int = config.vision_config.image_size
            target_num_frames = \
                self.info.get_num_frames_with_most_features(seq_len, mm_counts)
            num_videos = mm_counts.get("video", 0)
            dummy_video = {
                "video":
                self._get_dummy_videos(width=image_size,
                                       height=image_size,
                                       num_frames=target_num_frames,
                                       num_videos=num_videos)
            }
        else:
            dummy_video = {}
        return {**dummy_image, **dummy_video}


class InternS1MultiModalProcessor(
        BaseInternVLMultiModalProcessor[InternS1ProcessingInfo]):
    """InternVL MultiModalProcessor extended for video support"""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        processed_outputs = super()._call_hf_processor(prompt, mm_data,
                                                       mm_kwargs, tok_kwargs)

        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        if self.info.supports_video and (
                video_token_id := self.info.video_token_id) is not None:
            processed_outputs["video_token_id"] = torch.tensor(video_token_id)
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_fields = super()._get_mm_fields_config(hf_inputs,
                                                     hf_processor_mm_kwargs)
        if self.info.supports_video:
            video_num_patches = hf_inputs.get("video_num_patches",
                                              torch.empty(0))
            num_videos = len(video_num_patches)
            video_fields = dict(
                pixel_values_flat_video=MultiModalFieldConfig.flat_from_sizes(
                    "video", video_num_patches),
                video_num_patches=MultiModalFieldConfig.batched("video"),
                video_token_id=MultiModalFieldConfig.shared(
                    "video", num_videos),
            )
        else:
            video_fields = {}

        return image_fields | video_fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        prompt_repl: list[PromptUpdate] = super()._get_prompt_updates(
            mm_items, hf_processor_mm_kwargs, out_mm_kwargs)

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        if "video_num_patches" in out_mm_kwargs:
            video_num_patches = out_mm_kwargs["video_num_patches"]
            assert isinstance(video_num_patches, torch.Tensor)
            video_num_patches = video_num_patches.tolist()
        else:
            video_num_patches = []

        # !! TODO
        # def get_video_replacement_internvl(item_idx: int):
        #     # TODO: find `hf_processor.num_image_token`
        #     feature_size = hf_processor.num_image_token
        #     num_patches = video_num_patches[item_idx]
        #     if num_patches is not None:
        #         assert isinstance(num_patches, int)
        #     # TODO: find `hf_processor.get_video_repl`
        #     return hf_processor.get_video_repl(
        #         feature_size,
        #         num_patches,
        #         video_context_token=hf_processor.video_token)

        # if self.info.supports_video:
        #     prompt_repl.append(
        #         PromptReplacement(
        #             modality="video",
        #             target="<video>",
        #             replacement=get_video_replacement_internvl,
        #         ))
        return prompt_repl


@MULTIMODAL_REGISTRY.register_processor(
    InternS1MultiModalProcessor,
    info=InternS1ProcessingInfo,
    dummy_inputs=InternS1DummyInputsBuilder)
class InternS1ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<image>"
        if modality.startswith("video"):
            return "<video>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio

        self.llm_arch_name = config.text_config.architectures[0]
        self.vision_tower = self._init_vision_model(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.multi_modal_projector = self._init_mlp1(config)

        self.img_context_token_id = None
        self.video_context_token_id = None

        self.visual_token_mask = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        prefix: str,
    ):
        num_hidden_layers = config.vision_config.num_hidden_layers
        return InternS1VisionModel(
            config.vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            prefix=prefix,
        )
       
    def _init_mlp1(self, config: PretrainedConfig) -> nn.Sequential:
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vit_embeds = self.vision_tower(pixel_values=pixel_values)
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds,
                                        scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        
        vit_embeds = self.multi_modal_projector(vit_embeds)
        return vit_embeds

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[InternVLImageInputs]:
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return InternVLImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        image_token_id = kwargs["image_token_id"]
        assert isinstance(image_token_id, torch.Tensor)
        self.img_context_token_id = image_token_id.flatten().unique().item()

        if pixel_values_flat is not None:
            if not isinstance(pixel_values_flat, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values_flat)}")

            if not isinstance(image_num_patches, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image_num_patches. "
                                 f"Got type: {type(image_num_patches)}")

            pixel_values_flat = flatten_bn(pixel_values_flat, concat=True)
            image_num_patches = flatten_bn(image_num_patches, concat=True)

            return InternVLImagePixelInputs(
                type="pixel_values",
                pixel_values_flat=self._validate_pixel_values(
                    pixel_values_flat),
                num_patches=image_num_patches,
            )

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[InternVLVideoPixelInputs]:
        pixel_values_flat_video = kwargs.pop("pixel_values_flat_video", None)
        video_num_patches = kwargs.pop("video_num_patches", None)
        video_embeds = kwargs.pop("image_embeds", None)

        if pixel_values_flat_video is None and video_embeds is None:
            return None

        if video_embeds is not None:
            if not isinstance(video_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")

            return InternVLImageEmbeddingInputs(
                type="video_embeds",
                data=flatten_bn(video_embeds),
            )

        video_token_id = kwargs["video_token_id"]
        assert isinstance(video_token_id, torch.Tensor)
        self.video_context_token_id = video_token_id.flatten().unique().item()

        if pixel_values_flat_video is not None:
            if not isinstance(pixel_values_flat_video, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values_flat_video)}")

            if not isinstance(video_num_patches, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image_num_patches. "
                                 f"Got type: {type(video_num_patches)}")

            pixel_values_flat_video = flatten_bn(pixel_values_flat_video,
                                                 concat=True)
            video_num_patches = flatten_bn(video_num_patches, concat=True)

            return InternVLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_flat=self._validate_pixel_values(
                    pixel_values_flat_video),
                num_patches=video_num_patches,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: Union[InternVLImageInputs, InternVLVideoPixelInputs],
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_tower is not None

        image_embeds = self.extract_feature(image_input["pixel_values_flat"])

        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1,
                                      self.config.text_config.hidden_size), )

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1,
                                         self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values_flat",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("pixel_values_flat_video",
                             ) and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs)

        return modalities

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        if self.is_mono:
            assert self.img_context_token_id is not None
            self.visual_token_mask = (
                input_ids == self.img_context_token_id).reshape(-1, 1)
        else:
            self.visual_token_mask = None

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_image_input(video_input)
                multimodal_embeddings += video_embeddings

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            context_token_ids = [
                token_id for token_id in (self.img_context_token_id,
                                          self.video_context_token_id)
                if token_id is not None
            ]
            assert len(context_token_ids) >= 1
            self._set_visual_token_mask(input_ids)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                context_token_ids,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> IntermediateTensors:

        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        # Only required if the model is mono-architecture
        if self.visual_token_mask is not None:
            forward_kwargs.update(
                {"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(**forward_kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model="vision_tower")
