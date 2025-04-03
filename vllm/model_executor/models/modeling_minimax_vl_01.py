# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from functools import cached_property
from typing import (Final, Iterable, List, Literal, Mapping, Optional,
                    Protocol, Set, Tuple, TypedDict, TypeVar, Union)

import torch
import torch.nn as nn
from transformers import BatchFeature, LlavaNextConfig, LlavaNextProcessor
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape, unpad_image)
from typing_extensions import NotRequired
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import ImageSize
from vllm.sequence import IntermediateTensors

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .llava import (BaseLlavaMultiModalProcessor, BaseLlavaProcessingInfo,
                    LlavaDummyInputsBuilder, LlavaLikeConfig,
                    LlavaMultiModalProjector, init_vision_tower_for_llava)
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, embed_multimodal, flatten_bn,
                    init_vllm_registered_model, maybe_prefix)
from .image_processor import (ImageProcessor, get_hw_multiple_of, select_best_resolution,
                            get_num_token, split_special_tokens, CustomBatchFeature)

import os
import re
import numpy as np
from PIL import Image

logger = logging.get_logger(__name__)

LEGACY_PROCESSING = int(os.getenv('LEGACY_PROCESSING', 1))

class MiniMaxVL01ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
    }

class LlavaNextImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape:
    `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    image_sizes: NotRequired[torch.Tensor]
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """


class LlavaNextImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


LlavaNextImageInputs = Union[LlavaNextImagePixelInputs,
                             LlavaNextImageEmbeddingInputs]


class LlavaNextLikeConfig(LlavaLikeConfig, Protocol):
    image_grid_pinpoints: Final[list[list[int]]]


class LlavaNextProcessingInfo(BaseLlavaProcessingInfo):

    def get_hf_config(self) -> LlavaNextLikeConfig:
        from vllm.transformers_utils.configs.configuration_minimax_vl_01 import MiniMaxVL01Config
        return self.ctx.get_hf_config(MiniMaxVL01Config)

    def get_hf_processor(self, **kwargs: object):
        hf_processor = self.ctx.get_hf_processor(LlavaNextProcessor, **kwargs)

        # In case patch_size is omitted from `processor_config.json`
        # e.g. for E5-V: https://huggingface.co/royokong/e5-v
        if hf_processor.patch_size is None:
            patch_size = self.get_vision_encoder_info().get_patch_size()
            hf_processor.patch_size = patch_size

        return hf_processor

    # Based on: https://github.com/huggingface/text-generation-inference/blob/v3.0.1/server/text_generation_server/models/vlm_causal_lm.py#L113
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        base_feature_size = self._apply_feature_select_strategy(
            hf_config.vision_feature_select_strategy,
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
        )

        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            image_size=(image_height, image_width),
            grid_pinpoints=hf_config.image_grid_pinpoints,
            patch_size=vision_encoder_info.get_image_size(),
        )

        (
            unpadded_feature_size,
            newline_feature_size,
        ) = self._get_num_unpadded_features(
            original_height=image_height,
            original_width=image_width,
            npatches=vision_encoder_info.get_patch_grid_length(),
            num_patch_height=num_patch_height,
            num_patch_width=num_patch_width,
        )

        return unpadded_feature_size + newline_feature_size + base_feature_size

    # Based on: https://github.com/huggingface/text-generation-inference/blob/v3.0.1/server/text_generation_server/models/vlm_causal_lm.py#L86
    def _get_num_unpadded_features(
        self,
        *,
        original_height: int,
        original_width: int,
        npatches: int,
        num_patch_height: int,
        num_patch_width: int,
    ) -> tuple[int, int]:
        current_height = npatches * num_patch_height
        current_width = npatches * num_patch_width

        aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if aspect_ratio > current_aspect_ratio:
            new_height = (original_height * current_width) // original_width
            padding = (current_height - new_height) // 2
            current_height = current_height - (2 * padding)
        else:
            new_width = (original_width * current_height) // original_height
            padding = (current_width - new_width) // 2
            current_width = current_width - (2 * padding)

        unpadded_features = current_height * current_width
        newline_features = current_height

        return (unpadded_features, newline_features)

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()

        largest_feature_size, largest_feature_pinpoint = 0, None
        for (height, width) in hf_config.image_grid_pinpoints:
            feat_size = self.get_num_image_tokens(image_width=width,
                                                  image_height=height)
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width,
                                                     height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint


_I = TypeVar("_I", bound=LlavaNextProcessingInfo)


class BaseLlavaNextMultiModalProcessor(BaseLlavaMultiModalProcessor[_I]):

    # Copied from BaseMultiModalProcessor
    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        raise NotImplementedError


class LlavaNextMultiModalProcessor(
        BaseLlavaNextMultiModalProcessor[LlavaNextProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )


@MULTIMODAL_REGISTRY.register_processor(
    LlavaNextMultiModalProcessor,
    info=LlavaNextProcessingInfo,
    dummy_inputs=LlavaDummyInputsBuilder)
class MiniMaxVL01ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                        SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        # Convert MiniMaxVL01Config to LlavaNextConfig
        from transformers import LlavaNextConfig
        llava_config = LlavaNextConfig(
            vision_config=config.vision_config,
            text_config=config.text_config,
            ignore_index=config.ignore_index,
            image_token_index=config.image_token_index,
            projector_hidden_act=config.projector_hidden_act,
            vision_feature_select_strategy=config.vision_feature_select_strategy,
            vision_feature_layer=config.vision_feature_layer,
            image_grid_pinpoints=config.image_grid_pinpoints,
            tie_word_embeddings=config.tie_word_embeddings,
        )
        self.config = llava_config
        self.multimodal_config = multimodal_config

        vision_feature_layer = llava_config.vision_feature_layer
        # Determine the layer up to which we will initialize the vision tower
        if isinstance(vision_feature_layer, int):
            vision_hidden_size = config.vision_config.hidden_size
            self.feature_sample_layers = None
        # Used for multimodal granite models to control encoder outputs
        elif isinstance(vision_feature_layer, (list, tuple)):
            vision_hidden_size = config.vision_config.hidden_size * len(
                vision_feature_layer)
            self.feature_sample_layers = vision_feature_layer
        else:
            raise TypeError(
                f"vision_layer_feature type: {type(vision_feature_layer)}"
                " is not supported")

        self.vision_tower = init_vision_tower_for_llava(
            config,
            quant_config,
            require_post_norm=False,
            prefix=maybe_prefix(prefix, "vision_tower"))
        self.image_newline = nn.Parameter(
            torch.empty(config.text_config.hidden_size))
        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=True)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        expected_dims = (2, )

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    f"The expected shape of image sizes per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _validate_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)
            if len(actual_dims) == 2:
                # 如果是2D图像,添加通道维度
                d = d.unsqueeze(0)
            elif len(actual_dims) == 3 and actual_dims[0] != 3:
                # 如果通道维度不在第一维,调整维度顺序
                d = d.permute(2, 0, 1)
            
            if d.shape != expected_dims:
                raise ValueError(
                    f"The expected shape of pixel values is {expected_dims}. "
                    f"You supplied {d.shape}.")
            return d

        if isinstance(data, torch.Tensor):
            return _validate_shape(data)
        else:
            return [_validate_shape(d) for d in data]

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaNextImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            # 确保pixel_values具有正确的维度
            def _ensure_3d(img):
                if len(img.shape) == 2:
                    # 如果是2维的,添加通道维度
                    return img.unsqueeze(0)
                elif len(img.shape) == 3:
                    # 如果已经是3维的,检查通道维度
                    if img.shape[0] != 3:
                        return img.permute(2, 0, 1)
                return img

            if isinstance(pixel_values, torch.Tensor):
                if len(pixel_values.shape) == 4:  # (batch, channels, height, width)
                    pixel_values = [_ensure_3d(img) for img in pixel_values]
                else:
                    pixel_values = _ensure_3d(pixel_values)
            elif isinstance(pixel_values, list):
                pixel_values = [_ensure_3d(img) for img in pixel_values]

            return LlavaNextImagePixelInputs(
                type="pixel_values",
                pixel_values=self._validate_pixel_values(flatten_bn(pixel_values)),
                image_sizes=self._validate_image_sizes(flatten_bn(image_sizes, concat=True)) if image_sizes is not None else None,
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeds. "
                                 f"Got type: {type(image_embeds)}")

            return LlavaNextImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        raise AssertionError("This line should be unreachable.")

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/llava/modeling_llava.py#L421  # noqa
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(
            pixel_values, feature_sample_layers=self.feature_sample_layers)

        return self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

    # Based on: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py
    def _merge_image_patch_embeddings(self, image_size: torch.Tensor,
                                      patch_embeddings: torch.Tensor, *,
                                      strategy: str) -> torch.Tensor:
        if strategy == "flat":
            return patch_embeddings.flatten(0, 1)

        if strategy.startswith("spatial"):
            height = width = self.config.vision_config.image_size \
                // self.config.vision_config.patch_size

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError(
                    "The number of patches is not consistent with the "
                    "image size.")

            if patch_embeddings.shape[0] > 1:
                other_patch_embeds = patch_embeddings[1:]

                # Move to CPU to avoid floating-point errors
                orig_height, orig_width = image_size.tolist()

                # image_aspect_ratio == "anyres"
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    (orig_height, orig_width),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                num_patches = num_patch_height * num_patch_width

                # Image patches might be padded for batch processing
                other_patch_embeds = other_patch_embeds[:num_patches] \
                    .view(num_patch_height, num_patch_width, height, width, -1)

                if "unpad" in strategy:
                    other_patch_embeds = other_patch_embeds \
                        .permute(4, 0, 2, 1, 3).contiguous() \
                        .flatten(1, 2).flatten(2, 3)
                    other_patch_embeds = unpad_image(other_patch_embeds,
                                                     (orig_height, orig_width))
                    other_patch_embeds = torch.cat((
                        other_patch_embeds,
                        self.image_newline[:, None, None] \
                            .expand(*other_patch_embeds.shape[:-1], 1) \
                            .to(other_patch_embeds.device),
                    ), dim=-1)
                    other_patch_embeds = other_patch_embeds \
                        .flatten(1, 2).transpose(0, 1)
                else:
                    other_patch_embeds = other_patch_embeds \
                        .permute(0, 2, 1, 3, 4).contiguous() \
                        .flatten(0, 3)

                merged_patch_embeddings = torch.cat(
                    (base_patch_embeds, other_patch_embeds), dim=0)
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (base_patch_embeds,
                         self.image_newline[None] \
                            .to(base_patch_embeds.device)
                    ), dim=0)
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _process_image_pixels(
        self,
        inputs: LlavaNextImagePixelInputs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert self.vision_tower is not None

        pixel_values = inputs["pixel_values"]

        if isinstance(pixel_values, torch.Tensor):
            b, num_patches, c, h, w = pixel_values.shape
            stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)
            stacked_image_features = self._image_pixels_to_features(
                self.vision_tower, stacked_pixel_values)
            stacked_patch_embeddings = self.multi_modal_projector(
                stacked_image_features)

            return stacked_patch_embeddings.view(
                b, num_patches, *stacked_patch_embeddings.shape[1:])

        num_patches_per_batch = [v.shape[0] for v in pixel_values]
        stacked_pixel_values = torch.cat(pixel_values)
        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values)

        return torch.split(self.multi_modal_projector(stacked_image_features),
                           num_patches_per_batch)

    def _process_image_input(
        self,
        image_input: LlavaNextImageInputs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if image_input["type"] == "image_embeds":
            return [image_input["data"]]

        patch_embeddings = self._process_image_pixels(image_input)

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = len(image_input["data"])
            vision_config = self.config.vision_config
            default_height = default_width = vision_config.image_size
            image_sizes = torch.as_tensor([[default_height, default_width]
                                           for _ in range(batch_size)])

        return [
            self._merge_image_patch_embeddings(image_sizes[i],
                                               patch_features_batch,
                                               strategy="spatial_unpad")
            for i, patch_features_batch in enumerate(patch_embeddings)
        ]

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:

        if multimodal_embeddings is None:
            return self.language_model.get_input_embeddings(input_ids)

        inputs_embeds = embed_multimodal(
            input_ids,
            self.config.image_token_index,
            self.language_model.model.get_input_embeddings,
            multimodal_embeddings,
        )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for LlaVA-NeXT.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.

        Concretely, consider a text prompt:
        `"A chat between a curious human and an artificial intelligence
        assistant. The assistant gives helpful, detailed, and polite answers to
        the human's questions.
        USER: <image>\\nWhat is shown in this image? ASSISTANT:"`.

        Tokenizer outputs:
        `[1, 319, 13563, 1546, 263, 12758, 5199, 322, 385, 23116, 21082, 20255,
        29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568,
        6089, 304, 278, 5199, 29915, 29879, 5155, 29889, 3148, 1001, 29901,
        29871, 32000, 13, 5618, 338, 4318, 297, 445, 1967, 29973, 319, 1799,
        9047, 13566, 29901]`.

        To reserve space in KV cache, we have to insert placeholder tokens
        before they are inputted to the model, so the input processor prepends
        additional image tokens (denoted as `32000`), resulting in:
        `[1, 319, 13563, 1546, 263, 12758, 5199, 322, 385, 23116, 21082, 20255,
        29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568,
        6089, 304, 278, 5199, 29915, 29879, 5155, 29889, 3148, 1001, 29901,
        29871, 32000, ..., 32000, 13, 5618, 338, 4318, 297, 445, 1967, 29973,
        319, 1799, 9047, 13566, 29901]`.

        Unlike in LLaVA-1.5, the number of image tokens inputted to the language
        model depends on the original size of the input image. Including the
        original image token in the input, the required number of image tokens
        is given by :func:`get_llava_next_image_feature_size`.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values: The pixels in each grid patch for each input image.
            image_sizes: The original `(height, width)` for each input image.

        See also:
            :class:`LlavaNextImageInputs`
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class MiniMaxVL01Config(PretrainedConfig):
    model_type = "minimax_vl_01"

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_grid_pinpoints=None,
        tie_word_embeddings=False,
        image_seq_length=576,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.image_seq_length = image_seq_length

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        )
        self.image_grid_pinpoints = image_grid_pinpoints

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        self.vision_config = vision_config

        if text_config is not None:
            assert "model_type" in text_config, "text_config model_type is not specified"
            text_config = MiniMaxText01Config(**text_config)
        else:
            text_config = MiniMaxText01Config()

        self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class MiniMaxVL01ProcessingInfo(BaseLlavaProcessingInfo):

    def get_hf_config(self) -> MiniMaxVL01Config:
        from vllm.transformers_utils.configs.configuration_minimax_vl_01 import MiniMaxVL01Config
        return self.ctx.get_hf_config(MiniMaxVL01Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(
            MiniMaxVL01Processor,
            image_processor=self.get_image_processor(**kwargs),
            **kwargs,
        )

    def _get_image_processor_kwargs(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ):
        if self.ctx.model_config.mm_processor_kwargs:
            kwargs.update(self.ctx.model_config.mm_processor_kwargs)

        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels

            if size is None:
                size = {"shortest_edge": min_pixels}
            else:
                size["shortest_edge"] = min_pixels

        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels

            if size is None:
                size = {"longest_edge": max_pixels}
            else:
                size["longest_edge"] = max_pixels

        if size is not None:
            kwargs["size"] = size

        return kwargs

    def get_image_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> ImageProcessor:
        kwargs = self._get_image_processor_kwargs(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            size=size,
            **kwargs,
        )
        return ImageProcessor(**kwargs)


class MiniMaxVL01Processor(ProcessorMixin):
    r"""
    Constructs a MiniMaxVL01 processor which wraps a MiniMaxVL01 image processor and a MiniMaxVL01 tokenizer into a single processor.
    [`MiniMaxVL01Processor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~MiniMaxVL01Processor.__call__`] and [`~MiniMaxVL01Processor.decode`] for more information.
    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "patch_size", "vision_feature_select_strategy", "image_token"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        **kwargs,
    ):
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.patch_size = image_processor.patch_size
        self.grid_pinpoints = image_processor.image_grid_pinpoints
        self.max_size = image_processor.size
        self.process_image_mode = image_processor.process_image_mode

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.
        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            MiniMaxVL01ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            if self.process_image_mode == 'anyres':
                if LEGACY_PROCESSING:# 推理时不提前替换image token
                    pixel_values = image_inputs["pixel_values"]
                    image_sizes = image_inputs["image_sizes"]
                    all_image_tokens = []
                    for pixel_value, image_size in zip(pixel_values, image_sizes):
                        height, width = image_size
                        num_image_tokens = get_num_token(height, width, self.grid_pinpoints, self.patch_size)
                        all_image_tokens.append(num_image_tokens)
                    prompt_strings = []
                    image_index = 0
                    for sample in text:
                        split_text = split_special_tokens(sample, [self.image_token])
                        final_text = ''
                        for i, _sample in enumerate(split_text):
                            if _sample == self.image_token:
                                final_text += _sample * all_image_tokens[image_index]
                                image_index += 1
                            else:
                                final_text += _sample
                        prompt_strings.append(final_text)
            elif self.process_image_mode == 'resize':
                pixel_values = image_inputs["pixel_values"]
                all_image_tokens = []
                for pixel_value in pixel_values:
                    height, width = get_image_size(to_numpy_array(pixel_value))
                    all_image_tokens.append(int(height*width/self.patch_size**2))
                
                prompt_strings = []
                image_index = 0
                for sample in text:
                    split_text = split_special_tokens(sample, [self.image_token])
                    final_text = ''
                    for i, _sample in enumerate(split_text):
                        if _sample == self.image_token:
                            final_text += _sample * all_image_tokens[image_index]
                            image_index += 1
                        else:
                            final_text += _sample
                    prompt_strings.append(final_text)
            else:
                if self.patch_size is not None:
                    pixel_values = image_inputs["pixel_values"]
                    all_image_tokens = []
                    for pixel_value in pixel_values:
                        height, width = get_image_size(to_numpy_array(pixel_value))
                        new_width, new_height = get_hw_multiple_of((width, height), self.patch_size, self.max_size)
                        num_image_tokens = (new_height // self.patch_size) * (new_width // self.patch_size)
                        all_image_tokens.append(num_image_tokens)
                    
                    prompt_strings = []
                    image_index = 0
                    for sample in text:
                        split_text = split_special_tokens(sample, [self.image_token])
                        final_text = ''
                        for i, _sample in enumerate(split_text):
                            if _sample == self.image_token:
                                final_text += _sample * all_image_tokens[image_index]
                                image_index += 1
                            else:
                                final_text += _sample
                        prompt_strings.append(final_text)
                else:
                    logger.warning_once(
                        "Expanding inputs for image tokens in MiniMaxVL01 should be done in processing. "
                        "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                        "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                        "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                    )
                    raise ValueError(
                        "You need to provide `patch_size` and `vision_feature_select_strategy` in the model's processing config to expand inputs for image tokens."
                    )

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
        return CustomBatchFeature(data={**text_inputs, **image_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
