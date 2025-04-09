# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from functools import cached_property
from typing import (Final, Iterable, List, Literal, Mapping, Optional,
                    Protocol, Set, Tuple, TypedDict, TypeVar, Union, Dict, Any)

import torch
import torch.nn as nn
from transformers import (
    BatchFeature, 
    AutoProcessor,
    AutoImageProcessor,
)
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape, unpad_image)
from typing_extensions import NotRequired
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.processing_utils import ProcessingKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging, TensorType
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import ChannelDimension
from torchvision.transforms import Compose, ToTensor, Normalize

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import ImageSize
from vllm.sequence import IntermediateTensors
from .image_processer import ImageProcessor, resize_multiple_of, _convert_image_to_rgb
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .llava import (BaseLlavaMultiModalProcessor, BaseLlavaProcessingInfo,
                    LlavaDummyInputsBuilder, LlavaLikeConfig,
                    LlavaMultiModalProjector, init_vision_tower_for_llava)
from .llava_next import (LlavaNextMultiModalProcessor, LlavaNextProcessingInfo, LlavaNextImageEmbeddingInputs, LlavaNextImagePixelInputs, LlavaNextImageInputs)
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, embed_multimodal, flatten_bn,
                    init_vllm_registered_model, maybe_prefix)
from .processing_minimax_vl_01 import MiniMaxVL01Processor
from vllm.multimodal.processing import BaseProcessingInfo
import numpy as np
import numpy.typing as npt


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

        self.config = config
        self.multimodal_config = multimodal_config

        vision_feature_layer = config.vision_feature_layer
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

    @property
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
                d = d.unsqueeze(0)
            elif len(actual_dims) == 3 and actual_dims[0] != 3:
                d = d.permute(2, 0, 1)
            elif len(actual_dims) == 4 and actual_dims[0] == 1:
                d = d.squeeze(0)
            
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
        inputs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert self.vision_tower is not None

        pixel_values = inputs["pixel_values"]

        if isinstance(pixel_values, torch.Tensor):
            # 检查张量维度
            if len(pixel_values.shape) == 3:  # 如果是3维的 (c, h, w)
                # 添加批次和补丁维度
                pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)  # 变成 (1, 1, c, h, w)
            
            # # 确保有5个维度
            # if len(pixel_values.shape) != 5:
            #     raise ValueError(f"Expected pixel_values to have 5 dimensions (b, num_patches, c, h, w), but got shape {pixel_values.shape}")
                
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
        image_input,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if image_input["type"] == "image_embeds":
            return [image_input["data"]]

        patch_embeddings = self._process_image_pixels(image_input)

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = len(patch_embeddings)
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
