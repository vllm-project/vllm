# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Final, Optional, Protocol, Set, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (BatchFeature, CLIPVisionConfig, PixtralVisionConfig,
                          PretrainedConfig, SiglipVisionConfig)
from transformers.image_processing_utils import select_best_resolution

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.minimax_vl_01 import MiniMaxVL01Config

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .pixtral import PixtralHFVisionModel
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)
from .vision import get_vision_encoder_info

logger = init_logger(__name__)


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple,
    # otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError("image_size invalid type " +
                            f"{type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor,
    # must convert to into tuple,
    # otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                "image_size invalid type " +
                f"{type(image_size)} not valid, " +
                "should be either list, tuple, np.ndarray or tensor")
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def unpad_image(tensor, original_size):
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        new_height = int(original_height * current_width) // original_width
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        new_width = int(original_width * current_height) // original_height
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class MiniMaxVL01MultiModalProjector(nn.Module):

    def __init__(self,
                 vision_hidden_size: int,
                 text_hidden_size: int,
                 projector_hidden_act: str,
                 multimodal_projector_bias: bool,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.linear_1 = ColumnParallelLinear(vision_hidden_size,
                                             text_hidden_size,
                                             bias=multimodal_projector_bias,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.linear_1")
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(text_hidden_size,
                                          text_hidden_size,
                                          bias=multimodal_projector_bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.linear_2")

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class MiniMaxVL01LikeConfig(Protocol):
    vision_config: Final[PretrainedConfig]
    image_token_index: Final[int]
    vision_feature_select_strategy: Final[str]
    vision_feature_layer: Final[Union[int, list[int]]]


class MiniMaxVL01LikeProcessor(Protocol):
    image_token: Final[str]


_I = TypeVar("_I", bound=BaseProcessingInfo)


class MiniMaxVL01DummyInputsBuilder(BaseDummyInputsBuilder[_I]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        return "image:" + image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class MiniMaxVL01ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(MiniMaxVL01Config)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())

    def _apply_feature_select_strategy(
        self,
        strategy: str,
        encoder_num_image_tokens: int,
    ) -> int:
        if strategy == "default":
            return encoder_num_image_tokens - 1
        if strategy == "full":
            return encoder_num_image_tokens

        msg = f"Unexpected feature select strategy: {strategy!r}"
        raise NotImplementedError(msg)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        return self._apply_feature_select_strategy(
            hf_config.vision_feature_select_strategy,
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_encoder_info = self.get_vision_encoder_info()
        width = height = vision_encoder_info.get_image_size()
        return ImageSize(width=width, height=height)

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )


class BaseMiniMaxVL01MultiModalProcessor(BaseMultiModalProcessor[_I]):

    # Copied from BaseMultiModalProcessor
    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        raise NotImplementedError

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]


class MiniMaxVL01MultiModalProcessor(
        BaseMiniMaxVL01MultiModalProcessor[MiniMaxVL01ProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )
        pixel_values = processed_outputs.get("pixel_values")
        image_sizes = processed_outputs.get("image_sizes")

        if pixel_values is not None and isinstance(
                pixel_values, list) and len(pixel_values) > 0:
            first_item = pixel_values[0]
            processed_outputs["pixel_values"] = [first_item]

        if pixel_values is not None and image_sizes is not None and isinstance(
                image_sizes, torch.Tensor) and image_sizes.dim() > 0:
            processed_outputs["image_sizes"] = image_sizes[:1]
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "image_sizes": MultiModalFieldConfig.batched("image"),
        }


def _get_num_hidden_layers(hf_config: MiniMaxVL01LikeConfig) -> int:
    """Determine the number of hidden layers to initialize up to in the
    visual encoder.
    
    Args:
        hf_config: Model config with vision feature layer(s).
    """
    feature_layers = hf_config.vision_feature_layer
    num_hidden_layers = hf_config.vision_config.num_hidden_layers
    # If we have one feature layer, initialize up to that layer
    if isinstance(feature_layers, int):
        return _get_layer_index(feature_layers, num_hidden_layers)
    # If we have multiple feature layers, initialize up to the deepest one
    elif isinstance(feature_layers, (list, tuple)):
        return max(
            _get_layer_index(idx, num_hidden_layers) for idx in feature_layers)
    raise TypeError(f"vision_layer_feature type: {type(feature_layers)}"
                    " is not supported")


def _get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
    """Given a signed vision feature layer, get the number of hidden layers
    needed to leverage it.

    Args:
        feature_layer_index: Index of a required layer in the visual encoder.
        num_hidden_layers: The total number of hidden layers in the visual
            encoder.
    """
    if feature_layer_index < 0:
        return num_hidden_layers + feature_layer_index + 1
    return feature_layer_index


def init_vision_tower_for_MiniMaxVL01(
    hf_config: MiniMaxVL01LikeConfig,
    quant_config: Optional[QuantizationConfig],
    *,
    require_post_norm: Optional[bool] = None,
    prefix: str = "",
) -> Union[CLIPVisionModel, SiglipVisionModel, PixtralHFVisionModel]:
    vision_config = hf_config.vision_config

    # Initialize the vision tower only up to the deepest required feature layer
    num_hidden_layers = _get_num_hidden_layers(hf_config)

    if isinstance(vision_config, CLIPVisionConfig):
        return CLIPVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, SiglipVisionConfig):
        return SiglipVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, PixtralVisionConfig):
        return PixtralHFVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxVL01MultiModalProcessor,
    info=MiniMaxVL01ProcessingInfo,
    dummy_inputs=MiniMaxVL01DummyInputsBuilder)
class MiniMaxVL01ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                          SupportsPP):

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # TODO: Optionally initializes this for supporting embeddings.
        self.vision_tower = init_vision_tower_for_MiniMaxVL01(
            config,
            quant_config,
            require_post_norm=False,
            prefix=maybe_prefix(prefix, "vision_tower"))
        self.multi_modal_projector = MiniMaxVL01MultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "multi_modal_projector"))
        self.image_newline = nn.Parameter(
            torch.empty(config.text_config.hidden_size))
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.vision_feature_layer = config.vision_feature_layer
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = -1
        if self.config.pad_token_id is not None:
            self.pad_token_id = self.config.pad_token_id

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_index,
            )
        return inputs_embeds

    def pack_image_features(self,
                            image_features,
                            image_sizes,
                            image_newline=None):
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = (self.config.vision_config.image_size //
                                  self.config.vision_config.patch_size)
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not " +
                                     "consistent with the image size.")
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height,
                                                   num_patch_width, height,
                                                   width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1,
                                                      3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature,
                                            image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(
                                *image_feature.shape[:-1], 1).to(
                                    image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature),
                                          dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat(
                        (image_feature, image_newline[None].to(image_feature)),
                        dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens,
                                    dtype=torch.long,
                                    device=image_features.device)
        return image_features, feature_lens

    def _process_vision_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: Optional[list],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        vision_feature_select_strategy = (
            self.config.vision_feature_select_strategy)

        if image_sizes is not None:
            image_num_patches = [
                image_size_to_num_patches(
                    image_size=imsize,
                    grid_pinpoints=self.config.image_grid_pinpoints,
                    patch_size=self.config.vision_config.image_size,
                ) for imsize in image_sizes
            ]

        image_features = self.vision_tower(pixel_values,
                                           output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[
            self.vision_feature_layer]

        selected_image_feature = torch.chunk(selected_image_feature,
                                             len(pixel_values),
                                             dim=1)
        selected_image_feature = torch.cat(selected_image_feature, dim=0)

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature

        image_features = self.multi_modal_projector(selected_image_feature)

        if image_sizes is not None:
            image_features = torch.split(image_features,
                                         image_num_patches,
                                         dim=0)
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                image_newline=self.image_newline,
            )

        inputs_embeds = inputs_embeds.to(image_features.dtype)
        special_image_mask = ((input_ids == self.config.image_token_index
                               ).unsqueeze(-1).expand_as(inputs_embeds).to(
                                   inputs_embeds.device))
        image_features = image_features.to(inputs_embeds.device,
                                           inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask,
                                                     image_features)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        kv_caches = kwargs.pop("kv_caches", None)
        if inputs_embeds is None:
            # 1. Extract the input embeddings
            for_inputs_embeds_ids = input_ids.clone()
            for_inputs_embeds_ids[(
                input_ids == self.config.image_token_index)] = 0
            inputs_embeds = self.get_input_embeddings(for_inputs_embeds_ids)

            # 2. Merge text and images
            # 添加安全检查确保input_ids有正确的维度并且pixel_values不为空
            has_valid_shape = isinstance(input_ids, torch.Tensor) and len(
                input_ids.shape) > 1
            has_images = pixel_values is not None and (
                isinstance(pixel_values, torch.Tensor)
                and pixel_values.size(0) > 0
                or isinstance(pixel_values, list) and len(pixel_values) > 0)

            if has_images and has_valid_shape and input_ids.shape[1] != 1:
                inputs_embeds = self._process_vision_features(
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                )
            # pixel_values exists but is empty or
            # input_ids is a single token -> text only cases
            elif pixel_values is not None and (has_valid_shape
                                               and input_ids.shape[1]
                                               == 1) or not has_valid_shape:
                pass

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            kv_caches=kv_caches,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

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
