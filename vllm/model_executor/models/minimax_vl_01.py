# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable, Mapping
from typing import Literal, Optional, TypedDict, Union, cast

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.jsontree import json_map_leaves
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.minimax_vl_01 import MiniMaxVL01Config

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .llava import (BaseLlavaMultiModalProcessor, LlavaDummyInputsBuilder,
                    init_vision_tower_for_llava)
from .llava_next import LlavaNextProcessingInfo
from .pixtral import PixtralHFVisionModel
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)


class MiniMaxVL01ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """
    Shape: `(batch_size * num_images, num_channels, height, width)`

    Note that `height` or `width` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """


class MiniMaxVL01ImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


MiniMaxVL01ImageInputs = Union[MiniMaxVL01ImagePixelInputs,
                               MiniMaxVL01ImageEmbeddingInputs]


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


class MiniMaxVL01DummyInputsBuilder(LlavaDummyInputsBuilder):
    pass


class MiniMaxVL01ProcessingInfo(LlavaNextProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(MiniMaxVL01Config)

    def get_hf_processor(self, **kwargs: object):
        hf_processor = self.ctx.get_hf_processor(**kwargs)
        image_processor = hf_processor.image_processor
        image_processor.anyres_preprocess = (
            image_processor.anyres_for_vllm_preprocess)

        return hf_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}


class MiniMaxVL01MultiModalProcessor(
        BaseLlavaMultiModalProcessor[MiniMaxVL01ProcessingInfo]):

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
        if pixel_values is not None:
            # Avoid padding since we need the output for each image to be
            # independent of other images for the cache to work correctly
            image_sizes = processed_outputs["image_sizes"]
            assert len(pixel_values) == len(image_sizes)

            processed_outputs["pixel_values"] = [
                p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)
            ]

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "image_embeds": MultiModalFieldConfig.batched("image"),
        }


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
        self.vision_tower = init_vision_tower_for_llava(
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

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel,
                            PixtralHFVisionModel],
        pixel_values: Union[torch.Tensor, list[torch.Tensor]],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(pixel_values)

        def select_features(leaf: torch.Tensor):
            return self._select_image_features(
                leaf,
                strategy=self.config.vision_feature_select_strategy,
            )

        return cast(
            Union[torch.Tensor, tuple[torch.Tensor, ...]],
            json_map_leaves(select_features, image_features),
        )

    def _process_image_pixels(
        self,
        inputs: MiniMaxVL01ImagePixelInputs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert self.vision_tower is not None

        pixel_values = inputs["pixel_values"]

        return self._image_pixels_to_features(self.vision_tower, pixel_values)

    def _process_image_input(
        self,
        image_input: MiniMaxVL01ImageInputs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_tower is not None
        image_features = self._process_image_pixels(image_input)

        if isinstance(image_features, torch.Tensor):
            return self.multi_modal_projector(image_features)

        feature_sizes = [
            image_feature.shape[0] for image_feature in image_features
        ]

        image_embeds = self.multi_modal_projector(torch.cat(image_features))
        image_embeds = torch.split(image_embeds, feature_sizes)
        return image_embeds

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[MiniMaxVL01ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return MiniMaxVL01ImagePixelInputs(
                type="pixel_values",
                pixel_values=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return MiniMaxVL01ImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds, concat=True),
            )

        raise AssertionError("This line should be unreachable.")

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        return self._process_image_input(image_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if intermediate_tensors is not None:
            inputs_embeds = None
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

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
