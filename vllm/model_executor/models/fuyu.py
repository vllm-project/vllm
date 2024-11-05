# coding=utf-8
# adapted from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/fuyu/modeling_fuyu.py
# Copyright 2023 The vLLM team.
# Copyright 2023 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Fuyu model."""
import math
from array import array
from typing import Iterable, List, Literal, Mapping, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from PIL import Image
from transformers import FuyuConfig, FuyuImageProcessor

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.models.persimmon import PersimmonForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.image import cached_get_image_processor
from vllm.multimodal.utils import (cached_get_tokenizer,
                                   consecutive_placeholder_ranges)
from vllm.sequence import (VLLM_TOKEN_ID_ARRAY_TYPE, IntermediateTensors,
                           SequenceData)
from vllm.utils import is_list_of

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, flatten_bn, merge_multimodal_embeddings

# Cannot find the following 2 numbers from hf config.
_IMAGE_TOKEN_ID = 71011
_NEWLINE_TOKEN_ID = 71019

MAX_IMAGE_FEATURE_SIZE_HEIGHT = 1080
MAX_IMAGE_FEATURE_SIZE_WIDTH = 1920


class FuyuImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """
    Shape: 
    (batch_size, num_patches, patch_size_x * patch_size_y * num_channels)
    """


def _calculate_num_image_tokens(
    height: int,
    width: int,
) -> Tuple[int, int]:
    """
    calculate number of image tokens needed for a given image size
    The expected Fuyu image prompts is in format:
        (image_token * ncols + newline_token) * nrows
    args:
        image_size: Tuple[int, int] - (width, height) of the image
    returns:
        ncols: int - number of image tokens in x direction
        nrows: int - number of image tokens in y direction
    """
    ncol = math.ceil(width / 30)
    nrow = math.ceil(height / 30)
    return ncol, nrow


def get_max_fuyu_image_feature_size():

    return _calculate_num_image_tokens(
        height=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
        width=MAX_IMAGE_FEATURE_SIZE_WIDTH,
    )


def get_max_fuyu_image_tokens(ctx: InputContext):
    ncol, nrow = get_max_fuyu_image_feature_size()
    return (ncol + 1) * nrow


def dummy_seq_data_for_fuyu(ctx: InputContext, seq_len: int, num_images: int):
    ncol, nrow = get_max_fuyu_image_feature_size()
    image_feature_size = get_max_fuyu_image_tokens(ctx)

    image_token_ids = (
        array(VLLM_TOKEN_ID_ARRAY_TYPE, [_IMAGE_TOKEN_ID]) * ncol +
        array(VLLM_TOKEN_ID_ARRAY_TYPE, [_NEWLINE_TOKEN_ID])) * nrow
    token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE, image_token_ids) * num_images
    token_ids += array(VLLM_TOKEN_ID_ARRAY_TYPE,
                       [0]) * (seq_len - image_feature_size * num_images)
    return SequenceData(token_ids), {
        "image":
        consecutive_placeholder_ranges(num_items=num_images,
                                       item_size=image_feature_size)
    }


def dummy_image_for_fuyu(
    num_images: int,
    *,
    image_width: int,
    image_height: int,
):
    image = Image.new("RGB", (image_width, image_height), color=0)
    return {"image": image if num_images == 1 else [image] * num_images}


def dummy_data_for_fuyu(ctx: InputContext, seq_len: int,
                        mm_counts: Mapping[str, int]):
    num_images = mm_counts["image"]
    seq_data, ranges = dummy_seq_data_for_fuyu(ctx, seq_len, num_images)
    mm_data = dummy_image_for_fuyu(num_images,
                                   image_width=MAX_IMAGE_FEATURE_SIZE_WIDTH,
                                   image_height=MAX_IMAGE_FEATURE_SIZE_HEIGHT)
    return DummyData(seq_data, mm_data, ranges)


def _fuyu_image_preprocess(image_processor: FuyuImageProcessor,
                           data: List[Image.Image]):
    image_encoding = image_processor.preprocess(data, return_tensors="pt")
    batch_images = torch.stack([img[0] for img in image_encoding["images"]
                                ]).unsqueeze(1)
    image_unpadded_heights = torch.tensor(
        image_encoding["image_unpadded_heights"])
    image_unpadded_widths = torch.tensor(
        image_encoding["image_unpadded_widths"])

    batch_size = len(image_encoding["images"])
    image_present = torch.ones(batch_size, 1, 1)
    model_image_input = image_processor.preprocess_with_tokenizer_info(
        image_input=batch_images,
        image_present=image_present,
        image_unpadded_h=image_unpadded_heights,
        image_unpadded_w=image_unpadded_widths,
        image_placeholder_id=_IMAGE_TOKEN_ID,
        image_newline_id=_NEWLINE_TOKEN_ID,
        variable_sized=True,
    )
    return model_image_input


def input_processor_for_fuyu(ctx: InputContext, inputs: DecoderOnlyInputs):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return inputs

    model_config = ctx.model_config
    image_data = multi_modal_data["image"]
    new_multi_modal_data = {}
    image_list = image_data if isinstance(image_data, list) else [image_data]

    # process image data
    if is_list_of(image_list, Image.Image):
        # Fuyu's image_processor can also finish token padding
        image_processor: FuyuImageProcessor = cached_get_image_processor(
            model_config.model)

        model_image_input = _fuyu_image_preprocess(image_processor, image_data)
        image_patches = torch.cat([
            image_patch[0]
            for image_patch in model_image_input["image_patches"]
        ])
        new_multi_modal_data["image"] = image_patches

    elif is_list_of(image_list, torch.Tensor):
        raise NotImplementedError("Embeddings input is not supported yet")
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    # process prompts
    prompt = inputs.get("prompt")
    prompt_token_ids = inputs["prompt_token_ids"]
    tokenizer = cached_get_tokenizer(model_config.model)
    # dim0 is batch_size, dim1 is subseq_size which will always be 1
    image_input_ids: List[List[
        torch.Tensor]] = model_image_input["image_input_ids"]
    image_input_ids = image_input_ids[0][0].tolist()
    bos_token = tokenizer.encode("<s>", add_special_tokens=False)[1:]
    boa_token = tokenizer.encode("\x04", add_special_tokens=False)[1:]

    new_prompt = prompt + "\x04"
    new_prompt_token_ids = image_input_ids + bos_token + prompt_token_ids[
        1:] + boa_token

    return token_inputs(prompt=new_prompt,
                        prompt_token_ids=new_prompt_token_ids,
                        multi_modal_data=new_multi_modal_data)


def input_mapper_for_fuyu(ctx: InputContext, data: object):
    model_config = ctx.model_config
    data_list = data if isinstance(data, list) else [data]
    if is_list_of(data_list, Image.Image):
        # Fuyu's image_processor can also finish token padding
        image_processor: FuyuImageProcessor = cached_get_image_processor(
            model_config.model)

        model_image_input = _fuyu_image_preprocess(image_processor, data_list)
        data = torch.stack([
            image_patch[0]
            for image_patch in model_image_input["image_patches"]
        ])

    # image has been processed with prompt in input processor
    return MultiModalInputs({"pixel_values": data})


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_fuyu)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_fuyu_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_fuyu)
@INPUT_REGISTRY.register_input_processor(input_processor_for_fuyu)
class FuyuForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self,
                 config: FuyuConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.config = config
        self.multimodal_config = multimodal_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.text_config.vocab_size
        self.image_token_id = _IMAGE_TOKEN_ID
        self.image_feature_size = config.patch_size**2 * config.num_channels

        self.vision_embed_tokens = ColumnParallelLinear(
            self.image_feature_size,
            config.hidden_size,
            quant_config=quant_config,
            gather_output=True,
        )
        self.language_model = PersimmonForCausalLM(config.text_config,
                                                   cache_config=cache_config,
                                                   quant_config=quant_config)
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @property
    def sampler(self):
        return self.language_model.sampler

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:

        h = w = self.config.patch_size
        num_channels = self.config.num_channels
        expected_dims = num_channels * h * w

        def _validate_shape(d: torch.Tensor):
            actual_dims = d.size(-1)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data.to(self.vision_embed_tokens.weight.dtype)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[FuyuImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image patches. "
                                 f"Got type: {type(pixel_values)}")

            return FuyuImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        return None

    def _process_image_input(
            self, image_input: FuyuImagePixelInputs) -> torch.Tensor:

        assert self.vision_embed_tokens is not None
        vision_embeddings, _ = self.vision_embed_tokens(image_input["data"])
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ):
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            image_input = self._parse_and_validate_image_input(**kwargs)

            if image_input is not None:
                vision_embeddings = self._process_image_input(image_input)
                inputs_embeds = self.language_model.model.embed_tokens(
                    input_ids)
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, vision_embeddings,
                    self.image_token_id)

            else:
                inputs_embeds = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.language_model.logits_processor(
            self.language_model.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.language_model.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights)
