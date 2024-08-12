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
from typing import Iterable, List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from PIL import Image
from transformers import FuyuConfig, FuyuImageProcessor

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.persimmon import PersimmonForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.image import (cached_get_image_processor,
                                   cached_get_tokenizer)
from vllm.sequence import IntermediateTensors, SamplerOutput, SequenceData

from .interfaces import SupportsVision
from .utils import merge_vision_embeddings

logger = init_logger(__name__)

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


def dummy_seq_data_for_fuyu(ctx: InputContext, seq_len: int):
    ncol, nrow = get_max_fuyu_image_feature_size()
    image_feature_size = get_max_fuyu_image_tokens(ctx)

    token_ids = ([_IMAGE_TOKEN_ID] * ncol + [_NEWLINE_TOKEN_ID]) * nrow
    token_ids += [0] * (seq_len - image_feature_size)
    return SequenceData(token_ids)


def dummy_image_for_fuyu(
    image_width: int,
    image_height: int,
):
    image = Image.new("RGB", (image_width, image_height), color=0)
    return {"image": image}


def dummy_data_for_fuyu(ctx: InputContext, seq_len: int):
    seq_data = dummy_seq_data_for_fuyu(ctx, seq_len)
    mm_data = dummy_image_for_fuyu(MAX_IMAGE_FEATURE_SIZE_WIDTH,
                                   MAX_IMAGE_FEATURE_SIZE_HEIGHT)
    return seq_data, mm_data


def _fuyu_image_preprocess(image_processor: FuyuImageProcessor,
                           data: Image.Image):
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


def input_processor_for_fuyu(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config
    image_data = multi_modal_data["image"]
    new_multi_modal_data = {}
    # process image data
    if isinstance(image_data, Image.Image):
        # Fuyu's image_processor can also finish token padding
        image_processor: FuyuImageProcessor = cached_get_image_processor(
            model_config.model)

        model_image_input = _fuyu_image_preprocess(image_processor, image_data)
        image_patches = torch.stack([
            image_patch[0]
            for image_patch in model_image_input["image_patches"]
        ])
        new_multi_modal_data["image"] = image_patches

    elif isinstance(image_data, torch.Tensor):
        raise NotImplementedError("Embeddings input is not supported yet")
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    # process prompts
    prompt = llm_inputs.get("prompt")
    prompt_token_ids = llm_inputs["prompt_token_ids"]
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

    return LLMInputs(prompt=new_prompt,
                     prompt_token_ids=new_prompt_token_ids,
                     multi_modal_data=new_multi_modal_data)


def input_mapper_for_fuyu(ctx: InputContext, data: object):
    model_config = ctx.model_config
    if isinstance(data, Image.Image):
        # Fuyu's image_processor can also finish token padding
        image_processor: FuyuImageProcessor = cached_get_image_processor(
            model_config.model)

        model_image_input = _fuyu_image_preprocess(image_processor, data)
        data = torch.stack([
            image_patch[0]
            for image_patch in model_image_input["image_patches"]
        ])

    # image has been processed with prompt in input processor
    return MultiModalInputs({"image_patches": data})


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_fuyu)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_fuyu_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_fuyu)
@INPUT_REGISTRY.register_input_processor(input_processor_for_fuyu)
class FuyuForCausalLM(nn.Module, SupportsVision):

    def __init__(self,
                 config: FuyuConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.config = config
        self.multimodal_config = multimodal_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.image_token_id = _IMAGE_TOKEN_ID
        self.image_feature_size = config.patch_size**2 * config.num_channels

        self.vision_embed_tokens = ColumnParallelLinear(
            self.image_feature_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.language_model = PersimmonForCausalLM(config,
                                                   cache_config=cache_config,
                                                   quant_config=quant_config)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[FuyuImagePixelInputs]:
        image_patches = kwargs.pop("image_patches", None)

        if isinstance(image_patches, torch.Tensor):
            expected_feature_size = self.image_feature_size
            if image_patches.size(-1) != expected_feature_size:
                raise ValueError(
                    f"Expected image patches to have the last dimension of "
                    f"{expected_feature_size}, got {image_patches.size(-1)}")
            image_patches = image_patches.to(
                self.vision_embed_tokens.weight.dtype)
            return FuyuImagePixelInputs(type="pixel_values",
                                        data=image_patches)
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
        image_input = self._parse_and_validate_image_input(**kwargs)

        if image_input is not None:
            vision_embeddings = self._process_image_input(image_input)
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
            inputs_embeds = merge_vision_embeddings(input_ids, inputs_embeds,
                                                    vision_embeddings,
                                                    self.image_token_id)

        else:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
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
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            param = params_dict[name]

            if "query_key_value" in name:
                # copy from vllm/model_executor/models/bloom.py
                # NOTE: Fuyu's fused QKV's output_dim has the shape of
                # (num_heads * 3 * head_size), while the
                # required shape is (3 * num_heads * head_size).
                # Thus, we need weight conversion.
                output_dim = getattr(param, "output_dim", None)
                num_heads = self.config.num_attention_heads
                if output_dim is not None:
                    loaded_weight_shape = loaded_weight.shape
                    loaded_weight = loaded_weight.view(
                        loaded_weight_shape[:output_dim] + (num_heads, 3, -1) +
                        loaded_weight_shape[output_dim + 1:])
                    loaded_weight = loaded_weight.transpose(
                        output_dim, output_dim + 1)
                    loaded_weight = loaded_weight.reshape(loaded_weight_shape)

            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
