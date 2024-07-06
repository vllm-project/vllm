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
from transformers import FuyuConfig

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
from vllm.multimodal.image import (cached_get_tokenizer, cached_get_image_processor,
                                   repeat_and_pad_image_tokens, repeat_and_pad_token)
from vllm.sequence import IntermediateTensors, SamplerOutput, SequenceData

from .interfaces import SupportsVision

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


def get_max_fuyu_image_tokens():
    ncol, nrow = get_max_fuyu_image_feature_size()
    return (ncol + 1) * nrow


def dummy_seq_data_for_fuyu(seq_len: int):
    ncol, nrow = get_max_fuyu_image_feature_size()
    image_feature_size = get_max_fuyu_image_tokens()

    token_ids = ([_IMAGE_TOKEN_ID] * ncol + [_NEWLINE_TOKEN_ID]) * nrow
    token_ids += [0] * (seq_len - image_feature_size)
    return SequenceData(token_ids)


def dummy_image_for_fuyu(
    image_width: int,
    image_height: int,
):
    image = Image.new("RGB", (image_width, image_height), color=0)
    return {"image": image}


def dummy_data_for_fuyu(seq_len: int):
    seq_data = dummy_seq_data_for_fuyu(seq_len)
    mm_data = dummy_image_for_fuyu(
        MAX_IMAGE_FEATURE_SIZE_WIDTH, 
        MAX_IMAGE_FEATURE_SIZE_HEIGHT
    )
    return seq_data, mm_data


def input_processor_for_fuyu(
        ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs
    
    model_config = ctx.model_config
    image_data = multi_modal_data["image"]
    # process image data
    if isinstance(image_data, Image.Image):
        # we need to get transformed image_size for prompts padding
        image_processor = cached_get_image_processor(model_config.model)
        outputs = image_processor.preprocess(image_data, return_tensors="pt")
        if len(outputs["images"]) != 1:
            logger.warning("Multiple image input is not supported yet, "
                           "so any extra image tokens will be treated "
                           "as plain text.")

        # images: [[torch.Tensor(image1)], [torch.Tensor(image2)], ...]
        # unpadded_heights: [[torch.Tensor(h1)], [torch.Tensor(h2)], ...]
        # unpadded_widths: [[torch.Tensor(w1)], [torch.Tensor(w2)], ...]
        # NOTE: This only designed for single image
        image = torch.stack(outputs["images"][0])
        
        image_unpadded_h = outputs["image_unpadded_heights"][0][0].item()
        image_unpadded_w = outputs["image_unpadded_widths"][0][0].item()
        image_padded_h = math.ceil(image_unpadded_h / 30) * 30
        image_padded_w = math.ceil(image_unpadded_w / 30) * 30
        image_data = image[:, :, :image_padded_h, :image_padded_w]
        multi_modal_data["image"] = image_data

    elif isinstance(image_data, torch.Tensor):
        raise NotImplementedError("Embeddings input is not supported yet")
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    # process prompts
    prompt = llm_inputs["prompt"]
    prompt_token_ids = llm_inputs["prompt_token_ids"]
    ncol, nrow = _calculate_num_image_tokens(image_unpadded_h, image_unpadded_w)
    padding_ids = repeat_and_pad_token(_IMAGE_TOKEN_ID, repeat_count=ncol, pad_token_right=_NEWLINE_TOKEN_ID)

    # image tokens always left padded before text
    new_prompt_token_ids = padding_ids * nrow + prompt_token_ids

    return LLMInputs(prompt=prompt,
                     prompt_token_ids=new_prompt_token_ids,
                     multi_modal_data=multi_modal_data)


def input_mapper_for_fuyu(ctx: InputContext, data: object):
    model_config = ctx.model_config
    image_processor = cached_get_image_processor(model_config.model)
    # image has been processed in input_processor
    image_patches = image_processor.patchify_image(data)
    return MultiModalInputs({"image_patches": image_patches})


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

        self.vision_embed_tokens = ColumnParallelLinear(
            config.patch_size * config.patch_size * config.num_channels,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.language_model = PersimmonForCausalLM(config,
                                                   cache_config=cache_config,
                                                   quant_config=quant_config)

    def merge_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        vision_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        mask = input_ids == self.image_token_id
        inputs_embeds[mask] = vision_embeddings.view(
            -1, vision_embeddings.shape[-1])
        return inputs_embeds

    def _parse_and_validate_image_input(self, **kwargs: object):
        image_patches = kwargs.pop("image_patches", None)

        if isinstance(image_patches, torch.Tensor):
            image_patches = image_patches.to(
                self.vision_embed_tokens.weight.dtype)
            return FuyuImagePixelInputs(type="pixel_values",
                                        data=image_patches)
        return None

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
            vision_embeddings, _ = self.vision_embed_tokens(
                image_input["data"])
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
            inputs_embeds = self.merge_embeddings(
                input_ids,
                inputs_embeds,
                vision_embeddings,
            )

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
            self.language_model.lm_head, hidden_states,
            sampling_metadata)
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
