# coding=utf-8
# Copyright 2024 The vLLM team.
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
import re
from functools import lru_cache
from typing import Iterable, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionConfig, PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.clip import CLIPVisionModel
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, BatchedTensors
from vllm.multimodal.image import cached_get_tokenizer
from vllm.sequence import IntermediateTensors, SamplerOutput

from .clip import (dummy_image_for_clip, dummy_seq_data_for_clip,
                   input_processor_for_clip)
from .interfaces import SupportsVision

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "model.vision_embed_tokens": "vision_embed_tokens",
}

# Cannot find the following 2 numbers from hf config.
_IMAGE_TOKEN_ID = 32044

# Result in the max possible feature size (h:w = 16:1)
MAX_IMAGE_FEATURE_SIZE_HEIGHT = 8000
MAX_IMAGE_FEATURE_SIZE_WIDTH = 50

CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(dropout=0.0,
                                                     hidden_act="quick_gelu",
                                                     hidden_size=1024,
                                                     image_size=336,
                                                     intermediate_size=4096,
                                                     num_attention_heads=16,
                                                     num_channels=3,
                                                     num_hidden_layers=24,
                                                     patch_size=14,
                                                     projection_dim=768)


class Phi3ImageEmbeddingBase(nn.Module):

    def __init__(self, wte=None) -> None:
        super().__init__()
        self.wte = wte
        self.layer_idx: int
        self.type_feature: str
        self.img_processor: CLIPVisionModel

    def get_img_features(self,
                         img_embeds: torch.FloatTensor) -> torch.FloatTensor:
        LAYER_IDX = self.layer_idx
        TYPE_FEATURE = self.type_feature

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the img_processor
        img_feature = self.img_processor(img_embeds,
                                         vision_feature_layer=LAYER_IDX)

        if TYPE_FEATURE == "patch":
            patch_feature = img_feature[:, 1:]
            return patch_feature

        if TYPE_FEATURE == "cls_patch":
            return img_feature

        raise NotImplementedError


# adapted from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_embedding_phi3_v.py
class Phi3HDImageEmbedding(Phi3ImageEmbeddingBase):
    """Phi3 Image embedding with HD transform."""

    def __init__(self, config: PretrainedConfig, wte=None) -> None:
        super().__init__(wte)

        self.image_token_id = _IMAGE_TOKEN_ID
        # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(
            config, 'n_embd') else config.hidden_size

        clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
        self.img_processor = CLIPVisionModel(clip_config)
        image_dim_out = config.img_processor['image_dim_out']
        self.num_img_tokens = config.img_processor['num_img_tokens']

        self.image_dim_out = image_dim_out

        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = config.embd_layer.get('use_hd_transform',
                                                      False)
        self.with_learnable_separator = config.embd_layer.get(
            'with_learnable_separator', False)
        self.hd_transform_order = config.embd_layer.get(
            'hd_transform_order', 'glb_sub')
        # with_hd_transform and with_learnable_separator should have same value
        assert self.use_hd_transform and self.with_learnable_separator

        # 1024 * 4, merge spatial to channel dimension
        self.glb_GN = nn.Parameter(torch.empty([1, 1, self.image_dim_out * 4]))
        self.sub_GN = nn.Parameter(
            torch.empty([1, 1, 1, self.image_dim_out * 4]))

        dim_projection = hidden_size
        depth = 2
        layers = [nn.Linear(image_dim_out * 4, dim_projection)]
        for _ in range(1, depth):
            layers.extend(
                [nn.GELU(),
                 nn.Linear(dim_projection, dim_projection)])
        self.img_projection = nn.Sequential(*layers)

        self.vocab_size = config.vocab_size

        self.layer_idx = config.img_processor.get('layer_idx', -2)
        self.type_feature = config.img_processor.get('type_feature', 'patch')

    def forward(self, input_ids: torch.LongTensor,
                pixel_values: torch.FloatTensor,
                image_sizes: torch.Tensor) -> torch.FloatTensor:
        """process and merge text embeddings with image embeddings."""

        # (batch_size, max_num_crops, 3, height, width)
        img_embeds = pixel_values

        # (batch_size, 2)
        img_sizes = image_sizes

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        positions = torch.nonzero(input_ids == self.image_token_id)

        select = False

        target_dtype = self.img_projection[0].bias.dtype

        if len(positions.tolist()) > 0:
            # if self.use_hd_transform and img_sizes:
            # img_embeds: (num_images, max_num_crops, 3, H, W)
            # img_sizes: (num_images, 2).view(1, -1)

            bs = img_embeds.shape[0]
            # Nx(HW)xC
            img_features = self.get_img_features(img_embeds.flatten(0, 1))
            base_feat_height = base_feat_width = int(
                img_features.shape[1]**0.5)

            # bs x max_num_crops x (24x24) x C
            img_features = img_features.view(
                bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
            C = self.image_dim_out
            H = base_feat_height

            output_imgs = []
            output_len = []

            for _bs in range(bs):
                h, w = img_sizes[_bs]
                h = h // 336
                w = w // 336
                B_ = h * w

                # 1 x (24x24) x 1024
                global_img_feature = img_features[_bs, :1]

                # 1 x 12 x 12 x 4096
                glb_img = global_img_feature \
                    .reshape(1, H // 2, 2, H // 2, 2,C) \
                    .permute(0, 1, 3, 2, 4, 5) \
                    .reshape(1, H // 2, H // 2, 4 * C)
                temp_glb_GN = self.sub_GN.repeat(1, H // 2, 1, 1)

                # 1 x 156 x 4096
                glb_img = torch.cat([glb_img, temp_glb_GN],
                                    dim=2).reshape(1, -1, 4 * C)

                # (max_num_crops-1) x (12x12) x C
                sub_img = img_features[_bs, 1:]
                # 16x574x1024
                # get rid of padding sub_img
                sub_img = sub_img[:B_]

                sub_img = sub_img.reshape(B_, H // 2, 2, H // 2, 2, C) \
                    .permute(0, 1, 3, 2, 4, 5).reshape(B_, -1, 4 * C)
                sub_img = sub_img.reshape(1, h, w, 12, 12, -1) \
                    .permute(0, 1, 3, 2, 4, 5) \
                    .reshape(1, h * 12, w * 12, 4 * C)
                temp_sub_GN = self.sub_GN.repeat(1, h * 12, 1, 1)
                sub_img = torch.cat([sub_img, temp_sub_GN],
                                    dim=2).reshape(1, -1, 4 * C)
                # (1, num_img_tokens, 1024*4)

                # glb + sub
                if self.hd_transform_order == 'glb_sub':
                    output_imgs.append(
                        torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                elif self.hd_transform_order == 'sub_glb':
                    output_imgs.append(
                        torch.cat([sub_img, self.glb_GN, glb_img], dim=1))

                temp_len = int((h * w + 1) * 144 + 1 + (h + 1) * 12)
                output_len.append(temp_len)

            num_img_tokens = output_len
            img_set_tensor = []
            for _output_img in output_imgs:
                img_feature_proj = self.img_projection(
                    _output_img.to(target_dtype))
                img_set_tensor.append(img_feature_proj)
            select = True

        input_ids.clamp_min_(0).clamp_max_(self.vocab_size)

        hidden_states = self.wte(input_ids)

        if select:
            idx = 0
            for i, cnt in enumerate(num_img_tokens):
                hidden_states[positions[idx, 0],
                              positions[idx, 1]:positions[idx, 1] +
                              cnt] = (img_set_tensor[i].to(
                                  hidden_states.dtype))
                idx += cnt

        return hidden_states.squeeze(0)


class Phi3VImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: BatchedTensors
    """
    Shape: `(batch_size, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different for each batch, in which case
    the data is passed as a list instead of a batched tensor.
    """

    image_sizes: torch.Tensor
    """
    Shape: `(batch_size, 2)`

    This should be in `(height, width)` format.
    """


# Based on https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_processing_phi3_v.py#L57
def _calc_padded_size(*, width: int, height: int, padding_unit: int = 336):
    target_height = int(np.ceil(height / padding_unit) * padding_unit)
    top_padding = int((target_height - height) / 2)
    bottom_padding = target_height - height - top_padding
    padded_width = width
    padded_height = height + top_padding + bottom_padding
    return padded_width, padded_height


# Based on https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_processing_phi3_v.py#L90
def _calc_hd_transform_size(*, width: int, height: int, hd_num: int = 16):
    transposed = False
    if width < height:
        width, height = height, width
        transposed = True

    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1

    new_width = int(scale * 336)
    new_height = int(new_width / ratio)

    padded_width, padded_height = _calc_padded_size(width=new_width,
                                                    height=new_height)

    if transposed:
        padded_width, padded_height = padded_height, padded_width

    return padded_width, padded_height


# Based on https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_processing_phi3_v.py#L181
def get_phi3v_image_feature_size(
    hf_config: PretrainedConfig,
    *,
    input_height: int,
    input_width: int,
) -> int:
    num_crops = getattr(hf_config, "num_crops", 16)
    new_width, new_height = _calc_hd_transform_size(width=input_width,
                                                    height=input_height,
                                                    hd_num=num_crops)

    return (new_height // 336 * new_width // 336 + 1) * 144 + 1 \
        + (new_height // 336 + 1) * 12


def get_max_phi3v_image_tokens(ctx: InputContext):

    return get_phi3v_image_feature_size(
        ctx.get_hf_config(PretrainedConfig),
        input_height=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
        input_width=MAX_IMAGE_FEATURE_SIZE_WIDTH,
    )


def dummy_data_for_phi3v(ctx: InputContext, seq_len: int):

    image_feature_size = get_max_phi3v_image_tokens(ctx)

    seq_data = dummy_seq_data_for_clip(
        CLIP_VIT_LARGE_PATCH14_336_CONFIG,
        seq_len,
        image_token_id=_IMAGE_TOKEN_ID,
        image_feature_size_override=image_feature_size,
    )
    mm_data = dummy_image_for_clip(
        CLIP_VIT_LARGE_PATCH14_336_CONFIG,
        image_width_override=MAX_IMAGE_FEATURE_SIZE_WIDTH,
        image_height_override=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
    )

    return seq_data, mm_data


# Reserve this function to also handle placeholders for additional images
# [ref: PR #5820]
@lru_cache
def _get_image_placeholder_token_ids(model_config: ModelConfig,
                                     idx: int) -> List[int]:
    assert idx > 0

    tokenizer = cached_get_tokenizer(model_config.tokenizer)

    # We need to get the token for "<", not "▁<"
    # https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/raw/main/tokenizer.json
    a_token_id, = tokenizer.encode("a", add_special_tokens=False)
    a_token_id_, *image_placeholder_token_ids = tokenizer.encode(
        f"a<|image_{idx}|>", add_special_tokens=False)
    assert a_token_id == a_token_id_

    return image_placeholder_token_ids


def input_processor_for_phi3v(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(PretrainedConfig)

    image_data = multi_modal_data["image"]
    if isinstance(image_data, Image.Image):
        w, h = image_data.size
        w, h = _calc_hd_transform_size(width=w, height=h)

        image_feature_size = get_phi3v_image_feature_size(hf_config,
                                                          input_width=w,
                                                          input_height=h)
    elif isinstance(image_data, torch.Tensor):
        raise NotImplementedError("Embeddings input is not supported yet")
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    prompt = llm_inputs.get("prompt")
    if prompt is None:
        new_prompt = None
    else:
        if prompt.count("<|image|>") > 0:
            logger.warning("Please follow the prompt format that is "
                           "documented on HuggingFace which does not involve "
                           "repeating <|image|> tokens.")
        elif len(re.findall(r"(<\|image_\d+\|>)+", prompt)) > 1:
            logger.warning("Multiple image input is not supported yet, "
                           "so any extra image tokens will be treated "
                           "as plain text.")

        new_prompt = prompt

    prompt_token_ids = llm_inputs["prompt_token_ids"]
    image_1_token_ids = _get_image_placeholder_token_ids(model_config, idx=1)

    new_token_ids: List[int] = []
    for i in range(len(prompt_token_ids) - len(image_1_token_ids) + 1):
        if prompt_token_ids[i:i + len(image_1_token_ids)] == image_1_token_ids:
            new_token_ids.append(_IMAGE_TOKEN_ID)

            # No need to further scan the list since we only replace once
            new_token_ids.extend(prompt_token_ids[i + len(image_1_token_ids):])
            break
        else:
            new_token_ids.append(prompt_token_ids[i])

    # NOTE: Create a defensive copy of the original inputs
    llm_inputs = LLMInputs(prompt_token_ids=new_token_ids,
                           prompt=new_prompt,
                           multi_modal_data=multi_modal_data)

    return input_processor_for_clip(
        model_config,
        CLIP_VIT_LARGE_PATCH14_336_CONFIG,
        llm_inputs,
        image_token_id=_IMAGE_TOKEN_ID,
        image_feature_size_override=image_feature_size,
    )


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_phi3v_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_phi3v)
@INPUT_REGISTRY.register_input_processor(input_processor_for_phi3v)
class Phi3VForCausalLM(nn.Module, SupportsVision):

    def __init__(self,
                 config: PretrainedConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config

        self.model = LlamaModel(config, cache_config, quant_config)

        # TODO: Optionally initializes this for supporting embeddings.
        self.vision_embed_tokens = Phi3HDImageEmbedding(
            config, self.model.embed_tokens)
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        if list(data.shape[1:]) != [2]:
            raise ValueError(
                f"The expected shape of image sizes is batch dimension plus "
                f"{[2]}. You supplied {tuple(data.shape)}.")

        return data

    def _validate_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = CLIP_VIT_LARGE_PATCH14_336_CONFIG.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[1:])

            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values in each batch element "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Phi3VImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)

        if pixel_values is None:
            return None

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        if not isinstance(image_sizes, torch.Tensor):
            raise ValueError("Incorrect type of image sizes. "
                             f"Got type: {type(image_sizes)}")

        return Phi3VImagePixelInputs(
            type="pixel_values",
            data=self._validate_pixel_values(pixel_values),
            image_sizes=self._validate_image_sizes(image_sizes))

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                **kwargs: object):
        image_input = self._parse_and_validate_image_input(**kwargs)

        if image_input is not None:
            inputs_embeds = self.vision_embed_tokens(
                input_ids, image_input["data"], image_input["image_sizes"])

            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.model(input_ids,
                                   positions,
                                   kv_caches,
                                   attn_metadata,
                                   intermediate_tensors,
                                   inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # post_layernorm is not needed in CLIPVisionModel
            if "vision_model.post_layernorm" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # We only do sharding for language model
                # and not vision model for now.
                if "vision_embed_tokens" in name and self.vision_embed_tokens:
                    continue
                if weight_name not in name:
                    continue
                param = params_dict[name.replace(weight_name, param_name)]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
