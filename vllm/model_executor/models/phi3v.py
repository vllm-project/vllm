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
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import cached_get_tokenizer
from vllm.sequence import IntermediateTensors, SamplerOutput

from .clip import (dummy_image_for_clip, dummy_seq_data_for_clip,
                   input_processor_for_clip)
from .interfaces import SupportsVision
from .utils import merge_vision_embeddings

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


class Phi3VImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]
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


class Phi3VImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """Shape: `(batch_size, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


Phi3VImageInputs = Union[Phi3VImagePixelInputs, Phi3VImageEmbeddingInputs]


class Phi3ImageEmbeddingBase(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer_idx: int
        self.type_feature: str
        self.img_processor: CLIPVisionModel

    def get_img_features(self,
                         img_embeds: torch.FloatTensor) -> torch.FloatTensor:
        TYPE_FEATURE = self.type_feature

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the img_processor
        img_feature = self.img_processor(img_embeds)

        if TYPE_FEATURE == "patch":
            patch_feature = img_feature[:, 1:]
            return patch_feature

        if TYPE_FEATURE == "cls_patch":
            return img_feature

        raise NotImplementedError


# adapted from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_embedding_phi3_v.py
class Phi3HDImageEmbedding(Phi3ImageEmbeddingBase):
    """Phi3 Image embedding with HD transform."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(
            config, 'n_embd') else config.hidden_size

        clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
        self.layer_idx = config.img_processor.get('layer_idx', -2)

        # Initialize the CLIP only up to the required feature layer
        if self.layer_idx < 0:
            num_hidden_layers = clip_config.num_hidden_layers + \
                self.layer_idx + 1
        else:
            num_hidden_layers = self.layer_idx + 1

        self.img_processor = CLIPVisionModel(
            clip_config, num_hidden_layers_override=num_hidden_layers)
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

        self.type_feature = config.img_processor.get('type_feature', 'patch')

    def forward(self, pixel_values: torch.FloatTensor,
                image_sizes: torch.Tensor) -> torch.FloatTensor:
        """
        process image and return vision embeddings.

        pixel_values: (num_images, num_crops, c, h, w)
        output: (num_images, num_img_tokens, hidden_size)
        """
        num_images, num_crops, c, h, w = pixel_values.shape
        pixel_values = pixel_values.flatten(0, 1)
        img_features = self.get_img_features(pixel_values)
        img_features = img_features.reshape(num_images, num_crops, -1,
                                            self.image_dim_out)
        image_features_proj = self.hd_feature_transform(
            img_features, image_sizes)
        return image_features_proj

    def hd_feature_transform(self, image_features, image_sizes):
        """
        image_features: (num_images, num_crops+1, 24*24, 1024)
        """
        assert (
            self.hd_transform_order == 'sub_glb'
        ), f'hd_transform_order `{self.hd_transform_order}` not implemented'
        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype

        global_image_features = image_features[:,
                                               0]  # (num_images, 24*24, 1024)
        # global feature can be viewed as a special HD case with num_crops 1x1
        global_image_features_hd = self.reshape_hd_patches_2x2merge(
            global_image_features, 1, 1)
        global_image_features_hd_newline = self.add_image_newline(
            global_image_features_hd)

        batch_image_features_proj = []
        # need a for loop to process each image because of different image sizes
        # (patch arrangement is different for each image)
        for i, img_size in enumerate(image_sizes):
            h, w = img_size
            h_crop = h // 336
            w_crop = w // 336
            num_crops = h_crop * w_crop

            # NOTE: real num_crops is padded
            # (num_crops, 24*24, 1024)
            sub_image_features = image_features[i, 1:1 + num_crops]
            sub_image_features_hd = self.reshape_hd_patches_2x2merge(
                sub_image_features, h_crop, w_crop)
            sub_image_features_hd_newline = self.add_image_newline(
                sub_image_features_hd)

            # [sub features, separator, global features]
            image_embeddings = torch.cat([
                sub_image_features_hd_newline.squeeze(
                    0),  # (h_crop*12*(w_crop*12+1), 4096)
                self.glb_GN.squeeze(0),
                global_image_features_hd_newline[i],
            ])
            img_proj = self.img_projection(
                image_embeddings.to(target_device, target_dtype))
            batch_image_features_proj.append(img_proj)

        return batch_image_features_proj

    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        """
        image_features: (num_images*num_crops, 24*24, 1024)
        output: (num_images, h_crop*12, w_crop*12, 4096)
        where h_crop*w_crop == num_crops
        """
        N, L, C = image_features.shape
        assert L == 576 and C == 1024 and N % (h_crop * w_crop) == 0
        num_images = N // (h_crop * w_crop)
        H = int(L**0.5)
        image_features_hd = (
            image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
            .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
            .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
            .reshape(N, -1, 4 * C)  # N, 144, 4096
            .reshape(num_images, h_crop, w_crop, H // 2, H // 2,
                     -1)  # n_img, h_crop, w_crop, 12, 12, 4096
            .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
            .reshape(num_images, h_crop * H // 2, w_crop * H // 2,
                     4 * C)  # n_img, h_crop*12, w_crop*12, 4096
        )
        return image_features_hd

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape
        # add the newline token to the HD image feature patches
        newline_embeddings = self.sub_GN.expand(num_images, h, -1,
                                                -1)  # (n_img, h, 1, hid_dim)
        image_features_hd_newline = torch.cat(
            [image_features_hd, newline_embeddings],
            dim=2).reshape(num_images, -1, hid_dim)
        return image_features_hd_newline


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
        image_feature_size = image_data.shape[0]
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
        self.image_token_id = _IMAGE_TOKEN_ID

        self.model = LlamaModel(config, cache_config, quant_config)

        # TODO: Optionally initializes this for supporting embeddings.
        self.vision_embed_tokens = Phi3HDImageEmbedding(config)
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
            self, **kwargs: object) -> Optional[Phi3VImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None:
            return None

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
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

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Phi3VImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: Phi3VImageInputs,
    ) -> torch.Tensor:

        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_embed_tokens is not None
        image_embeds = self.vision_embed_tokens(image_input["data"],
                                                image_input["image_sizes"])

        return image_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                **kwargs: object):
        image_input = self._parse_and_validate_image_input(**kwargs)

        if image_input is not None:
            vision_embeddings = self._process_image_input(image_input)
            inputs_embeds = self.model.get_input_embeddings(input_ids)
            inputs_embeds = merge_vision_embeddings(input_ids, inputs_embeds,
                                                    vision_embeddings,
                                                    self.image_token_id)
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
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
