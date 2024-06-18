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
from typing import Iterable, List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel, PretrainedConfig
from transformers.utils import logging

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VisionLanguageConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.vlm_base import VisionLanguageModelBase
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import get_dummy_image_data
from vllm.sequence import SamplerOutput

logger = logging.get_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "model.vision_embed_tokens": "vision_embed_tokens",
}

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

    def set_img_features(self, img_features: torch.FloatTensor) -> None:
        self.img_features = img_features

    def set_img_sizes(self, img_sizes: torch.LongTensor) -> None:
        self.img_sizes = img_sizes

    def get_img_features(self,
                         img_embeds: torch.FloatTensor) -> torch.FloatTensor:
        LAYER_IDX = self.layer_idx
        TYPE_FEATURE = self.type_feature

        img_processor_output = self.img_processor(img_embeds,
                                                  output_hidden_states=True)
        img_feature = img_processor_output.hidden_states[LAYER_IDX]

        if TYPE_FEATURE == "patch":
            patch_feature = img_feature[:, 1:]
            return patch_feature

        if TYPE_FEATURE == "cls_patch":
            return img_feature

        raise NotImplementedError


# adapted from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_embedding_phi3_v.py
class Phi3HDImageEmbedding(Phi3ImageEmbeddingBase):
    """Phi3 Image embedding with HD transform."""

    def __init__(self,
                 vision_language_config: VisionLanguageConfig,
                 config: PretrainedConfig,
                 wte=None) -> None:
        super().__init__(wte)

        self.image_token_id = vision_language_config.image_token_id
        # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(
            config, 'n_embd') else config.hidden_size

        clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
        self.img_processor = CLIPVisionModel(clip_config)
        image_dim_out = config.img_processor['image_dim_out']
        self.num_img_tokens = config.img_processor['num_img_tokens']

        self.image_dim_out = image_dim_out
        self.img_sizes = None

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
        self.img_features = None

        self.layer_idx = config.img_processor.get('layer_idx', -2)
        self.type_feature = config.img_processor.get('type_feature', 'patch')

    def forward(self,
                input_ids: torch.LongTensor,
                pixel_values: torch.FloatTensor,
                image_sizes=None) -> torch.FloatTensor:
        """process and merge text embeddings with image embeddings."""

        img_embeds = pixel_values
        img_sizes = image_sizes

        if self.img_features is not None:
            img_embeds = self.img_features.clone()
            self.img_features = None

        if self.img_sizes is not None:
            img_sizes = self.img_sizes

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        positions = torch.nonzero(input_ids == self.image_token_id)

        select = False

        target_device = self.img_projection[0].bias.device
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

            if isinstance(img_sizes, torch.Tensor):
                img_sizes.squeeze_(0)

            for _bs in range(bs):
                h, w = img_sizes
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
                    _output_img.to(target_device, target_dtype))
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
                                  hidden_states.device, hidden_states.dtype))
                idx += cnt

        return hidden_states.squeeze(0)


class Phi3VImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: (batch_size, 1 + num_patches, num_channels, height, width)"""

    image_sizes: torch.Tensor
    """Shape: (batch_size, 2)"""


@MULTIMODAL_REGISTRY.register_image_pixel_input()
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)
class Phi3VForCausalLM(VisionLanguageModelBase):

    def __init__(self,
                 config: PretrainedConfig,
                 vision_language_config: VisionLanguageConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__(vision_language_config)
        self.config = config
        self.model = LlamaModel(config, cache_config, quant_config)
        self.vision_embed_tokens = Phi3HDImageEmbedding(
            vision_language_config, config, self.model.embed_tokens)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Phi3VImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)

        expected_input_type = self.vision_language_config.image_input_type
        ImageInputType = VisionLanguageConfig.ImageInputType

        if expected_input_type != ImageInputType.PIXEL_VALUES:
            raise ValueError(
                f"Unexpected image input type: {expected_input_type}."
                "Phi3v only support pixel_values input currently.")

        if pixel_values is not None and image_sizes is not None:
            return Phi3VImagePixelInputs(type="pixel_values",
                                         data=pixel_values,
                                         image_sizes=image_sizes)

        return None

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata, **kwargs: object):
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
                                   inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
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
