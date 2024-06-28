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
from typing import Dict, Iterable, List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from PIL import Image
from transformers import FuyuConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, VisionLanguageConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.persimmon import PersimmonForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import ImagePixelData, get_dummy_image_data
from vllm.sequence import SamplerOutput

from .interfaces import SupportsVision

logger = init_logger(__name__)


class FuyuImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """
    Shape: 
    (batch_size, num_patches, patch_size_x * patch_size_y * num_channels)
    """


def calculate_num_image_tokens(image_size: Tuple[int, int]) -> Tuple[int, int]:
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
    W, H = image_size
    ncol = math.ceil(W / 30)
    nrow = math.ceil(H / 30)
    return ncol, nrow


def _image_processor(
    data: ImagePixelData,
    model_config: ModelConfig,
    vlm_config: VisionLanguageConfig,
) -> Dict[str, torch.Tensor]:
    image = data.image

    img_processor = MULTIMODAL_REGISTRY \
                    ._get_plugin_for_data_type(ImagePixelData) \
                    ._get_hf_image_processor(model_config, vlm_config)

    if isinstance(image, Image.Image):
        # Temporary patch before dynamic number of image tokens is supported
        # It's difficult to infer number of image tokens from image size for
        # image larger than (1920, 1080)
        _, _, h, w = vlm_config.image_input_shape
        if image.width > w or image.height > h:
            h, w = min(h, image.height), min(w, image.width)
            logger.warning(
                "Dynamic image larger than (1920, 1080) currently unsupported. "
                "Resizing input image to (%d, %d).", w, h)
            data.image = image.resize((w, h))

        # FuyuImageProcessor's preprocess returns unpatched image,
        # we need to call patchify_image manually
        outputs = MULTIMODAL_REGISTRY \
                    ._get_plugin_for_data_type(ImagePixelData) \
                    ._default_input_processor(data, model_config, vlm_config)

        image = torch.stack(outputs["images"][0])
        _, _, h, w = image.shape
        image_unpadded_h = outputs["image_unpadded_heights"]
        image_unpadded_w = outputs["image_unpadded_widths"]
        new_h = min(h, math.ceil(image_unpadded_h[0][0] / 30) * 30)
        new_w = min(w, math.ceil(image_unpadded_w[0][0] / 30) * 30)
        image = image[:, :, :new_h, :new_w]

    image_patches = img_processor.patchify_image(image)
    return {"image_patches": image_patches}


@MULTIMODAL_REGISTRY.register_image_pixel_input(_image_processor)
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)
class FuyuForCausalLM(nn.Module, SupportsVision):

    def __init__(self,
                 config: FuyuConfig,
                 vlm_config: VisionLanguageConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vlm_config = vlm_config
        self.image_token_id = vlm_config.image_token_id

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

        expected_input_type = self.vlm_config.image_input_type
        ImageInputType = VisionLanguageConfig.ImageInputType

        if expected_input_type != ImageInputType.PIXEL_VALUES:
            raise ValueError(
                f"Unexpected image input type: {expected_input_type}."
                "Phi3v only support pixel_values input currently.")

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
            self.language_model.lm_head.weight, hidden_states,
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
