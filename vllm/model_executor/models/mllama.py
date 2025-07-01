# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Mllama model."""
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Optional, TypedDict, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers.models.mllama.configuration_mllama as config_mllama
from PIL.Image import Image
from torch import nn
from transformers import BatchFeature, MllamaConfig
from transformers.modeling_outputs import (BaseModelOutput,
                                           CausalLMOutputWithPast)
from transformers.models.mllama.image_processing_mllama import (
    get_optimal_tiled_canvas)
from transformers.models.mllama.processing_mllama import (
    MllamaProcessor, get_cross_attention_token_mask)

import vllm.distributed.parallel_state as ps
from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.selector import _Backend
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVCrossParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalEncDecInputs,
                                    MultiModalFieldConfig, MultiModalKwargs)
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseProcessingInfo,
                                        EncDecMultiModalProcessor,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from .clip import CLIPMLP
from .interfaces import SupportsMultiModal, SupportsV0Only
from .llama import LlamaDecoderLayer, LlamaMLP
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

logger = init_logger(__name__)


class MllamaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: """
    """(batch_size, max_num_image, max_num_chunk, num_channel, height, width)"""
    aspect_ratio_ids: torch.Tensor
    """Shape: `(batch_size, max_num_image)`"""
    aspect_ratio_mask: torch.Tensor
    """Shape: `(batch_size, max_num_image, max_num_tiles)`"""


# TODO: support LlamaImageEmbeddingInputs


def calc_token_per_chunk(image_size: int) -> int:
    assert image_size % 14 == 0, "chunk size should be multiple of 14"
    token_per_chunk = (image_size // 14)**2 + 1
    return token_per_chunk


class MllamaProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> MllamaConfig:
        return self.ctx.get_hf_config(MllamaConfig)

    def get_hf_processor(self, **kwargs: object) -> MllamaProcessor:
        return self.ctx.get_hf_processor(MllamaProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_token_per_chunk_from_config(self) -> int:
        image_size = self.get_hf_config().vision_config.image_size
        return calc_token_per_chunk(image_size)

    def get_num_tiles_per_image(self, image_height: int,
                                image_width: int) -> int:
        vision_config = self.get_hf_config().vision_config
        max_num_tiles = vision_config.max_num_tiles
        image_size = vision_config.image_size
        tiled_height, tiled_width = get_optimal_tiled_canvas(
            image_height,
            image_width,
            max_num_tiles,
            tile_size=image_size,
        )
        num_tiles_height = tiled_height // image_size
        num_tiles_width = tiled_width // image_size
        return num_tiles_height * num_tiles_width

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_config = self.get_hf_config().vision_config
        image_size = vision_config.image_size
        max_num_tiles = vision_config.max_num_tiles
        # Result in the max possible feature size (h:w = 16:1)
        return ImageSize(height=max_num_tiles * image_size, width=image_size)


class MllamaDummyInputsBuilder(BaseDummyInputsBuilder[MllamaProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

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


class MllamaMultiModalProcessor(EncDecMultiModalProcessor[MllamaProcessingInfo]
                                ):

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        return_mm_hashes: bool = False,
    ) -> MultiModalEncDecInputs:
        mm_inputs = super().apply(prompt, mm_data, hf_processor_mm_kwargs,
                                  tokenization_kwargs, return_mm_hashes)

        image_token_id = self.info.get_hf_config().image_token_index
        # Check that the number of image tokens in the decoder prompt matches
        # the number of images provided in mm_data
        num_image_tokens = mm_inputs['prompt_token_ids'].count(image_token_id)
        image_data = mm_data.get("image", [])
        num_images = 1 if isinstance(image_data, Image) else len(image_data)
        if num_image_tokens != num_images:
            raise ValueError(
                f"The number of image tokens ({num_image_tokens}) must be"
                f" the same as the number of images ({num_images})")

        # Given prompt: <IMG0> P0 P1 <IMG1> <IMG2> P3 P4 D5 D6...., (P-prefill, D-decode)  # noqa: E501
        # P0 & P1 do cross attention with placeholder of <IMG0>
        # P3 P4 D5 D6 do cross attention with placeholder of <IMG1> and <IMG2>
        # Example input to encoder and decoder:
        # {
        #     'encoder': {
        #         'type': 'token',
        #         'prompt_token_ids': [128256, 128256, ..., 128256],
        #         'prompt': '<|image|><|image|>...<|image|>',
        #         'multi_modal_data': {'image': <PIL.Image.Image image mode=RGB size=1770x1180 at 0x7FDE2C624880>},  # noqa: E501
        #     },
        #     'decoder': {
        #         'type': 'token',
        #         'prompt_token_ids': [128000, 128256, 128000, 3923, 374, 279, 2262, 315, 420, 2217, 30],  # noqa: E501
        #         'prompt': '<|image|><|begin_of_text|>What is the content of this image?',  # noqa: E501
        #         'multi_modal_data': {'image': <PIL.Image.Image image mode=RGB size=1770x1180 at 0x7FDE2C624880>},  # noqa: E501
        #     },
        # }

        if mm_data:
            hf_processor = self.info.get_hf_processor()
            image_token: str = hf_processor.image_token

            # Since only the last group of consecutive images
            # are attended by the decoded tokens, we only need to
            # get the number of tokens for those images.
            token_per_chunk = self.info.get_token_per_chunk_from_config()
            num_decode_images = self._get_num_image_in_last_group(
                mm_inputs["prompt_token_ids"])
            num_encode_images = num_images - num_decode_images

            # Set encoder prompt length based on the number of tiles.
            # This tells the block manager to allocate correct number
            # of slots for encoder tokens.
            num_tiles = mm_inputs["mm_kwargs"]["num_tiles"]
            decode_tiles = num_tiles[num_encode_images:num_images].sum().item()
            num_tokens = decode_tiles * token_per_chunk
            mm_inputs["encoder_prompt_token_ids"] = [image_token_id
                                                     ] * num_tokens
            mm_inputs["encoder_prompt"] = image_token * num_tokens

        return mm_inputs

    def _get_num_image_in_last_group(self, prompt_token_ids: list[int]) -> int:
        num_images = 0
        for token_id in prompt_token_ids[::-1]:
            if token_id == self.info.get_hf_config().image_token_index:
                num_images += 1
            elif num_images > 0:
                break
        return num_images

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        if mm_data:
            num_tiles = [
                self.info.get_num_tiles_per_image(img.height, img.width)
                for img in mm_data["images"]
            ]
            processed_outputs = super()._call_hf_processor(
                prompt, mm_data, mm_kwargs, tok_kwargs)
            processed_outputs["num_tiles"] = torch.tensor(num_tiles)
            for k in ('pixel_values', 'aspect_ratio_ids', "aspect_ratio_mask"):
                processed_outputs[k] = processed_outputs[k].squeeze(0)

            processed_token_ids = processed_outputs.pop("input_ids")
            start_idx, end_idx = 0, processed_token_ids.size(1)
            processed_prompt_text = tokenizer.decode(processed_token_ids[0])

            hf_processor = self.info.get_hf_processor()
            bos_token = hf_processor.bos_token
            # Remove the bos_token from the start of prompt,
            # because we all know there would be image_token.
            if processed_prompt_text.startswith(bos_token):
                start_idx += 1
            # Remove the bos_token from the end of prompt,
            # because text is empty in this case.
            if processed_prompt_text.endswith(bos_token):
                end_idx -= 1
            processed_outputs[
                "input_ids"] = processed_token_ids[:, start_idx:end_idx]
        else:
            processed_outputs = tokenizer(prompt,
                                          add_special_tokens=False,
                                          return_tensors="pt")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            aspect_ratio_ids=MultiModalFieldConfig.batched("image"),
            aspect_ratio_mask=MultiModalFieldConfig.batched("image"),
            num_tiles=MultiModalFieldConfig.batched("image"),
        )

    def create_encoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:
        data = mm_data.get("image", [])
        num_images = 1 if isinstance(data, Image) else len(data)
        image_token_id = self.info.get_hf_config().image_token_index
        return [image_token_id] * num_images

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        token_per_chunk = self.info.get_token_per_chunk_from_config()
        image_token_id = self.info.get_hf_config().image_token_index

        def get_replacement_mllama(item_idx):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)
            num_tile = self.info.get_num_tiles_per_image(
                image_height=image_size.height,
                image_width=image_size.width,
            )
            num_tokens = num_tile * token_per_chunk
            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_mllama,
            )
        ]


def _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: torch.Tensor,
    num_patches: int,
    target_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1,
                                            1).to(dtype)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles*target_length, max_num_tiles*target_length)
    attention_mask = attention_mask.reshape(batch_size,
                                            max_num_tiles * target_length, 1)
    attention_mask = attention_mask @ attention_mask.transpose(
        -1, -2) * torch.finfo(dtype).min
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask


class ColumnParallelConv2dPatch(torch.nn.Module):
    """Conv2D Patching layer with model parallelism.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]],
        bias: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        self._linear = ColumnParallelLinear(
            in_channels * kernel_size[0] * kernel_size[1],
            out_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        x, _ = self._linear(x)
        return x


class MllamaPrecomputedAspectRatioEmbedding(nn.Module):

    def __init__(self,
                 config: config_mllama.MllamaVisionConfig,
                 is_gated: bool = True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(self.max_aspect_ratio_id + 1,
                                      self.max_num_tiles * self.hidden_size)
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_state: torch.Tensor,
                aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1,
                                        self.hidden_size)

        if self.is_gated:
            embeddings = embeddings * self.gate.tanh()

        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaPrecomputedPositionEmbedding(nn.Module):

    def __init__(self, config: config_mllama.MllamaVisionConfig):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size)**2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size**-0.5

        self.gate = nn.Parameter(torch.zeros(1))

        # position embedding
        position_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.embedding = nn.Parameter(self.scale * position_embedding)

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1,
            self.max_num_tiles * self.num_patches * self.hidden_size)

    def forward(self, hidden_state: torch.Tensor,
                aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        # position embeddings
        gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.view(
            1, 1, self.num_patches, self.hidden_size)

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size)
        gated_tile_position_embedding = self.gate.tanh(
        ) * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state


# TODO: support other attention backends for attention in vision model
class MllamaVisionSdpaAttention(nn.Module):

    def __init__(self,
                 config: config_mllama.MllamaVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        tensor_parallel_size = get_tp_group().world_size
        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads
        self.num_local_heads = self.num_heads // tensor_parallel_size
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_state)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(q.shape[0], q.shape[1], self.num_local_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_local_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_local_heads,
                   self.head_dim).transpose(1, 2)

        # TODO: remove padding in image encoder
        attn_output = F.scaled_dot_product_attention(q,
                                                     k,
                                                     v,
                                                     attn_mask=attention_mask,
                                                     dropout_p=0.0)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(attn_output.shape[0],
                                          attn_output.shape[1], -1)
        output, _ = self.o_proj(attn_output)
        return output


class MllamaVisionEncoderLayer(nn.Module):

    def __init__(
        self,
        config: config_mllama.MllamaVisionConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
        is_gated: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.attention_heads
        self.is_gated = is_gated
        self.intermediate_size = config.intermediate_size

        self.self_attn = MllamaVisionSdpaAttention(
            config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = CLIPMLP(config,
                           quant_config=quant_config,
                           prefix=f"{prefix}.mlp")

        self.input_layernorm = nn.LayerNorm(self.hidden_size,
                                            eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size,
                                                     eps=config.norm_eps)

        # there used to be an if else here, no code path
        if is_gated:
            self.gate_attn = nn.Parameter(torch.ones(1) * math.pi / 4)
            self.gate_ffn = nn.Parameter(torch.ones(1) * math.pi / 4)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state,
                                      attention_mask=attention_mask)
        gate_attn = 1 if not self.is_gated else self.gate_attn.tanh()
        hidden_state = residual + gate_attn * hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        gate_ffn = 1 if not self.is_gated else self.gate_ffn.tanh()
        hidden_state = residual + gate_ffn * hidden_state

        return hidden_state


class MllamaVisionEncoder(nn.Module):

    def __init__(
        self,
        config: config_mllama.MllamaVisionConfig,
        quant_config: Optional[QuantizationConfig],
        num_layers: int = 32,
        is_gated: bool = False,
        output_hidden_states=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            MllamaVisionEncoderLayer(config,
                                     quant_config=quant_config,
                                     is_gated=is_gated,
                                     prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(num_layers)
        ])
        self.output_hidden_states = output_hidden_states or []

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[BaseModelOutput]:
        encoder_states = ()

        for i, encoder_layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                encoder_states = encoder_states + (hidden_states, )
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
            )

        if len(self.layers) - 1 in self.output_hidden_states:
            encoder_states = encoder_states + (hidden_states, )

        return hidden_states, encoder_states


class MllamaVisionModel(nn.Module):

    def __init__(
        self,
        config: config_mllama.MllamaVisionConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.in_channels = config.num_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices

        self.num_patches = (self.image_size // self.patch_size)**2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = ColumnParallelConv2dPatch(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.class_embedding = nn.Parameter(self.scale *
                                            torch.randn(self.hidden_size))
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(
            config)

        self.pre_tile_positional_embedding = \
            MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)
        self.post_tile_positional_embedding = \
            MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size)
        self.layernorm_post = nn.LayerNorm(self.hidden_size)

        # encoders
        self.transformer = MllamaVisionEncoder(
            config,
            quant_config,
            config.num_hidden_layers,
            is_gated=False,
            output_hidden_states=config.intermediate_layers_indices,
            prefix=f"{prefix}.transformer",
        )
        self.global_transformer = MllamaVisionEncoder(
            config,
            quant_config,
            config.num_global_layers,
            is_gated=True,
            prefix=f"{prefix}.global_transformer",
        )

    def apply_class_embedding(self,
                              hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1,
                                                      hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    def forward(self, pixel_values: torch.Tensor,
                aspect_ratio_ids: torch.Tensor,
                aspect_ratio_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_concurrent_media, num_tiles, num_channels, \
            height, width = pixel_values.shape

        pixel_values = pixel_values.reshape(
            batch_size * num_concurrent_media * num_tiles, num_channels,
            height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(
            batch_size * num_concurrent_media, -1)

        # patch embedding
        patch_embeds = self.patch_embedding(
            pixel_values.to(self.layernorm_pre.weight.dtype))
        hidden_state = patch_embeds
        hidden_state = ps.get_tp_group().all_gather(hidden_state)

        # tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media,
                                            num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(
            hidden_state, aspect_ratio_ids)

        # apply cls token
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # apply position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media,
                                            num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state,
                                                       aspect_ratio_ids)

        # apply encoder
        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (
            0, 0, 0, num_padding_patches
        )  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        attention_mask = aspect_ratio_mask.reshape(
            batch_size * num_concurrent_media, -1)
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.layernorm_pre.weight.dtype,
        )

        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1,
                                         dim)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
        )
        hidden_state, intermediate_hidden_states = output[0], output[1]
        intermediate_hidden_states = torch.stack(intermediate_hidden_states,
                                                 dim=-1)

        # apply global encoder
        hidden_state = self.layernorm_post(hidden_state)
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media,
                                            num_tiles,
                                            num_patches + num_padding_patches,
                                            dim)
        hidden_state = self.post_tile_positional_embedding(
            hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles * (num_patches + num_padding_patches), dim)
        hidden_state = self.global_transformer(
            hidden_state, attention_mask=attention_mask)[0]
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media,
                                            num_tiles,
                                            num_patches + num_padding_patches,
                                            dim)
        hidden_state = hidden_state[:, :, :slice_index]

        # adding intermediate layer outputs
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media,
                                            num_tiles, num_patches, dim)
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles,
            num_patches + num_padding_patches, -1)
        intermediate_hidden_states = intermediate_hidden_states[:, :, :
                                                                slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1)
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states],
                                 dim=-1)
        return hidden_state

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        updated_params: set[str] = set()
        for name, loaded_weight in weights:
            if 'patch_embedding._linear.weight' in name:
                loaded_weight = loaded_weight.view(loaded_weight.shape[0], -1)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                updated_params.add(name)
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict.pop(name)
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                updated_params.add(name)
        return updated_params


class MllamaTextRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """
        MllamaTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MllamaTextCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Optional[config_mllama.MllamaTextConfig] = None,
        layer_idx: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pipeline_parallel_rank = get_pp_group().rank_in_group
        self.tensor_parallel_size = get_tp_group().world_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.num_local_heads = self.num_heads // self.tensor_parallel_size
        self.num_local_key_value_heads = \
            self.num_key_value_heads // self.tensor_parallel_size
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.layer_idx = layer_idx
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_local_size = self.num_local_heads * self.head_dim
        self.kv_local_size = self.num_local_key_value_heads * self.head_dim

        self.qkv_proj = QKVCrossParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # vllm.model_executor.layers.layernorm.RMSNorm has precision issue,
        # use huggingface's instead
        self.q_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scaling = self.head_dim**-0.5

        self.attn = Attention(
            self.num_local_heads,
            self.head_dim,
            self.scaling,
            self.num_local_key_value_heads,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv_range_for_decode: Optional[list[tuple[int, int]]],
        cross_attention_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q, k, v = self.qkv_proj(hidden_states, cross_attention_states)
        if cross_attention_states is not None:
            k = k.view(-1, self.num_local_key_value_heads, self.head_dim)
            v = v.view(-1, self.num_local_key_value_heads, self.head_dim)
            k = self.k_norm(k)

        q = q.view(-1, self.num_local_heads, self.head_dim)
        q = self.q_norm(q)

        if attention_mask is not None:
            output = self._attention_with_mask(q, k, v, attention_mask,
                                               kv_range_for_decode)
        else:
            output = self.attn(
                q.view(-1, self.num_local_heads * self.head_dim), k, v)
        out, _ = self.o_proj(output)
        return out

    def _attention_with_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_range_for_decode: list[tuple[int, int]],
    ) -> torch.Tensor:
        kv_cache = self.attn.kv_cache[self.pipeline_parallel_rank]
        attn_metadata: AttentionMetadata = get_forward_context().attn_metadata
        # Skip writing kv-cache for the initial profiling run.
        # TODO (NickLucche) replace with custom attn bias and use standard attn
        if len(kv_cache.shape) > 1:
            i = torch.ones(1, dtype=torch.float32)
            if self.attn.backend in (_Backend.FLASH_ATTN,
                                     _Backend.FLASH_ATTN_VLLM_V1):
                cached_k = torch.cat([k[s:e] for s, e in kv_range_for_decode])
                cached_v = torch.cat([v[s:e] for s, e in kv_range_for_decode])
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    cached_k,
                    cached_v,
                    kv_cache[0],
                    kv_cache[1],
                    attn_metadata.
                    cross_slot_mapping,  # type: ignore[union-attr]
                    "auto",
                    i,
                    i,
                )
            elif self.attn.backend in (_Backend.XFORMERS, _Backend.ROCM_FLASH,
                                       _Backend.TORCH_SDPA):
                key_cache, value_cache = PagedAttention.split_kv_cache(
                    kv_cache, self.num_local_key_value_heads, self.head_dim)
                cached_k = torch.cat([k[s:e] for s, e in kv_range_for_decode])
                cached_v = torch.cat([v[s:e] for s, e in kv_range_for_decode])
                PagedAttention.write_to_paged_cache(
                    cached_k, cached_v, key_cache, value_cache,
                    attn_metadata.cross_slot_mapping, "auto", i, i)
            else:
                raise ValueError(
                    f"Unsupported Attention backend {self.attn.backend} "
                    "enum found. Expected the Attention backend to be "
                    "FLASH_ATTN, FLASH_ATTN_VLLM_V1, "
                    "XFORMERS or TORCH_SDPA.")

        # We have to call torch.sdpa for prefill when using a
        # custom cross-attention mask. Because the mask is not a
        # standard causal mask, neither a block diagonal mask which
        # can be optimized by xformers.BlockDiagonalMask.
        # The mask is specially calculated for supporting multi
        # images and interleaved images.
        q_len = q.shape[0]
        kv_len = k.shape[0]
        q = q.transpose(0, 1).view(self.num_local_key_value_heads,
                                   self.num_key_value_groups, q_len,
                                   self.head_dim).contiguous()
        k = k.transpose(0,
                        1)[:,
                           None, :, :].expand(self.num_local_key_value_heads,
                                              self.num_key_value_groups,
                                              kv_len,
                                              self.head_dim).contiguous()
        v = v.transpose(0,
                        1)[:,
                           None, :, :].expand(self.num_local_key_value_heads,
                                              self.num_key_value_groups,
                                              kv_len,
                                              self.head_dim).contiguous()
        attention_mask = attention_mask.view(1, 1, q_len, kv_len)
        output = F.scaled_dot_product_attention(q,
                                                k,
                                                v,
                                                attn_mask=attention_mask,
                                                is_causal=False)
        output = output.permute(2, 0, 1, 3).reshape(
            q_len, self.num_local_heads * self.head_dim)
        return output


class MllamaCrossAttentionDecoderLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention
    and feedforward."""

    def __init__(
        self,
        config: config_mllama.MllamaTextConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.cross_attn = MllamaTextCrossAttention(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attn",
        )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))

        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        kv_range_for_decode: Optional[list[tuple[int, int]]],
        full_text_row_masked_out_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            kv_range_for_decode=kv_range_for_decode,
            cross_attention_states=cross_attention_states,
        )
        hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_attn_gate.tanh(
        ) * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_mlp_gate.tanh(
        ) * hidden_states
        return hidden_states


class MllamaTextModel(nn.Module):
    config_class = config_mllama.MllamaTextConfig
    base_model_prefix = "model"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size + 8,
                                                   config.hidden_size)
        self.cross_attention_layers = config.cross_attention_layers

        layers = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.cross_attention_layers:
                layers.append(
                    MllamaCrossAttentionDecoderLayer(
                        config,
                        layer_idx,
                        quant_config=quant_config,
                        prefix=f"{prefix}.layers.{layer_idx}",
                    ))
            else:
                # TODO: force LlamaDecoderLayer to config.attention_bias=False
                layers.append(
                    LlamaDecoderLayer(
                        config,
                        cache_config=cache_config,
                        quant_config=quant_config,
                        prefix=f"{prefix}.layers.{layer_idx}",
                    ))

        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        cross_attention_states: Optional[torch.LongTensor],
        cross_attention_mask: Optional[torch.LongTensor],
        kv_range_for_decode: Optional[list[tuple[int, int]]],
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor,
                                                      torch.Tensor]],
        skip_cross_attention: bool,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            if idx in self.cross_attention_layers:
                if not skip_cross_attention:
                    hidden_states = decoder_layer(
                        hidden_states=hidden_states,
                        cross_attention_states=cross_attention_states,
                        cross_attention_mask=cross_attention_mask,
                        kv_range_for_decode=kv_range_for_decode,
                        full_text_row_masked_out_mask=
                        full_text_row_masked_out_mask,
                    )
            else:
                hidden_states, residual = decoder_layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=None,
                )
                hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MllamaForCausalLM(nn.Module):
    config_class = config_mllama.MllamaTextConfig
    base_model_prefix = "language_model"
    _no_split_modules = [
        "MllamaCrossAttentionDecoderLayer", "MllamaSelfAttentionDecoderLayer"
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.text_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size
        self.model = MllamaTextModel(vllm_config=vllm_config,
                                     prefix=f"{prefix}.model")
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head",
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        cross_attention_states: Optional[torch.LongTensor],
        cross_attention_mask: Optional[torch.LongTensor],
        kv_range_for_decode: Optional[list[tuple[int, int]]],
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor,
                                                      torch.Tensor]],
        skip_cross_attention: bool,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            kv_range_for_decode=kv_range_for_decode,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            skip_cross_attention=skip_cross_attention,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        updated_params: set[str] = set()
        for name, loaded_weight in weights:
            if 'patch_embedding.weight' in name:
                name = name.replace('patch_embedding.weight',
                                    'patch_embedding._linear.weight')
                loaded_weight = loaded_weight.view(loaded_weight.shape[0], -1)
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                updated_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                updated_params.add(name)
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                orig_name = name
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    logger.debug("Missing name %s, orig name %s", name,
                                 orig_name)
                    continue

                param = params_dict.pop(name)
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                updated_params.add(name)
        return updated_params


@MULTIMODAL_REGISTRY.register_processor(MllamaMultiModalProcessor,
                                        info=MllamaProcessingInfo,
                                        dummy_inputs=MllamaDummyInputsBuilder)
class MllamaForConditionalGeneration(nn.Module, SupportsMultiModal,
                                     SupportsV0Only):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.vision_model.": "vision_model.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        },
        orig_to_new_suffix={
            "patch_embedding.weight": "patch_embedding._linear.weight",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: MllamaConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim
        self.pad_token_id = \
            config.pad_token_id if config.pad_token_id is not None else -1
        self.image_size = config.vision_config.image_size
        self.image_token_id = config.image_token_index

        self.vision_model = MllamaVisionModel(config.vision_config,
                                              quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "vision_model"))
        self.language_model = MllamaForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.multi_modal_projector = ColumnParallelLinear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
            quant_config=quant_config,
            gather_output=True,
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )
        self.logits_processor = LogitsProcessor(config.output_hidden_states,
                                                config.text_config.vocab_size)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.language_model.lm_head,
                                       hidden_states, sampling_metadata)
        return logits

    def unpack_data(self,
                    image_data: Union[list[torch.Tensor], torch.Tensor],
                    padding_value=0) -> torch.Tensor:
        if isinstance(image_data, torch.Tensor):
            # torch.Tensor
            return image_data
        else:
            assert isinstance(
                image_data[0],
                torch.Tensor), "Image data is not properly batched."
            # list[torch.Tensor]
            bsz = len(image_data)
            max_length = max(t.size(0) for t in image_data)
            trailing_dims = image_data[0].shape[1:]
            for data in image_data:
                cur_trailing_dims = data.shape[1:]
                assert cur_trailing_dims == trailing_dims
            output_tensor = torch.full((bsz, max_length, *trailing_dims),
                                       padding_value,
                                       dtype=image_data[0].dtype,
                                       device=image_data[0].device)
            for i, t in enumerate(image_data):
                output_tensor[i, :t.size(0)] = t
            return output_tensor

    def _parse_and_validate_image_input(self, **kwargs: object):
        # tensor with the same shape will be batched together by
        # MultiModalKwargs.batch, so pixel_values here can be:
        #   - list[torch.Tensor]:
        #       with shape (num_image, num_tiles, 3, image_res, image_res)
        #   - torch.Tensor:
        #       with shape (bs, num_image, num_tiles, 3, image_res, image_res)
        pixel_values: Optional[Union[list[list[torch.Tensor]],
                                     list[torch.Tensor],
                                     torch.Tensor]] = kwargs.pop(
                                         "pixel_values", None)
        image_embeds: Optional[Union[list[list[torch.Tensor]],
                                     list[torch.Tensor],
                                     torch.Tensor]] = kwargs.pop(
                                         "image_embeds", None)
        aspect_ratio_ids: Optional[Union[list[list[torch.Tensor]],
                                         list[torch.Tensor],
                                         torch.Tensor]] = kwargs.pop(
                                             "aspect_ratio_ids", None)
        aspect_ratio_mask: Optional[Union[list[list[torch.Tensor]],
                                          list[torch.Tensor],
                                          torch.Tensor]] = kwargs.pop(
                                              "aspect_ratio_mask", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None and image_embeds is not None:
            raise ValueError(
                "Both pixel values and image embeds are provided.")

        if pixel_values is not None:
            assert aspect_ratio_ids is not None
            assert aspect_ratio_mask is not None

            return MllamaImagePixelInputs(
                type="pixel_values",
                data=self.unpack_data(pixel_values),
                aspect_ratio_ids=self.unpack_data(aspect_ratio_ids),
                aspect_ratio_mask=self.unpack_data(aspect_ratio_mask))

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

    def _get_and_validate_encoder_lens(
        self,
        encoder_seq_lens: list[int],
        num_tiles: list[list[int]],
        num_tokens_per_tile: int,
    ) -> list[int]:
        # Get the actual number of encoder tokens for each sample.
        # Because attn_metadata.encoder_seq_lens only counts the last
        # group of images for each sample, which is used to cheat the
        # block manager to allocate blocks for those images only.
        # See MllamaMultiModalProcessor for more details.
        actual_encoder_seq_lens = [
            sum(num_tile) * num_tokens_per_tile for num_tile in num_tiles
        ]

        # remove 0 encoder len entries for text-only requests for these
        # assertions
        attn_metadata_lens = [x for x in encoder_seq_lens if x > 0]
        assert len(actual_encoder_seq_lens) == len(attn_metadata_lens)
        for actual_len, last_group_len in zip(actual_encoder_seq_lens,
                                              attn_metadata_lens):
            assert actual_len >= last_group_len

        return actual_encoder_seq_lens

    def flat_encoder_result(self, cross_attention_states: torch.Tensor,
                            attn_metadata: AttentionMetadata,
                            actual_encoder_seq_lens: list[int]):

        cross_attention_states_flat = torch.zeros(
            sum(actual_encoder_seq_lens),
            cross_attention_states.shape[-1],
            device=cross_attention_states.device,
            dtype=cross_attention_states.dtype)
        start_pos = 0
        for seq_len, vision_token_in_batch in zip(actual_encoder_seq_lens,
                                                  cross_attention_states):
            end_pos = start_pos + seq_len
            cross_attention_states_flat[
                start_pos:end_pos] = vision_token_in_batch[:seq_len]
            start_pos = end_pos
        cross_attention_states = cross_attention_states_flat
        return cross_attention_states

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_cross_attention_states(
        self,
        image_inputs: MllamaImagePixelInputs,
        attn_metadata: AttentionMetadata,
        actual_encoder_seq_lens: list[int],
    ) -> tuple[torch.Tensor]:
        # NOTE: llama's reference implementation runs vision model on CPU
        pixel_values = image_inputs['data']
        aspect_ratio_ids = image_inputs['aspect_ratio_ids']
        aspect_ratio_mask = image_inputs['aspect_ratio_mask']
        cross_attention_states = self.vision_model(pixel_values,
                                                   aspect_ratio_ids,
                                                   aspect_ratio_mask)
        cross_attention_states, _ = self.multi_modal_projector(
            cross_attention_states)

        bsz, _, _, _, image_token_dim = tuple(cross_attention_states.shape)
        cross_attention_states = cross_attention_states.view(
            bsz, -1, image_token_dim)

        cross_attention_states = self.flat_encoder_result(
            cross_attention_states, attn_metadata, actual_encoder_seq_lens)

        return cross_attention_states

    def get_cross_attention_mask(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        num_tiles: list[list[int]],
        num_tokens_per_tile: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = input_ids.tolist()
        start = 0
        batch_token_ids = []
        for seq_len in attn_metadata.seq_lens:
            batch_token_ids.append(token_ids[start:start + seq_len])
            start += seq_len
        sparse_mask = [
            get_cross_attention_token_mask(t, self.image_token_id)
            for t in batch_token_ids
        ]

        # Skip generating cross-attention mask if all samples
        # are text-only or have only 1 leading image.
        if skip_attention_mask(sparse_mask):
            return None, None

        dense_mask, tile_range_for_decode = \
            convert_sparse_cross_attention_mask_to_dense(
                sparse_mask, num_tiles, attn_metadata.seq_lens)
        cross_attention_mask = \
            convert_dense_cross_attention_mask_to_tensor(
                dense_mask, num_tokens_per_tile, input_ids.device, dtype)
        kv_range_for_decode = [[
            t[0] * num_tokens_per_tile, t[1] * num_tokens_per_tile
        ] for t in tile_range_for_decode]

        return cross_attention_mask, kv_range_for_decode

    def get_full_text_row_masked_out_mask(
        self,
        attn_metadata: AttentionMetadata,
        device: torch.device,
    ) -> torch.Tensor:
        full_text_row_masked_out_mask = torch.ones(
            (attn_metadata.num_prefill_tokens, 1), dtype=torch.bool)
        start_pos = 0
        for seq_len, encoder_seq_len in zip(attn_metadata.seq_lens,
                                            attn_metadata.encoder_seq_lens):
            if encoder_seq_len == 0:
                full_text_row_masked_out_mask[start_pos:start_pos +
                                              seq_len] = False
            start_pos += seq_len
        full_text_row_masked_out_mask = full_text_row_masked_out_mask.to(
            device)
        return full_text_row_masked_out_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs: object,
    ) -> Union[CausalLMOutputWithPast]:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata.num_prefill_tokens > 0 and \
            attn_metadata.num_decode_tokens > 0:
            raise ValueError("Chunk prefill not supported")
        image_inputs = self._parse_and_validate_image_input(**kwargs)
        cross_attention_states = None
        cross_attention_mask = None
        kv_range_for_decode = None

        # For 1) text-only prefill and decode, 2) image-present decode.
        if image_inputs is None:
            full_text_row_masked_out_mask = (
                attn_metadata.encoder_seq_lens_tensor
                != 0).reshape(-1, 1).to(input_ids.device)
            skip_cross_attention = attn_metadata.max_encoder_seq_len == 0

        # For image-present prefill.
        else:
            skip_cross_attention = False

            num_tiles = [t.tolist() for t in kwargs.pop("num_tiles")]
            num_tokens_per_tile = calc_token_per_chunk(self.image_size)

            actual_encoder_seq_lens = self._get_and_validate_encoder_lens(
                attn_metadata.encoder_seq_lens,
                num_tiles,
                num_tokens_per_tile,
            )

            cross_attention_states = self.get_cross_attention_states(
                image_inputs, attn_metadata, actual_encoder_seq_lens)

            full_text_row_masked_out_mask = \
                self.get_full_text_row_masked_out_mask(
                    attn_metadata, input_ids.device)

            cross_attention_mask, kv_range_for_decode = \
                self.get_cross_attention_mask(
                    input_ids, attn_metadata, num_tiles,
                    num_tokens_per_tile, cross_attention_states.dtype)

        outputs = self.language_model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            kv_range_for_decode=kv_range_for_decode,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            skip_cross_attention=skip_cross_attention,
        )

        return outputs

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model="vision_model")


def skip_attention_mask(sparse_mask: list[list[int]]) -> bool:
    for mask in sparse_mask:
        # Skip text-only samples.
        if len(mask) == 0:
            continue
        # If the sample contains more than 1 images,
        # we can't skip mask.
        if len(mask) != 1:
            return False
        # If the sample contains only 1 image,
        # but the image is not the leading one,
        # we can't skip mask.
        if mask[0][0] != 0 or mask[0][1] != -1:
            return False
    return True


def convert_sparse_cross_attention_mask_to_dense(
    sparse_mask: list[list[list[int]]],
    num_tiles: list[list[int]],
    lengths: list[int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    total_length = sum(lengths)
    total_tiles = sum([sum(tiles) for tiles in num_tiles])
    dense_mask = np.zeros(shape=(total_length, total_tiles), dtype=np.int64)
    # A list of ranges, range[i] = [start, end] means that the i-th image will
    # use tiles[start, end] for cross-attention decoding.
    tile_range_for_decode = []

    seq_start = 0
    tile_start = 0

    # sparse_mask has an [] entry for each sequence that does not have images,
    # but num_tiles does not have these entries...
    num_tiles_idx = 0
    for masks, length in zip(sparse_mask, lengths):
        if len(masks) == 0:
            # Text only
            continue

        tiles = num_tiles[num_tiles_idx]
        num_tiles_idx += 1
        ts, td = -1, 0
        for mask, tile in zip(masks, tiles):
            if len(mask) != 2:
                continue
            start, end = mask
            end = min(end, length)
            if end == -1:
                end = length
            if end == length:
                if ts == -1:
                    ts = tile_start
                td += tile
            dense_mask[seq_start + start:seq_start + end,
                       tile_start:tile_start + tile] = 1
            tile_start += tile
        assert ts != -1
        assert td != 0
        tile_range_for_decode.append((ts, ts + td))
        seq_start += length
    assert num_tiles_idx == len(num_tiles)

    return dense_mask, tile_range_for_decode


def convert_dense_cross_attention_mask_to_tensor(
    cross_attention_token_mask: np.ndarray,
    num_tokens_per_tile: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.tensor(cross_attention_token_mask, dtype=dtype, device=device)
    mask = mask.repeat_interleave(num_tokens_per_tile, dim=1)

    mask = 1.0 - mask
    mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(dtype).min)

    ninf = torch.finfo(dtype).min
    full_text_mask = ((mask != ninf).any(dim=-1).type_as(mask)[..., None])
    mask *= full_text_mask
    # (num_prompt_tokens, num_encoder_tokens)
    return mask
