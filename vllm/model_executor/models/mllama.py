# coding=utf-8
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
from array import array
import math
from PIL import Image
from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
                    TypedDict, Union, Callable, Dict, Any, Set)

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaTextConfig, MllamaVisionConfig
from transformers.models.mllama.image_processing_mllama import MllamaImageProcessor
from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.attention.ops.paged_attn import PagedAttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from .interfaces import SupportsMultiModal
from .llama import LlamaAttention, LlamaMLP
# from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               ColumnParallelLinear)
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)

import vllm.distributed.parallel_state as ps
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, SequenceData


logger = init_logger(__name__)
MP_SCALE = 8
IMAGE_RES = 224
LLAMA_IMAGE_TOKEN_ID = 128256

class MllamaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size, max_num_image, max_num_chunk, num_channels, height, width)`"""
    aspect_ratio_ids: torch.Tensor
    """Shape: `(batch_size, max_num_image)`"""
    aspect_ratio_mask: torch.Tensor
    """Shape: `(batch_size, max_num_image, max_num_tiles)`"""

# TODO: support LlamaImageEmbeddingInputs

image_processor = None

def recursive_sum(x):
    if isinstance(x, torch.Tensor):
        return x.sum()
    if isinstance(x, (list, tuple)):
        return sum(recursive_sum(v) for v in x)
    if isinstance(x, (int, float)):
        return x
    return 0

def input_processor_for_mllama(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("encoder_multi_modal_data")
    hf_config = ctx.model_config.hf_config
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs
    global image_processor
    if image_processor is None:
        image_processor = MllamaImageProcessor(
            ctx.model_config.model,
            size={"height": hf_config.vision_config.image_size, "width": hf_config.vision_config.image_size},
        )
    processed_image = image_processor(multi_modal_data["image"])
    llm_inputs["encoder_multi_modal_data"]["image"] = processed_image
    num_tiles = recursive_sum(processed_image["num_tiles"])
    assert hf_config.vision_config.image_size % 14 == 0, "chunk size should be multiple of 14"
    token_per_chunk = (hf_config.vision_config.image_size // 14) ** 2 + 1
    num_tokens = num_tiles * token_per_chunk
    llm_inputs["prompt"] = llm_inputs["encoder_prompt"]
    llm_inputs["prompt_token_ids"] = llm_inputs["encoder_prompt_token_ids"]
    llm_inputs["encoder_prompt"] = "<|image|>" * num_tokens
    llm_inputs["encoder_prompt_token_ids"] = [LLAMA_IMAGE_TOKEN_ID] * num_tokens

    assert "decoder_multi_modal_data" not in llm_inputs, "multi-modal data should be put in encoder message of LLaMA Vision"

    return llm_inputs


def dummy_seq_data(
    seq_len: int,
    num_images: int
):
    assert seq_len >= num_images, "seq_len should be greater than or equal to num_images"
    token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                      [LLAMA_IMAGE_TOKEN_ID]) * num_images
    token_ids += array(VLLM_TOKEN_ID_ARRAY_TYPE,
                       [0]) * (seq_len - num_images)
    return SequenceData(token_ids)


def dummy_image(
    num_images: int,
):
    width = height = 512
    image = Image.new("RGB", (width, height), color=0)
    return {"image": image if num_images == 1 else [image] * num_images}

def dummy_data_for_mllama(ctx: InputContext, seq_len: int, mm_counts: Mapping[str, int]):
    num_images = mm_counts["image"]
    return dummy_seq_data(seq_len, num_images), dummy_image(num_images)

def get_max_mllama_image_tokens(ctx: InputContext) -> int:
    hf_config = ctx.model_config.hf_config
    token_per_chunk = (hf_config.vision_config.image_size // 14) ** 2 + 1
    return hf_config.vision_config.max_num_tiles * token_per_chunk 


# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


def _prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    past_key_values: Cache,
    num_vision_tokens: int,
    cross_attention_states: torch.Tensor,
    cross_attention_layers: List[int],
    device: str,
    dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cross_attention_mask is None:
        # should we raise error or prepare a full attn mask with all ones?
        return None, None
    else:
        # reshape so it can be used by attn module
        batch_size, text_total_length, *_ = cross_attention_mask.shape
        cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
        cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
        cross_attention_mask = cross_attention_mask.unsqueeze(1)

    # invert the mask
    inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(
        inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
    )

    # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.finfo(dtype).min
    full_text_row_masked_out_mask = (
        (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
    )
    cross_attention_mask *= full_text_row_masked_out_mask

    # In case we receive a new image but already have previous cross-attention key/values in cache,
    # then we need to extend the attention-mask and add previous images' lengths
    if (
        past_key_values is not None
        and cross_attention_states is not None
        and past_key_values.get_seq_length(cross_attention_layers[0]) != 0
    ):
        # make all zeros mask for cross-attn-mask from previuos cached hidden_states, all zeros right?
        # i.e. extend current cross-attn-mask on image-seq-length dimension to account for past_seen_tokens
        past_cross_attn_kv_length = past_key_values.get_seq_length(cross_attention_layers[0])
        past_cross_attn_mask = torch.zeros(
            (*cross_attention_mask.shape[:-1], past_cross_attn_kv_length), dtype=dtype, device=device
        )
        # concatenate both on image-seq-length dimension
        cross_attention_mask = torch.cat([past_cross_attn_mask, cross_attention_mask], dim=-1)

    return cross_attention_mask, full_text_row_masked_out_mask


def _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: torch.Tensor,
    num_patches: int,
    target_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1, 1).to(dtype)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = attention_mask @ attention_mask.transpose(-1, -2) * torch.finfo(dtype).min
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
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
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
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size)
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        if self.is_gated:
            embeddings = embeddings * self.gate.tanh()

        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size**-0.5

        self.gate = nn.Parameter(torch.zeros(1))

        # position embedding
        position_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.embedding = nn.Parameter(self.scale * position_embedding)

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.num_patches * self.hidden_size
        )

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        # position embeddings
        gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.view(1, 1, self.num_patches, self.hidden_size)

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )
        gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->MllamaVision
class MllamaVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class MllamaVisionSdpaAttention(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()

        model_parallel_size = get_tensor_model_parallel_world_size()
        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads
        self.num_local_heads = self.num_heads // model_parallel_size
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=False,
            input_is_parallel=True,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_state)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(q.shape[0], q.shape[1], self.num_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_local_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_local_heads, self.head_dim).transpose(1, 2)

        # TODO: remove padding in image encoder
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], -1)
        output, _ = self.o_proj(attn_output)
        return output


class MllamaVisionEncoderLayer(nn.Module):
    def __init__(self, config, is_gated: bool = False):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.attention_heads
        self.is_gated = is_gated
        self.intermediate_size = config.intermediate_size

        self.self_attn = MllamaVisionSdpaAttention(config)
        self.mlp = MllamaVisionMLP(config)

        self.input_layernorm = nn.LayerNorm(self.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size)

        # there used to be an if else here, no code path
        if is_gated:
            self.gate_attn = nn.Parameter(torch.ones(1) * math.pi / 4)
            self.gate_ffn = nn.Parameter(torch.ones(1) * math.pi / 4)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = None,
    ):
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state, attention_mask=attention_mask)
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
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MllamaEncoderLayer`].

    Args:
        config: MllamaConfig
    """

    def __init__(self, config: MllamaVisionConfig, num_layers=32, is_gated=False):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MllamaVisionEncoderLayer(config, is_gated) for _ in range(num_layers)])
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            # SDPA never returns attn weights, so the kwarg isn't used at all
            # TODO: fix this
            # if output_attentions:
            #     all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MllamaVisionModel(nn.Module):
    config_class = MllamaVisionConfig
    base_model_prefix = "vision_encoder"
    _no_split_modules = ["MllamaVisionSdpaAttention"]
    _supports_sdpa = True

    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.in_channels = config.in_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = ColumnParallelConv2dPatch(
            in_channels=config.in_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        
        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(config)

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size)
        self.layernorm_post = nn.LayerNorm(self.hidden_size)

        # encoders
        self.transformer = MllamaVisionEncoder(config, config.num_hidden_layers, is_gated=False)
        self.global_transformer = MllamaVisionEncoder(config, config.num_global_layers, is_gated=True)

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    def forward(
        self, pixel_values: torch.Tensor, aspect_ratio_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)

        # patch embedding
        patch_embeds = self.patch_embedding(pixel_values.to(self.layernorm_pre.weight.dtype))
        hidden_state = patch_embeds
        hidden_state = ps.get_tp_group().all_gather(hidden_state)

        # tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

        # apply cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # apply position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        # apply encoder
        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size * num_concurrent_media, -1)
            attention_mask = _prepare_aspect_ratio_attention_mask(
                aspect_ratio_mask=attention_mask,
                num_patches=self.num_patches,
                target_length=hidden_state.shape[2],
                dtype=self.layernorm_pre.weight.dtype,
            )

        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_state, all_intermediate_hidden_states = output[0], output[1]
        intermediate_hidden_states = [
            hidden_state
            for idx, hidden_state in enumerate(all_intermediate_hidden_states)
            if idx in self.intermediate_layers_indices
        ]
        intermediate_hidden_states = torch.stack(intermediate_hidden_states, dim=-1)

        # apply global encoder
        hidden_state = self.layernorm_post(hidden_state)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim
        )
        hidden_state = self.global_transformer(hidden_state, attention_mask=attention_mask)[0]
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = hidden_state[:, :, :slice_index]

        # adding intermediate layer outputs
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)
        return hidden_state


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->MllamaText
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
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MllamaTextCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Optional[MllamaTextConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        self.num_heads = self.config.num_attention_heads
        self.num_local_heads = self.num_heads // self.model_parallel_size
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_local_key_value_heads = self.num_key_value_heads // self.model_parallel_size
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_local_size = self.num_local_heads * self.head_dim
        self.kv_local_size = self.num_local_key_value_heads * self.head_dim

        # TODO(heheda12345): change to Q/KV seperate linear after #7448 is merged
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_key_value_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
        )

        self.q_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scaling = self.head_dim**-0.5

        self.attn = Attention(
            self.num_local_heads,
            self.head_dim,
            self.scaling,
            self.num_local_key_value_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cross_attention_states: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        qkv_dec, _ = self.qkv_proj(hidden_states)
        q, _, _ = qkv_dec.split([self.q_local_size, self.kv_local_size, self.kv_local_size],
                                dim=-1)
        if cross_attention_states is None:
            k = None
            v = None
        else:
            qkv_enc, _ = self.qkv_proj(cross_attention_states)
            _, k, v = qkv_enc.split([self.q_local_size, self.kv_local_size, self.kv_local_size],
                                    dim=-1)
            k = k.view(-1, self.num_local_key_value_heads, self.head_dim)
            v = v.view(-1, self.num_local_key_value_heads, self.head_dim)
            k = self.k_norm(k)
        q = q.view(-1, self.num_local_heads, self.head_dim)
        q = self.q_norm(q)

        output = self.attn(q, k, v, kv_cache, attn_metadata, attn_type=AttentionType.ENCODER_DECODER)
        out, _ = self.o_proj(output)
        return out

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaDecoderLayer with LlamaDecoder->MllamaSelfAttentionDecoder, Llama->MllamaText, LLAMA->MLLAMA_TEXT
class MllamaSelfAttentionDecoderLayer(nn.Module):
    def __init__(self, config: MllamaTextConfig, layer_idx: int, cache_config: Optional[CacheConfig] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=None,
            bias=False,
            cache_config=cache_config)

        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size, hidden_act=config.hidden_activation)
        self.input_layernorm = MllamaTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MllamaTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Ignore copy
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: Optional[torch.LongTensor],
        kv_cache: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MllamaCrossAttentionDecoderLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(self, config: MllamaTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.cross_attn = MllamaTextCrossAttention(
            config=config,
            layer_idx=layer_idx,
        )

        self.input_layernorm = MllamaTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))

        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_activation,
        )
        self.post_attention_layernorm = MllamaTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states
        return hidden_states

class MllamaTextModel(nn.Module):
    config_class = MllamaTextConfig
    base_model_prefix = "model"
    _no_split_modules = ["MllamaCrossAttentionDecoderLayer", "MllamaSelfAttentionDecoderLayer"]

    def __init__(self, config: MllamaTextConfig, cache_config:Optional[CacheConfig]):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size + 8, config.hidden_size)
        self.cross_attention_layers = config.cross_attention_layers

        layers = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.cross_attention_layers:
                layers.append(MllamaCrossAttentionDecoderLayer(config, layer_idx))
            else:
                layers.append(MllamaSelfAttentionDecoderLayer(config, layer_idx, cache_config=cache_config))

        self.layers = nn.ModuleList(layers)
        self.norm = MllamaTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.rotary_emb = MllamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        cross_attention_states: Optional[torch.LongTensor],
        cross_attention_mask: Optional[torch.LongTensor],
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            if isinstance(decoder_layer, MllamaCrossAttentionDecoderLayer):
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    cross_attention_states=cross_attention_states,
                    cross_attention_mask=cross_attention_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    # xattn_cache=xattn_caches[xattn_layer_idx] if xattn_caches is not None else None,
                    kv_cache=kv_caches[idx],
                    attn_metadata=attn_metadata,
                )
            elif isinstance(decoder_layer, MllamaSelfAttentionDecoderLayer):
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    positions=positions,
                    kv_cache=kv_caches[idx],
                    attn_metadata=attn_metadata,
                )
            else:
                raise ValueError(f"Unknown decoder layer type {type(decoder_layer)}")
        hidden_states = self.norm(hidden_states)
        return hidden_states
    

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # TODO: we have only SDPA currently and there's a bug when attn-bias is passed. Need to add eager attn and return the line
        # self.config._attn_implementation == "sdpa" and
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class MllamaForCausalLM(nn.Module):
    config_class = MllamaTextConfig
    base_model_prefix = "language_model"
    _no_split_modules = ["MllamaCrossAttentionDecoderLayer", "MllamaSelfAttentionDecoderLayer"]

    def __init__(self, config: MllamaTextConfig, cache_config:Optional[CacheConfig], quant_config: Optional[QuantizationConfig]):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.model = MllamaTextModel(config, cache_config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
        )


    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        cross_attention_states: Optional[torch.LongTensor],
        cross_attention_mask: Optional[torch.LongTensor],
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        return hidden_states

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_mllama_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_mllama)
@INPUT_REGISTRY.register_input_processor(input_processor_for_mllama)
class MllamaForConditionalGeneration(nn.Module, SupportsMultiModal):
    def __init__(self, config: MllamaConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
        self.image_size = config.vision_config.image_size

        self.vision_model = MllamaVisionModel(
            config.vision_config,
        )
        self.language_model = MllamaForCausalLM(
            config.text_config,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.multi_modal_projector = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
        )
        self.logits_processor = LogitsProcessor(config.output_hidden_states, config.text_config.vocab_size)
        self.sampler = Sampler()


    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.language_model.lm_head, hidden_states,
                                       sampling_metadata)
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def _parse_and_validate_image_input(
            self, **kwargs: object):
        # tensor with the same shape will be batched together by MultiModalInputs.batch, so pixel_values here can be: 
        #   - List[List[torch.Tensor]]: with shape (num_tiles, 3, image_res, image_res)
        #   - List[torch.Tensor]: with shape (num_image_in_batch, num_tiles, 3, image_res, image_res)
        #   - torch.Tensor: with shape (bs, num_image_in_batch, num_tiles, 3, image_res, image_res)
        pixel_values: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]] = kwargs.pop("pixel_values", None)
        image_embeds: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]] = kwargs.pop("image_embeds", None)
        aspect_ratio_ids: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]] = kwargs.pop("aspect_ratio_ids", None)
        aspect_ratio_mask: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]] = kwargs.pop("aspect_ratio_mask", None)

        if pixel_values is None and image_embeds is None:
            return None
        
        if pixel_values is not None and image_embeds is not None:
            raise ValueError("Both pixel values and image embeds are provided.")

        if pixel_values is not None:
            assert aspect_ratio_ids is not None
            assert aspect_ratio_mask is not None
            max_num_images = max([len(x[0]) for x in pixel_values])
            if max_num_images == 0:
                raise ValueError("No images provided.")
            max_num_tiles = max(max([len(x) for x in y[0]]) for y in pixel_values)
            device = self.multi_modal_projector.weight.device
            bsz = len(pixel_values)
            out_num_tiles = []
            out_images = torch.zeros(
                bsz,
                max_num_images,
                max_num_tiles,
                3,
                self.image_size,
                self.image_size,
                dtype=torch.float32,
                device=device,
            )
            out_ar_ids = torch.ones(bsz, max_num_images, dtype=torch.int64, device=device)
            out_ar_mask = torch.zeros(bsz, max_num_images, max_num_tiles, dtype=torch.int64, device=device)
            for b in range(len(pixel_values)):
                _num_tiles = []
                for i in range(len(pixel_values[b][0])):
                    img = pixel_values[b][0][i]
                    out_images[b, i, :img.shape[0]] = img
                    out_ar_ids[b, i] = aspect_ratio_ids[b][0][i]
                    out_ar_mask[b, i] = aspect_ratio_mask[b][0][i]
                    _num_tiles.append(img.shape[0])
                out_num_tiles.append(_num_tiles)

            return MllamaImagePixelInputs(
                type="pixel_values",
                data=out_images,
                aspect_ratio_ids=out_ar_ids,
                aspect_ratio_mask=out_ar_mask,
            )

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

        

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        print("input_ids", input_ids)
        if attn_metadata.num_prefill_tokens > 0 and attn_metadata.num_decode_tokens > 0:
            raise ValueError("Chunk prefill not supported")
        image_inputs = self._parse_and_validate_image_input(**kwargs)
        if image_inputs is None:
            cross_attention_mask = None
            run_xattn_mask = (attn_metadata.encoder_seq_lens_tensor != 0).reshape(-1, 1).cuda()
            xattn_caches = None
            vision_tokens = None
            cross_attention_states = None
            full_text_row_masked_out_mask = None
        else:
            # llama's reference implementation runs the vision model on CPU
            pixel_values = image_inputs['data']
            aspect_ratio_ids = image_inputs['aspect_ratio_ids']
            aspect_ratio_mask = image_inputs['aspect_ratio_mask']
            cross_attention_states = self.vision_model(pixel_values, aspect_ratio_ids, aspect_ratio_mask)
            cross_attention_states = self.multi_modal_projector(cross_attention_states)

            bsz, _, _, _, image_token_dim = tuple(cross_attention_states.shape)
            cross_attention_states = cross_attention_states.view(bsz, -1, image_token_dim)

            cross_attention_states_flat = torch.zeros(sum(attn_metadata.encoder_seq_lens), image_token_dim, device=cross_attention_states.device, dtype=cross_attention_states.dtype)
            start_pos = 0
            for seq_len, vision_token_in_batch in zip(attn_metadata.encoder_seq_lens, cross_attention_states):
                end_pos = start_pos + seq_len
                cross_attention_states_flat[start_pos:end_pos] = vision_token_in_batch[:seq_len]
                start_pos = end_pos
            cross_attention_states = cross_attention_states_flat
            cross_attention_mask = None # TODO
            full_text_row_masked_out_mask = None # TODO

            # run_xattn_mask = torch.ones((attn_metadata.num_prefill_tokens, 1), dtype=torch.bool, device=cross_attention_states.device) 
            # start_pos = 0
            # for seq_len, encoder_seq_len in zip(attn_metadata.seq_lens_tensor, attn_metadata.encoder_seq_lens):
            #     if encoder_seq_len == 0:
            #         run_xattn_mask[start_pos:start_pos+seq_len] = False
            #     start_pos += seq_len

        # if pixel_values is not None:
        #     if aspect_ratio_ids is None:
        #         raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
        #     # get vision tokens from vision model
        #     cross_attention_states = self.vision_model(pixel_values, aspect_ratio_ids, aspect_ratio_mask)
        #     cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
        #         -1, cross_attention_states.shape[-2], self.hidden_size
        #     )

        # cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
        #     cross_attention_mask,
        #     past_key_values=past_key_values,
        #     num_vision_tokens=self.vision_model.num_patches,
        #     cross_attention_layers=self.language_model.model.cross_attention_layers,
        #     cross_attention_states=cross_attention_states,
        #     device=self.device,
        #     dtype=self.dtype,
        # )

        # if cross_attention_mask is not None and cache_position is not None:
        #     cross_attention_mask = cross_attention_mask[:, :, cache_position]
        #     full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]

        # print("input_ids", input_ids, cross_attention_states is None)
        # if positions.numel() == 1:
        #     global step_name
        #     step_name = f"decode_{positions.item()}"

        outputs = self.language_model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        # if positions.numel() == 1 and positions.item() == 20:
        #     exit(0)

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # TODO: we have no attention_mask so this won't work, check if we really won't need attention mask and find another way
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.get_output_embeddings().weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cross_attention_mask": cross_attention_mask,
            }
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cached within each cross attn layer
        if (input_ids == self.config.image_token_index).any():
            model_inputs["pixel_values"] = pixel_values
            model_inputs["aspect_ratio_ids"] = aspect_ratio_ids
            model_inputs["aspect_ratio_mask"] = aspect_ratio_mask

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            model_kwargs["cross_attention_mask"] = torch.cat(
                [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
            )
        return model_kwargs

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
        updated_params = set()
        for name, loaded_weight in weights:
            if 'patch_embedding.weight' in name:
                name = name.replace('patch_embedding.weight', 'patch_embedding._linear.weight')
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
