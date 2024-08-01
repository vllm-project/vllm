# coding=utf-8

# adapted from https://github.com/huggingface/transformers/blob/v4.43.2/src/transformers/models/idefics2/modeling_idefics2.py
# Copyright 2024 The vLLM team.
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
"""PyTorch Idefics2 model."""

from typing import Optional

import torch
from torch import nn
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.idefics2.configuration_idefics2 import (
    Idefics2Config, Idefics2VisionConfig)
from xformers import ops as xops

from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig

_CONFIG_FOR_DOC = "Idefics2Config"


class Idefics2VisionEmbeddings(nn.Module):
    """
    This is a modified version of `siglip.modelign_siglip.SiglipVisionEmbeddings
    ` to enable images of variable
    resolution.

    The modifications are adapted from [Patch n' Pack: NaViT, a Vision
    Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)
    which allows treating images in their native aspect ratio and without the
    need to resize them to the same fixed size. In particular, we start from the
    original pre-trained SigLIP model(which uses images of fixed-size square
    images) and adapt it by training on images of variable resolutions.
    """

    def __init__(self, config: Idefics2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions,
                                               self.embed_dim)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        patch_attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h, max_nb_patches_w = (
            max_im_h // self.patch_size,
            max_im_w // self.patch_size,
        )
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0,
                                  1 / self.num_patches_per_side)
        position_ids = torch.full(size=(batch_size,
                                        max_nb_patches_h * max_nb_patches_w),
                                  fill_value=0)

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h,
                                              boundaries,
                                              right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w,
                                              boundaries,
                                              right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side +
                       bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class Idefics2VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Idefics2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"  # noqa: E501
                f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            quant_config=quant_config,
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.size()
        qkv, _ = self.qkv_proj(
            hidden_states
        )  # batch_size, q_len, 3 * num_heads_per_partition * head_dim

        query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        query_states = query_states.view(batch_size, q_len,
                                         self.num_heads_per_partition,
                                         self.head_dim)
        key_states = key_states.view(batch_size, q_len,
                                     self.num_heads_per_partition,
                                     self.head_dim)
        value_states = value_states.view(batch_size, q_len,
                                         self.num_heads_per_partition,
                                         self.head_dim)
        # see: https://facebookresearch.github.io/xformers/components/ops.html
        out = xops.memory_efficient_attention_forward(
            query_states,
            key_states,
            value_states,
            p=self.dropout,
            scale=self.scale,
        )
        out = out.view(batch_size, q_len, -1)
        attn_output, _ = self.out_proj(out)

        return attn_output


class Idefics2VisionMLP(nn.Module):

    def __init__(
        self,
        config: Idefics2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class Idefics2EncoderLayer(nn.Module):

    def __init__(self, config: Idefics2Config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Idefics2VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)
        self.mlp = Idefics2VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where
                padding elements are indicated by very large negative values.

        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Idefics2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention
    layers. Each layer is a
    [`Idefics2EncoderLayer`].

    Args:
        config: Idefics2Config
    """

    def __init__(self, config: Idefics2Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Idefics2EncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            inputs_embeds (torch.Tensor):
                Optionally, instead of passing `input_ids` you can choose to
                directly pass an embedded representation.
                This is useful if you want more control over how to convert
                `input_ids` indices into associated vectorsthan the model's
                internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` 
                , *optional*):
                Mask to avoid performing attention on padding token indices.
                Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
        """  # noqa: E501

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
            )

            hidden_states = layer_outputs
        return hidden_states


class Idefics2VisionTransformer(nn.Module):

    def __init__(self, config: Idefics2VisionConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.config = config
        self.embeddings = Idefics2VisionEmbeddings(config)
        self.encoder = Idefics2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim,
                                           eps=config.layer_norm_eps)

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.tensor:
        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.config.patch_size
            patch_attention_mask = torch.ones((
                batch_size,
                pixel_values.size(2) // patch_size,
                pixel_values.size(3) // patch_size,
            ))
            patch_attention_mask = patch_attention_mask.to(
                dtype=torch.bool, device=pixel_values.device)

        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to
        # the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to
        # attending to the full sequence
        if not torch.any(~patch_attention_mask):
            patch_attention_mask = None
        elif not self._use_flash_attention_2:
            patch_attention_mask = _prepare_4d_attention_mask(
                patch_attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
        )

        last_hidden_state = self.post_layernorm(encoder_outputs)
        return last_hidden_state
