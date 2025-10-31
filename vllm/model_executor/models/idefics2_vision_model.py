# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

from collections.abc import Iterable

import torch
from torch import nn
from transformers.models.idefics2.configuration_idefics2 import (
    Idefics2Config,
    Idefics2VisionConfig,
)

from vllm.attention.layer import MultiHeadAttention
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .vision import run_dp_sharded_vision_model


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
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        patch_attention_mask: torch.BoolTensor,
        tgt_sizes: torch.IntTensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        max_nb_patches_h, max_nb_patches_w = (
            max_im_h // self.patch_size,
            max_im_w // self.patch_size,
        )
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side
        )
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()
            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)
            bucket_coords_h = torch.bucketize(
                fractional_coords_h, boundaries, right=True
            )
            bucket_coords_w = torch.bucketize(
                fractional_coords_w, boundaries, right=True
            )
            pos_ids = (
                bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w
            ).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings += self.position_embedding(position_ids)
        return embeddings


class Idefics2VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Idefics2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"  # noqa: E501
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_heads_per_partition = self.num_heads // tp_size

        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=use_data_parallel,
        )
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=use_data_parallel,
        )
        # Use unified MultiHeadAttention with Flash Attention support
        self.attn = MultiHeadAttention(
            self.num_heads_per_partition, self.head_dim, self.scale
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(
            hidden_states
        )  # batch_size, q_len, 3 * num_heads_per_partition * head_dim
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        # Use unified MultiHeadAttention implementation
        out = self.attn(query_states, key_states, value_states)
        attn_output, _ = self.out_proj(out)
        return attn_output


class Idefics2VisionMLP(nn.Module):
    def __init__(
        self,
        config: Idefics2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Idefics2EncoderLayer(nn.Module):
    def __init__(
        self,
        config: Idefics2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Idefics2VisionAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            use_data_parallel=use_data_parallel,
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Idefics2VisionMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=use_data_parallel,
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.

        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        return hidden_states


class Idefics2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention
    layers. Each layer is a
    [`Idefics2EncoderLayer`].

    Args:
        config: Idefics2Config
    """

    def __init__(
        self,
        config: Idefics2Config,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList(
            [
                Idefics2EncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    use_data_parallel=use_data_parallel,
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            inputs_embeds (torch.Tensor):
                Optionally, instead of passing `input_ids` you can choose to
                directly pass an embedded representation.
                This is useful if you want more control over how to convert
                `input_ids` indices into associated vectorsthan the model's
                internal embedding lookup matrix.
        """
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs
        return hidden_states


class Idefics2VisionTransformer(nn.Module):
    def __init__(
        self,
        config: Idefics2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool = True,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        embed_dim = config.hidden_size
        self.config = config
        self.use_data_parallel = use_data_parallel
        self.embeddings = Idefics2VisionEmbeddings(config)
        self.encoder = Idefics2Encoder(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
            use_data_parallel=use_data_parallel,
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        self.require_post_norm = require_post_norm
        self.post_layernorm = (
            nn.LayerNorm(
                embed_dim,
                eps=config.layer_norm_eps,
            )
            if require_post_norm
            else nn.Identity()
        )

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values,
        patch_attention_mask: torch.BoolTensor | None = None,
        tgt_sizes: torch.IntTensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            tgt_sizes=tgt_sizes,
        )
        if self.use_data_parallel:
            encoder_outputs = run_dp_sharded_vision_model(hidden_states, self.encoder)
        else:
            encoder_outputs = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(encoder_outputs)
        return last_hidden_state

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.encoder.layers)

        for name, loaded_weight in weights:
            # skip pooling header
            if name.startswith("head."):
                continue

            # post_layernorm is optional
            if name.startswith("post_layernorm.") and not self.require_post_norm:
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("encoder.layers."):
                layer_idx = int(name.split(".")[2])
                if layer_idx >= layer_count:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or self.use_data_parallel:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
