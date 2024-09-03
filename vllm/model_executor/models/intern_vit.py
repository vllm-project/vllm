# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

try:
    from xformers import ops as xops
    USE_XFORMERS_OPS = True
except ImportError:
    USE_XFORMERS_OPS = False

NORM2FN = {
    'rms_norm': RMSNorm,
    'layer_norm': nn.LayerNorm,
}


class InternVisionEmbeddings(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(in_channels=3,
                                         out_channels=self.embed_dim,
                                         kernel_size=self.patch_size,
                                         stride=self.patch_size)

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size,
            self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed,
                                  size=(H, W),
                                  mode='bicubic',
                                  align_corners=False)
        pos_embed = pos_embed.reshape(1, -1, H * W).permute(0, 2,
                                                            1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            target_dtype))  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1,
                                                   -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height,
                                width)
        ],
                                       dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternParallelAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads '
                f'(got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')

        self.scale = self.head_dim**-0.5
        self.qkv = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
        )

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            quant_config=quant_config,
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

    def forward(self, x):
        B, N, C = x.shape
        qkv, _ = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads_per_partition, self.head_dim)
        k = k.view(B, N, self.num_heads_per_partition, self.head_dim)
        v = v.view(B, N, self.num_heads_per_partition, self.head_dim)

        if self.qk_normalization:
            B_, N_, H_, D_ = q.shape
            q = self.q_norm.forward_native(q.flatten(-2,
                                                     -1)).view(B_, N_, H_, D_)
            k = self.k_norm.forward_native(k.flatten(-2,
                                                     -1)).view(B_, N_, H_, D_)

        x = xops.memory_efficient_attention_forward(q, k, v, scale=self.scale)
        x = x.view(B, N, -1)

        x, _ = self.proj(x)
        return x


class InternSdpaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads '
                f'(got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim,
                             3 * self.embed_dim,
                             bias=config.qkv_bias)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        if self.qk_normalization:
            B_, N_, H_, D_ = q.shape
            q = self.q_norm.forward_native(q.flatten(-2,
                                                     -1)).view(B_, N_, H_, D_)
            k = self.k_norm.forward_native(k.flatten(-2,
                                                     -1)).view(B_, N_, H_, D_)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(1, 2).view(B, N, -1)

        x = self.proj(x)
        return x


class InternMLP(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config)
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class InternVisionEncoderLayer(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        # fallback to sdpa attention if tp unavailable
        tp_size = get_tensor_model_parallel_world_size()
        num_heads = config.num_attention_heads
        if USE_XFORMERS_OPS and num_heads % tp_size == 0:
            self.attn = InternParallelAttention(config,
                                                quant_config=quant_config)
        else:
            self.attn = InternSdpaAttention(config)
        self.mlp = InternMLP(config, quant_config=quant_config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim,
                                             eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim,
                                             eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor *
                                torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor *
                                torch.ones(self.embed_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states)) * self.ls1

        hidden_states = hidden_states + self.mlp(
            self.norm2(hidden_states)) * self.ls2

        return hidden_states


class InternVisionEncoder(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 num_hidden_layers_override: Optional[int] = None):
        super().__init__()
        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override
        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(config=config, quant_config=quant_config)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, inputs_embeds: torch.Tensor):

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class InternVisionModel(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 num_hidden_layers_override: Optional[int] = None):
        super().__init__()
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size,
                                            old_size // patch_size,
                                            -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(),
                                size=new_size // patch_size,
                                mode='bicubic',
                                align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim,
                                                    -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is None and pixel_embeds is None:
            raise ValueError(
                'You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        elif pixel_values is not None:
            if pixel_values.ndim == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(
                    f'wrong pixel_values size: {pixel_values.shape}')

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)

        return encoder_outputs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
