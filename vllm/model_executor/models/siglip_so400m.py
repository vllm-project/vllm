# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import SiglipVisionConfig

from vllm.attention.layer import MultiHeadAttention
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.vision import VisionEncoderInfo


class SiglipSo400mEncoderInfo(VisionEncoderInfo[SiglipVisionConfig]):

    def get_num_image_tokens(self, *, image_width: int,
                             image_height: int) -> int:
        return (image_width // self.get_patch_size()) * (image_height //
                                                         self.get_patch_size())

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        return self.vision_config.patch_size


class SiglipSo400mVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
        )
        self.position_embedding = VocabParallelEmbedding(
            num_embeddings=(config.image_size // config.patch_size)**2,
            embedding_dim=self.embed_dim,
        )
        self.register_buffer(
            "position_ids",
            torch.arange(self.position_embedding.num_embeddings).unsqueeze(0),
            persistent=False)

    def forward(self,
                pixel_values: torch.Tensor,
                interpolate_pos_encoding: bool = False) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # [B, C, H, W]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)  # [B, N, C]
        embeddings += self.position_embedding(
            self.position_ids[:, :embeddings.size(1)])
        return embeddings


class SiglipSo400mAttention(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)
        self.attn = MultiHeadAttention(self.num_heads_per_partition,
                                       self.head_dim, self.scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        out = self.attn(q, k, v)
        out, _ = self.out_proj(out)
        return out, None


class SiglipSo400mMLP(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            quant_config=quant_config,
        )
        self.act = get_act_fn(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class SiglipSo400mEncoderLayer(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size,
                                  eps=config.layer_norm_eps)
        self.attn = SiglipSo400mAttention(config, quant_config)
        self.norm2 = nn.LayerNorm(config.hidden_size,
                                  eps=config.layer_norm_eps)
        self.mlp = SiglipSo400mMLP(config, quant_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SiglipSo400mVisionTransformer(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.embeddings = SiglipSo400mVisionEmbeddings(config)
        self.encoder_layers = nn.ModuleList([
            SiglipSo400mEncoderLayer(config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        return x.mean(dim=1)  # mean pooling


class SiglipSo400mVisionModel(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.encoder = SiglipSo400mVisionTransformer(config, quant_config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, tensor)
                loaded.add(name)
        return loaded
