# SPDX-License-Identifier: Apache-2.0

# A modified implementation of the AIMv2 Transformer
# inserted here also the image tokenizer used by Ovis2
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.transformers_utils.configs.ovis import AIMv2Config


class AIMv2SwiGLUFFN(nn.Module):

    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig,
                 prefix: str):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.hidden_size
        bias = config.use_bias

        # TODO(Isotr0py): investigate if we can add TP to visual tokenizer
        self.fc1 = ReplicatedLinear(in_features,
                                    hidden_features,
                                    bias=bias,
                                    quant_config=quant_config,
                                    prefix=f"{prefix}.fc1")
        self.fc2 = ReplicatedLinear(hidden_features,
                                    in_features,
                                    bias=bias,
                                    quant_config=quant_config,
                                    prefix=f"{prefix}.fc2")
        self.fc3 = ReplicatedLinear(in_features,
                                    hidden_features,
                                    bias=bias,
                                    quant_config=quant_config,
                                    prefix=f"{prefix}.fc3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel, _ = self.fc1(x)
        gate, _ = self.fc3(x)
        x_parallel = F.silu(x_parallel) * gate
        out, _ = self.fc2(x_parallel)
        return out


class AIMv2PatchEmbed(nn.Module):

    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm.forward_native(x)
        return x


class AIMv2ViTPreprocessor(nn.Module):

    def __init__(self, config: AIMv2Config):
        super().__init__()
        num_patches = (config.image_size // config.patch_size)**2

        self.patchifier = AIMv2PatchEmbed(config)
        self.pos_embed = nn.Parameter(
            torch.zeros((1, num_patches, config.hidden_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patchifier(x)
        _, N, _ = tokens.shape
        pos_embed = self.pos_embed.to(tokens.device)
        tokens = tokens + pos_embed[:, :N]
        return tokens


class AIMv2Attention(nn.Module):

    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig,
                 prefix: str):
        super().__init__()
        dim = config.hidden_size

        # TODO(Isotr0py): investigate if we can add TP to visual tokenizer
        self.num_heads = config.num_attention_heads
        self.qkv = ReplicatedLinear(dim, dim * 3, bias=config.qkv_bias)
        # self.qkv = QKVParallelLinear(
        #               hidden_size=dim,
        #               head_size=dim // config.num_attention_heads,
        #               total_num_heads=config.num_attention_heads,
        #               bias=config.qkv_bias,
        #               quant_config=quant_config,
        #               prefix=f"{prefix}.qkv")
        self.proj = ReplicatedLinear(dim, dim, bias=config.use_bias)
        # self.proj = RowParallelLinear(input_size=dim,
        #                  output_size=dim,
        #                  bias = config.use_bias,
        #                  quant_config=quant_config,
        #                  prefix=f"{prefix}.proj")

    def forward(  # todo might implement multiple attn implementations
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv, _ = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads,
                          C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x, _ = self.proj(x)
        return x


class AIMv2Block(nn.Module):

    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig,
                 prefix: str):
        super().__init__()
        self.attn = AIMv2Attention(config,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.attn")
        self.norm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = AIMv2SwiGLUFFN(config,
                                  quant_config=quant_config,
                                  prefix=f"{prefix}.mlp")
        self.norm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm_1.forward_native(x), mask)
        x = x + self.mlp(self.norm_2.forward_native(x))
        return x


class AIMv2Transformer(nn.Module):

    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig,
                 prefix: str):
        super().__init__()

        self.blocks = nn.ModuleList([
            AIMv2Block(config, quant_config, prefix=f"{prefix}.blocks.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.post_trunk_norm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # they take the -1 as the ref embeddings, like a clip skip
        for block in self.blocks:
            tokens = block(tokens, mask)
        # NO NORM IN THE OG IMPLEMENTATION
        # tokens = self.post_trunk_norm(tokens)
        return tokens


class AIMv2Model(torch.nn.Module):

    def __init__(self,
                 config: AIMv2Config,
                 quant_config: QuantizationConfig,
                 prefix: str = ""):
        super().__init__()
        self.preprocessor = AIMv2ViTPreprocessor(config)
        self.trunk = AIMv2Transformer(config,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.trunk")

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = self.preprocessor(pixel_values)
        x = self.trunk(x, mask)

        return x
