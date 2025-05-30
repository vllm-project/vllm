# SPDX-License-Identifier: Apache-2.0

# A modified implementation of the AIMv2 Transformer
# inserted here also the image tokenizer used by Ovis2

import torch
import torch.nn as nn

from vllm.attention.layer import MultiHeadAttention
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.utils import divide
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
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

        self.gate_up_proj = MergedColumnParallelLinear(
            in_features,
            [hidden_features] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=hidden_features,
            output_size=in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


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
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

        self.attn = MultiHeadAttention(self.num_heads_per_partition,
                                       self.head_dim, self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv, _ = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads,
                          C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        x = self.attn(q, k, v)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1.forward_native(x))
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # they take the -1 as the ref embeddings, like a clip skip
        for block in self.blocks:
            tokens = block(tokens)
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

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        x = self.preprocessor(pixel_values)
        x = self.trunk(x)

        return x
