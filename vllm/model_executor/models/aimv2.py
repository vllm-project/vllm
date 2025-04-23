# SPDX-License-Identifier: Apache-2.0

# A modified implementation of the AIMv2 Transformer
# inserted here also the image tokenizer used by Ovis2
from typing import Optional

import torch
from torch import nn, softmax
from torch.nn import functional as F
from torch.nn.functional import gumbel_softmax, pad

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.transformers_utils.configs.ovis2 import (AIMv2Config,
                                                   Aimv2VisualTokenizerConfig)

IMAGE_INDICATOR_IDS = [-301, -302, -303, -304,
                       -305]  # kept for vocab prefixed tokens


def st_argmax(y_soft: torch.Tensor, dim: int):  # straight-through softmax
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(
        y_soft, memory_format=torch.legacy_contiguous_format).scatter_(
            dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret


class Aimv2VisualTokenizer(torch.nn.Module):

    def __init__(self,
                 config: Aimv2VisualTokenizerConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 **kwargs):
        super().__init__()
        self.config = config
        self.backbone = AIMv2Model(
            config=config.backbone_config,  # noqa
            quant_config=quant_config,
            prefix=f"{prefix}.visual_tokenizer")
        # reserved tokens for IMAGE_INDICATORS
        head_dim = config.vocab_size - len(IMAGE_INDICATOR_IDS)
        self.head = torch.nn.Sequential(
            ReplicatedLinear(
                config.backbone_config.hidden_size * config.hidden_stride *
                config.hidden_stride,
                head_dim,
                bias=False,
            ), torch.nn.LayerNorm(head_dim))

    @property
    def dtype(self):
        return self.backbone.dtype

    @property
    def device(self):
        return self.backbone.device

    def tokenize(self, logits):
        if self.config.tokenize_function == 'softmax':
            tokens = softmax(logits, dim=-1)
        elif self.config.tokenize_function == 'gumbel_argmax':
            tokens = gumbel_softmax(logits, tau=self.config.tau, hard=True)
        elif self.config.tokenize_function == 'st_argmax':
            tokens = st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                'Invalid `max_type`, expected softmax or gumbel_argmax '
                f'or st_argmax, but got {self.config.tokenize_function}')
        return tokens

    def encode(self, pixel_values):
        features = self.backbone(pixel_values)
        if self.config.drop_cls_token:
            features = features[:, 1:, :]

        # merge number of `hidden_stride * hidden_stride` hidden states together
        # to reduce token sequence length
        # e.g., for hidden_stride=2, this leads to a token length reduction:
        # 1024 -> 256 for aimv2
        if self.config.hidden_stride > 1:
            # this `d` maybe different from the above `d``
            n, L, d = features.shape
            sqrt_l = int(L**0.5)
            assert sqrt_l**2 == L, (
                "The token sequence length should be a perfect square.")
            features = features.reshape(n, sqrt_l, sqrt_l, d)
            pl = (self.config.hidden_stride -
                  (sqrt_l %
                   self.config.hidden_stride)) % self.config.hidden_stride
            features = pad(features, (0, 0, 0, pl, 0, pl), "constant", 0)
            sqrt_l += pl
            features = features.reshape(n, sqrt_l // self.config.hidden_stride,
                                        self.config.hidden_stride,
                                        sqrt_l // self.config.hidden_stride,
                                        self.config.hidden_stride, d)
            # [n, sqrt_l/hs, sqrt_l/hs, hs, hs, d]
            features = features.permute(0, 1, 3, 2, 4, 5)
            # [n, sqrt_l/hs, sqrt_l/hs, hs*hs*d]
            features = features.flatten(3)
            # [n, sqrt_l/hs*sqrt_l/hs, hs*hs*d]
            features = features.reshape(
                n, -1,
                self.config.hidden_stride * self.config.hidden_stride * d)

        return features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[BatchSize, ImageShape] -> [BatchSize, Token, VocabSize]"""
        features = self.encode(pixel_values)
        logits, _ = self.head[0](
            features)  # we spllit the sequncial here for not throwing an error
        logits = self.head[1](logits)
        tokens = self.tokenize(logits)
        # tokens' shape is [BatchSize, #Token, VocabSize-5], so padding with
        # [BatchSize, #Token, 5], after which, tokens' shape should become
        # [BatchSize, #Token, VocabSize]
        batch_size, token_len, _ = tokens.shape
        padding_tensor = torch.zeros(size=(batch_size, token_len,
                                           len(IMAGE_INDICATOR_IDS)),
                                     dtype=tokens.dtype,
                                     device=tokens.device,
                                     layout=tokens.layout,
                                     requires_grad=False)
        tokens = torch.cat((tokens, padding_tensor), dim=2)
        return tokens


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

    @property
    def dtype(self):
        return self.trunk.blocks[0].attn.qkv.weight.dtype

    @property
    def device(self):
        return self.trunk.blocks[0].attn.qkv.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = self.preprocessor(pixel_values)
        x = self.trunk(x, mask)

        return x
