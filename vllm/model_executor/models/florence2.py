# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, TypedDict

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BatchFeature, PretrainedConfig
from transformers.models.bart.modeling_bart import BartEncoder

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.nemotron_parse import (
    BartDecoderLayer,
    BartParallelLMHead,
    BartScaledWordEmbedding,
)
from vllm.model_executor.models.whisper import WhisperAttention
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.v1.attention.backend import AttentionType

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .utils import AutoWeightsLoader, _merge_multimodal_embeddings


class Florence2ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: (batch_size, num_channel, height, width)"""


class Florence2EncoderInputIds(TypedDict):
    type: Literal["input_ids"]
    data: torch.Tensor
    """Shape: (batch_size, seq_len)"""


# ViT implementation are all copied from
# https://huggingface.co/microsoft/Florence-2-base/blob/main/modeling_florence2.py
class LearnedAbsolutePositionEmbedding2D(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256, num_pos=50):
        super().__init__()
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(
            num_pos, embedding_dim - (embedding_dim // 2)
        )

    def forward(self, pixel_values):
        """
        pixel_values: (batch_size, height, width, num_channels)
        returns: (batch_size, height, width, embedding_dim * 2)
        """
        if len(pixel_values.shape) != 4:
            raise ValueError("pixel_values must be a 4D tensor")
        height, width = pixel_values.shape[1:3]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        # (height, width, embedding_dim * 2)
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).repeat(height, 1, 1),
                y_emb.unsqueeze(1).repeat(1, width, 1),
            ],
            dim=-1,
        )
        # (embedding_dim * 2, height, width)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        # (batch_size, embedding_dim * 2, height, width)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # (batch_size, height, width, embedding_dim * 2)
        pos = pos.permute(0, 2, 3, 1)
        return pos


class PositionalEmbeddingCosine1D(nn.Module):
    """
    This class implements a very simple positional encoding. It follows closely
    the encoder from the link below:
    https://pytorch.org/tutorials/beginner/translation_transformer.html
    Args:
        embed_dim: The dimension of the embeddings.
        dropout_prob: The dropout probability.
        max_seq_len: The maximum length to precompute the positional encodings.
    """

    def __init__(self, embed_dim: int = 512, max_seq_len: int = 1024) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        # Generate the sinusoidal arrays.
        factor = math.log(10000)
        denominator = torch.exp(
            -factor * torch.arange(0, self.embed_dim, 2) / self.embed_dim
        )
        # Matrix where rows correspond to a positional embedding as a function
        # of the position index (i.e., the row index).
        frequencies = (
            torch.arange(0, self.max_seq_len).reshape(self.max_seq_len, 1) * denominator
        )
        pos_idx_to_embed = torch.zeros((self.max_seq_len, self.embed_dim))
        # Populate uneven entries.
        pos_idx_to_embed[:, 0::2] = torch.sin(frequencies)
        pos_idx_to_embed[:, 1::2] = torch.cos(frequencies)
        # Save the positional embeddings in a constant buffer.
        # self.register_buffer("pos_idx_to_embed", pos_idx_to_embed)
        self.pos_idx_to_embed = nn.Parameter(pos_idx_to_embed, requires_grad=False)

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_embeds: The sequence embeddings in order. Allowed size:
                1. [T, D], where T is the length of the sequence, and D is the
                frame embedding dimension.
                2. [B, T, D], where B is the batch size and T and D are the
                same as above.
        Returns a tensor of with the same dimensions as the input: i.e.,
        [1, T, D] or [T, D].
        """
        shape_len = len(seq_embeds.shape)
        assert 2 <= shape_len <= 3
        len_seq = seq_embeds.size(-2)
        assert len_seq <= self.max_seq_len
        pos_embeds = self.pos_idx_to_embed[0 : seq_embeds.size(-2), :]
        # Adapt pre-computed positional embeddings to the input.
        if shape_len == 3:
            pos_embeds = pos_embeds.view((1, pos_embeds.size(0), pos_embeds.size(1)))
        return pos_embeds


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            inputs = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
        return inputs


class PreNorm(nn.Module):
    def __init__(self, norm, fn):
        super().__init__()
        self.norm = norm
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        shortcut = x
        if self.norm is not None:
            x, size = self.fn(self.norm(x), *args, **kwargs)
        else:
            x, size = self.fn(x, *args, **kwargs)

        x = shortcut + x

        return x, size


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(in_features, hidden_features)),
                    ("act", act_layer()),
                    ("fc2", nn.Linear(hidden_features, out_features)),
                ]
            )
        )

    def forward(self, x, size):
        return self.net(x), size


class DepthWiseConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_size,
        padding,
        stride,
        bias=True,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim_in,
            stride=stride,
            bias=bias,
        )

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = self.dw(x.transpose(1, 2).view(B, C, H, W))
        size = (x.size(-2), x.size(-1))
        x = x.flatten(2).transpose(1, 2)
        return x, size


class ConvEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        norm_layer=None,
        pre_norm=True,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )

        dim_norm = in_chans if pre_norm else embed_dim
        self.norm = norm_layer(dim_norm) if norm_layer else None

        self.pre_norm = pre_norm

    def forward(self, x, size):
        H, W = size
        if len(x.size()) == 3:
            if self.norm and self.pre_norm:
                x = self.norm(x)
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        x = self.proj(x)

        _, _, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm and not self.pre_norm:
            x = self.norm(x)

        return x, (H, W)


class ChannelAttention(nn.Module):
    def __init__(self, dim, groups=8, qkv_bias=True):
        super().__init__()

        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.groups, C // self.groups)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (float(N) ** -0.5)
        attention = q.transpose(-1, -2) @ k
        attention = attention.softmax(dim=-1)
        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, size


class ChannelBlock(nn.Module):
    def __init__(
        self,
        dim,
        groups,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        conv_at_attn=True,
        conv_at_ffn=True,
    ):
        super().__init__()

        self.conv1 = (
            PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        )
        self.channel_attn = PreNorm(
            norm_layer(dim),
            ChannelAttention(dim, groups=groups, qkv_bias=qkv_bias),
        )
        self.conv2 = (
            PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        )
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
            ),
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)

        return x, size


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, batch_size: int, window_size: int, H: int, W: int):
    B = batch_size

    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # attn_windows = self.attn(x_windows)

        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        # merge windows
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, B, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        return x, size


class SpatialBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        conv_at_attn=True,
        conv_at_ffn=True,
    ):
        super().__init__()

        self.conv1 = (
            PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        )
        self.window_attn = PreNorm(
            norm_layer(dim),
            WindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias),
        )
        self.conv2 = (
            PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        )
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
            ),
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class DaViT(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(1, 1, 3, 1),
        patch_size=(7, 2, 2, 2),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 0, 0, 0),
        patch_prenorm=(False, False, False, False),
        embed_dims=(64, 128, 192, 256),
        num_heads=(3, 6, 12, 24),
        num_groups=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
        conv_at_attn=True,
        conv_at_ffn=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_stages = len(self.embed_dims)
        self.enable_checkpoint = enable_checkpoint
        assert self.num_stages == len(self.num_heads) == len(self.num_groups)

        num_stages = len(embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)]

        depth_offset = 0
        convs = []
        blocks = []
        for i in range(num_stages):
            conv_embed = ConvEmbed(
                patch_size=patch_size[i],
                stride=patch_stride[i],
                padding=patch_padding[i],
                in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                embed_dim=self.embed_dims[i],
                norm_layer=norm_layer,
                pre_norm=patch_prenorm[i],
            )
            convs.append(conv_embed)

            block = MySequential(
                *[
                    MySequential(
                        OrderedDict(
                            [
                                (
                                    "spatial_block",
                                    SpatialBlock(
                                        embed_dims[i],
                                        num_heads[i],
                                        window_size,
                                        drop_path_rate=dpr[depth_offset + j * 2],
                                        qkv_bias=qkv_bias,
                                        mlp_ratio=mlp_ratio,
                                        conv_at_attn=conv_at_attn,
                                        conv_at_ffn=conv_at_ffn,
                                    ),
                                ),
                                (
                                    "channel_block",
                                    ChannelBlock(
                                        embed_dims[i],
                                        num_groups[i],
                                        drop_path_rate=dpr[depth_offset + j * 2 + 1],
                                        qkv_bias=qkv_bias,
                                        mlp_ratio=mlp_ratio,
                                        conv_at_attn=conv_at_attn,
                                        conv_at_ffn=conv_at_ffn,
                                    ),
                                ),
                            ]
                        )
                    )
                    for j in range(depths[i])
                ]
            )
            blocks.append(block)
            depth_offset += depths[i] * 2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    @property
    def dim_out(self):
        return self.embed_dims[-1]

    def forward_features_unpool(self, x):
        """
        forward until avg pooling
        Args:
            x (_type_): input image tensor
        """
        input_size = (x.size(2), x.size(3))
        for conv, block in zip(self.convs, self.blocks):
            x, input_size = conv(x, input_size)
            x, input_size = block(x, input_size)
        return x

    def forward_features(self, x):
        x = self.forward_features_unpool(x)

        # (batch_size, num_tokens, token_dim)
        x = self.avgpool(x.transpose(1, 2))
        # (batch_size, 1, num_tokens)
        x = torch.flatten(x, 1)
        x = self.norms(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    @classmethod
    def from_config(cls, config):
        return cls(
            depths=config.depths,
            embed_dims=config.dim_embed,
            num_heads=config.num_heads,
            num_groups=config.num_groups,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            patch_padding=config.patch_padding,
            patch_prenorm=config.patch_prenorm,
            drop_path_rate=config.drop_path_rate,
            window_size=config.window_size,
        )


# Language backbone and processor implementation
class Florence2TextEncoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            attn_type=AttentionType.ENCODER,
            prefix="self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = get_act_fn(config.activation_function)
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.encoder_ffn_dim,
            bias=True,
            prefix="fc1",
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            self.embed_dim,
            bias=True,
            prefix="fc2",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class Florence2TextEncoder(nn.Module):
    def __init__(self, config: PretrainedConfig, embed_tokens: nn.Module):
        super().__init__()
        self.embed_tokens = embed_tokens
        max_positions = config.max_position_embeddings + 2
        self.embed_positions = nn.Embedding(max_positions, config.d_model)
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layers = nn.ModuleList(
            [
                Florence2TextEncoderLayer(config, prefix=f"layers.{i}")
                for i in range(config.encoder_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]
        positions = (
            torch.arange(seq_len, device=inputs_embeds.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        hidden_states = inputs_embeds + self.embed_positions(positions + 2)
        hidden_states = self.layernorm_embedding(hidden_states)
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class Florence2BartDecoderNoPos(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: VllmConfig | None = None,
        quant_config: VllmConfig | None = None,
        embed_tokens: nn.Embedding | None = None,
        prefix: str = "",
    ):
        super().__init__()
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, config.d_model, embed_scale=embed_scale
        )
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.layers = nn.ModuleList(
            [
                BartDecoderLayer(
                    config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.decoder_layers)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(
        self,
        decoder_input_ids: torch.Tensor | None,
        *,
        encoder_hidden_states: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(decoder_input_ids)

        hidden_states = self.layernorm_embedding(inputs_embeds)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                decoder_hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        return hidden_states


class Florence2LanguageModel(nn.Module):
    """Lightweight Florence2 decoder wrapper used by vLLM runtime.

    Florence2 is encoder-decoder, but in vLLM v1 the visual branch is executed
    through multimodal embedding paths and the decoder consumes encoder outputs
    as cross-attention memory.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.shared = BartScaledWordEmbedding(self.vocab_size, config.d_model)
        self.encoder = BartEncoder(config, embed_tokens=self.shared)
        self.decoder_embed_positions = nn.Embedding(
            config.max_position_embeddings + 2,
            config.d_model,
        )
        self.decoder = Florence2BartDecoderNoPos(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            embed_tokens=self.shared,
            prefix=f"{prefix}.decoder",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids
                Indices of *decoder* input sequence tokens in the vocabulary.
                Padding will be ignored by default should you
                provide it.
            positions
                Positions of *decoder* input sequence tokens.
            encoder_input_ids
                Indices of *encoder* input sequence tokens in the vocabulary.
            encoder_positions:
                Positions of *encoder* input sequence tokens.
        Returns:
            Model output torch.Tensor
        """

        if inputs_embeds is not None:
            encoder_inputs_embeds = inputs_embeds
            squeezed = False
            if encoder_inputs_embeds.ndim == 2:
                encoder_inputs_embeds = encoder_inputs_embeds.unsqueeze(0)
                squeezed = True
            encoder_hidden_states = self.encoder(
                input_ids=None,
                attention_mask=None,
                inputs_embeds=encoder_inputs_embeds,
                return_dict=True,
            ).last_hidden_state
            if squeezed:
                encoder_hidden_states = encoder_hidden_states.squeeze(0)
        else:
            encoder_hidden_states = encoder_outputs
        decoder_inputs_embeds = None
        if input_ids is not None:
            decoder_inputs_embeds = self.decoder.embed_tokens(input_ids)
            pos_ids = positions.clamp(
                min=0,
                max=self.decoder_embed_positions.num_embeddings - 3,
            )
            decoder_inputs_embeds = (
                decoder_inputs_embeds + self.decoder_embed_positions(pos_ids + 2)
            )
        decoder_outputs = self.decoder(
            decoder_input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            inputs_embeds=decoder_inputs_embeds,
        )

        return decoder_outputs


class Florence2LanguageForConditionalGeneration(nn.Module):
    """Text generation head for Florence2 decoder.

    This class keeps load-time compatibility with HF Florence2 checkpoints by
    remapping attention projection names to vLLM fused projection parameters.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.config = config
        self.model = Florence2LanguageModel(
            vllm_config=vllm_config, prefix=f"{prefix}.model"
        )
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.vocab_size = config.vocab_size
        self.lm_head = BartParallelLMHead(
            self.vocab_size, config.d_model, embed_scale=embed_scale
        )

        self.logits_processor = LogitsProcessor(self.vocab_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids
                torch.Tensor of *decoder* input token ids.
            positions
                torch.Tensor of *decoder* position indices.
            encoder_input_ids
                torch.Tensor of *encoder* input token ids.
            encoder_positions
                torch.Tensor of *encoder* position indices
        Returns:
            Output torch.Tensor
        """

        return self.model(
            input_ids,
            positions,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.encoder.embed_tokens(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load decoder weights with HF->vLLM projection remapping.

        Remaps:
        - self_attn.{q,k,v}_proj -> self_attn.qkv_proj
        - encoder_attn.{k,v}_proj -> encoder_attn.kv_proj

        Decoder final layer norm is replaced with Identity in this port.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        loaded_shared_weight = False
        for name, loaded_weight in weights:
            if "final_logits_bias" in name:
                continue

            if self.config.tie_word_embeddings and "embed_tokens" in name:
                continue

            target_name = name
            shard_id = None

            if ".decoder.layers." in name and ".self_attn.q_proj" in name:
                target_name = name.replace(".self_attn.q_proj", ".self_attn.qkv_proj")
                shard_id = "q"
            elif ".decoder.layers." in name and ".self_attn.k_proj" in name:
                target_name = name.replace(".self_attn.k_proj", ".self_attn.qkv_proj")
                shard_id = "k"
            elif ".decoder.layers." in name and ".self_attn.v_proj" in name:
                target_name = name.replace(".self_attn.v_proj", ".self_attn.qkv_proj")
                shard_id = "v"
            elif ".encoder_attn.k_proj" in name:
                target_name = name.replace(
                    ".encoder_attn.k_proj", ".encoder_attn.kv_proj"
                )
                shard_id = "k"
            elif ".decoder.embed_positions.weight" in name:
                target_name = name.replace(
                    ".decoder.embed_positions.weight",
                    ".decoder_embed_positions.weight",
                )
            elif ".encoder_attn.v_proj" in name:
                target_name = name.replace(
                    ".encoder_attn.v_proj", ".encoder_attn.kv_proj"
                )
                shard_id = "v"
            if target_name not in params_dict:
                continue

            param = params_dict[target_name]
            if shard_id is None:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

            loaded_params.add(target_name)
            if target_name == "model.shared.weight":
                loaded_shared_weight = True

        if self.config.tie_word_embeddings and loaded_shared_weight:
            shared_name = "model.shared.weight"
            lm_head_name = "lm_head.weight"
            if shared_name in params_dict and lm_head_name in params_dict:
                shared_weight = params_dict[shared_name]
                lm_head_weight = params_dict[lm_head_name]
                shared_rows = shared_weight.shape[0]
                lm_head_weight.data[:shared_rows, :].copy_(shared_weight.data)
                if lm_head_weight.shape[0] > shared_rows:
                    lm_head_weight.data[shared_rows:, :].zero_()
                loaded_params.add(lm_head_name)

        return loaded_params


class Florence2ProcessingInfo(BaseProcessingInfo):
    """Processor metadata used by multimodal preprocessing.

    Florence2 v1 currently targets single-image inputs and derives the image
    token budget from HF image processor config.
    """

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self):
        return self.ctx.get_hf_processor()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(self) -> int:
        processor_config = self.ctx.get_hf_image_processor_config()
        return processor_config["image_seq_length"]


class Florence2DummyInputsBuilder(BaseDummyInputsBuilder[Florence2ProcessingInfo]):
    """Builds worst-case dummy multimodal inputs for profile run."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
        mm_processor_kwargs: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width = target_height = self.info.get_hf_config().projection_dim

        return {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            )
        }


class Florence2MultiModalProcessor(EncDecMultiModalProcessor[Florence2ProcessingInfo]):
    """Florence2 encoder-decoder multimodal processor for v1.

    Key behavior:
    - encoder prompt carries task token prompt text (e.g. <CAPTION>)
    - decoder prompt starts from decoder start token (eos_token_id in Florence2)
    - image placeholders are inserted as pad-token spans on encoder side
    """

    def _resolve_task_token_case(
        self,
        task_token: str,
        has_suffix_input: bool,
    ) -> str:
        hf_processor = self.info.get_hf_processor()

        prompts_without_input = getattr(hf_processor, "task_prompts_without_inputs", {})
        prompts_with_input = getattr(hf_processor, "task_prompts_with_input", {})

        tokenizer = getattr(hf_processor, "tokenizer", None)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)

        def _tokenizer_recognizes(token: str) -> bool:
            if tokenizer is None:
                return False
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None:
                return False
            return unk_token_id is None or token_id != unk_token_id

        candidates = [task_token, task_token.upper(), task_token.lower()]
        deduped_candidates = list(dict.fromkeys(candidates))

        if has_suffix_input:
            for candidate in deduped_candidates:
                if candidate in prompts_with_input:
                    return candidate
        else:
            for candidate in deduped_candidates:
                if candidate in prompts_without_input:
                    return candidate

        for candidate in deduped_candidates:
            if (
                candidate in prompts_with_input
                or candidate in prompts_without_input
                or _tokenizer_recognizes(candidate)
            ):
                return candidate

        return task_token

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        if isinstance(prompt, list):
            tokenizer = self.info.get_hf_processor().tokenizer
            prompt = tokenizer.decode(prompt)

        if isinstance(prompt, str):
            prompt = prompt.strip()
            match = re.match(r"^(<[^<>]+>)(.*)$", prompt)
            if match is not None:
                task_token, suffix = match.groups()
                resolved_task_token = self._resolve_task_token_case(
                    task_token,
                    has_suffix_input=bool(suffix.strip()),
                )
                prompt = resolved_task_token + suffix
        return prompt

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [self.info.get_hf_config().eos_token_id]

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        hf_processor = self.info.get_hf_processor()
        tokenizer = hf_processor.tokenizer
        prompt_text = tokenizer.decode(prompt_tokens)
        prompt_text = hf_processor._construct_prompts([prompt_text])[0]
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        return prompt_tokens

    def _call_hf_processor(
        self,
        prompt: str | list[int],
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_processor = self.info.get_hf_processor()
        tokenizer = hf_processor.tokenizer

        if isinstance(prompt, list):
            prompt = tokenizer.decode(prompt)

        if mm_data:
            processed_outputs = super()._call_hf_processor(
                prompt, mm_data, mm_kwargs, tok_kwargs
            )
        else:
            prompt = hf_processor._construct_prompts([prompt])[0]
            processed_outputs = tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt"
            )
        processed_outputs["encoder_input_ids"] = processed_outputs["input_ids"].clone()
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_images = int(hf_inputs["pixel_values"].shape[0])
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            encoder_input_ids=MultiModalFieldConfig.shared(
                "image", batch_size=num_images
            ),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        pad_token_id = hf_config.pad_token_id
        num_image_tokens = self.info.get_num_image_tokens()
        image_tokens = [pad_token_id] * num_image_tokens
        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.start(),
                insertion=image_tokens,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Florence2MultiModalProcessor,
    info=Florence2ProcessingInfo,
    dummy_inputs=Florence2DummyInputsBuilder,
)
class Florence2ForConditionalGeneration(nn.Module, SupportsMultiModal):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        processor_config = vllm_config.model_config.hf_image_processor_config

        self.config = config
        self.vision_config = config.vision_config
        self.processor_config = processor_config
        assert config.vision_config.model_type == "davit", (
            "only DaViT is supported for now"
        )
        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = DaViT.from_config(config=config.vision_config)
        self._build_image_projection_layers(config)
        with self._mark_language_model(vllm_config):
            self.language_model = Florence2LanguageForConditionalGeneration(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=f"{prefix}.language_model",
            )
        self.pad_token_id = config.pad_token_id

    def _build_image_projection_layers(self, config: PretrainedConfig):
        """Initialize visual projection and positional/temporal embeddings."""
        image_dim_out = config.vision_config.dim_embed[-1]
        dim_projection = config.vision_config.projection_dim
        self.image_projection = nn.Parameter(torch.empty(image_dim_out, dim_projection))
        self.image_proj_norm = nn.LayerNorm(dim_projection)
        image_pos_embed_config = config.vision_config.image_pos_embed
        if image_pos_embed_config["type"] == "learned_abs_2d":
            self.image_pos_embed = LearnedAbsolutePositionEmbedding2D(
                embedding_dim=image_dim_out,
                num_pos=image_pos_embed_config["max_pos_embeddings"],
            )
        else:
            raise NotImplementedError(
                "Florence2 only supports learned_abs_2d as image position embedding."
            )

        self.image_feature_source = config.vision_config.image_feature_source

        # Temporal embedding is applied over frame axis (T). Florence2 image
        # mode uses T=1, but we preserve the original formulation.
        visual_temporal_embedding_config = self.vision_config.visual_temporal_embedding
        if visual_temporal_embedding_config["type"] == "COSINE":
            self.visual_temporal_embed = PositionalEmbeddingCosine1D(
                embed_dim=image_dim_out,
                max_seq_len=visual_temporal_embedding_config["max_temporal_embeddings"],
            )
        else:
            raise NotImplementedError(
                "Florence2 only supports COSINE as temporal embedding."
            )

    def _validate_pixel_values(
        self, data: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor | list[torch.Tensor]:
        """Validate pixel tensor shapes against HF processor image size.

        Accepts either a single CHW tensor, a batched BCHW tensor, or a list
        of CHW tensors. Returns normalized tensor/list format for downstream
        image encoding.
        """

        size = self.processor_config["size"]
        h, w = size["height"], size["width"]
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                raise ValueError(
                    "The expected shape of pixel values per batch "
                    f"is {expected_dims}. You supplied {tuple(d.shape)}."
                )

        if isinstance(data, torch.Tensor):
            if data.ndim == 3:
                _validate_shape(data)
                return data.unsqueeze(0)

            if data.ndim == 4:
                for d in data:
                    _validate_shape(d)
                return data

            raise ValueError(
                "The expected tensor rank for pixel values is 3 or 4. "
                f"You supplied rank {data.ndim}."
            )

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(self, **kwargs: object):
        """Parse image inputs from runtime kwargs and normalize to BCHW.

        v1 profile and serving paths may provide image tensors with different
        ranks (3/4/5) or nested list structures. This method canonicalizes them
        before validation.
        """
        pixel_values: (
            list[list[torch.Tensor]] | list[torch.Tensor] | torch.Tensor | None
        ) = kwargs.pop("pixel_values", None)
        encoder_input_ids: torch.Tensor | None = kwargs.pop("encoder_input_ids", None)
        image_embeds: (
            list[list[torch.Tensor]] | list[torch.Tensor] | torch.Tensor | None
        ) = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None and image_embeds is not None:
            raise ValueError("Both pixel values and image embeds are provided.")

        if pixel_values is not None:
            if isinstance(pixel_values, torch.Tensor):
                if pixel_values.ndim == 5:
                    flattened_pixel_values = pixel_values.flatten(0, 1)
                elif pixel_values.ndim in (3, 4):
                    flattened_pixel_values = pixel_values
                else:
                    raise ValueError(
                        "Unsupported pixel_values ndim. "
                        f"Expected 3/4/5, got {pixel_values.ndim}."
                    )
            else:
                flattened_list: list[torch.Tensor] = []
                for batch_items in pixel_values:
                    if isinstance(batch_items, list):
                        flattened_list.extend(batch_items)
                    else:
                        flattened_list.append(batch_items)
                flattened_pixel_values = torch.stack(flattened_list, dim=0)

            return (
                Florence2ImagePixelInputs(
                    type="pixel_values",
                    data=self._validate_pixel_values(flattened_pixel_values),
                ),
                (
                    None
                    if encoder_input_ids is None
                    else Florence2EncoderInputIds(
                        type="input_ids",
                        data=encoder_input_ids,
                    )
                ),
            )

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images with DaViT and project to decoder hidden size."""
        dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(dtype)

        batch_size, T = pixel_values.size(0), 1
        x = self.vision_tower.forward_features_unpool(pixel_values)
        if self.image_pos_embed is not None:
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h, w = int(num_tokens**0.5), int(num_tokens**0.5)
            assert h * w == num_tokens, "only support square feature maps for now"
            x = x.view(batch_size * T, h, w, x.shape[-1])
            pos_embed = self.image_pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, T * h * w, x.shape[-1])

        if self.visual_temporal_embed is not None:
            visual_temporal_embed = self.visual_temporal_embed(
                x.view(batch_size, T, -1, x.shape[-1])[:, :, 0]
            )
            x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(
                1, T, 1, x.shape[-1]
            )

        x_feat_dict = {}

        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
        x_feat_dict["spatial_avg_pool"] = spatial_avg_pool_x

        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
        x_feat_dict["temporal_avg_pool"] = temporal_avg_pool_x

        x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
        x_feat_dict["last_frame"] = x

        new_x = []
        for _image_feature_source in self.image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError(
                    "invalid image feature source: {}".format(_image_feature_source)
                )
            new_x.append(x_feat_dict[_image_feature_source])

        x = torch.cat(new_x, dim=1)

        x = x @ self.image_projection
        x = self.image_proj_norm(x)

        return x

    def _process_image_input(
        self, image_input: Florence2ImagePixelInputs
    ) -> torch.Tensor:
        assert image_input["type"] == "pixel_values"
        pixel_values = image_input["data"]
        return self._encode_image(pixel_values)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        """Return encoder-side multimodal embeddings for image inputs."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        parsed_image_input, encoder_input_ids = image_input
        vision_embeddings = self._process_image_input(parsed_image_input)

        if encoder_input_ids is None:
            return vision_embeddings

        encoder_ids = encoder_input_ids["data"]
        if encoder_ids.ndim == 1:
            encoder_ids = encoder_ids.unsqueeze(0)

        num_images = vision_embeddings.shape[0]
        if encoder_ids.shape[0] == 1 and num_images > 1:
            encoder_ids = encoder_ids.expand(num_images, -1).contiguous()

        encoder_token_embeddings = self.language_model.get_input_embeddings(encoder_ids)
        is_image_placeholder = encoder_ids.eq(self.pad_token_id)
        num_placeholder_tokens = int(is_image_placeholder.sum().item())
        num_vision_tokens = int(vision_embeddings.shape[0] * vision_embeddings.shape[1])
        if num_placeholder_tokens == num_vision_tokens and num_placeholder_tokens > 0:
            encoder_inputs_embeds = _merge_multimodal_embeddings(
                encoder_token_embeddings,
                vision_embeddings,
                is_image_placeholder,
            )
        else:
            encoder_inputs_embeds = torch.cat(
                [vision_embeddings, encoder_token_embeddings],
                dim=1,
            )

        encoder_hidden_states = self.language_model.model.encoder(
            input_ids=None,
            attention_mask=None,
            inputs_embeds=encoder_inputs_embeds,
            return_dict=True,
        ).last_hidden_state
        return encoder_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run decoder forward pass using provided encoder outputs.

        In v1, multimodal encoder outputs are produced by `embed_multimodal`
        and injected into this method through `encoder_outputs`.
        """
        if encoder_outputs is None:
            encoder_outputs = []
        inputs_embeds = torch.cat(encoder_outputs, dim=0) if encoder_outputs else None
        hidden_states = self.language_model(
            input_ids,
            positions,
            encoder_outputs=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
