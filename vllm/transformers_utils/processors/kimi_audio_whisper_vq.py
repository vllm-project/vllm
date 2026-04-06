# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal WhisperVQ encoder used by Kimi-Audio speech tokenization.

This module intentionally vendors only the encoder-side inference path needed
to extract discrete speech tokens from `THUDM/glm-4-voice-tokenizer`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import WhisperConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoderLayer,
    WhisperPreTrainedModel,
)


@dataclass
class QuantizedBaseModelOutput(BaseModelOutput):
    quantized_token_ids: Optional[torch.LongTensor] = None


class WhisperVQConfig(WhisperConfig):
    def __init__(
        self,
        pooling_kernel_size=None,
        pooling_type="max",
        pooling_position=0,
        quantize_vocab_size=None,
        quantize_position=16,
        quantize_commit_coefficient=0.25,
        quantize_loss_scale=1.0,
        quantize_ema_decay=None,
        quantize_restart_interval=None,
        quantize_encoder_only=False,
        quantize_causal_encoder=False,
        quantize_causal_block_size=None,
        skip_language_detection=False,
        encoder_causal_attention=False,
        encoder_causal_convolution=False,
        **kwargs,
    ):
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_type = pooling_type
        self.pooling_position = pooling_position
        self.quantize_vocab_size = quantize_vocab_size
        self.quantize_position = quantize_position
        self.quantize_commit_coefficient = quantize_commit_coefficient
        self.quantize_loss_scale = quantize_loss_scale
        self.quantize_ema_decay = quantize_ema_decay
        self.quantize_restart_interval = quantize_restart_interval
        self.quantize_encoder_only = quantize_encoder_only
        self.quantize_causal_encoder = quantize_causal_encoder
        self.quantize_causal_block_size = quantize_causal_block_size
        self.skip_language_detection = skip_language_detection
        self.encoder_causal_attention = encoder_causal_attention
        self.encoder_causal_convolution = encoder_causal_convolution
        super().__init__(**kwargs)


def vector_quantize(
    inputs: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding_size = codebook.size(1)
    flattened = inputs.reshape(-1, embedding_size)
    codebook_sqr = torch.sum(codebook**2, dim=1)
    inputs_sqr = torch.sum(flattened**2, dim=1, keepdim=True)
    distances = torch.addmm(
        codebook_sqr + inputs_sqr,
        flattened,
        codebook.t(),
        alpha=-2.0,
        beta=1.0,
    )
    indices = torch.argmin(distances, dim=1)
    codes = torch.index_select(codebook, dim=0, index=indices).view_as(inputs)
    return codes, indices


def sinusoids(
    length: int,
    channels: int,
    max_timescale: float = 10000,
) -> torch.Tensor:
    if channels % 2 != 0:
        raise ValueError(
            "Number of channels must be divisible by 2 for sinusoidal "
            f"positional embeddings, got {channels}."
        )
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2)
    )
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(inp, (self.left_padding, 0)))


class WhisperVQEncoder(WhisperPreTrainedModel):
    config_class = WhisperVQConfig

    def __init__(self, config: WhisperVQConfig):
        if not getattr(config, "_attn_implementation", None):
            config._attn_implementation = (
                "sdpa"
                if hasattr(torch.nn.functional, "scaled_dot_product_attention")
                else "eager"
            )
        super().__init__(config)
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        conv_class = CausalConv1d if config.encoder_causal_convolution else nn.Conv1d
        self.conv1 = conv_class(
            self.num_mel_bins,
            embed_dim,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = conv_class(
            embed_dim,
            embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        num_layers = (
            config.quantize_position
            if config.quantize_encoder_only
            else config.encoder_layers
        )
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(num_layers)]
        )
        self.layer_norm = (
            None
            if config.quantize_encoder_only
            else nn.LayerNorm(config.d_model)
        )

        self.gradient_checkpointing = False
        self.pooling_layer: nn.Module | None = None
        self.codebook: nn.Embedding | None = None
        self.embed_positions2: nn.Embedding | None = None

        self._init_pooling_layer()
        self._init_quantize_layer()
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, WhisperVQEncoder):
            with torch.no_grad():
                module.embed_positions.weight.copy_(
                    sinusoids(*module.embed_positions.weight.shape)
                )
                if module.embed_positions2 is not None:
                    max_positions = module.embed_positions2.weight.shape[0]
                    module.embed_positions2.weight.copy_(
                        module.embed_positions.weight[:max_positions]
                    )

    def _init_pooling_layer(self) -> None:
        if self.config.pooling_kernel_size is None:
            return
        if self.config.pooling_type == "max":
            self.pooling_layer = nn.MaxPool1d(
                kernel_size=self.config.pooling_kernel_size
            )
        elif self.config.pooling_type == "avg":
            self.pooling_layer = nn.AvgPool1d(
                kernel_size=self.config.pooling_kernel_size
            )
        else:
            raise NotImplementedError(
                f"Pooling type {self.config.pooling_type} not implemented"
            )

    def _init_quantize_layer(self) -> None:
        if self.config.quantize_vocab_size is None:
            return
        if self.config.pooling_position is not None:
            assert self.config.quantize_position >= self.config.pooling_position
        self.codebook = nn.Embedding(
            self.config.quantize_vocab_size,
            self.config.d_model,
        )
        max_source_positions = self.max_source_positions
        if self.config.pooling_kernel_size is not None:
            max_source_positions = math.ceil(
                max_source_positions / self.config.pooling_kernel_size
            )
        self.embed_positions2 = nn.Embedding(
            max_source_positions,
            self.config.d_model,
        )
        if self.config.quantize_ema_decay is not None:
            self.codebook.weight.requires_grad = False
            self.register_buffer(
                "ema_count",
                torch.ones(self.config.quantize_vocab_size, dtype=torch.float),
            )
            self.register_buffer(
                "ema_weight",
                self.codebook.weight.data.clone().float(),
            )

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def get_block_causal_attention_mask(
        self,
        attention_mask: torch.Tensor,
        *,
        block_size: int,
    ) -> torch.Tensor:
        dtype = self.conv1.weight.dtype
        _, seq_length = attention_mask.shape
        causal_mask = torch.tril(
            torch.ones(
                1,
                seq_length,
                seq_length,
                dtype=torch.bool,
                device=attention_mask.device,
            )
        )
        block_square_mask = []
        for start in range(0, seq_length, block_size):
            end = min(start + block_size, seq_length)
            length = end - start
            block_square_mask.append(causal_mask.new_ones((length, length)))
        block_square_mask = torch.block_diag(*block_square_mask)
        block_causal_mask = causal_mask | block_square_mask
        block_causal_mask = block_causal_mask & attention_mask[:, None, :]
        block_causal_mask = block_causal_mask.to(dtype=dtype)
        block_causal_mask = (1.0 - block_causal_mask) * torch.finfo(dtype).min
        return block_causal_mask.unsqueeze(1)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        quantized_token_ids: torch.LongTensor | None = None,
    ):
        batch_size, _, seq_length = input_features.shape
        stride = self.conv1.stride[0] * self.conv2.stride[0]
        seq_length = seq_length // stride

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, input_features.shape[-1]),
                dtype=torch.long,
                device=input_features.device,
            )
        attention_mask = attention_mask[:, ::stride]

        if self.config.quantize_causal_block_size is not None:
            extended_attention_mask = self.get_block_causal_attention_mask(
                attention_mask,
                block_size=self.config.quantize_causal_block_size,
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask,
                (batch_size, seq_length),
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        hidden_states = inputs_embeds.permute(0, 2, 1)
        hidden_states = hidden_states + self.embed_positions.weight[:seq_length]
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers), (
                "The head_mask should be specified for "
                f"{len(self.layers)} layers, but got {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            to_drop = False
            if self.training:
                if torch.rand([]) < self.layerdrop:
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask=(
                        head_mask[idx] if head_mask is not None else None
                    ),
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            if (
                idx + 1 == self.config.pooling_position
                and self.config.pooling_kernel_size is not None
            ):
                hidden_states = hidden_states.permute(0, 2, 1)
                if (
                    hidden_states.shape[-1] % self.config.pooling_kernel_size
                    != 0
                ):
                    hidden_states = F.pad(
                        hidden_states,
                        (
                            0,
                            self.config.pooling_kernel_size
                            - hidden_states.shape[-1]
                            % self.config.pooling_kernel_size,
                        ),
                    )
                assert self.pooling_layer is not None
                hidden_states = self.pooling_layer(hidden_states).permute(0, 2, 1)
                attention_mask = attention_mask[:, ::self.config.pooling_kernel_size]
                if self.config.quantize_causal_block_size is not None:
                    extended_attention_mask = self.get_block_causal_attention_mask(
                        attention_mask,
                        block_size=(
                            self.config.quantize_causal_block_size
                            // self.config.pooling_kernel_size
                        ),
                    )
                else:
                    extended_attention_mask = self.get_extended_attention_mask(
                        attention_mask,
                        (
                            batch_size,
                            seq_length // self.config.pooling_kernel_size,
                        ),
                    )

            if (
                idx + 1 == self.config.quantize_position
                and self.config.quantize_vocab_size is not None
            ):
                assert self.codebook is not None
                assert self.embed_positions2 is not None
                if quantized_token_ids is not None:
                    hidden_states = self.codebook(quantized_token_ids)
                else:
                    hidden_quantized, flat_indices = vector_quantize(
                        hidden_states,
                        self.codebook.weight,
                    )
                    quantized_token_ids = flat_indices.reshape(
                        batch_size,
                        hidden_quantized.shape[1],
                    )
                    hidden_states = hidden_quantized
                hidden_states = hidden_states + self.embed_positions2.weight[
                    : hidden_states.shape[1]
                ]

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in (
                    hidden_states,
                    encoder_states,
                    all_attentions,
                    quantized_token_ids,
                )
                if value is not None
            )

        return QuantizedBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            quantized_token_ids=quantized_token_ids,
        )
