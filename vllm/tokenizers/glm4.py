# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team.
# Copyright contributors to the vLLM project
# adapted from https://huggingface.co/cydxg/glm-4-voice-9b-int4/resolve/98d7318b5ef3103b7d11cc86ab80cbb8423ece10/speech_tokenizer/modeling_whisper.py
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
"""PyTorch Whisper model."""

import math
import os.path
import random
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from torch import nn
from transformers import WhisperFeatureExtractor
from transformers.activations import ACT2FN
from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from vllm.transformers_utils.configs import WhisperVQConfig

logger = logging.get_logger(__name__)


@dataclass
class QuantizedBaseModelOutput(BaseModelOutput):
    quantized_token_ids: torch.LongTensor | None = None


def vector_quantize(inputs, codebook):
    embedding_size = codebook.size(1)
    inputs_flatten = inputs.reshape(-1, embedding_size)
    codebook_sqr = torch.sum(codebook**2, dim=1)
    inputs_sqr = torch.sum(inputs_flatten**2, dim=1, keepdim=True)
    # Compute the distances to the codebook
    distances = torch.addmm(
        codebook_sqr + inputs_sqr,
        inputs_flatten,
        codebook.t(),
        alpha=-2.0,
        beta=1.0,
    )

    _, indices_flatten = torch.min(distances, dim=1)
    codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
    codes = codes_flatten.view_as(inputs)
    return codes, indices_flatten, distances


def mse_loss_with_mask(input, target, mask):
    loss = torch.nn.functional.mse_loss(input, target, reduction="none")
    loss = loss.mean(dim=-1)
    loss = loss * mask
    return loss.sum() / mask.sum()


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

    def forward(self, inp):
        x = torch.nn.functional.pad(
            inp.unsqueeze(2), (self.left_padding, 0, 0, 0)
        ).squeeze(2)

        return super().forward(x)


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """Returns sinusoids for positional embedding"""
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 \
            for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: int | None = None,
        config: WhisperVQConfig | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads \
                (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} \
                without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. "
                "Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # Copied from transformers.models.bart.modeling_bart.
    # BartAttention._shape with BART->whisper
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        past_key_value: EncoderDecoderCache | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used
        # as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self._shape(
            self.q_proj(hidden_states) * self.scaling, tgt_len, bsz
        )

        is_updated = False
        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently
                # re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = (
            key_value_states if key_value_states is not None else hidden_states
        )
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                # save all key/value_states to cache to be
                # re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size \
                    {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size \
                {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        # Use the `embed_dim` from the config (stored in the class)
        # rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class WhisperSdpaAttention(WhisperAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        past_key_value: EncoderDecoderCache | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions or layer_head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config.
            # _attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "WhisperModel is using WhisperSdpaAttention, "
                "but `torch.nn.functional.scaled_dot_product_attention` "
                "does not support `output_attentions=True` or `layer_head_mask`"
                "not None. Falling back to the manual attention"
                " implementation, but specifying the manual implementation "
                "will be required from Transformers version v5.0.0 onwards. "
                "This warning can be removed using the argument "
                '`attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

        # if key_value_states are provided this layer
        # is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)

        is_updated = False
        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently
                # re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = (
            key_value_states if key_value_states is not None else hidden_states
        )
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for
                # fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this
        # `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full
        # graph options. An inline conditional prevents dynamic shapes
        # from compiling. The tgt_len > 1 is necessary to match with
        # AttentionMaskConverter.to_causal_4d that does not create
        # a causal mask in case tgt_len == 1.
        is_causal = bool(self.is_causal and causal_mask is None and tgt_len > 1)

        # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2)
        # bugged when using non-contiguous inputs and a custom attn_mask,
        # but we are fine here as `_shape` do call `.contiguous()`.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class)
        # rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


WHISPER_ATTENTION_CLASSES = {
    "eager": WhisperAttention,
    # "flash_attention_2": WhisperFlashAttention2,
    "sdpa": WhisperSdpaAttention,
}


# Copied from transformers.models.mbart.modeling_mbart.
# MBartEncoderLayer with MBart->Whisper, MBART->WHISPER
class WhisperVQEncoderLayer(nn.Module):
    def __init__(self, config: WhisperVQConfig, is_causal=False):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            is_causal=is_causal,
        )
        self.is_causal = is_causal
        if self.is_causal:
            assert isinstance(self.self_attn, WhisperSdpaAttention), (
                "Causal attention is only supported for SDPA"
            )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer
            of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements
                are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention
            heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors
                of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask if not self.is_causal else None,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class WhisperPreTrainedModel(PreTrainedModel):
    config_class = WhisperVQConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoderLayer", "WhisperDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear | nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, WhisperVQEncoder):
            with torch.no_grad():
                embed_positions = module.embed_positions.weight
                embed_positions.copy_(sinusoids(*embed_positions.shape))

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths


class WhisperVQEncoder(WhisperPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers*
    self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperVQConfig):
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
        self.conv1 = conv_class(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = conv_class(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
        )

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        if config.quantize_encoder_only:
            self.layers = nn.ModuleList(
                [
                    WhisperVQEncoderLayer(
                        config,
                        is_causal=config.encoder_causal_attention
                        or config.quantize_causal_encoder,
                    )
                    for _ in range(config.quantize_position)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    WhisperVQEncoderLayer(
                        config,
                        is_causal=config.encoder_causal_attention
                        or (
                            config.quantize_causal_encoder
                            and layer_id < config.quantize_position
                        ),
                    )
                    for layer_id in range(config.encoder_layers)
                ]
            )
            self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Parameters related to pooling layer
        self.pooling_layer = None
        # Parameters related to quantization layer
        self.codebook = None
        self.embed_positions2 = None
        self.quantize_loss = None
        self.num_active_codes = None
        self.quantize_ema_count = 0
        # Save hiddens
        self.save_hidden_dir = None
        self.save_hidden_position = None
        # Initialize weights and apply final processing
        self.init_pooling_layer(config)
        self.init_quantize_layer(config)
        self.post_init()

    def init_pooling_layer(self, config: WhisperVQConfig):
        if config.pooling_kernel_size is not None:
            if config.pooling_type == "max":
                self.pooling_layer = nn.MaxPool1d(
                    kernel_size=config.pooling_kernel_size
                )
            elif config.pooling_type == "avg":
                self.pooling_layer = nn.AvgPool1d(
                    kernel_size=config.pooling_kernel_size
                )
            else:
                raise NotImplementedError(
                    f"Pooling type {config.pooling_type} not implemented"
                )

    def init_quantize_layer(self, config: WhisperVQConfig, quantize_load_codebook=None):
        if config.quantize_vocab_size is not None:
            if config.pooling_position is not None:
                assert config.quantize_position >= config.pooling_position
            self.codebook = nn.Embedding(
                config.quantize_vocab_size, self.config.d_model
            )
            if quantize_load_codebook is not None:
                init_codes = np.load(quantize_load_codebook)
                self.codebook.weight.data.copy_(torch.from_numpy(init_codes))
            max_source_positions = self.max_source_positions
            if config.pooling_kernel_size is not None:
                max_source_positions = math.ceil(
                    max_source_positions / self.config.pooling_kernel_size
                )
            self.embed_positions2 = nn.Embedding(
                max_source_positions, self.config.d_model
            )
            self.embed_positions2.weight.data.copy_(
                self.embed_positions.weight.data[:max_source_positions]
            )
            if config.quantize_ema_decay is not None:
                self.codebook.weight.requires_grad = False
                self.register_buffer(
                    "ema_count",
                    torch.ones(config.quantize_vocab_size, dtype=torch.float),
                )
                self.register_buffer(
                    "ema_weight", self.codebook.weight.data.clone().float()
                )

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def get_block_causal_attention_mask(self, attention_mask, block_size=50):
        dtype = self.dtype
        batch_size, seq_length = attention_mask.shape
        causal_mask = torch.torch.tril(
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
        block_causal_mask = block_causal_mask.unsqueeze(1)
        return block_causal_mask

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        quantized_token_ids=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size,
            feature_size, sequence_length)`):
                Float values of mel features extracted from the raw
                speech waveform. Raw speech waveform can be obtained by
                loading a `.flac` or `.wav` audio file into an array of
                type `List[float]` or a `numpy.ndarray`, *e.g.* via the
                soundfile library (`pip install soundfile`). To prepare the
                array into `input_features`, the [`AutoFeatureExtractor`]
                should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`.
                See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`,
                this argument is preserved for compatibility, but it is not
                used. By default the silence in the input log mel spectrogram
                are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers,
            encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules.
                Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all
                attention layers. See `attentions` under returned tensors
                for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
                See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead
                of a plain tuple.
        """

        # expected_seq_length = self.config.max_source_positions * self.conv1.
        # stride[0] * self.conv2.stride[0]
        # if input_features.shape[-1] != expected_seq_length:
        #     raise ValueError(
        #         f"Whisper expects the mel input features to be of
        # length {expected_seq_length}, but found {input_features.shape[-1]}.
        # Make sure to pad the input mel features to {expected_seq_length}."
        #     )

        batch_size, feature_size, seq_length = input_features.shape
        seq_length = seq_length // (self.conv1.stride[0] * self.conv2.stride[0])

        attention_mask = attention_mask[
            :, :: self.conv1.stride[0] * self.conv2.stride[0]
        ]
        if self.config.quantize_causal_block_size is not None:
            extended_attention_mask = self.get_block_causal_attention_mask(
                attention_mask,
                block_size=self.config.quantize_causal_block_size,
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, (batch_size, seq_length)
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
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos[:seq_length]
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        assert attention_mask.shape[-1] == hidden_states.shape[1]
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} \
            layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        extended_attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
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
                if hidden_states.shape[-1] % self.config.pooling_kernel_size != 0:
                    hidden_states = torch.nn.functional.pad(
                        hidden_states,
                        (
                            0,
                            self.config.pooling_kernel_size
                            - hidden_states.shape[-1] % self.config.pooling_kernel_size,
                        ),
                    )
                hidden_states = self.pooling_layer(hidden_states).permute(0, 2, 1)
                attention_mask = attention_mask[:, :: self.config.pooling_kernel_size]
                if self.config.quantize_causal_block_size is not None:
                    extended_attention_mask = self.get_block_causal_attention_mask(
                        attention_mask,
                        block_size=self.config.quantize_causal_block_size
                        // self.config.pooling_kernel_size,
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
                if quantized_token_ids is not None:
                    hidden_states = self.codebook(quantized_token_ids)
                else:
                    hidden_quantized, indices_flat, distances = vector_quantize(
                        hidden_states, self.codebook.weight
                    )
                    quantized_token_ids = indices_flat.reshape(
                        batch_size, hidden_quantized.shape[1]
                    )
                    if self.training:
                        encodings = torch.nn.functional.one_hot(
                            indices_flat, self.config.quantize_vocab_size
                        ).float()
                        encodings = encodings * attention_mask.reshape(-1, 1)
                        n = torch.sum(encodings, dim=0)
                        torch.distributed.all_reduce(
                            n, op=torch.distributed.ReduceOp.SUM
                        )
                        self.num_active_codes = n.nonzero().shape[0]
                        if self.config.quantize_ema_decay:
                            hidden_flat = (
                                hidden_states.detach()
                                .float()
                                .reshape(-1, hidden_states.shape[-1])
                            )
                            with torch.autocast(
                                device_type="cuda", dtype=torch.float32
                            ):
                                dw = torch.matmul(encodings.t(), hidden_flat)
                            torch.distributed.all_reduce(
                                dw, op=torch.distributed.ReduceOp.SUM
                            )
                            self.ema_count = (
                                self.ema_count * self.config.quantize_ema_decay
                                + (1 - self.config.quantize_ema_decay) * n
                            )
                            total_count = torch.sum(self.ema_count)
                            self.ema_count = (
                                (self.ema_count + 1e-5)
                                / (total_count + self.config.quantize_vocab_size * 1e-5)
                                * total_count
                            )
                            self.ema_weight = (
                                self.ema_weight * self.config.quantize_ema_decay
                                + (1 - self.config.quantize_ema_decay) * dw
                            )
                            self.codebook.weight.data = (
                                self.ema_weight / self.ema_count.unsqueeze(1)
                            )
                            self.quantize_loss = (
                                self.config.quantize_loss_scale
                                * self.config.quantize_commit_coefficient
                                * mse_loss_with_mask(
                                    hidden_states,
                                    hidden_quantized.detach(),
                                    attention_mask,
                                )
                            )
                            self.quantize_ema_count += 1
                            if (
                                self.config.quantize_restart_interval is not None
                                and self.quantize_ema_count
                                % self.config.quantize_restart_interval
                                == 0
                            ):
                                rank, world_size = (
                                    torch.distributed.get_rank(),
                                    torch.distributed.get_world_size(),
                                )
                                segment_vocab_size = (
                                    self.config.quantize_vocab_size // world_size
                                )
                                start_idx = segment_vocab_size * rank
                                ema_count_segment = self.ema_count[
                                    start_idx : start_idx + segment_vocab_size
                                ]
                                threshold = 1 * (
                                    self.config.quantize_ema_decay
                                    ** self.config.quantize_restart_interval
                                )
                                update_indices = (
                                    ema_count_segment < threshold
                                ).nonzero()[:, 0] + start_idx
                                num_update = update_indices.shape[0]
                                mask_flat = attention_mask.reshape(-1) > 0
                                hidden_selected = hidden_flat[mask_flat]
                                hidden_update = hidden_selected[
                                    random.sample(
                                        range(len(hidden_selected)), num_update
                                    )
                                ]
                                num_update = torch.as_tensor(
                                    [num_update],
                                    dtype=torch.long,
                                    device=hidden_states.device,
                                )
                                num_update_list = [
                                    torch.as_tensor(
                                        [0],
                                        dtype=torch.long,
                                        device=hidden_states.device,
                                    )
                                    for _ in range(world_size)
                                ]
                                torch.distributed.all_gather(
                                    num_update_list, num_update
                                )
                                update_indices_list = [
                                    torch.zeros(
                                        num.item(),
                                        dtype=torch.long,
                                        device=hidden_states.device,
                                    )
                                    for num in num_update_list
                                ]
                                torch.distributed.all_gather(
                                    update_indices_list, update_indices
                                )
                                update_indices = torch.cat(update_indices_list)
                                hidden_update_list = [
                                    torch.zeros(
                                        num.item(),
                                        hidden_flat.shape[-1],
                                        dtype=hidden_update.dtype,
                                        device=hidden_states.device,
                                    )
                                    for num in num_update_list
                                ]
                                torch.distributed.all_gather(
                                    hidden_update_list, hidden_update
                                )
                                hidden_update = torch.cat(hidden_update_list)
                                self.codebook.weight.data[update_indices] = (
                                    hidden_update
                                )
                                self.ema_count[update_indices] = 1
                                self.ema_weight[update_indices] = hidden_update
                                if torch.distributed.get_rank() == 0:
                                    print(f"restart {len(update_indices)} tokens")
                        else:
                            loss = self.config.quantize_loss_scale * (
                                self.config.quantize_commit_coefficient
                                * mse_loss_with_mask(
                                    hidden_states,
                                    hidden_quantized.detach(),
                                    attention_mask,
                                )
                                + mse_loss_with_mask(
                                    hidden_quantized,
                                    hidden_states.detach(),
                                    attention_mask,
                                )
                            )
                            self.quantize_loss = loss
                        hidden_states = (
                            hidden_states + (hidden_quantized - hidden_states).detach()
                        )
                    else:
                        hidden_states = hidden_quantized
                hidden_states = (
                    hidden_states
                    + self.embed_positions2.weight[: hidden_states.shape[1]]
                )

            if idx + 1 == self.save_hidden_position:
                import uuid

                import numpy as np

                to_save = []
                for batch_idx, hidden_state in enumerate(hidden_states):
                    for seq_idx, hidden in enumerate(hidden_state):
                        if attention_mask[batch_idx, seq_idx]:
                            to_save.append(hidden.detach().cpu().numpy())
                np.save(
                    os.path.join(self.save_hidden_dir, f"{str(uuid.uuid4())}.npy"),
                    to_save,
                )
        if not self.config.quantize_encoder_only:
            hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return QuantizedBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            quantized_token_ids=quantized_token_ids,
        )


_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}


def extract_speech_token(
    model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor, utts
):
    dtype = model.conv1.weight.dtype
    device = model.conv1.weight.device
    with torch.no_grad():
        audios, indices = [], []
        for idx, utt in enumerate(utts):
            if isinstance(utt, tuple):
                audio, sample_rate = utt
            else:
                audio, sample_rate = torchaudio.load(utt)
            audio = audio.to(device)
            if sample_rate != 16000:
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=16000
                    ).to(device)
                audio = _resample_buffer[sample_rate](audio)
            # if audio.shape[0] > 1:
            #     audio = audio[:1]
            audio = audio[0]
            audio = audio.cpu().numpy()
            time_step = 0
            while time_step * 16000 < audio.shape[0]:
                audio_segment = audio[time_step * 16000 : (time_step + 30) * 16000]
                audios.append(audio_segment)
                indices.append(idx)
                time_step += 30
        pooling_kernel_size = model.config.pooling_kernel_size or 1
        stride = (
            model.conv1.stride[0]
            * model.conv2.stride[0]
            * pooling_kernel_size
            * feature_extractor.hop_length
        )
        all_speech_tokens = [[] for _ in range(len(utts))]
        batch_size = 128
        for start in range(0, len(audios), batch_size):
            features = feature_extractor(
                audios[start : start + batch_size],
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
                device=device,
                padding="longest",
                pad_to_multiple_of=stride,
            )
            features["input_features"] = features["input_features"].to(device).to(dtype)
            features["attention_mask"] = features["attention_mask"].to(device)
            # import ipdb; ipdb.set_trace()
            outputs = model(**features)
            speech_tokens = outputs.quantized_token_ids
            attention_mask = features.attention_mask[
                :, :: model.conv1.stride[0] * model.conv2.stride[0]
            ]
            attention_mask = attention_mask[:, :: model.config.pooling_kernel_size]
            assert attention_mask.shape == speech_tokens.shape
            for i in range(len(speech_tokens)):
                idx = indices[start + i]
                speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                all_speech_tokens[idx].extend(speech_token)
        return all_speech_tokens


class Glm4Tokenizer(nn.Module):
    def __init__(self, tokenizer_path):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.whisper_model = None
        self.feature_extractor = None

    def tokenize(self, speech=None, audio_path=None, sr=16000):
        if self.whisper_model is None:
            self.whisper_model = WhisperVQEncoder.from_pretrained(
                self.tokenizer_path
            ).eval()
        if self.feature_extractor is None:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self.tokenizer_path
            )

        if audio_path:
            # Use torchaudio instead of librosa to avoid fork issues
            audio, orig_sr = torchaudio.load(audio_path)
            # Resample if necessary
            if orig_sr != 16000:
                if orig_sr not in _resample_buffer:
                    _resample_buffer[orig_sr] = torchaudio.transforms.Resample(
                        orig_freq=orig_sr, new_freq=16000
                    )
                audio = _resample_buffer[orig_sr](audio)
            # Take first channel if stereo
            if audio.shape[0] > 1:
                audio = audio[:1]
            audio_info = (audio, 16000)
        else:
            assert speech is not None
            assert sr
            if isinstance(speech, list):
                speech = torch.tensor(speech).unsqueeze(0)
            if len(speech.shape) == 1:
                speech = speech.unsqueeze(0)
            audio_info = (speech, sr)

        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [audio_info]
        )[0]
        audio_tokens = torch.tensor(audio_tokens).unsqueeze(0)
        return audio_tokens
