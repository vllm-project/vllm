# Derived from BART implementation posted on HuggingFace; license below:
#
# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team.
# All rights reserved.
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
"""PyTorch BART model."""
import math
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import BartConfig
from transformers.utils import logging

from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

logger = logging.get_logger(__name__)


def get_bsz_seq_len(input_ids):
    shp = input_ids.shape
    ndim = len(shp)
    if ndim == 1:
        return 1, input_ids.numel()
    else:
        return shp[:2]


class BartLearnedPositionalEmbedding(VocabParallelEmbedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is
        # specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately.
        # Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(
        self,
        positions: torch.Tensor,
        attn_type: AttentionType,
    ) -> torch.Tensor:
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        assert attn_type != AttentionType.ENCODER_DECODER

        return super().forward(positions + self.offset)


class BartScaledWordEmbedding(VocabParallelEmbedding):
    """
    This module overrides VocabParallelEmbedding's 
    forward by multiplying with embeddings scale.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class BartParallelLMHead(ParallelLMHead):
    """
    This module overrides ParallelLMHead's
    forward by dividing by embeddings scale,
    yielding effectively the inverse of
    BartScaledWordEmbedding
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) / self.embed_scale


class BartEncoderAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: Optional[BartConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads "
                             f"(got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q,
                                k,
                                v,
                                kv_cache,
                                attn_metadata,
                                attn_type=AttentionType.ENCODER)

        output, _ = self.out_proj(attn_output)
        return output


class BartDecoderSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: Optional[BartConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads "
                             f"(got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q,
                                k,
                                v,
                                kv_cache,
                                attn_metadata,
                                attn_type=AttentionType.DECODER)

        output, _ = self.out_proj(attn_output)
        return output


class BartCrossAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: Optional[BartConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads "
                             f"(got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        # (afeldman-nm 2024/07/22) TODO:
        # Need a more efficient solution for q/k/v
        qkv_dec, _ = self.qkv_proj(decoder_hidden_states)
        q, _, _ = qkv_dec.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)
        if encoder_hidden_states is None:
            k = None
            v = None
        else:
            qkv_enc, _ = self.qkv_proj(encoder_hidden_states)
            _, k, v = qkv_enc.split([self.q_size, self.kv_size, self.kv_size],
                                    dim=-1)

        attn_output = self.attn(q,
                                k,
                                v,
                                kv_cache,
                                attn_metadata,
                                attn_type=AttentionType.ENCODER_DECODER)

        output, _ = self.out_proj(attn_output)
        return output


class BartEncoderLayer(nn.Module):

    def __init__(
        self,
        config: BartConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartEncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = get_act_fn(config.activation_function,
                                        quant_config)

        ffn_hidden_size = self.embed_dim
        ffn_intermediate_size = config.encoder_ffn_dim
        ffn_has_bias = True
        self.fc1 = ColumnParallelLinear(
            ffn_hidden_size,
            ffn_intermediate_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )
        self.act = get_act_fn("gelu", quant_config, ffn_intermediate_size)
        self.fc2 = RowParallelLinear(
            ffn_intermediate_size,
            ffn_hidden_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        r"""
        Args:
            hidden_states
                torch.Tensor of *encoder* input embeddings.
            kv_cache:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Encoder layer output torch.Tensor
        """
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata)

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)

        hidden_states, _ = self.fc2(hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any()
                or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states,
                                        min=-clamp_value,
                                        max=clamp_value)

        return hidden_states


class BartDecoderLayer(nn.Module):

    def __init__(
        self,
        config: BartConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartDecoderSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config)
        self.activation_fn = get_act_fn(config.activation_function,
                                        quant_config)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        '''
        afeldman-nm: personally I would call this "cross-attention",
        however I left the name as "encoder_attn" to maintain consistency
        with the name of the pretrained weights.
        '''
        self.encoder_attn = BartCrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        ffn_hidden_size = self.embed_dim
        ffn_intermediate_size = config.encoder_ffn_dim
        ffn_has_bias = True
        self.fc1 = ColumnParallelLinear(
            ffn_hidden_size,
            ffn_intermediate_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            ffn_intermediate_size,
            ffn_hidden_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            decoder_hidden_states
                torch.Tensor of *decoder* input embeddings.
            kv_cache:
                KV cache tensor
            attn_metadata:
                vLLM Attention metadata structure
            encoder_hidden_states
                torch.Tensor of *encoder* input embeddings.
        Returns:
            Decoder layer output torch.Tensor
        """
        residual = decoder_hidden_states

        # Self Attention
        hidden_states = self.self_attn(hidden_states=decoder_hidden_states,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata)

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block

        residual = hidden_states

        hidden_states = self.encoder_attn(
            decoder_hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)

        hidden_states, _ = self.fc2(hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers*
    self attention layers. Each layer is a [`BartEncoderLayer`].
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self,
                 config: BartConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 lora_config: Optional[LoRAConfig] = None,
                 embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()

        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        embed_dim = config.d_model
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(config.vocab_size,
                                                    embed_dim,
                                                    embed_scale=embed_scale)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList(
            [BartEncoderLayer(config,cache_config,quant_config) \
             for _ in range(config.encoder_layers)])

        self.layernorm_embedding = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        r"""
        Args:
            input_ids
                Indices of *encoder* input sequence tokens in the vocabulary.
                Padding will be ignored by default should you
                provide it.
            positions
                Positions of *encoder* input sequence tokens.
            kv_caches:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Decoder output torch.Tensor
        """
        # retrieve input_ids and inputs_embeds

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(
            positions,
            AttentionType.ENCODER,
        )
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states=hidden_states,
                kv_cache=kv_caches[idx],
                attn_metadata=attn_metadata,
            )

        return hidden_states


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers.
    Each layer is a [`BartDecoderLayer`]
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(
        self,
        config: BartConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(
            config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(config.vocab_size,
                                                    config.d_model,
                                                    embed_scale=embed_scale)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )

        self.layers = nn.ModuleList(
            [BartDecoderLayer(config,cache_config,quant_config) \
             for _ in range(config.decoder_layers)])

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(self, decoder_input_ids: torch.Tensor,
                decoder_positions: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor],
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        r"""
        Args:
            decoder_input_ids
                Indices of *decoder* input sequence tokens in the vocabulary.
                Padding will be ignored by default should you
                provide it.
            decoder_positions
                Positions of *decoder* input sequence tokens.
            encoder_hidden_states:
                Tensor of encoder output embeddings
            kv_caches:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Decoder output torch.Tensor
        """

        inputs_embeds = self.embed_tokens(decoder_input_ids)

        # embed positions
        embed_pos = self.embed_positions(
            decoder_positions,
            AttentionType.DECODER,
        )
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)

        # decoder layers

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                decoder_hidden_states=hidden_states,
                kv_cache=kv_caches[idx],
                attn_metadata=attn_metadata,
                encoder_hidden_states=encoder_hidden_states,
            )

        return hidden_states


class BartModel(nn.Module):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight", "decoder.embed_tokens.weight"
    ]

    def __init__(self,
                 config: BartConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 lora_config: Optional[LoRAConfig] = None):
        super().__init__()

        self.config = config

        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.encoder = BartEncoder(config,
                                   cache_config,
                                   quant_config=quant_config)
        self.decoder = BartDecoder(config,
                                   cache_config,
                                   quant_config=quant_config)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                encoder_input_ids: torch.Tensor,
                encoder_positions: torch.Tensor, kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata) -> torch.Tensor:
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
            kv_caches:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Model output torch.Tensor
        """

        encoder_hidden_states = None

        if encoder_input_ids.numel() > 0:
            # Run encoder attention if a non-zero number of encoder tokens
            # are provided as input
            encoder_hidden_states = self.encoder(input_ids=encoder_input_ids,
                                                 positions=encoder_positions,
                                                 kv_caches=kv_caches,
                                                 attn_metadata=attn_metadata)

        # decoder outputs consists of
        # (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids=input_ids,
            decoder_positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata)

        return decoder_outputs


class BartForConditionalGeneration(nn.Module):
    base_model_prefix = "model"

    def __init__(self,
                 config: BartConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 lora_config: Optional[LoRAConfig] = None):

        super().__init__()
        # currently all existing BART models have `tie_word_embeddings` enabled
        assert config.tie_word_embeddings
        self.config = config
        self.model = BartModel(config,
                               cache_config,
                               quant_config,
                               lora_config=lora_config)

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        embed_scale = math.sqrt(
            config.d_model) if config.scale_embedding else 1.0

        self.lm_head = BartParallelLMHead(config.vocab_size,
                                          config.d_model,
                                          embed_scale=embed_scale)

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        *,
        encoder_input_ids: torch.Tensor,
        encoder_positions: torch.Tensor,
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
            kv_caches:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Output torch.Tensor
        """
        return self.model(input_ids, positions, encoder_input_ids,
                          encoder_positions, kv_caches, attn_metadata)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    stacked_params_mapping = {
        "q_proj": {
            "param_name": "qkv_proj",
            "shard_id": "q",
        },
        "k_proj": {
            "param_name": "qkv_proj",
            "shard_id": "k",
        },
        "v_proj": {
            "param_name": "qkv_proj",
            "shard_id": "v",
        },
    }

    params_mapping = {
        "beta": "bias",
        "gamma": "weight",
        "LayerNorm": "layernorm",
    }

    def _rename_key(self, key: str):
        prefix = f"{self.base_model_prefix}."
        key = key[len(prefix):] if key.startswith(prefix) else key

        for src, dst in self.params_mapping.items():
            key = key.replace(src, dst)

        return key

    def _rename_stacked_param(
        self,
        name: str,
    ) -> Tuple[str, Optional[str]]:
        for key, mapping in self.stacked_params_mapping.items():
            if key in name:
                name = name.replace(key, mapping["param_name"])
                return name, mapping["shard_id"]
        return name, None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        model_params_dict = dict(self.model.named_parameters())
        top_params_dict = dict(self.named_parameters())

        weights_tuple_list = list(weights)

        shared_embedding_weight = None
        shared_embedding_shard_id = None

        for name, loaded_weight in weights_tuple_list:

            name = self._rename_key(name)
            name, shard_id = self._rename_stacked_param(name)

            if ('shared.weight' in name
                    or 'encoder.embed_tokens.weight' in name
                    or 'decoder.embed_tokens.weight' in name
                    or 'lm_head.weight' in name):
                assert shared_embedding_weight is None, (
                    "Conflicting embedding weights.")
                shared_embedding_weight = loaded_weight
                shared_embedding_shard_id = shard_id
            else:
                # Skip the specific downstream task weight.
                if name.startswith('cls.'):
                    continue
                # use Pooler instead.
                if name.startswith('pooler.'):
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in model_params_dict:
                    continue

                param = model_params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if shard_id:
                    weight_loader(param, loaded_weight, shard_id)
                else:
                    weight_loader(param, loaded_weight)

        # Assign shared weight values
        encoder_in_param = model_params_dict['encoder.embed_tokens.weight']
        encoder_in_weight_loader = getattr(encoder_in_param, "weight_loader",
                                           default_weight_loader)

        decoder_in_param = model_params_dict['decoder.embed_tokens.weight']
        decoder_in_weight_loader = getattr(decoder_in_param, "weight_loader",
                                           default_weight_loader)

        lm_head_in_param = top_params_dict['lm_head.weight']
        lm_head_in_weight_loader = getattr(lm_head_in_param, "weight_loader",
                                           default_weight_loader)

        assert shared_embedding_weight is not None

        if shared_embedding_shard_id:
            encoder_in_weight_loader(encoder_in_param, shared_embedding_weight,
                                     shared_embedding_shard_id)
            decoder_in_weight_loader(decoder_in_param, shared_embedding_weight,
                                     shared_embedding_shard_id)
            lm_head_in_weight_loader(lm_head_in_param, shared_embedding_weight,
                                     shared_embedding_shard_id)
        else:
            encoder_in_weight_loader(encoder_in_param, shared_embedding_weight)
            decoder_in_weight_loader(decoder_in_param, shared_embedding_weight)
            lm_head_in_weight_loader(lm_head_in_param, shared_embedding_weight)
