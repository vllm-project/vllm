# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import DebertaV2Config
from transformers.modeling_outputs import BaseModelOutput

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.pooler import ClassifierPooler
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.transformers_utils.config import (
    get_cross_encoder_activation_function)

from .interfaces import SupportsCrossEncoding, SupportsQuant, SupportsV0Only
from .utils import maybe_prefix


def build_relative_position(
    query_size: int,
    key_size: int,
    bucket_size: int,
    max_position: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build relative position matrix for disentangled attention.

    Computes relative position indices for query-key pairs, where each position
    represents the relative distance between query position i and key position
    j.

    Args:
        query_size (int): Length of the query sequence
        key_size (int): Length of the key sequence
        bucket_size (int): Number of position buckets for encoding
        max_position (int): Maximum allowed absolute position
        device (torch.device): Device for tensor allocation

    Returns:
        torch.LongTensor: Relative position matrix with shape 
            [1, query_size, key_size]
    """

    q_ids = torch.arange(0, query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(0, key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


def make_log_bucket_position(relative_pos: torch.Tensor, bucket_size: int,
                             max_position: int) -> torch.Tensor:
    """
    Map relative positions to position buckets using logarithmic scaling.

    For positions within the middle range, use linear mapping.
    For positions beyond the middle range, use logarithmic compression
    to efficiently represent distant positions.
    """
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        mid - 1,
        torch.abs(relative_pos),
    )

    # Convert scalars to tensors for proper computation
    mid_tensor = torch.tensor(mid, dtype=abs_pos.dtype, device=abs_pos.device)
    max_pos_tensor = torch.tensor(max_position - 1,
                                  dtype=abs_pos.dtype,
                                  device=abs_pos.device)

    # Add epsilon to prevent log(0) and division by 0
    eps = 1e-8
    abs_pos = torch.clamp(abs_pos, min=eps)
    mid_tensor = torch.clamp(mid_tensor, min=eps)
    max_pos_tensor = torch.clamp(max_pos_tensor, min=eps)

    # Compute log position with numerical stability
    log_ratio = torch.log(abs_pos / mid_tensor) / torch.log(
        max_pos_tensor / mid_tensor)
    log_pos = torch.ceil(log_ratio * (mid - 1)) + mid

    # Ensure log_pos is finite
    log_pos = torch.where(
        torch.isfinite(log_pos),
        log_pos,
        torch.tensor(mid, dtype=log_pos.dtype, device=log_pos.device),
    )

    bucket_pos = torch.where(abs_pos <= mid, relative_pos,
                             log_pos * sign).long()

    # Clamp to valid bucket range
    bucket_pos = torch.clamp(bucket_pos, 0, bucket_size - 1)

    return bucket_pos


class DebertaV2Embeddings(nn.Module):

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size,
                                                      config.hidden_size)

        # DeBERTa v2 doesn't use token_type_embeddings by default
        if config.type_vocab_size > 0:
            self.token_type_embeddings = VocabParallelEmbedding(
                config.type_vocab_size, config.hidden_size)
        else:
            self.token_type_embeddings = None

        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # DeBERTa v2 uses relative position encoding, check
        # position_biased_input from config
        self.position_biased_input = getattr(config, "position_biased_input",
                                             False)
        if self.position_biased_input:
            self.position_embeddings = VocabParallelEmbedding(
                config.max_position_embeddings, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = input_ids.size()

        # Input embeddings
        inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        # Add position embeddings only if position_biased_input is True
        if self.position_biased_input and hasattr(self, "position_embeddings"):
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        # Add token type embeddings if available
        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape,
                                             dtype=torch.long,
                                             device=inputs_embeds.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaV2DisentangledSelfAttention(nn.Module):
    """
    DeBERTa v2 disentangled self-attention mechanism.

    Implements three-component attention computation:
    1. Content-to-content attention (standard self-attention)
    2. Content-to-position attention (query content attending to relative 
       positions)
    3. Position-to-content attention (relative positions attending to key 
       content)

    The attention score is computed as the sum of all three components,
    scaled by 1/√(3d) where d is the head dimension.
    """

    def __init__(
        self,
        config: DebertaV2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = (config.hidden_size //
                                    config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Content projection matrices
        self.query_proj = ColumnParallelLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.query_proj",
        )
        self.key_proj = ColumnParallelLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ColumnParallelLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )

        # Position projection matrices (shared with content if specified in
        # config)
        if getattr(config, "share_att_key", False):
            self.pos_key_proj = self.key_proj
            self.pos_query_proj = self.query_proj
        else:
            self.pos_key_proj = ColumnParallelLinear(
                config.hidden_size,
                self.all_head_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.pos_key_proj",
            )
            self.pos_query_proj = ColumnParallelLinear(
                config.hidden_size,
                self.all_head_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.pos_query_proj",
            )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Relative position parameters
        self.max_relative_positions = getattr(config, "max_relative_positions",
                                              512)
        self.position_biased_input = getattr(config, "position_biased_input",
                                             True)

        # Scaling factor for three-component attention: 1/√(3d)
        self.scale_factor = 1.0 / math.sqrt(3.0 * self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention computation."""
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_size]

    def build_relative_position(self, query_size: int, key_size: int,
                                device: torch.device) -> torch.Tensor:
        """
        Build relative position matrix with position bucketing.

        Maps relative distances to bucket indices using the bucketing strategy:
        - For small distances: direct mapping
        - For large distances: logarithmic compression
        """
        k = self.max_relative_positions

        # Create position indices
        q_ids = torch.arange(query_size, dtype=torch.long, device=device)
        k_ids = torch.arange(key_size, dtype=torch.long, device=device)

        # Compute relative distances: i - j
        rel_pos = q_ids[:, None] - k_ids[None, :]

        # Apply bucketing with clamping
        rel_pos = torch.clamp(rel_pos, -k, k)
        rel_pos = rel_pos + k  # Shift to make it 0-indexed: [0, 2k]

        return rel_pos

    def disentangled_attention_bias(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        relative_pos: torch.Tensor,
        rel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute disentangled attention scores with three components:

        1. Content-to-content: Standard self-attention between content 
           representations
        2. Content-to-position: Query content attending to relative position 
           embeddings
        3. Position-to-content: Relative position embeddings attending to key 
           content

        Returns the sum of all three attention components.
        """
        batch_size, num_heads, seq_len, head_size = query_layer.shape

        # Project relative position embeddings
        # rel_embeddings: [2*max_rel_pos+1, hidden_size]
        rel_k, _ = self.pos_key_proj(
            rel_embeddings)  # [2*max_rel_pos+1, all_head_size]
        rel_q, _ = self.pos_query_proj(
            rel_embeddings)  # [2*max_rel_pos+1, all_head_size]

        # Reshape for multi-head attention
        rel_k = rel_k.view(-1, self.num_attention_heads,
                           self.attention_head_size
                           )  # [2*max_rel_pos+1, num_heads, head_size]
        rel_q = rel_q.view(-1, self.num_attention_heads,
                           self.attention_head_size
                           )  # [2*max_rel_pos+1, num_heads, head_size]

        # 1. Content-to-content attention: standard self-attention
        content_to_content = torch.matmul(query_layer,
                                          key_layer.transpose(-2, -1))

        # 2. Content-to-position attention: query content to relative positions
        content_to_position = torch.zeros_like(content_to_content,
                                               device=query_layer.device)

        # For each query position i, compute attention to all relative positions
        for i in range(seq_len):
            # Q_i^c: [batch_size, num_heads, head_size]
            q_i = query_layer[:, :, i, :]  # [batch_size, num_heads, head_size]

            # Compute attention to all relative positions
            # rel_k: [2*max_rel_pos+1, num_heads, head_size]
            # Result: [batch_size, num_heads, 2*max_rel_pos+1]
            c2p_scores = torch.einsum("bnh,rnh->bnr", q_i, rel_k)

            # Extract the correct relative position for each j
            for j in range(seq_len):
                rel_idx = relative_pos[i, j].item()
                content_to_position[:, :, i, j] = c2p_scores[:, :, rel_idx]

        # 3. Position-to-content attention: relative positions to key content
        position_to_content = torch.zeros_like(content_to_content,
                                               device=query_layer.device)

        # For each key position j, compute attention from relative positions
        for j in range(seq_len):
            # K_j^c: [batch_size, num_heads, head_size]
            k_j = key_layer[:, :, j, :]  # [batch_size, num_heads, head_size]

            # Compute attention from all relative positions
            # rel_q: [2*max_rel_pos+1, num_heads, head_size]
            # Result: [batch_size, num_heads, 2*max_rel_pos+1]
            p2c_scores = torch.einsum("bnh,rnh->bnr", k_j, rel_q)

            # Extract the correct relative position for each i (note: j,i not
            # i,j)
            for i in range(seq_len):
                rel_idx = relative_pos[
                    j, i].item()  # Reverse direction for position-to-content
                position_to_content[:, :, i, j] = p2c_scores[:, :, rel_idx]

        # Sum all three components
        attention_scores = (content_to_content + content_to_position +
                            position_to_content)

        return attention_scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        rel_embeddings: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass implementing DeBERTa v2 disentangled attention.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Project to query, key, value
        query_layer, _ = self.query_proj(hidden_states)
        key_layer, _ = self.key_proj(hidden_states)
        value_layer, _ = self.value_proj(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Build relative position matrix if not provided
        if rel_pos is None:
            rel_pos = self.build_relative_position(seq_len, seq_len,
                                                   hidden_states.device)

        # Compute disentangled attention scores
        if rel_embeddings is not None:
            attention_scores = self.disentangled_attention_bias(
                query_layer, key_layer, rel_pos, rel_embeddings)
        else:
            # DeBERTa v2 REQUIRES disentangled attention - this should never
            # happen
            raise ValueError(
                "DeBERTa v2 requires relative embeddings for disentangled "
                "attention. This indicates a configuration or implementation "
                "error.")

        # Apply scaling factor for three-component attention
        attention_scores = attention_scores * self.scale_factor

        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax to get attention probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape back to original format
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = ((context_layer, attention_probs) if output_attentions else
                   (context_layer, ))
        return outputs


class DebertaV2SelfOutput(nn.Module):

    def __init__(
        self,
        config: DebertaV2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Attention(nn.Module):
    """DeBERTa v2 attention module."""

    def __init__(
        self,
        config: DebertaV2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.self = DebertaV2DisentangledSelfAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self",
            layer_idx=0,  # Will be set properly in the layer
        )
        self.output = DebertaV2SelfOutput(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.output",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        rel_embeddings: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            rel_embeddings,
            rel_pos,
        )

        if output_attentions:
            self_output, attention_probs = self_outputs
        else:
            self_output = self_outputs[0]
            attention_probs = None

        output = self.output(self_output, hidden_states)

        if output_attentions:
            return output, attention_probs
        else:
            return output, None


class DebertaV2Intermediate(nn.Module):

    def __init__(
        self,
        config: DebertaV2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )
        self.intermediate_act_fn = get_act_fn(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaV2Output(nn.Module):

    def __init__(
        self,
        config: DebertaV2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dense = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Layer(nn.Module):
    """DeBERTa v2 transformer layer."""

    def __init__(
        self,
        config: DebertaV2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.attention = DebertaV2Attention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        self.intermediate = DebertaV2Intermediate(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.intermediate",
        )
        self.output = DebertaV2Output(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.output",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        rel_embeddings: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
    ):
        attn_output, attention_probs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions,
            rel_embeddings,
            rel_pos,
        )
        intermediate_output = self.intermediate(attn_output)
        output = self.output(intermediate_output, attn_output)

        if output_attentions:
            return output, attention_probs
        else:
            return output, None


@support_torch_compile
class DebertaV2Encoder(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # Transformer layers
        self.layer = nn.ModuleList([
            DebertaV2Layer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layer.{layer_idx}",
            ) for layer_idx in range(config.num_hidden_layers)
        ])

        # Relative position embeddings
        self.relative_attention = getattr(config, "relative_attention", True)
        if self.relative_attention:
            self.max_relative_positions = getattr(config,
                                                  "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.position_buckets = getattr(config, "position_buckets", 256)

            # Relative embeddings for position encoding
            self.rel_embeddings = VocabParallelEmbedding(
                self.position_buckets, config.hidden_size)

        # Encoder-level LayerNorm (applied after all layers)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)

        # Convolutional layer for enhanced position encoding (DeBERTa v2
        # specific)
        conv_kernel_size = getattr(config, "conv_kernel_size", 3)
        self.conv = DebertaV2ConvLayer(config.hidden_size, conv_kernel_size,
                                       config.hidden_dropout_prob)

    def get_relative_positions(self,
                               hidden_states: torch.Tensor) -> torch.Tensor:
        """Generate relative position matrix for the sequence."""
        if len(hidden_states.shape) == 2:
            # vLLM format: (seq_len, hidden_size)
            seq_len = hidden_states.size(0)
        else:
            # Standard format: (batch_size, seq_len, hidden_size)
            seq_len = hidden_states.size(1)

        return build_relative_position(
            seq_len,
            seq_len,
            self.position_buckets,
            self.max_relative_positions,
            hidden_states.device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Build relative position matrix and embeddings
        seq_len = hidden_states.size(1)
        relative_pos = build_relative_position(
            seq_len,
            seq_len,
            self.position_buckets,
            self.max_relative_positions,
            hidden_states.device,
        )
        rel_embeddings = self.rel_embeddings.weight

        # Apply convolutional layer for enhanced position encoding
        # This captures n-gram patterns in the input sequence
        hidden_states = self.conv(hidden_states)

        # Pass through transformer layers
        for layer in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_output, attention_probs = layer(
                hidden_states,
                attention_mask,
                output_attentions,
                rel_embeddings,
                relative_pos,
            )

            hidden_states = layer_output

            if output_attentions:
                all_attentions = all_attentions + (attention_probs, )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        # Apply final LayerNorm
        hidden_states = self.LayerNorm(hidden_states)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class DebertaV2ConvLayer(nn.Module):
    """
    Convolutional layer for enhanced position encoding in DeBERTa v2.
    This layer applies grouped convolution to capture local dependencies.
    """

    def __init__(self, hidden_size: int, kernel_size: int,
                 dropout_prob: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=hidden_size,
        )
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Handle different tensor formats
        if len(hidden_states.shape) == 2:
            # vLLM format: (seq_len, hidden_size)
            # Add batch dimension for conv operation
            conv_input = hidden_states.unsqueeze(0).transpose(
                1, 2)  # (1, hidden_size, seq_len)
            conv_output = self.conv(conv_input)
            conv_output = conv_output.transpose(1, 2).squeeze(
                0)  # (seq_len, hidden_size)
        else:
            # Standard format: (batch_size, seq_len, hidden_size)
            conv_input = hidden_states.transpose(
                1, 2)  # (batch_size, hidden_size, seq_len)
            conv_output = self.conv(conv_input)
            conv_output = conv_output.transpose(
                1, 2)  # (batch_size, seq_len, hidden_size)

        # Apply activation and normalization
        conv_output = self.activation(conv_output)
        conv_output = self.dropout(conv_output)

        # Residual connection and layer norm
        output = self.LayerNorm(conv_output + hidden_states)
        return output


class DebertaV2Model(nn.Module, SupportsQuant):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        embedding_class: type = DebertaV2Embeddings,
        add_pooling_layer: bool = False,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embeddings = embedding_class(config)
        self.encoder = DebertaV2Encoder(vllm_config=vllm_config,
                                        prefix=f"{prefix}.encoder")

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            attn_metadata = get_forward_context().attn_metadata
            assert hasattr(attn_metadata, "seq_lens_tensor")
            hidden_states = self.embeddings(
                input_ids=input_ids,
                seq_lens=attn_metadata.seq_lens_tensor,
                position_ids=positions,
                token_type_ids=token_type_ids,
            )
        return self.encoder(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # Handle DeBERTa v2 specific weights
            if "relative_attention_bias" in name:
                # These are handled by the disentangled attention mechanism
                continue
            if "disentangled_attention_bias" in name:
                # These are computed dynamically in disentangled attention
                continue

            if "rel_embeddings" in name:
                param_name = name.replace("encoder.rel_embeddings",
                                          "encoder.rel_embeddings.weight")
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                continue

            if "encoder.LayerNorm" in name:
                param_name = name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                continue

            if "encoder.conv" in name:
                if "encoder.conv.weight" in name:
                    param_name = "encoder.conv.conv.weight"
                elif "encoder.conv.bias" in name:
                    param_name = "encoder.conv.conv.bias"
                else:
                    param_name = name

                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                continue

            # Handle position projection weights for disentangled attention
            if "pos_query_proj" in name or "pos_key_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                continue

            # Skip loading extra bias for packed layers
            if name.endswith(".bias") and name not in params_dict:
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)


class DebertaV2CrossEncoderActivation(nn.Module):
    """Custom activation function for DeBERTa v2 cross-encoder models.

    For multi-class models like MNLI, this extracts the appropriate score
    for similarity ranking. For MNLI, we use the ENTAILMENT score (index 2).
    """

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.num_labels = config.num_labels

        # For MNLI models, use ENTAILMENT score (index 2)
        if (self.num_labels == 3 and hasattr(config, "id2label")
                and config.id2label.get(2) == "ENTAILMENT"):
            self.entailment_index = 2
        else:
            self.entailment_index = None

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_labels == 1:
            # Single label case - return as is
            return logits
        elif self.entailment_index is not None:
            # MNLI case - extract ENTAILMENT score
            return logits[..., self.entailment_index:self.entailment_index + 1]
        else:
            # Other multi-class cases - use softmax and take max probability
            probs = torch.softmax(logits, dim=-1)
            max_prob, _ = torch.max(probs, dim=-1, keepdim=True)
            return max_prob


class DebertaV2ContextPooler(nn.Module):
    """Context pooler for DeBERTa v2 that matches the original 
    implementation."""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        pooler_hidden_size = getattr(config, "pooler_hidden_size",
                                     config.hidden_size)
        pooler_dropout = getattr(config, "pooler_dropout", 0.0)

        self.dense = nn.Linear(config.hidden_size, pooler_hidden_size)
        self.dropout = nn.Dropout(pooler_dropout)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Handle vLLM's tensor format: (seq_len, hidden_size) for a single
        # sequence. The ClassifierPooler already extracts the sequence, so we
        # just need the first token
        if len(hidden_states.shape) == 2:
            # vLLM format: (seq_len, hidden_size) - take the first token (CLS)
            pooled_output = hidden_states[0]
        else:
            # Standard format: (batch_size, seq_len, hidden_size) - take first
            # token of each sequence
            pooled_output = hidden_states[:, 0]

        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        return pooled_output


class DebertaV2ForSequenceClassification(nn.Module, SupportsV0Only,
                                         SupportsCrossEncoding, SupportsQuant):
    """
    DeBERTa v2 model for sequence classification tasks.

    Supports both single-label and multi-label classification with proper
    pooling and activation functions for cross-encoder applications.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.default_activation_function = (
            get_cross_encoder_activation_function(config))
        self.num_labels = config.num_labels

        # Build the main DeBERTa model
        self.deberta = DebertaV2Model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "deberta"))

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Pooler for classification
        self._pooler = ClassifierPooler(
            vllm_config.model_config,
            self.classifier,
            None,  # No built-in pooler in DeBERTa v2
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for sequence classification."""
        hidden_states = self.deberta(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
            token_type_ids=token_type_ids,
        )

        # Apply classification head
        logits = self.classifier(hidden_states[:, 0])  # Use [CLS] token
        return logits

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights from checkpoint."""
        self_weights = []

        def weight_filter():
            for name, weight in weights:
                if name.startswith("deberta."):
                    yield (name[len("deberta."):], weight)
                else:
                    self_weights.append((name, weight))

        self.deberta.load_weights(weight_filter())

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in self_weights:
            if name.startswith("classifier"):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
