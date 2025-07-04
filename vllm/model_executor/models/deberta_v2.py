# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
DeBERTa v2 model implementation for vLLM.

Based on the HuggingFace transformers implementation and Microsoft's official DeBERTa repository.
Supports disentangled attention with proper relative position encoding.
"""

import math
from collections.abc import Iterable
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Config

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               LinearBase)
from vllm.model_executor.layers.pooler import Pooler, PoolingType, ClassifierPooler
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import PoolerOutput
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors


def build_relative_position(query_size: int, key_size: int, bucket_size: int = -1, max_position: int = -1, device=None):
    """Build relative position according to the query and key sizes."""
    q_ids = torch.arange(0, query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(0, key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    
    return rel_pos_ids


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    """Make relative position into log buckets like T5."""
    mid = bucket_size // 2
    abs_pos = torch.abs(relative_pos)
    
    # Small positions use linear buckets (0 to mid-1)
    is_small = abs_pos < mid
    
    # Large positions use log buckets (mid to bucket_size-1)
    # Use T5-style logarithmic bucketing for positions beyond mid
    # Avoid log(0) by using max(abs_pos, 1.0) and ensure mid > 0
    safe_abs_pos = torch.clamp(abs_pos.float(), min=1.0)
    safe_mid = max(mid, 1)
    safe_max_pos = max(max_position, safe_mid + 1)
    
    log_pos = mid + (torch.log(safe_abs_pos / safe_mid) / math.log(safe_max_pos / safe_mid) * (mid - 1)).long()
    log_pos = torch.clamp(log_pos, mid, bucket_size - 1)
    
    # Use linear for small, log for large
    bucket_pos = torch.where(is_small, abs_pos.long(), log_pos)
    
    # Handle negative positions by mirroring around bucket_size
    # Negative positions get indices [bucket_size, 2*bucket_size-1]
    bucket_pos = torch.where(relative_pos < 0, bucket_size + bucket_pos, bucket_pos)
    
    return bucket_pos


class DebertaV2Embeddings(nn.Module):
    """DeBERTa v2 embeddings with position-biased input."""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            self.embedding_size,
        )
        
        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = VocabParallelEmbedding(
                config.max_position_embeddings,
                self.embedding_size,
            )

        if getattr(config, "type_vocab_size", 0) > 0:
            self.token_type_embeddings = VocabParallelEmbedding(
                config.type_vocab_size,
                self.embedding_size,
            )

        if self.embedding_size != config.hidden_size:
            self.embed_proj = RowParallelLinear(
                self.embedding_size,
                config.hidden_size,
                bias=False,
            )
        else:
            self.embed_proj = None

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_length = input_ids.size(-1)
        
        # Word embeddings
        words_embeddings = self.word_embeddings(input_ids)
        
        # Position embeddings
        embeddings = words_embeddings
        if self.position_biased_input:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Token type embeddings
        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        # Project to hidden size if needed
        if self.embed_proj is not None:
            embeddings = self.embed_proj(embeddings)[0]

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaV2DisentangledSelfAttention(nn.Module):
    """DeBERTa v2 disentangled self-attention with proper relative position encoding."""

    def __init__(self, config: DebertaV2Config, layer_idx: int = 0):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Content projections
        self.query_proj = RowParallelLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
        )
        self.key_proj = RowParallelLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
        )
        self.value_proj = RowParallelLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Position attention configuration
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else ["c2p", "p2c"]
        self.relative_attention = getattr(config, "relative_attention", False)
        self.talking_head = getattr(config, "talking_head", False)
        
        # Ensure pos_att_type is a list for consistent behavior
        if isinstance(self.pos_att_type, str):
            self.pos_att_type = [self.pos_att_type]
        
        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(new_x_shape)
        return x.permute(0, 2, 1, 3)


    def _get_sequence_lengths(self):
        """Get sequence lengths from vLLM forward context."""
        try:
            from vllm.forward_context import get_forward_context
            forward_context = get_forward_context()
            
            if forward_context and hasattr(forward_context, 'attn_metadata'):
                attn_metadata = forward_context.attn_metadata
                if hasattr(attn_metadata, 'seq_lens') and attn_metadata.seq_lens is not None:
                    return attn_metadata.seq_lens
        except (ImportError, AttributeError):
            pass
        return None

    def _compute_isolated_attention_bias(
        self, 
        query_layer: torch.Tensor, 
        key_layer: torch.Tensor,
        relative_pos: torch.Tensor, 
        rel_embeddings: torch.Tensor,
        seq_lens: list,
        total_seq_len: int
    ) -> torch.Tensor:
        """Compute attention bias with sequence isolation to prevent cross-contamination."""
        # Ensure relative_pos indices are within bounds
        max_idx = rel_embeddings.shape[0] - 1
        relative_pos = torch.clamp(relative_pos.long(), 0, max_idx)
        
        att_span = torch.zeros_like(query_layer @ key_layer.transpose(-1, -2))
        
        # Process each sequence separately
        start_pos = 0
        for seq_len in seq_lens:
            end_pos = start_pos + seq_len
            
            # Extract sub-tensors for this sequence
            query_seq = query_layer[:, :, start_pos:end_pos, :]  # [batch, heads, seq_len, head_dim]
            key_seq = key_layer[:, :, start_pos:end_pos, :]
            rel_pos_seq = relative_pos[start_pos:end_pos, start_pos:end_pos]
            
            # Compute attention bias for this sequence only
            att_span_seq = torch.zeros_like(query_seq @ key_seq.transpose(-1, -2))
            
            # Content-to-position attention (c2p)
            if "c2p" in self.pos_att_type:
                pos_key_embeddings = rel_embeddings[rel_pos_seq]  # [seq_len, seq_len, hidden_size]
                
                # Project position embeddings through key projection
                pos_key_flat = pos_key_embeddings.reshape(-1, pos_key_embeddings.size(-1))
                pos_key_proj = self.key_proj(pos_key_flat)[0]  # [seq_len*seq_len, hidden_size]
                pos_key = pos_key_proj.reshape(seq_len, seq_len, self.all_head_size)
                
                # Reshape for multi-head attention: [seq_len, seq_len, num_heads, head_dim]
                pos_key = pos_key.reshape(seq_len, seq_len, self.num_attention_heads, self.attention_head_size)
                pos_key = pos_key.permute(2, 0, 1, 3)  # [heads, seq, seq, head_dim]
                
                # Compute c2p attention: query content @ position key
                c2p_att = torch.einsum('bhiq,hijq->bhij', query_seq, pos_key)
                att_span_seq += c2p_att

            # Position-to-content attention (p2c)  
            if "p2c" in self.pos_att_type:
                # We need relative positions from key to query (transpose of rel_pos_seq)
                rel_pos_seq_t = rel_pos_seq.transpose(0, 1)  # [seq_len, seq_len]
                
                # Extract relative position embeddings for p2c
                pos_query_embeddings = rel_embeddings[rel_pos_seq_t]  # [seq_len, seq_len, hidden_size]
                
                # Project position embeddings through query projection
                pos_query_flat = pos_query_embeddings.reshape(-1, pos_query_embeddings.size(-1))
                pos_query_proj = self.query_proj(pos_query_flat)[0]  # [seq_len*seq_len, hidden_size]
                pos_query = pos_query_proj.reshape(seq_len, seq_len, self.all_head_size)
                
                # Reshape for multi-head attention: [seq_len, seq_len, num_heads, head_dim]
                pos_query = pos_query.reshape(seq_len, seq_len, self.num_attention_heads, self.attention_head_size)
                pos_query = pos_query.permute(2, 0, 1, 3)  # [heads, seq, seq, head_dim]
                
                # Compute p2c attention: position query @ content key
                p2c_att = torch.einsum('hijq,bhjq->bhij', pos_query, key_seq)
                att_span_seq += p2c_att
            
            # Place the computed bias in the correct position in the full attention matrix
            att_span[:, :, start_pos:end_pos, start_pos:end_pos] = att_span_seq
            
            start_pos = end_pos
        
        return att_span

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rel_embeddings: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Handle 2D vs 3D tensor shapes for vLLM compatibility
        if hidden_states.dim() == 2:
            seq_len, hidden_size = hidden_states.shape
            batch_size = 1
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_output = True
        else:
            batch_size, seq_len, hidden_size = hidden_states.shape
            squeeze_output = False

        # Content projections
        query_layer = self.transpose_for_scores(self.query_proj(hidden_states)[0])
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states)[0])
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states)[0])

        # Content-to-content attention (standard self-attention)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Add relative position bias if available
        if self.relative_attention and rel_embeddings is not None and rel_pos is not None:
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, rel_pos, rel_embeddings, batch_size, seq_len
            )
            attention_scores = attention_scores + rel_att



        # DeBERTa v2 scaling: 1/√(3d) when using all three attention components, otherwise 1/√d
        num_attention_components = 1  # content-to-content
        if self.relative_attention and rel_embeddings is not None and rel_pos is not None:
            if "c2p" in self.pos_att_type:
                num_attention_components += 1  # content-to-position
            if "p2c" in self.pos_att_type:
                num_attention_components += 1  # position-to-content
        
        scale_factor = math.sqrt(num_attention_components * self.attention_head_size)
        attention_scores = attention_scores / scale_factor

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Convert 2D mask to 4D
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            attention_scores = attention_scores + attention_mask

        # Clamp attention scores to prevent numerical instability
        attention_scores = torch.clamp(attention_scores, min=-50.0, max=50.0)
        
        # Normalize attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape output
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        if squeeze_output:
            context_layer = context_layer.squeeze(0)
        return context_layer

    def disentangled_attention_bias(
        self, 
        query_layer: torch.Tensor, 
        key_layer: torch.Tensor,
        relative_pos: torch.Tensor, 
        rel_embeddings: torch.Tensor,
        batch_size: int,
        seq_len: int
    ) -> torch.Tensor:
        """Compute disentangled attention bias for content-to-position and position-to-content attention."""
        if relative_pos is None or rel_embeddings is None:
            return 0

        # Check for vLLM sequence isolation
        seq_lens = self._get_sequence_lengths()
        if seq_lens and len(seq_lens) > 1:
            # Multiple sequences - compute bias per sequence to prevent cross-contamination
            return self._compute_isolated_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, seq_lens, seq_len
            )

        # Single sequence - use original logic
        # Ensure relative_pos indices are within bounds
        max_idx = rel_embeddings.shape[0] - 1
        relative_pos = torch.clamp(relative_pos.long(), 0, max_idx)

        att_span = torch.zeros_like(query_layer @ key_layer.transpose(-1, -2))
        
        # Handle the case where batch_size might not match the actual tensor batch dimension
        actual_batch_size = query_layer.shape[0]
        if actual_batch_size != batch_size:
            # This might happen in vLLM when batching is handled differently
            batch_size = actual_batch_size
        
        # For batch processing, we need to ensure each sequence gets its own position bias
        # The issue is that when batch_size > 1, all sequences share the same relative_pos matrix
        # which causes cross-contamination. We need to isolate the computation per sequence.
        if batch_size > 1:
            # Process each sequence in the batch separately to avoid cross-contamination
            att_spans = []
            for b in range(batch_size):
                # Extract single sequence tensors
                query_b = query_layer[b:b+1]  # [1, heads, seq, head_dim]
                key_b = key_layer[b:b+1]      # [1, heads, seq, head_dim]
                
                # Compute attention bias for this sequence only
                att_span_b = torch.zeros_like(query_b @ key_b.transpose(-1, -2))
                
                # Compute c2p and p2c for this sequence
                if "c2p" in self.pos_att_type:
                    pos_key_embeddings = rel_embeddings[relative_pos]
                    pos_key_flat = pos_key_embeddings.reshape(-1, pos_key_embeddings.size(-1))
                    pos_key_proj = self.key_proj(pos_key_flat)[0]
                    pos_key = pos_key_proj.reshape(seq_len, seq_len, self.all_head_size)
                    pos_key = pos_key.reshape(seq_len, seq_len, self.num_attention_heads, self.attention_head_size)
                    pos_key = pos_key.permute(2, 0, 1, 3)
                    c2p_att = torch.einsum('bhiq,hijq->bhij', query_b, pos_key)
                    att_span_b += c2p_att
                
                if "p2c" in self.pos_att_type:
                    relative_pos_t = relative_pos.transpose(0, 1)
                    pos_query_embeddings = rel_embeddings[relative_pos_t]
                    pos_query_flat = pos_query_embeddings.reshape(-1, pos_query_embeddings.size(-1))
                    pos_query_proj = self.query_proj(pos_query_flat)[0]
                    pos_query = pos_query_proj.reshape(seq_len, seq_len, self.all_head_size)
                    pos_query = pos_query.reshape(seq_len, seq_len, self.num_attention_heads, self.attention_head_size)
                    pos_query = pos_query.permute(2, 0, 1, 3)
                    p2c_att = torch.einsum('hijq,bhjq->bhij', pos_query, key_b)
                    att_span_b += p2c_att
                
                att_spans.append(att_span_b)
            
            # Concatenate results from all sequences
            att_span = torch.cat(att_spans, dim=0)
            return att_span

        # DeBERTa v2 uses shared query/key projections for efficiency
        # The same projections are used for both content and position attention
        
        # Content-to-position attention (c2p)
        if "c2p" in self.pos_att_type:
            # Content-to-position: query content attends to relative position keys
            pos_key_embeddings = rel_embeddings[relative_pos]  # [seq_len, seq_len, hidden_size]
            
            # Project position embeddings through key projection
            pos_key_flat = pos_key_embeddings.reshape(-1, pos_key_embeddings.size(-1))
            pos_key_proj = self.key_proj(pos_key_flat)[0]  # [seq_len*seq_len, hidden_size]
            pos_key = pos_key_proj.reshape(seq_len, seq_len, self.all_head_size)
            
            # Reshape for multi-head attention: [seq_len, seq_len, num_heads, head_dim]
            pos_key = pos_key.reshape(seq_len, seq_len, self.num_attention_heads, self.attention_head_size)
            pos_key = pos_key.permute(2, 0, 1, 3)  # [heads, seq, seq, head_dim]
            
            # Compute c2p attention: query content @ position key
            c2p_att = torch.einsum('bhiq,hijq->bhij', query_layer, pos_key)
            att_span += c2p_att

        # Position-to-content attention (p2c)  
        if "p2c" in self.pos_att_type:
            # Position-to-content: relative position queries attend to content keys
            # We need relative positions from key to query (transpose of relative_pos)
            relative_pos_t = relative_pos.transpose(0, 1)  # [seq_len, seq_len]
            
            # Extract relative position embeddings for p2c
            pos_query_embeddings = rel_embeddings[relative_pos_t]  # [seq_len, seq_len, hidden_size]
            
            # Project position embeddings through query projection
            pos_query_flat = pos_query_embeddings.reshape(-1, pos_query_embeddings.size(-1))
            pos_query_proj = self.query_proj(pos_query_flat)[0]  # [seq_len*seq_len, hidden_size]
            pos_query = pos_query_proj.reshape(seq_len, seq_len, self.all_head_size)
            
            # Reshape for multi-head attention: [seq_len, seq_len, num_heads, head_dim]
            pos_query = pos_query.reshape(seq_len, seq_len, self.num_attention_heads, self.attention_head_size)
            pos_query = pos_query.permute(2, 0, 1, 3)  # [heads, seq, seq, head_dim]
            
            # Compute p2c attention: position query @ content key
            p2c_att = torch.einsum('hijq,bhjq->bhij', pos_query, key_layer)
            att_span += p2c_att

        return att_span


class DebertaV2Attention(nn.Module):
    """DeBERTa v2 attention layer."""

    def __init__(self, config: DebertaV2Config, layer_idx: int = 0):
        super().__init__()
        self.self = DebertaV2DisentangledSelfAttention(config, layer_idx)
        self.output = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Add LayerNorm to match HuggingFace DeBERTa v2 structure
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rel_embeddings: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            rel_embeddings,
            rel_pos,
        )
        
        attention_output = self.output(self_outputs)[0]
        attention_output = self.dropout(attention_output)
        
        # Apply LayerNorm after residual connection (post-norm architecture)
        attention_output = self.LayerNorm(attention_output + hidden_states)
        
        return attention_output


class DebertaV2Intermediate(nn.Module):
    """DeBERTa v2 intermediate (feed-forward) layer."""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_act_fn(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)[0]  # ColumnParallelLinear returns tuple
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaV2Output(nn.Module):
    """DeBERTa v2 output layer."""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Layer(nn.Module):
    """DeBERTa v2 transformer layer."""

    def __init__(self, config: DebertaV2Config, layer_idx: int = 0):
        super().__init__()
        self.attention = DebertaV2Attention(config, layer_idx)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rel_embeddings: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention (residual connection handled inside attention module)
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            rel_embeddings,
            rel_pos,
        )
        
        # Feed-forward network (residual connection handled inside output module)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output


class DebertaV2Encoder(nn.Module):
    """DeBERTa v2 transformer encoder."""

    def __init__(self, config: DebertaV2Config, cache_config: Optional[CacheConfig] = None):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([
            DebertaV2Layer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Relative position embeddings
        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            # DeBERTa v2 uses 2 * pos_ebd_size for bidirectional relative positions
            rel_embeddings_size = self.pos_ebd_size * 2
            self.rel_embeddings = VocabParallelEmbedding(
                rel_embeddings_size,
                config.hidden_size,
            )

    def get_rel_embedding(self):
        """Get relative position embeddings."""
        if self.relative_attention:
            rel_embeddings = self.rel_embeddings.weight
            return rel_embeddings
        return None

    def get_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Convert attention mask to the format expected by attention layers."""
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = attention_mask
        return extended_attention_mask

    def get_rel_pos(self, hidden_states: torch.Tensor, query_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get relative position matrix."""
        if self.relative_attention and hidden_states.size():
            if hidden_states.dim() == 2:
                # vLLM 2D format: [seq_len, hidden_size]
                seq_len = hidden_states.size(0)
            else:
                # Standard 3D format: [batch_size, seq_len, hidden_size]
                seq_len = hidden_states.size(-2)
            
            q = query_states.size(-2) if query_states is not None else seq_len
            
            rel_pos = build_relative_position(
                q,
                seq_len,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=hidden_states.device,
            )
            return rel_pos
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get relative position information
        rel_embeddings = self.get_rel_embedding()
        rel_pos = self.get_rel_pos(hidden_states)
        
        # Convert attention mask
        if attention_mask is not None:
            attention_mask = self.get_attention_mask(attention_mask)

        # Pass through transformer layers
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                rel_embeddings,
                rel_pos,
            )

        return hidden_states


class DebertaV2ConvLayer(nn.Module):
    """DeBERTa v2 convolution layer for nGiE (nGram Induced Input Encoding)."""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=groups,
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = get_act_fn(self.conv_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply convolution
        hidden_states = hidden_states.transpose(1, 2)  # (batch, hidden, seq)
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)  # (batch, seq, hidden)
        
        # Apply activation, layer norm, and dropout
        hidden_states = self.act(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class DebertaV2Model(nn.Module):
    """DeBERTa v2 model."""

    def __init__(
        self,
        config: DebertaV2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config=None,
    ):
        super().__init__()
        self.config = config
        
        self.embeddings = DebertaV2Embeddings(config)
        
        # Optional convolution layer for nGiE
        self.conv = None
        if getattr(config, "conv_kernel_size", 0) > 0:
            self.conv = DebertaV2ConvLayer(config)
            
        self.encoder = DebertaV2Encoder(config, cache_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # Apply nGiE convolution if available
        if self.conv is not None:
            embedding_output = self.conv(embedding_output)
            
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        return encoder_outputs


class DebertaV2ContextPooler(nn.Module):
    """DeBERTa v2 context pooler for sequence classification."""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
        )
        self.dropout = nn.Dropout(config.pooler_dropout)
        self.act_fn = get_act_fn(config.pooler_hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Handle 2D tensor case for vLLM pooling models
        if hidden_states.dim() == 2:
            # For 2D input [seq_len, hidden_size], take first token
            context_token = hidden_states[0]  # Shape: [hidden_size]
            is_batch = False
        else:
            # For 3D input [batch, seq_len, hidden_size], take first token
            context_token = hidden_states[:, 0]  # Shape: [batch, hidden_size]
            is_batch = True
        
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)[0]  # ColumnParallelLinear returns tuple
        pooled_output = self.act_fn(pooled_output)
        
        return pooled_output


class DebertaV2ForSequenceClassification(nn.Module):
    """DeBERTa v2 model for sequence classification."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        
        self.config = config
        self.num_labels = config.num_labels
        
        self.deberta = DebertaV2Model(config, cache_config, quant_config)
        
        # Pooler and classifier
        self.context_pooler = DebertaV2ContextPooler(config)
        # Use regular Linear layer for classifier to work with ClassifierPooler
        self.classifier = nn.Linear(
            config.hidden_size,
            config.num_labels,
            bias=True,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize pooler for vLLM
        self._pooler = ClassifierPooler(
            vllm_config.model_config,
            self.classifier,
            self.context_pooler,
        )
        

        
        # Don't initialize weights - they will be loaded from checkpoint

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
            hidden_states = self.deberta(input_ids, token_type_ids=token_type_ids)
        
        return hidden_states

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        
        weight_list = list(weights)
        
        for name, loaded_weight in weight_list:
            # DON'T skip the "deberta." prefix - our model parameters include it!
            # The checkpoint already has the correct "deberta." prefix
            # if name.startswith("deberta."):
            #     name = name[len("deberta."):]
            
            # Handle HuggingFace -> vLLM parameter name mapping
            original_name = name
            # Map HuggingFace parameter names to vLLM parameter names
            if '.attention.output.dense.' in name:
                # HF: deberta.encoder.layer.X.attention.output.dense.{weight,bias}
                # vLLM: deberta.encoder.layer.X.attention.output.{weight,bias}
                name = name.replace('.attention.output.dense.', '.attention.output.')
            elif '.attention.output.LayerNorm.' in name:
                # HF: deberta.encoder.layer.X.attention.output.LayerNorm.{weight,bias}
                # vLLM: deberta.encoder.layer.X.attention.LayerNorm.{weight,bias}
                name = name.replace('.attention.output.LayerNorm.', '.attention.LayerNorm.')
            elif '.output.LayerNorm.' in name and '.attention.' not in name:
                # HF: deberta.encoder.layer.X.output.LayerNorm.{weight,bias}
                # vLLM: deberta.encoder.layer.X.output.LayerNorm.{weight,bias} (same)
                pass  # No change needed
            elif 'pooler.dense.' in name:
                name = name.replace('pooler.dense.', 'context_pooler.dense.')
            
            # Skip known unused parameters from HuggingFace models
            skip_params = [
                'cls.predictions.bias', 'cls.predictions.transform.dense.weight', 
                'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 
                'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight', 
                'cls.predictions.decoder.bias'
            ]
            
            if original_name in skip_params:
                continue
                
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            else:
                pass  # Parameter not found - this is expected for some HuggingFace parameters
        
        # Check for missing critical parameters
        critical_params = [
            'encoder.rel_embeddings.weight',
            'context_pooler.dense.weight',
            'classifier.weight',
            'classifier.bias'
        ]
        
        # Verify critical parameters are loaded
        missing_critical = [param for param in critical_params 
                           if param in params_dict and param not in loaded_params]
        if missing_critical:
            raise RuntimeError(f"Critical parameters not loaded: {missing_critical}")

    def _init_weights(self):
        """Initialize weights that might not be loaded correctly."""
        # DISABLED: This was overwriting correctly loaded weights!
        # The weight loading should handle all initialization
        pass


def _get_model_type() -> str:
    return "deberta_v2"
