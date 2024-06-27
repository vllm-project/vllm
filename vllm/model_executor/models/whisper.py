from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import WhisperConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import FastGELU
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.utils import is_hip, print_warning_once

def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """Returns sinusoids for positional embedding"""
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)

class WhisperPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, position_ids=None):
        return self.weight[position_ids]

class WhisperAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        is_decoder: bool = False,
        is_causal: bool = False,
        bias: bool = True,
        config: Optional[WhisperConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = max(1, self.num_heads // tp_size)
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = RowParallelLinear(
            input_size = embed_dim,
            output_size = embed_dim,
            bias = False,
            quant_config=quant_config,
        )
        self.k_proj = RowParallelLinear(
            input_size = embed_dim,
            output_size = embed_dim,
            bias = bias,
            quant_config=quant_config
        )
        self.q_proj = RowParallelLinear(
            input_size = embed_dim,
            output_size = embed_dim,
            bias = bias,
            quant_config=quant_config
        )
        self.out_proj = RowParallelLinear(
            input_size = embed_dim,
            output_size = embed_dim,
            bias = bias,
            quant_config=quant_config
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config
        )

    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states = None,
        past_key_value = None,
        kv_cache: torch.Tensor = None,
        attn_metadata: AttentionMetadata = None,
    ):
        is_cross_attention = encoder_hidden_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        q, _ = self.q_proj(hidden_states) * self.scaling
        
        past_key_value = None
        
        if kv_cache is None:
            q = self._shape(q, tgt_len, bsz)

            if is_cross_attention:
                if past_key_value is not None:
                    k = past_key_value[0]
                    v = past_key_value[1]
                else:
                    k, _ = self.k_proj(encoder_hidden_states)
                    v, _ = self.v_proj(encoder_hidden_states)
                    k = self._shape(k, -1, bsz)
                    v = self._shape(v, -1, bsz)

                    past_key_value = (k, v)
            else:
                k, _ = self.k_proj(key_value_states)
                v, _ = self.v_proj(hidden_states)
                k = self._shape(k, -1, bsz)
                v = self._shape(v, -1, bsz)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=self.is_causal and tgt_len > 1,
            )
            attn_output = attn_output.reshape(bsz, q_len, -1)
            output = self.out_proj(attn_output)
        else:
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

            output, _ = self.o_proj(attn_output)
        return output, past_key_value

class WhisperEncoderLayer(nn.Module):
    def __init__(
        self, 
        config: WhisperConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = FastGELU()
        self.fc1 = RowParallelLinear(
            input_size = self.embed_dim,
            output_size = config.encoder_ffn_dim,
            bias = True,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            input_size = config.encoder_ffn_dim,
            output_size = self.embed_dim,
            bias = True,
            quant_config=quant_config,
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class WhisperDecoderLayer(nn.Module):
    def __init__(
        self, 
        config: WhisperConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn, _ = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            is_decoder=True,
            is_causal=True,
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
        )
        self.activation_fn = FastGELU()

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            is_decoder=True,
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = RowParallelLinear(
            input_size = self.embed_dim,
            output_size = config.decoder_ffn_dim,
            bias = True,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            input_size = config.decoder_ffn_dim,
            output_size = self.embed_dim,
            bias = True,
            quant_config=quant_config,
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata
        )
        hidden_states = residual + hidden_states

        hidden_states, cross_attention_past_key_value = self.self_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

    return outputs, cross_attention_past_key_value

class WhisperEncoder(nn.Module):
    def __init__(
        self, 
        config: WhisperConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config, quant_config=quant_config, cache_config=cache_config)
                                     for layer_idx in range(config.decoder_layers)])
        
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        input_features,
    ):
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states)
        
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

class WhisperDecoder(nn.Module):
    def __init__(
        self, 
        config: WhisperConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)
        self.layers = nn.ModuleList([WhisperDecoderLayer(config, quant_config=quant_config, cache_config=cache_config)
                                     for layer_idx in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        input_ids,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        past_key_values = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        positions = self.embed_positions(input_ids, positions)
        hidden_states = inputs_embeds + positions

        cross_attention_past_key_values = []

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, cross_attention_past_key_value = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=None if past_key_values is None else past_key_values[idx],
                kv_cache=kv_caches[idx],
                output_attentions=output_attentions,
                attn_metadata=attn_metadata
            )
            cross_attention_past_key_values.append(cross_attention_past_key_value)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states, cross_attention_past_key_values

class WhisperModel(nn.Module):
    def __init__(
        self, 
        config: WhisperConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(config)

        self.encoder = WhisperEncoder(config, cache_config=cache_config, quant_config=quant_config)
        self.decoder = WhisperDecoder(config, cache_config=cache_config, quant_config=quant_config)
    
    def forward(
        self,
        input_features: torch.FloatTensor,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        past_key_values = None,
    ):

        encoder_outputs = self.encoder(
            input_features,
        )
        decoder_outputs, cross_attention_past_key_values = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_outputs,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            past_key_values=past_key_values
        )
        return decoder_outputs, cross_attention_past_key_values
        

class WhisperForConditionalGeneration(nn.Module):
    def __init__(
        self, 
        config: WhisperConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(config)
        self.model = WhisperModel(config, cache_config=cache_config, quant_config=quant_config)
        self.proj_out = RowParallelLinear(
            input_size = config.d_model,
            output_size = config.vocab_size,
            bias = False,
            quant_config=quant_config,
        )

    def forward(
        self,
        input_features: torch.FloatTensor,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        past_key_values = None,
    ):
        outputs = self.model(
            input_features=input_features,
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            past_key_values=past_key_values,
        )