from typing import Iterable, List, Literal, Optional, Tuple, TypedDict, Union

import math
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
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.audio import get_dummy_audio_data
from vllm.sequence import SamplerOutput
from vllm.utils import is_hip, print_warning_once
from xformers import ops as xops
from vllm.utils import is_hip

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
        self.v_proj = RowParallelLinear(
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
        if self.is_causal:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config
            )
        else:
            self.attn = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states = None,
        past_key_value = None,
        kv_cache: torch.Tensor = None,
        attn_metadata: AttentionMetadata = None,
        is_cross_attention = False,
    ):
        sizes = hidden_states.size()
        if len(sizes) == 3:
            bsz, tgt_len, _ = sizes
        else:
            tgt_len, _ = sizes
        q, _ = self.q_proj(hidden_states)
        
        past_key_value = None

        if is_cross_attention or not self.is_decoder:
            if is_cross_attention and encoder_hidden_states is not None:
                if past_key_value is not None:
                    k = past_key_value[0]
                    v = past_key_value[1]
                else:
                    k, _ = self.k_proj(encoder_hidden_states)
                    v, _ = self.v_proj(encoder_hidden_states)

                    past_key_value = (k, v)
            else:
                k, _ = self.k_proj(hidden_states)
                v, _ = self.v_proj(hidden_states)

            q = self._shape(q, -1, 1)
            k = self._shape(k, -1, 1)
            v = self._shape(v, -1, 1)

            attn_output = xops.memory_efficient_attention_forward(
                q,
                k,
                v,
                attn_bias=None,
                p=0.0,
                scale=None,
                op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0] if
                (is_hip()) else None,
            )
            
            attn_output = attn_output.reshape(-1, self.embed_dim)

        else:
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        output, _ = self.out_proj(attn_output)

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
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

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
        self.self_attn = WhisperAttention(
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

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata
        )
        hidden_states = residual + hidden_states

        hidden_states, cross_attention_past_key_value = self.self_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
            is_cross_attention=True,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cross_attention_past_key_value

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
                                     for layer_idx in range(config.encoder_layers)])
        
        self.layer_norm = nn.LayerNorm(config.d_model)

        with torch.no_grad():
            self.embed_positions.weight.copy_(sinusoids(*self.embed_positions.weight.shape))
    
    def forward(
        self,
        input_features,
    ):
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(1, 0)
    
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
        super().__init__()

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
        if input_features is not None:
            encoder_outputs = self.encoder(
                input_features[0],
            )
        else:
            encoder_outputs = None
        decoder_outputs, cross_attention_past_key_values = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_outputs,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            past_key_values=past_key_values
        )
        return decoder_outputs, cross_attention_past_key_values

@MULTIMODAL_REGISTRY.register_audio_input()
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_audio_data)
class WhisperForConditionalGeneration(nn.Module):
    def __init__(
        self, 
        config: WhisperConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.model = WhisperModel(config, cache_config=cache_config, quant_config=quant_config)
        self.unpadded_vocab_size = config.vocab_size
        self.proj_out = RowParallelLinear(
            input_size = config.d_model,
            output_size = config.vocab_size,
            bias = False,
            quant_config=quant_config,
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> torch.Tensor:
        input_features = kwargs.pop("input_features", None)

        return input_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        past_key_values = None,
        **kwargs: object,
    ) -> SamplerOutput:

        input_features = self._parse_and_validate_audio_input(**kwargs)

        decoder_outputs, cross_attention_past_key_values = self.model(
            input_features=input_features,
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            past_key_values=past_key_values,
        )
        return decoder_outputs
    
    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.proj_out.weight, hidden_states,
                                       sampling_metadata)
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

            if name == 'model.decoder.embed_tokens.weight':
                param = params_dict['proj_out.weight']
                weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
                weight_loader(param, loaded_weight)

        param = params_dict[name]