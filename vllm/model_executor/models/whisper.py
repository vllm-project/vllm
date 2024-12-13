import math
from functools import lru_cache
from typing import Iterable, List, Literal, Mapping, Optional, Tuple, TypedDict, Union

import librosa
import numpy as np
import torch
from torch import nn
from transformers import WhisperConfig, WhisperProcessor

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import FastGELU
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.utils import consecutive_placeholder_ranges
from vllm.sequence import SequenceData
from vllm.vllm_flash_attn import flash_attn_func
from xformers import ops as xops

from .interfaces import SupportsMultiModal
from .utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)


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
    def __init__(self, num_positions: int, embedding_dim: int,
                 padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)

    def forward(self, position_ids):
        return self.weight[position_ids]


class WhisperAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = max(1, self.num_heads // tp_size)
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = RowParallelLinear(
            input_size = embed_dim,
            output_size = embed_dim,
            bias = False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = RowParallelLinear(
            input_size = embed_dim,
            output_size = embed_dim,
            bias = bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.q_proj = RowParallelLinear(
            input_size = embed_dim,
            output_size = embed_dim,
            bias = bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size = embed_dim,
            output_size = embed_dim,
            bias = bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous()


class WhisperEncoderAttention(WhisperAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=prefix,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        bsz, seq = hidden_states.size()[:2]

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # q = self._shape(q, -1, 1)
        # k = self._shape(k, -1, 1)
        # v = self._shape(v, -1, 1)

        q = q.view(bsz, seq, self.num_heads, self.head_dim)
        k = k.view(bsz, seq, self.num_heads, self.head_dim)
        v = v.view(bsz, seq, self.num_heads, self.head_dim)

        attn_output = xops.memory_efficient_attention_forward(
            q,
            k,
            v,
            attn_bias=None,
            p=0.0,
            scale=None,
            op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0],
        )

        # attn_output = flash_attn_func(
        #     q=q,
        #     k=k,
        #     v=v,
        #     softmax_scale=None,
        #     causal=False,
        #     window_size=(-1, -1),
        #     alibi_slopes=None,
        #     softcap=0,
        # )

        attn_output = attn_output.reshape(bsz, seq, self.embed_dim)
        output, _ = self.out_proj(attn_output)
        return output


class WhisperDecoderAttention(WhisperAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=prefix,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor = None,
        attn_metadata: AttentionMetadata = None,
    ):
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        output, _ = self.out_proj(attn_output)

        return output


class WhisperDecoderCrossAttention(WhisperAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ):
        # HACK
        query_lens = attn_metadata.query_start_loc.diff().tolist()
        hidden_states = list(hidden_states.split(query_lens))
        padded_size = max(query_lens)
        for i in range(len(hidden_states)):
            hidden_states[i] = torch.nn.functional.pad(hidden_states[i], (0, 0, 0, padded_size - hidden_states[i].size(0)))
        hidden_states = torch.stack(hidden_states, dim=0)

        bsz, seq = hidden_states.size()[:2]
        bsz2, seq2 = encoder_hidden_states.size()[:2]

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(encoder_hidden_states)
        v, _ = self.v_proj(encoder_hidden_states)

        q = q.view(bsz, seq, self.num_heads, self.head_dim)
        k = k.view(bsz2, seq2, self.num_heads, self.head_dim)
        v = v.view(bsz2, seq2, self.num_heads, self.head_dim)

        attn_output = xops.memory_efficient_attention_forward(
            q,
            k,
            v,
            attn_bias=None,
            p=0.0,
            scale=None,
            op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0],
        )

        # HACK
        attn_output = list(torch.unbind(attn_output))
        for i in range(len(attn_output)):
            attn_output[i] = attn_output[i][:query_lens[i], :]
        attn_output = torch.cat(attn_output, dim=0)

        attn_output = attn_output.reshape(-1, self.embed_dim)
        output, _ = self.out_proj(attn_output)
        return output


class WhisperEncoderLayer(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.embed_dim = config.d_model
        self.self_attn = WhisperEncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = FastGELU()
        self.fc1 = RowParallelLinear(
            input_size = self.embed_dim,
            output_size = config.encoder_ffn_dim,
            bias = True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            input_size = config.encoder_ffn_dim,
            output_size = self.embed_dim,
            bias = True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.embed_dim = config.d_model
        self.self_attn = WhisperDecoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.activation_fn = FastGELU()

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperDecoderCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = RowParallelLinear(
            input_size = self.embed_dim,
            output_size = config.decoder_ffn_dim,
            bias = True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            input_size = config.decoder_ffn_dim,
            output_size = self.embed_dim,
            bias = True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
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

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoder(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.encoder_layers,
            lambda prefix: WhisperEncoderLayer(vllm_config=vllm_config,
                                               prefix=f"{prefix}.layers"),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

        with torch.no_grad():
            self.embed_positions.weight.copy_(sinusoids(*self.embed_positions.weight.shape))
    
    def forward(
        self,
        input_features,
    ):
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
    
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperDecoder(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.decoder_layers,
            lambda prefix: WhisperDecoderLayer(vllm_config=vllm_config,
                                               prefix=f"{prefix}.layers"),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        input_ids,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        kv_caches: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        positions = self.embed_positions(positions)
        hidden_states = inputs_embeds + positions

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                kv_cache=kv_caches[idx],
                attn_metadata=attn_metadata
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = WhisperEncoder(vllm_config=vllm_config,
                                      prefix=f"{prefix}.encoder")
        self.decoder = WhisperDecoder(vllm_config=vllm_config,
                                      prefix=f"{prefix}.decoder")

    def forward(
        self,
        input_features: torch.FloatTensor,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor: 
        encoder_outputs = self.encoder(input_features)

        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_outputs,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        return decoder_outputs


def get_max_whisper_audio_tokens(ctx: InputContext) -> int:
    return ctx.model_config.hf_config.max_source_positions


def dummy_encoder_data_for_whisper(ctx: InputContext, seq_len: int,
                                   mm_counts: Mapping[str, int]):
    assert mm_counts["audio"] == 1
    sample_rate = 16000
    return DummyData(
        SequenceData.from_prompt_token_counts((0, seq_len)),
        {"audio": [(np.zeros(30 * sample_rate), sample_rate)]},
    )


@lru_cache
def get_whisper_processor(
    processor_name: str,
    *args,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    **kwargs,
) -> WhisperProcessor:
    """Gets an whisper processor for the given model name via HuggingFace."""
    try:
        processor: WhisperProcessor = WhisperProcessor.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs)
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoImageProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the whisper processor. If the whisper processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return processor


def input_processor_for_whisper(ctx: InputContext, inputs: DecoderOnlyInputs) -> DecoderOnlyInputs:
    return inputs


def input_mapper_for_whisper(
    ctx: InputContext,
    multi_modal_data: Union[np.ndarray, List[np.ndarray]],
) -> MultiModalKwargs:
    if not isinstance(multi_modal_data, list):
        multi_modal_data = [multi_modal_data]

    assert len(multi_modal_data) == 1

    if len(multi_modal_data) == 0:
        return MultiModalKwargs()

    processor = get_whisper_processor(ctx.model_config.model)
    target_sampling_rate = processor.feature_extractor.sampling_rate

    resampled_audios = [
        librosa.resample(audio, orig_sr=sampling_rate,
                         target_sr=target_sampling_rate)
        for audio, sampling_rate in multi_modal_data
    ]

    kwargs = processor(resampled_audios, sampling_rate=target_sampling_rate,
                       return_tensors="pt")
    kwargs["input_features"] = kwargs["input_features"].squeeze(0)

    return MultiModalKwargs(kwargs)


#@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("audio", get_max_whisper_audio_tokens)
@INPUT_REGISTRY.register_dummy_encoder_data(dummy_encoder_data_for_whisper)
@INPUT_REGISTRY.register_input_processor(input_processor_for_whisper)
@MULTIMODAL_REGISTRY.register_input_mapper("audio", input_mapper_for_whisper)
class WhisperForConditionalGeneration(nn.Module, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config

        self.model = WhisperModel(vllm_config=vllm_config, prefix=prefix)
        self.unpadded_vocab_size = config.vocab_size
        # self.proj_out = RowParallelLinear(
        #     input_size = config.d_model,
        #     output_size = config.vocab_size,
        #     bias = False,
        #     quant_config=quant_config,
        # )
        self.proj_out = ParallelLMHead(config.vocab_size,
                                          config.d_model,
                                          quant_config=quant_config)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        #whisper_data: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:        
        decoder_outputs = self.model(
            input_features=kwargs["input_features"].to(torch.float16),
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        return decoder_outputs
    
    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.proj_out, hidden_states,
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
