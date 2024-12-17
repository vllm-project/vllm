import math
from functools import lru_cache
from typing import Iterable, List, Mapping, Optional, Set, Tuple, Union

import librosa
import numpy as np
import torch
from torch import nn
from transformers import WhisperConfig, WhisperProcessor

from vllm.attention import Attention, AttentionMetadata, AttentionType
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

from .interfaces import SupportsMultiModal
from .utils import AutoWeightsLoader, make_layers, maybe_prefix

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
        attn_type: AttentionType = AttentionType.DECODER,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            # Number of heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_heads % tp_size == 0
        else:
            # Number of heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = attn_type

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self._init_qkv(embed_dim, bias, quant_config, prefix=prefix)
        self.out_proj = RowParallelLinear(
            input_size = embed_dim,
            output_size = embed_dim,
            bias = bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
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

    def _init_qkv(self,
        embed_dim: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                                attn_type=self.attn_type)

        output, _ = self.out_proj(attn_output)

        return output


class WhisperCrossAttention(WhisperAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _init_qkv(self,
        embed_dim: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.q_proj = RowParallelLinear(
            input_size = embed_dim,
            output_size = embed_dim,
            bias = bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.kv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        q, _ = self.q_proj(hidden_states)

        if encoder_hidden_states is not None:
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

        attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                                attn_type=AttentionType.ENCODER_DECODER)

        output, _ = self.out_proj(attn_output)

        return output


class WhisperEncoderLayer(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            attn_type=AttentionType.ENCODER,
            cache_config=cache_config,
            quant_config=quant_config,
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
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
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
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.activation_fn = FastGELU()

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
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
        encoder_hidden_states: Optional[torch.Tensor],
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
            kv_cache=kv_cache,
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
        input_features: Union[torch.Tensor, List[torch.Tensor]],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        hidden_states = []
        for features in input_features:
            embeds = nn.functional.gelu(self.conv1(features))
            embeds = nn.functional.gelu(self.conv2(embeds))
            embeds = embeds.permute(1, 0)
            embeds = embeds + self.embed_positions.weight[:embeds.size(0), :]
            hidden_states.append(embeds)
        hidden_states = torch.cat(hidden_states)

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                kv_cache=kv_caches[idx],
                attn_metadata=attn_metadata,
            )

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
        encoder_hidden_states: Optional[torch.Tensor],
        kv_caches: List[torch.Tensor],
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
                attn_metadata=attn_metadata,
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
        input_features: Optional[torch.FloatTensor],
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if input_features is not None:
            # Prefill encoder kv-caches
            encoder_outputs = self.encoder(
                input_features,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
            )
        else:
            encoder_outputs = None

        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_outputs,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        return decoder_outputs

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


def get_max_whisper_audio_tokens(ctx: InputContext) -> int:
    return ctx.model_config.hf_config.max_source_positions


def dummy_encoder_data_for_whisper(ctx: InputContext, seq_len: int,
                                   mm_counts: Mapping[str, int]):
    assert mm_counts["audio"] == 1
    sample_rate = 16000
    num_tokens = ctx.model_config.hf_config.max_source_positions
    return DummyData(
        SequenceData.from_prompt_token_counts((0, num_tokens)),
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
    multi_modal_data = inputs["encoder"]["multi_modal_data"]
    if isinstance(multi_modal_data["audio"], list):
        assert len(multi_modal_data["audio"]) == 1
        multi_modal_data["audio"] = multi_modal_data["audio"][0]
    # Resample and process audio
    audio, orig_sr = multi_modal_data["audio"]
    processor = get_whisper_processor(ctx.model_config.model)
    target_sr = processor.feature_extractor.sampling_rate
    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    if audio.size > 30 * target_sr:
        # Truncate audio to 30 seconds
        audio = audio[:30 * target_sr]
    multi_modal_data["audio"] = (audio, target_sr)
    # Calculate number of tokens after convolutions
    num_tokens = (audio.size // 80 - 1) // 2 + 1
    num_tokens = (num_tokens - 2) // 2 + 1
    # Pre-allocate placeholder tokens in encoder sequence
    inputs["encoder"]["prompt_token_ids"] = [0] * num_tokens
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
    sampling_rate = processor.feature_extractor.sampling_rate

    audios = [audio for audio, _ in multi_modal_data]

    kwargs = processor(audios, sampling_rate=sampling_rate,
                       padding=False, return_tensors="pt")
    kwargs["input_features"] = kwargs["input_features"].squeeze(0)
    kwargs["input_features"] = kwargs["input_features"].to(torch.float16)

    return MultiModalKwargs(kwargs)


@INPUT_REGISTRY.register_dummy_encoder_data(dummy_encoder_data_for_whisper)
@INPUT_REGISTRY.register_input_processor(input_processor_for_whisper)
@MULTIMODAL_REGISTRY.register_input_mapper("audio", input_mapper_for_whisper)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("audio", get_max_whisper_audio_tokens)
class WhisperForConditionalGeneration(nn.Module, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config

        self.model = WhisperModel(vllm_config=vllm_config, prefix=prefix)
        self.unpadded_vocab_size = config.vocab_size
        self.proj_out = ParallelLMHead(config.vocab_size,
                                       config.d_model,
                                       quant_config=quant_config)
        self.proj_out = self.proj_out.tie_weights(
            self.model.decoder.embed_tokens)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        input_features = kwargs.get("input_features")
        decoder_outputs = self.model(
            input_features=input_features,
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

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["proj_out."])
        return loader.load_weights((name, loaded_weight)
                                   for name, loaded_weight in weights)