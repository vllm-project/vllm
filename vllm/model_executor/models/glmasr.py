# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias, cast

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.glmasr import GlmAsrConfig, GlmAsrProcessor
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.inputs.data import PromptType
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .glmasr_utils import (
    DEFAULT_CONV_PARAMS,
    DEFAULT_MAX_AUDIO_LEN_S,
    DEFAULT_MERGE_FACTOR,
    _flatten_audio_features_by_length,
    _get_audio_output_lengths_for_tower,
    _group_audio_embeddings,
    _normalize_chunk_counts,
    _repeat_kv,
)
from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from .whisper import ISO639_1_SUPPORTED_LANGS


# Optimized vLLM Native GlmAsrEncoder Implementation
class GlmAsrRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for GLM-ASR encoder.

    Pre-computes cos/sin cache and uses vLLM's ApplyRotaryEmb CustomOp
    for efficient rotary embedding application.
    """

    def __init__(self, config, device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings

        # Compute inverse frequencies following transformers implementation
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        # Handle rope_parameters if present (for compatibility with transformers config)
        if hasattr(config, "rope_parameters") and config.rope_parameters:
            base = config.rope_parameters.get("rope_theta", 10000.0)
            partial_rotary_factor = config.rope_parameters.get(
                "partial_rotary_factor", 1.0
            )
            dim = int(head_dim * partial_rotary_factor)
            self.attention_scaling = config.rope_parameters.get(
                "attention_scaling", 1.0
            )
        else:
            base = getattr(config, "rope_theta", 10000.0)
            dim = head_dim
            self.attention_scaling = 1.0

        self.dim = dim
        self.head_dim = head_dim
        self.base = base

        # Compute the inverse frequencies exactly as transformers does
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache for efficiency
        self._set_cos_sin_cache(self.max_seq_len_cached, device)

        # Use vLLM's ApplyRotaryEmb CustomOp for efficient rotary embedding
        # enforce_enable=True ensures the op is always enabled (important for ViT)
        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True,
            is_neox_style=True,
        )

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device | None = None
    ) -> None:
        """Pre-compute cos and sin cache for given sequence length."""
        self.max_seq_len_cached = seq_len

        # Create position indices
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Compute frequencies: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))

        # Compute and cache cos/sin (shape: [seq_len, dim/2])
        # ApplyRotaryEmb expects cos/sin with shape [seq_len, rotary_dim/2]
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def get_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin tensors for a given sequence length.

        Args:
            seq_len: The sequence length to get embeddings for.

        Returns:
            Tuple of (cos, sin) tensors with shape [seq_len, dim/2]
        """
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=self.cos_cached.device)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class GlmAsrAttention(nn.Module):
    """
    Optimized Multi-headed Grouped Query Attention for GLM-ASR.
    Uses vLLM's QKVParallelLinear and ApplyRotaryEmb for better performance.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_rank = self.num_heads // self.tp_size
        self.num_kv_heads_per_rank = max(1, self.num_kv_heads // self.tp_size)

        # Use QKVParallelLinear for fused QKV projection
        # Note: GLM-ASR uses bias on Q and V, but not K
        # For simplicity with QKVParallelLinear, we use bias=True for all
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Use vLLM's ApplyRotaryEmb CustomOp
        # enforce_enable=True ensures the op is always enabled (important for ViT)
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            cos: [seq_len, rotary_dim/2] - cosine part of rotary embeddings
            sin: [seq_len, rotary_dim/2] - sine part of rotary embeddings

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection - fused for efficiency
        qkv, _ = self.qkv_proj(hidden_states)

        # Split into q, k, v
        q_size = self.num_heads_per_rank * self.head_dim
        kv_size = self.num_kv_heads_per_rank * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape to [batch, seq, num_heads, head_dim] for ApplyRotaryEmb
        q = q.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads_per_rank, self.head_dim)

        # Apply rotary position embeddings using vLLM's ApplyRotaryEmb
        # ApplyRotaryEmb expects x: [batch, seq, heads, head_dim]
        # cos/sin: [seq_len, rotary_dim/2]
        q = self.apply_rotary_emb(q, cos, sin)
        k = self.apply_rotary_emb(k, cos, sin)

        # Transpose to [batch, num_heads, seq, head_dim] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle GQA: repeat k/v if needed
        if self.num_kv_groups > 1:
            k = _repeat_kv(k, self.num_kv_groups)
            v = _repeat_kv(v, self.num_kv_groups)

        # Ensure contiguous for optimal SDPA/Flash Attention performance
        # Non-contiguous tensors can cause fallback to slower implementations
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class GlmAsrMLP(nn.Module):
    """
    Optimized MLP for GLM-ASR encoder.
    Uses vLLM's parallel linear layers for better performance.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )

        self.act_fn = get_act_fn(config.hidden_act)

        self.fc2 = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class GlmAsrEncoderLayer(nn.Module):
    """
    Optimized Transformer encoder layer for GLM-ASR.
    Combines attention and MLP with residual connections and layer norms.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GlmAsrAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = GlmAsrMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        layer_norm_eps = getattr(config, "layer_norm_eps", 1e-5)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, eps=layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            cos: [seq_len, rotary_dim/2] - cosine part of rotary embeddings
            sin: [seq_len, rotary_dim/2] - sine part of rotary embeddings

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class _GlmAsrEncoderOutput:
    """Simple output container compatible with transformers' BaseModelOutput."""

    __slots__ = ("last_hidden_state",)

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class GlmAsrEncoder(nn.Module):
    """
    Optimized GLM-ASR Audio Encoder with vLLM native implementation.

    This encoder processes audio features through convolutional layers
    followed by transformer layers with rotary position embeddings.
    Optimized for performance with:
    - QKVParallelLinear for fused attention projections
    - Tensor parallelism support via ColumnParallelLinear/RowParallelLinear
    - Quantization support
    - Flash Attention (SDPA)
    """

    # Mapping for weight loading: transformers uses separate q/k/v, we use fused qkv
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        # Convolutional feature extraction layers
        self.conv1 = nn.Conv1d(
            config.num_mel_bins,
            config.hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                GlmAsrEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final layer norm
        layer_norm_eps = getattr(config, "layer_norm_eps", 1e-5)
        self.norm = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)

        # Rotary position embeddings
        self.rotary_emb = GlmAsrRotaryEmbedding(config)

    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the output length after convolutions.

        Args:
            input_lengths: Input sequence lengths [batch_size]

        Returns:
            Tuple of (output after conv1, output after conv2)
        """
        # Conv1: kernel=3, stride=1, padding=1
        output_lengths = (input_lengths + 2 * 1 - 3) // 1 + 1

        # Conv2: kernel=3, stride=2, padding=1
        output_lengths = (output_lengths + 2 * 1 - 3) // 2 + 1

        return input_lengths, output_lengths

    def forward(self, input_features: torch.Tensor):
        """
        Forward pass through the encoder.

        Args:
            input_features: [batch_size, num_mel_bins, seq_len]

        Returns:
            _GlmAsrEncoderOutput: Object with .last_hidden_state attribute \
                containing [batch_size, seq_len', hidden_size] where seq_len' \
                is the sequence length after convolutions
        """
        # Apply convolutional layers with GELU activation
        hidden_states = torch.nn.functional.gelu(self.conv1(input_features))
        hidden_states = torch.nn.functional.gelu(self.conv2(hidden_states))

        # Transpose to [batch_size, seq_len, hidden_size]
        hidden_states = hidden_states.transpose(1, 2)
        output_seq_len = hidden_states.shape[1]

        # Get cos/sin from rotary embedding cache
        cos, sin = self.rotary_emb.get_cos_sin(output_seq_len)
        # Match dtype with hidden states
        cos = cos.to(dtype=hidden_states.dtype)
        sin = sin.to(dtype=hidden_states.dtype)

        # Apply transformer layers
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cos, sin)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Return in a format compatible with transformers' BaseModelOutput
        return _GlmAsrEncoderOutput(last_hidden_state=hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Custom weight loading to handle q_proj/k_proj/v_proj -> qkv_proj mapping."""
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

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
                # Default weight loading for non-stacked params
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class GlmAsrFeatureInputs(TensorSchema):
    """
    Dimensions:
        - num_chunks: Number of audio chunks (flattened)
        - nmb: Number of mel bins
        - num_audios: Number of original audio files
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("num_chunks", "nmb", "chunk_length", dynamic_dims={"chunk_length"}),
    ]
    feature_attention_mask: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("num_chunks", "chunk_length", dynamic_dims={"chunk_length"}),
    ]
    chunk_counts: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("num_audios"),
    ]


class GlmAsrEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size
        - naf: Number of audio features
        - hs: Hidden size (must match the hidden size of language model
          backbone)
    """

    type: Literal["audio_embeds"] = "audio_embeds"
    audio_embeds: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "naf", "hs", dynamic_dims={"naf"}),
    ]


GlmAsrInputs: TypeAlias = GlmAsrFeatureInputs | GlmAsrEmbeddingInputs


class GlmAsrMultiModalProjector(nn.Module):
    def __init__(
        self,
        config: GlmAsrConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            input_size=config.audio_config.intermediate_size,
            output_size=config.text_config.hidden_size * 2,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            input_size=config.text_config.hidden_size * 2,
            output_size=config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class GlmAsrProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> GlmAsrConfig:
        return self.ctx.get_hf_config(GlmAsrConfig)

    def get_hf_processor(self, **kwargs: object) -> GlmAsrProcessor:
        return self.ctx.get_hf_processor(GlmAsrProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class GlmAsrDummyInputsBuilder(BaseDummyInputsBuilder[GlmAsrProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        hf_processor = self.info.get_hf_processor()
        return hf_processor.audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None

        max_audio_len = getattr(
            self.info.get_hf_processor(), "max_audio_len", DEFAULT_MAX_AUDIO_LEN_S
        )
        audio_len = int(max_audio_len * sampling_rate)

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


def _glmasr_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    chunk_counts = hf_inputs.get("chunk_counts")
    if chunk_counts is not None:
        return dict(
            audio_embeds=MultiModalFieldConfig.batched("audio"),
            input_features=MultiModalFieldConfig.flat_from_sizes(
                "audio", chunk_counts, dim=0
            ),
            feature_attention_mask=MultiModalFieldConfig.flat_from_sizes(
                "audio", chunk_counts, dim=0
            ),
            chunk_counts=MultiModalFieldConfig.batched("audio"),
        )
    return dict(
        audio_embeds=MultiModalFieldConfig.batched("audio"),
        input_features=MultiModalFieldConfig.batched("audio"),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        chunk_counts=MultiModalFieldConfig.batched("audio"),
    )


class GlmAsrMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[Any],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_glmasr_field_config,
            )
        return super()._parse_audio_data(data)


class GlmAsrMultiModalProcessor(BaseMultiModalProcessor["GlmAsrProcessingInfo"]):
    """
    GLM-ASR processor that inherits directly from BaseMultiModalProcessor
    for better performance and cleaner implementation.
    """

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return GlmAsrMultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _calculate_chunk_counts(
        self,
        audio_list: list[Any],
        feature_extractor: WhisperFeatureExtractor,
        processor: GlmAsrProcessor,
    ) -> list[int]:
        """Calculate chunk counts for each audio."""
        sampling_rate = feature_extractor.sampling_rate
        chunk_length = feature_extractor.chunk_length
        max_audio_len = getattr(processor, "max_audio_len", DEFAULT_MAX_AUDIO_LEN_S)
        window_size = int(sampling_rate * chunk_length)
        max_windows = int(max_audio_len // chunk_length)

        chunk_counts = []
        for audio in audio_list:
            n_samples = len(audio) if isinstance(audio, list) else audio.shape[0]
            n_chunks = max(1, (n_samples + window_size - 1) // window_size)
            chunk_counts.append(min(n_chunks, max_windows))
        return chunk_counts

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: dict[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Normalize input: handle deprecated key and list conversion.
        if "audios" in mm_data:
            mm_data["audio"] = mm_data.pop("audios")

        audio = mm_data.get("audio", [])
        audio_list = [audio] if audio and not isinstance(audio, list) else audio

        # Early return for text-only.
        if not audio_list:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # Handle sampling_rate
        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )

        # Call parent method
        outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        # Postprocess: rename mask and add chunk counts
        # Handle different key names from different transformers versions
        if "input_feature_mask" in outputs:
            outputs["feature_attention_mask"] = outputs.pop("input_feature_mask")
        elif "feature_attention_mask" not in outputs and "input_features" in outputs:
            # If no mask is provided, create one from input_features
            input_features = outputs["input_features"]
            if isinstance(input_features, torch.Tensor):
                # Create a mask of all ones matching the sequence length
                mask = torch.ones(
                    input_features.shape[0],
                    input_features.shape[-1],
                    dtype=torch.long,
                )
                outputs["feature_attention_mask"] = mask

        # Get processor for chunk counts calculation
        processor = self.info.get_hf_processor(**mm_kwargs)

        # Override chunk counts calculation with GLM-ASR specific logic
        chunk_counts = self._calculate_chunk_counts(
            audio_list, processor.feature_extractor, processor
        )
        outputs["chunk_counts"] = torch.tensor(chunk_counts, dtype=torch.long)

        return outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _glmasr_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        config = self.info.get_hf_config()

        audio_token = getattr(processor, "audio_token", "<|pad|>")
        audio_token_id = vocab.get(audio_token)
        if audio_token_id is None:
            audio_token_id = processor.audio_token_id

        merge_factor = getattr(config, "merge_factor", DEFAULT_MERGE_FACTOR)
        conv_params = getattr(config, "conv_params", DEFAULT_CONV_PARAMS)
        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        chunk_counts = out_mm_data.get("chunk_counts")

        # Pre-compute audio output lengths if feature_attention_mask is available
        audio_output_lengths: list[int] = []
        if feature_attention_mask is not None:
            # Compute output lengths for all audio items
            from .glmasr_utils import (
                _as_list_chunk_counts,
                _get_audio_output_lengths_from_mask,
            )

            if chunk_counts is not None:
                counts_list = _as_list_chunk_counts(chunk_counts)
                start_idx = 0
                for count in counts_list:
                    end_idx = start_idx + count
                    if isinstance(feature_attention_mask, torch.Tensor):
                        mask = feature_attention_mask[start_idx:end_idx]
                    else:
                        mask = feature_attention_mask[start_idx:end_idx]
                        if isinstance(mask, list):
                            mask = torch.stack(mask)
                    lengths = _get_audio_output_lengths_from_mask(
                        mask, merge_factor, conv_params
                    )
                    audio_output_lengths.append(int(lengths.sum().item()))
                    start_idx = end_idx
            else:
                # Single chunk per audio
                for idx in range(len(feature_attention_mask)):
                    if isinstance(feature_attention_mask, torch.Tensor):
                        mask = feature_attention_mask[idx : idx + 1]
                    else:
                        mask = feature_attention_mask[idx]
                        if not isinstance(mask, torch.Tensor):
                            mask = torch.tensor(mask)
                        mask = mask.unsqueeze(0)
                    lengths = _get_audio_output_lengths_from_mask(
                        mask, merge_factor, conv_params
                    )
                    audio_output_lengths.append(int(lengths.sum().item()))

        def get_replacement_glmasr(item_idx: int):
            # Use pre-computed lengths if available, otherwise fall back to audio_embeds
            if audio_output_lengths:
                num_features = audio_output_lengths[item_idx]
            else:
                audio_embeds = out_mm_data.get("audio_embeds")
                if audio_embeds is not None:
                    embed = audio_embeds[item_idx]
                    num_features = embed.shape[0]
                else:
                    raise ValueError(
                        "Either feature_attention_mask or audio_embeds must be provided"
                    )

            if num_features == 0:
                raise ValueError("Audio is too short")

            audio_tokens = [audio_token_id] * int(num_features)
            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_glmasr,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    GlmAsrMultiModalProcessor,
    info=GlmAsrProcessingInfo,
    dummy_inputs=GlmAsrDummyInputsBuilder,
)
class GlmAsrForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, SupportsTranscription
):
    supported_languages = ISO639_1_SUPPORTED_LANGS

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        # Use optimized vLLM native encoder
        self.audio_tower = GlmAsrEncoder(
            config.audio_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "audio_tower"),
        )
        self.multi_modal_projector = GlmAsrMultiModalProjector(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )
        self.quant_config = quant_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["LlamaForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|begin_of_audio|><|pad|><|end_of_audio|>"

        raise ValueError("Only audio modality is supported")

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="multi_modal_projector.",
            tower_model="audio_tower.",
        )

    def _parse_and_validate_audio_input(self, **kwargs: object) -> GlmAsrInputs | None:
        audio_embeds = kwargs.pop("audio_embeds", None)
        if audio_embeds is not None:
            return GlmAsrEmbeddingInputs(type="audio_embeds", audio_embeds=audio_embeds)

        input_features = kwargs.pop("input_features", None)
        if input_features is None:
            return None

        return GlmAsrFeatureInputs(
            type="audio_features",
            input_features=input_features,
            feature_attention_mask=kwargs.pop("feature_attention_mask", None),
            chunk_counts=kwargs.pop("chunk_counts", None),
        )

    def _process_audio_input(
        self, audio_input: GlmAsrInputs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return tuple(audio_input["audio_embeds"])

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]

        if isinstance(input_features, list):
            input_features = torch.cat(input_features, dim=0)
            feature_attention_mask = torch.cat(feature_attention_mask, dim=0)

        num_chunks = input_features.shape[0]
        chunk_counts = _normalize_chunk_counts(
            audio_input.get("chunk_counts"), num_chunks=num_chunks
        )

        # Convert input_features to model dtype (e.g., bfloat16) to match model weights
        input_features = input_features.to(dtype=self.audio_tower.conv1.weight.dtype)

        # audio_tower returns [batch_size, seq_len, hidden_size] where hidden_size=1280
        audio_hidden_states = self.audio_tower(input_features).last_hidden_state

        # GLM-ASR merges consecutive frames: 4 frames with hidden_size=1280
        # -> 1 frame with intermediate_size=5120
        hidden_size = self.config.audio_config.hidden_size
        intermediate_size = self.config.audio_config.intermediate_size
        merge_ratio = intermediate_size // hidden_size

        # Truncate sequence length to be divisible by merge_ratio
        seq_len = audio_hidden_states.shape[1]
        seq_len_truncated = (seq_len // merge_ratio) * merge_ratio
        if seq_len_truncated < seq_len:
            audio_hidden_states = audio_hidden_states[:, :seq_len_truncated, :]

        # Reshape to merge consecutive frames
        audio_hidden_states = audio_hidden_states.reshape(
            num_chunks,
            -1,
            intermediate_size,
        )

        audio_features = self.multi_modal_projector(audio_hidden_states)

        merge_factor = getattr(self.config, "merge_factor", DEFAULT_MERGE_FACTOR)
        conv_params = getattr(self.config, "conv_params", DEFAULT_CONV_PARAMS)

        audio_output_lengths = _get_audio_output_lengths_for_tower(
            self.audio_tower,
            feature_attention_mask.sum(-1),
            merge_factor,
            conv_params,
        )

        masked_audio_features = _flatten_audio_features_by_length(
            audio_features, audio_output_lengths
        )

        chunk_embeddings = torch.split(
            masked_audio_features, audio_output_lengths.flatten().tolist()
        )
        result = _group_audio_embeddings(chunk_embeddings, chunk_counts)

        return result

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        masked_audio_features = self._process_audio_input(audio_input)

        return masked_audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["audio_tower.embed_positions"]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)

    @classmethod
    def _get_audio_token(cls, model_config: ModelConfig) -> str:
        """Get the audio token from processor.

        Similar to get_placeholder_str but returns single token.
        """
        processor = cached_processor_from_config(model_config)
        return getattr(processor, "audio_token", "<|pad|>")

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        max_audio_clip_s = getattr(processor, "max_audio_len", DEFAULT_MAX_AUDIO_LEN_S)
        return SpeechToTextConfig(
            max_audio_clip_s=max_audio_clip_s,
            sample_rate=feature_extractor.sampling_rate,
        )

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        """Get the generation prompt to be used for transcription requests."""
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_token = cls._get_audio_token(model_config)

        if task_type == "translate":
            full_lang_name_to = cls.supported_languages.get(to_language, to_language)
            user_content = f"{audio_token}translate the speech to {full_lang_name_to}"
        elif task_type == "transcribe":
            user_content = (
                f"{audio_token}can you transcribe the speech into a written format?"
            )
        else:
            raise ValueError(f"Unsupported task type {task_type}")

        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompt_token_ids = tokenizer.encode(prompt)
        prompt_dict = {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"audio": audio},
        }
        return cast(PromptType, prompt_dict)
