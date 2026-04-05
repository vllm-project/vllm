# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers import WhisperConfig as HFWhisperConfig
from transformers.activations import ACT2FN
from transformers.models.whisper.modeling_whisper import sinusoids

try:
    from flash_attn import flash_attn_func
except ModuleNotFoundError:
    flash_attn_func = None

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.inputs import PromptType, TokensPrompt
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.kimi_audio_prompt import KimiAudioPromptBuilder
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.whisper_utils import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import (
    AudioItem,
    DictEmbeddingItems,
    ModalityData,
    ModalityDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    ProcessorInputs,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.kimi_audio import KimiAudioTokenizer
from vllm.transformers_utils.processor import cached_feature_extractor_from_config
from vllm.transformers_utils.processors.kimi_audio import KimiAudioProcessor
from vllm.transformers_utils.processors.kimi_audio_speech import (
    cached_get_kimi_audio_speech_tokenizer,
)
from vllm.v1.sample.metadata import SamplingMetadata

# Kimi-Audio constants
KIMIA_WHISPER_SUBFOLDER = "whisper-large-v3"


def _get_kimi_audio_token_length(sample_length: int) -> int:
    if sample_length <= 0:
        return 0
    return (sample_length - 1) // (160 * 8) + 1


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Compute output lengths after Whisper feature extraction.

    Whisper processes audio through multiple conv layers with stride=2,
    producing 13 output features per 100 input samples.
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths


class KimiAudioWhisperAttention(nn.Module):
    def __init__(self, config: HFWhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        key_states = self.k_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        if flash_attn_func is not None:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=0.0,
                softmax_scale=self.scaling,
                causal=False,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            ).transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class KimiAudioWhisperEncoderLayer(nn.Module):
    def __init__(self, config: HFWhisperConfig):
        super().__init__()
        self.self_attn = KimiAudioWhisperAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class KimiAudioWhisperEncoder(nn.Module):
    """Kimi-Audio specific Whisper encoder aligned to the official custom code."""

    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", init_in_fp32: bool = False
    ):
        super().__init__()
        model_path = vllm_config.model_config.model
        whisper_config = HFWhisperConfig.from_pretrained(
            model_path,
            subfolder=KIMIA_WHISPER_SUBFOLDER,
        )
        self.config = whisper_config
        embed_dim = whisper_config.d_model
        self.num_mel_bins = whisper_config.num_mel_bins
        self.max_source_positions = whisper_config.max_source_positions
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        with torch.no_grad():
            self.embed_positions.weight.copy_(
                sinusoids(*self.embed_positions.weight.shape)
            )
        self.layers = nn.ModuleList(
            [
                KimiAudioWhisperEncoderLayer(whisper_config)
                for _ in range(whisper_config.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(whisper_config.d_model)

    def forward(
        self,
        input_features: torch.Tensor | list[torch.Tensor],
        input_lengths: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(input_lengths, torch.Tensor):
            normalized_lengths = [int(length) for length in input_lengths.tolist()]
        elif input_lengths is None:
            normalized_lengths = []
        else:
            normalized_lengths = [int(length) for length in input_lengths]

        hidden_states = []
        input_is_batched = False
        if isinstance(input_features, torch.Tensor):
            if input_features.dim() == 3:
                feature_iter = list(input_features.unbind(dim=0))
            elif input_features.dim() == 2:
                feature_iter = [input_features]
            else:
                raise ValueError(
                    "input_features must be 2D or 3D, "
                    f"but got shape {tuple(input_features.shape)}."
                )
        else:
            feature_iter = list(input_features)
        for idx, features in enumerate(feature_iter):
            embeds = nn.functional.gelu(self.conv1(features))
            embeds = nn.functional.gelu(self.conv2(embeds))
            embeds = embeds.transpose(-1, -2)

            target_length = embeds.shape[-2]
            if normalized_lengths and idx < len(normalized_lengths):
                target_length = min(target_length, max(0, normalized_lengths[idx]))
            embeds = embeds[:, :target_length, :]

            embeds = (
                embeds + self.embed_positions.weight[: embeds.size(-2), :]
            ).to(embeds.dtype)

            hidden_states.append(embeds)
            input_is_batched = embeds.ndim > 2

        if input_is_batched:
            hidden_states = torch.cat(hidden_states)
        else:
            hidden_states = torch.stack(hidden_states, dim=0)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


# -----------------------------------------------------------------------------
# Processing Info, Dummy Inputs, and MultiModal Processor
# (Following Qwen3ASR pattern - same file as model)
# -----------------------------------------------------------------------------


class KimiAudioProcessingInfo(BaseProcessingInfo):
    """Processing info for vLLM registry."""

    def get_hf_processor(self, **kwargs: object) -> KimiAudioProcessor:
        feature_extractor = cached_feature_extractor_from_config(
            self.ctx.model_config,
            subfolder=KIMIA_WHISPER_SUBFOLDER,
        )

        return KimiAudioProcessor(
            feature_extractor=feature_extractor,
            tokenizer=self.get_tokenizer(),
            speech_tokenizer=cached_get_kimi_audio_speech_tokenizer(
                getattr(self.ctx.model_config.hf_config, "kimia_token_offset", 152064),
            ),
        )

    def get_feature_extractor(self, **kwargs: object):
        return cached_feature_extractor_from_config(
            self.ctx.model_config, subfolder=KIMIA_WHISPER_SUBFOLDER
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self) -> "KimiAudioMultiModalDataParser":
        feature_extractor = self.get_feature_extractor()
        return KimiAudioMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class KimiAudioDummyInputsBuilder(BaseDummyInputsBuilder[KimiAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        num_audios = mm_counts.get("audio", 0)
        if num_audios == 0:
            return {}

        feature_extractor = self.info.get_feature_extractor()
        target_audio_length = (
            min(feature_extractor.chunk_length, 30) * feature_extractor.sampling_rate
        )

        return {
            "audio": self._get_dummy_audios(
                length=target_audio_length, num_audios=num_audios
            ),
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> ProcessorInputs:
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        dummy_mm_items = self.info.parse_mm_data(dummy_mm_data)

        num_audios = mm_counts.get("audio", 0)
        dummy_tokens = (
            [198]
            if num_audios == 0
            else [
                KimiAudioProcessor.KIMIA_MEDIA_BEGIN,
                KimiAudioProcessor.KIMIA_TEXT_BLANK,
                KimiAudioProcessor.KIMIA_MEDIA_END,
                KimiAudioProcessor.KIMIA_SPEECH_CT_ID,
            ]
            * num_audios
        )

        return ProcessorInputs(prompt=dummy_tokens, mm_data_items=dummy_mm_items)


# Field config for Kimi-Audio multimodal data
_KIMIAUDIO_FIELD_CONFIG = {
    "whisper_input_features": MultiModalFieldConfig.batched("audio"),
    "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
    "audio_sample_lengths": MultiModalFieldConfig.batched("audio"),
    "speech_token_ids": MultiModalFieldConfig.batched("audio"),
    "speech_attention_mask": MultiModalFieldConfig.batched("audio"),
    "audio_token_ids": MultiModalFieldConfig.batched("audio"),
    "text_token_ids": MultiModalFieldConfig.batched("audio"),
    "is_continuous_mask": MultiModalFieldConfig.batched("audio"),
}


class KimiAudioMultiModalDataParser(MultiModalDataParser):
    """Custom data parser for Kimi-Audio multimodal data."""

    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"whisper_input_features", "feature_attention_mask"},
                fields_factory=lambda hf_inputs: _KIMIAUDIO_FIELD_CONFIG,
            )

        return super()._parse_audio_data(data)


class KimiAudioMultiModalProcessor(BaseMultiModalProcessor[KimiAudioProcessingInfo]):
    """vLLM multi-modal processor wrapper for Kimi-Audio."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Call the HuggingFace processor."""
        # Convert mm_data format: {'audios': [...]} -> {'audio': ...}
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])
        processor_kwargs = dict(**mm_kwargs, **tok_kwargs)

        # Convert audio format: [(array, sr), ...] -> [array, ...]
        # KimiAudioProcessor expects raw numpy arrays
        if audios:
            audio_arrays = []
            for aud in audios:
                if isinstance(aud, (tuple, list)) and len(aud) == 2:
                    # Format: (audio_array, sampling_rate)
                    audio_arrays.append(aud[0])
                elif isinstance(aud, np.ndarray):
                    audio_arrays.append(aud)
                else:
                    audio_arrays.append(aud)
            mm_data["audio"] = audio_arrays

        if (
            processor_kwargs.get("messages") is not None
            and "return_packed_kimi_tokens" not in processor_kwargs
        ):
            processor_kwargs["return_packed_kimi_tokens"] = True

        # Use the context's call_hf_processor for proper handling
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            processor_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, Any]:
        """Get multi-modal field configuration."""
        return _KIMIAUDIO_FIELD_CONFIG

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ) -> Sequence[PromptReplacement]:
        """Get prompt updates for audio tokens."""
        out_mm_data = out_mm_kwargs.get_data()
        speech_attention_mask = out_mm_data.get("speech_attention_mask")
        speech_token_ids = out_mm_data.get("speech_token_ids")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        audio_sample_lengths = out_mm_data.get("audio_sample_lengths")

        prompt_audio_lengths: list[int] = []
        if speech_attention_mask is not None:
            prompt_audio_lengths = speech_attention_mask.sum(-1).tolist()
        elif speech_token_ids is not None:
            prompt_audio_lengths = speech_token_ids.ne(-1).sum(-1).tolist()
        elif audio_sample_lengths is not None:
            prompt_audio_lengths = [
                _get_kimi_audio_token_length(int(length))
                for length in audio_sample_lengths.tolist()
            ]
        elif feature_attention_mask is not None:
            prompt_audio_lengths = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            ).tolist()

        def get_replacement_kimiaudio(item_idx: int):
            num_features = (
                prompt_audio_lengths[item_idx]
                if item_idx < len(prompt_audio_lengths)
                else 376
            )
            if num_features == 0:
                num_features = 376  # Default Kimi-Audio sequence length
            # Return the placeholder token ID repeated num_features times
            return [KimiAudioProcessor.KIMIA_TEXT_BLANK] * num_features

        # Use the token ID as target (as a list)
        return [
            PromptReplacement(
                modality="audio",
                target=[KimiAudioProcessor.KIMIA_TEXT_BLANK],
                replacement=get_replacement_kimiaudio,
            ),
        ]


# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------


class KimiAudioMultiModalProjector(nn.Module):
    """Projects Whisper features to LLM embedding space.

    Kimi-Audio VQ-Adaptor architecture:
    Custom Whisper (5120) → Linear[5120→3584] → Linear[3584→3584] → LayerNorm
    """

    def __init__(
        self,
        whisper_dim: int = 5120,  # Kimi-Audio custom Whisper encoder dim
        llm_dim: int = 3584,
        norm_eps: float = 1e-6,
        prefix: str = "",
    ):
        super().__init__()
        self.whisper_dim = whisper_dim
        self.llm_dim = llm_dim

        # VQ-Adaptor layers (exact checkpoint structure)
        # layers.0: Linear[5120 → 3584]
        self.vq_adaptor_layers_0 = nn.Linear(whisper_dim, llm_dim)
        self.vq_adaptor_activation = nn.SiLU()
        # layers.3: Linear[3584 → 3584]
        self.vq_adaptor_layers_3 = nn.Linear(llm_dim, llm_dim)
        # layers.4: LayerNorm[3584]
        self.vq_adaptor_layers_4 = nn.LayerNorm(llm_dim, eps=norm_eps)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        # Project: [B, T, 5120] → [B, T, 3584]
        hidden = self.vq_adaptor_layers_0(audio_features)
        hidden = self.vq_adaptor_activation(hidden)
        hidden = self.vq_adaptor_layers_3(hidden)
        hidden = self.vq_adaptor_layers_4(hidden)
        return hidden


class KimiAudioMimoModel(nn.Module):
    """Kimi-Audio text-output decoder stack kept local to this model."""

    def __init__(self, *, config, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.num_layers = getattr(config, "kimia_mimo_layers", 0)
        if get_pp_group().is_last_rank and self.num_layers > 0:
            self.layers = nn.ModuleList(
                [
                    Qwen2DecoderLayer(
                        config=config,
                        cache_config=cache_config,
                        quant_config=quant_config,
                        prefix=f"{prefix}.layers.{idx}",
                    )
                    for idx in range(self.num_layers)
                ]
            )
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.layers = nn.ModuleList()
            self.norm = PPMissingLayer()

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if residual is None:
            return hidden_states

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params


@MULTIMODAL_REGISTRY.register_processor(
    KimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder,
)
class KimiAudioForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
):
    """Kimi-Audio model for ASR transcription."""
    supports_multimodal_raw_input_only: ClassVar[bool] = True
    builds_multimodal_inputs_embeds_in_forward: ClassVar[bool] = True

    # Kimi-Audio supports a subset of Whisper's supported languages
    supported_languages: ClassVar[Mapping[str, str]] = {
        k: ISO639_1_SUPPORTED_LANGS[k]
        for k in ["zh", "en", "ja", "ko", "de", "fr", "es", "it", "pt", "ru", "ar"]
    }
    supports_transcription: ClassVar[Literal[True]] = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # audio tower
            "model.encoder.": "audio_tower.",
            # Audio projector (VQ-Adaptor)
            "model.vq_adaptor.layers.0.": "multi_modal_projector.vq_adaptor_layers_0.",
            "model.vq_adaptor.layers.3.": "multi_modal_projector.vq_adaptor_layers_3.",
            "model.vq_adaptor.layers.4.": "multi_modal_projector.vq_adaptor_layers_4.",
            # Language model
            "model.layers.": "language_model.model.layers.",
            # Embeddings and output
            "model.embed_tokens.": "language_model.model.embed_tokens.",
            "model.norm.": "language_model.model.norm.",
            # Kimi-Audio text-output branch
            "model.mimo_layers.": "mimo_model.layers.",
            "model.mimo_norm.": "mimo_model.norm.",
            "lm_head.": "language_model.lm_head.",
            "mimo_output.": "mimo_output.",
        },
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Audio placeholder token sequence
    AUDIO_PLACEHOLDER = KimiAudioPromptBuilder.AUDIO_PLACEHOLDER

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        return cls.AUDIO_PLACEHOLDER if modality.startswith("audio") else None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.model_path = vllm_config.model_config.model

        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=vllm_config.model_config.model,
                subfolder="whisper-large-v3",
                revision=None,
            )
        ]

        self.audio_tower = KimiAudioWhisperEncoder(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "audio_tower"),
        )

        self.multi_modal_projector = KimiAudioMultiModalProjector(
            whisper_dim=getattr(self.config, "kimia_adaptor_input_dim", 5120),
            llm_dim=self.config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config.with_hf_config(
                self.config, architectures=["Qwen2ForCausalLM"]
            ),
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Official Kimi-Audio remote code uses lm_head for text tokens and
        # mimo_output for the audio stream. Keep the MIMO modules loaded for
        # future audio-output work, but do not route the text-output subset
        # through them.
        self.use_mimo_text_path = (
            get_pp_group().world_size == 1
            and getattr(self.config, "kimia_mimo_layers", 0) > 0
        )
        self.kimia_mimo_transformer_from_layer_index = getattr(
            self.config,
            "kimia_mimo_transformer_from_layer_index",
            -1,
        )
        self.mimo_model = KimiAudioMimoModel(
            config=self.config,
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mimo_model"),
        )
        if get_pp_group().is_last_rank:
            self.mimo_output = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "mimo_output"),
            )
        else:
            self.mimo_output = PPMissingLayer()

        if self.use_mimo_text_path:
            tapped_layer = self.kimia_mimo_transformer_from_layer_index + 1
            start_layer = getattr(self.language_model.model, "start_layer", 0)
            end_layer = getattr(self.language_model.model, "end_layer", 0)
            if start_layer < tapped_layer <= end_layer:
                self.language_model.model._set_aux_hidden_state_layers(
                    (tapped_layer - start_layer,)
                )
            else:
                self.use_mimo_text_path = False

        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            self.config.vocab_size,
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> dict[str, torch.Tensor] | None:
        whisper_input_features = kwargs.pop("whisper_input_features", None)
        if whisper_input_features is None:
            return None
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        audio_sample_lengths = kwargs.pop("audio_sample_lengths", None)

        return {
            "whisper_input_features": whisper_input_features,
            "feature_attention_mask": feature_attention_mask,
            "audio_sample_lengths": audio_sample_lengths,
        }

    def _project_audio_features(
        self,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        # Reshape for 4x downsampling (Whisper outputs at 50Hz, need 12.5Hz)
        B, T, D = audio_features.shape
        if T % 4 != 0:
            pad_len = 4 - (T % 4)
            audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad_len))
            T = audio_features.shape[1]

        audio_features = audio_features.reshape(B, T // 4, D * 4)
        return self.multi_modal_projector(audio_features)

    def _iter_single_audio_features(
        self,
        input_features: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        feature_attention_mask: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> list[torch.Tensor]:
        if isinstance(feature_attention_mask, torch.Tensor):
            if feature_attention_mask.dim() == 1:
                feature_attention_masks = [feature_attention_mask]
            elif feature_attention_mask.dim() == 2:
                feature_attention_masks = list(feature_attention_mask.unbind(dim=0))
            else:
                msg = (
                    "feature_attention_mask must be a 1D or 2D tensor, but got "
                    f"shape {tuple(feature_attention_mask.shape)}."
                )
                raise ValueError(msg)
        elif feature_attention_mask is None:
            feature_attention_masks = None
        else:
            feature_attention_masks = list(feature_attention_mask)

        if isinstance(input_features, torch.Tensor):
            if input_features.dim() == 2:
                return [
                    self._trim_single_audio_features(
                        input_features,
                        None if feature_attention_masks is None else feature_attention_masks[0],
                    ).unsqueeze(0)
                ]
            if input_features.dim() == 3:
                normalized_features: list[torch.Tensor] = []
                for idx, feature in enumerate(input_features.unbind(dim=0)):
                    feature_mask = (
                        None
                        if feature_attention_masks is None
                        else feature_attention_masks[idx]
                    )
                    normalized_features.append(
                        self._trim_single_audio_features(feature, feature_mask).unsqueeze(0)
                    )
                return normalized_features
            msg = (
                "whisper_input_features must be a 2D or 3D tensor when passed "
                f"directly, but got shape {tuple(input_features.shape)}."
            )
            raise ValueError(msg)

        normalized_features: list[torch.Tensor] = []
        for idx, features in enumerate(input_features):
            if not isinstance(features, torch.Tensor):
                msg = (
                    "whisper_input_features must contain tensors, but found "
                    f"{type(features)!r}."
                )
                raise TypeError(msg)
            feature_mask = (
                None
                if feature_attention_masks is None or idx >= len(feature_attention_masks)
                else feature_attention_masks[idx]
            )
            if features.dim() == 2:
                normalized_features.append(
                    self._trim_single_audio_features(features, feature_mask).unsqueeze(0)
                )
            elif features.dim() == 3:
                for inner_idx, feature in enumerate(features.unbind(dim=0)):
                    inner_mask = feature_mask
                    if (
                        isinstance(feature_mask, torch.Tensor)
                        and feature_mask.dim() == 2
                        and inner_idx < feature_mask.shape[0]
                    ):
                        inner_mask = feature_mask[inner_idx]
                    normalized_features.append(
                        self._trim_single_audio_features(feature, inner_mask).unsqueeze(0)
                    )
            else:
                msg = (
                    "whisper_input_features entries must be 2D or 3D tensors, "
                    f"but got shape {tuple(features.shape)}."
                )
                raise ValueError(msg)

        return normalized_features

    def _trim_single_audio_features(
        self,
        features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if feature_attention_mask is None:
            return features

        valid_length = int(feature_attention_mask.sum().item())
        if valid_length <= 0:
            return features

        trimmed_length = min(valid_length, features.shape[-1])
        return features[..., :trimmed_length]

    def _process_audio_input(
        self, audio_input: dict[str, torch.Tensor]
    ) -> list[torch.Tensor]:
        input_features = audio_input["whisper_input_features"]
        feature_attention_mask = audio_input.get("feature_attention_mask")
        audio_sample_lengths = audio_input.get("audio_sample_lengths")
        if isinstance(audio_sample_lengths, torch.Tensor):
            sample_lengths = [int(length) for length in audio_sample_lengths.tolist()]
        else:
            sample_lengths = [] if audio_sample_lengths is None else [
                int(length) for length in audio_sample_lengths
            ]
        projected_embeds: list[torch.Tensor] = []
        normalized_features = self._iter_single_audio_features(
            input_features,
            None if sample_lengths else feature_attention_mask,
        )
        for idx, single_feature in enumerate(normalized_features):
            token_length = (
                _get_kimi_audio_token_length(sample_lengths[idx])
                if idx < len(sample_lengths)
                else 0
            )
            # Match the official tokenize_waveform path: run the full Whisper
            # encoder sequence first, then slice to token_len * 4 afterwards.
            audio_features = self.audio_tower([single_feature])
            if token_length > 0:
                audio_features = audio_features[:, : token_length * 4, :]
            audio_embeds = self._project_audio_features(audio_features)
            projected_embeds.append(audio_embeds.squeeze(0))
        return projected_embeds

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor] | None:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        audio_embeds = self._process_audio_input(audio_input)
        return audio_embeds

    def _normalize_multimodal_embeddings(
        self,
        multimodal_embeddings: tuple[torch.Tensor, ...] | list[torch.Tensor] | None,
        batch_size: int,
    ) -> list[torch.Tensor]:
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return []

        if len(multimodal_embeddings) == batch_size:
            return [embed for embed in multimodal_embeddings]

        normalized_embeds: list[torch.Tensor] = []
        for embed in multimodal_embeddings:
            if not isinstance(embed, torch.Tensor):
                continue
            if embed.dim() == 3:
                normalized_embeds.extend(embed.unbind(dim=0))
            else:
                normalized_embeds.append(embed)

        if normalized_embeds:
            return normalized_embeds

        first = multimodal_embeddings[0]
        return [first] if isinstance(first, torch.Tensor) else []

    def _build_kimi_audio_inputs_embeds(
        self,
        *,
        audio_token_ids: torch.Tensor,
        text_input_ids: torch.Tensor | None,
        is_continuous_mask: torch.Tensor | None,
        multimodal_embeddings: tuple[torch.Tensor, ...] | list[torch.Tensor] | None,
    ) -> torch.Tensor:
        flatten_runtime_batch = audio_token_ids.dim() == 1
        if flatten_runtime_batch:
            audio_token_ids = audio_token_ids.unsqueeze(0)
            if text_input_ids is not None and text_input_ids.dim() == 1:
                text_input_ids = text_input_ids.unsqueeze(0)
            if is_continuous_mask is not None and is_continuous_mask.dim() == 1:
                is_continuous_mask = is_continuous_mask.unsqueeze(0)

        audio_inputs_embeds = self.language_model.model.embed_tokens(audio_token_ids)

        if is_continuous_mask is not None and is_continuous_mask.any():
            normalized_mm_embeds = self._normalize_multimodal_embeddings(
                multimodal_embeddings,
                batch_size=audio_inputs_embeds.shape[0],
            )
            if normalized_mm_embeds:
                whisper_embeds = torch.zeros_like(audio_inputs_embeds)
                for batch_idx, mm_embeds in enumerate(normalized_mm_embeds):
                    batch_mask = is_continuous_mask[batch_idx].to(torch.bool)
                    if not batch_mask.any():
                        continue
                    num_mm_tokens = int(batch_mask.sum().item())
                    num_audio_embeds = int(mm_embeds.shape[0])
                    num_to_use = min(num_mm_tokens, num_audio_embeds)
                    if num_to_use <= 0:
                        continue
                    positions = batch_mask.nonzero(as_tuple=True)[0][:num_to_use]
                    used_mm_embeds = mm_embeds[:num_to_use].to(
                        dtype=audio_inputs_embeds.dtype,
                        device=audio_inputs_embeds.device,
                    )
                    whisper_embeds[batch_idx, positions] = used_mm_embeds

                continuous_mask = is_continuous_mask[:, :, None].to(torch.bool)
                sqrt_two = torch.sqrt(
                    torch.tensor(
                        2.0,
                        dtype=audio_inputs_embeds.dtype,
                        device=audio_inputs_embeds.device,
                    ))
                encoder_input_addwith_discrete_token = (
                    audio_inputs_embeds + whisper_embeds
                ) * sqrt_two
                audio_inputs_embeds = (
                    audio_inputs_embeds * (~continuous_mask)
                    + encoder_input_addwith_discrete_token * continuous_mask
                )

        if text_input_ids is not None and torch.any(text_input_ids != 0):
            text_inputs_embeds = self.language_model.model.embed_tokens(text_input_ids)
            output = audio_inputs_embeds + text_inputs_embeds
            return output.squeeze(0) if flatten_runtime_batch else output

        return audio_inputs_embeds.squeeze(0) if flatten_runtime_batch else audio_inputs_embeds

    def _pad_runtime_kimi_inputs_embeds(
        self,
        input_ids: torch.Tensor | None,
        kimi_inputs_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        if input_ids is None:
            return kimi_inputs_embeds, False

        expected_tokens = input_ids.shape[-1]
        actual_tokens = kimi_inputs_embeds.shape[-2]
        if expected_tokens == actual_tokens:
            return kimi_inputs_embeds, False

        # The v1 runtime may pad prompt tokens for scheduling/compilation while
        # Kimi raw dual-stream tensors still describe only the scheduled prefix.
        padded_inputs_embeds = self.embed_input_ids(input_ids)
        if input_ids.dim() == 1 and kimi_inputs_embeds.dim() == 2:
            padded_inputs_embeds[:actual_tokens] = kimi_inputs_embeds
        elif input_ids.dim() == 2 and kimi_inputs_embeds.dim() == 3:
            padded_inputs_embeds[:, :actual_tokens] = kimi_inputs_embeds
        else:
            msg = (
                "Unsupported Kimi runtime embedding padding shapes: "
                f"input_ids={tuple(input_ids.shape)}, "
                f"inputs_embeds={tuple(kimi_inputs_embeds.shape)}."
            )
            raise ValueError(msg)

        return padded_inputs_embeds, True

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: tuple[torch.Tensor, ...] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Embed input IDs and fuse with audio embeddings.

        Kimi-Audio fusion: inputs_embeds = (text_emb + audio_emb) × √2

        For PP compatibility, we use the is_multimodal mask from vLLM engine
        which is correctly computed per pipeline stage.
        """
        blank_audio_ids = torch.full_like(
            input_ids,
            fill_value=KimiAudioProcessor.KIMIA_TEXT_BLANK,
        )
        inputs_embeds = (
            self.language_model.model.embed_tokens(input_ids)
            + self.language_model.model.embed_tokens(blank_audio_ids)
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None or not is_multimodal.any():
            return inputs_embeds

        embedding_items: list[torch.Tensor] = []
        for embed in multimodal_embeddings:
            if isinstance(embed, (list, tuple)):
                embedding_items.extend(
                    tensor if tensor.dim() == 2 else tensor.reshape(-1, tensor.shape[-1])
                    for tensor in embed
                    if isinstance(tensor, torch.Tensor)
                )
            elif isinstance(embed, torch.Tensor):
                if embed.dim() == 3:
                    embedding_items.extend(embed.unbind(dim=0))
                else:
                    embedding_items.append(embed)

        if not embedding_items:
            return inputs_embeds

        mm_positions = is_multimodal.nonzero(as_tuple=True)[0]
        if mm_positions.numel() == 0:
            return inputs_embeds

        split_points = (mm_positions[1:] != mm_positions[:-1] + 1).nonzero(
            as_tuple=True
        )[0] + 1
        mm_segments = torch.tensor_split(mm_positions, split_points.tolist())

        for segment_positions, audio_embeds in zip(mm_segments, embedding_items):
            if segment_positions.numel() == 0:
                continue
            if audio_embeds.dim() != 2:
                audio_embeds = audio_embeds.reshape(-1, audio_embeds.shape[-1])
            num_to_use = min(int(segment_positions.numel()), int(audio_embeds.shape[0]))
            if num_to_use <= 0:
                continue

            actual_positions = segment_positions[:num_to_use]
            used_audio_embeds = audio_embeds[:num_to_use].to(dtype=inputs_embeds.dtype)
            text_at_positions = inputs_embeds[actual_positions].clone()
            inputs_embeds[actual_positions] = (
                used_audio_embeds + text_at_positions
            ) * (2**0.5)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        audio_token_ids = kwargs.pop("audio_token_ids", None)
        text_input_ids = kwargs.pop("text_token_ids", None)
        is_continuous_mask = kwargs.pop("is_continuous_mask", None)
        multimodal_embeddings = kwargs.get("multimodal_embeddings")

        model_input_ids = input_ids
        if audio_token_ids is not None:
            if input_ids is not None and input_ids.dim() == 1:
                if audio_token_ids.dim() == 2 and audio_token_ids.shape[0] == 1:
                    audio_token_ids = audio_token_ids.squeeze(0)
                if (
                    text_input_ids is not None
                    and text_input_ids.dim() == 2
                    and text_input_ids.shape[0] == 1
                ):
                    text_input_ids = text_input_ids.squeeze(0)
                if (
                    is_continuous_mask is not None
                    and is_continuous_mask.dim() == 2
                    and is_continuous_mask.shape[0] == 1
                ):
                    is_continuous_mask = is_continuous_mask.squeeze(0)
            if inputs_embeds is None:
                kimi_inputs_embeds = self._build_kimi_audio_inputs_embeds(
                    audio_token_ids=audio_token_ids,
                    text_input_ids=text_input_ids,
                    is_continuous_mask=is_continuous_mask,
                    multimodal_embeddings=multimodal_embeddings,
                )
                inputs_embeds, used_runtime_padding = (
                    self._pad_runtime_kimi_inputs_embeds(
                        input_ids=input_ids,
                        kimi_inputs_embeds=kimi_inputs_embeds,
                    ))
                model_input_ids = input_ids if used_runtime_padding else audio_token_ids
            else:
                model_input_ids = input_ids if input_ids is not None else audio_token_ids
        elif inputs_embeds is None and model_input_ids is not None:
            inputs_embeds = self.embed_input_ids(
                model_input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=kwargs.get("is_multimodal"),
            )

        model_output = self.language_model.model(
            model_input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        if isinstance(model_output, tuple):
            hidden_states, _aux_hidden_states = model_output
            return hidden_states

        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        if not get_pp_group().is_last_rank:
            return None

        logits = self.logits_processor(self.language_model.lm_head,
                                       hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for the Kimi-Audio text-output subset."""
        skipped_patterns = [
            # Audio tower
            "model.",
        ]

        loader = AutoWeightsLoader(self, skip_prefixes=skipped_patterns)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return loaded

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        """Get speech-to-text config with custom processor."""
        # Load feature extractor for config values
        feature_extractor = cached_feature_extractor_from_config(
            model_config,
            subfolder=KIMIA_WHISPER_SUBFOLDER,
        )

        return SpeechToTextConfig(
            max_audio_clip_s=feature_extractor.chunk_length,
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
        tokenizer = cached_get_tokenizer(
            model_config.tokenizer,
            tokenizer_cls=KimiAudioTokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            revision=model_config.tokenizer_revision,
            trust_remote_code=model_config.trust_remote_code,
        )

        if task_type not in ("transcribe", "translate"):
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                "Supported task types are 'transcribe' and 'translate'."
            )

        prompt = KimiAudioPromptBuilder.build_transcription_prompt(
            request_prompt=request_prompt,
        )
        messages: list[dict[str, object]] = [
            {"role": "user", "message_type": "text", "content": request_prompt},
            {"role": "user", "message_type": "audio"},
        ]

        prompt_token_ids = tokenizer.encode(prompt)

        return TokensPrompt(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data={"audio": audio},
            mm_processor_kwargs={
                "messages": messages,
                "output_type": "text",
            },
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        if not text:
            return ""
        text = text.split("<|im_kimia_text_eos|>", 1)[0]
        return text
