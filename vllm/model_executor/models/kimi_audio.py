# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers import WhisperConfig as HFWhisperConfig

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import PromptType, TokensPrompt
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.whisper import WhisperEncoder
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

# Kimi-Audio constants
KIMIA_WHISPER_SUBFOLDER = "whisper-large-v3"


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


class KimiAudioWhisperEncoder(WhisperEncoder):
    """WhisperEncoder for Kimi-Audio with packed_modules_mapping."""

    # packed_modules_mapping for Q/K/V fusion during weight loading
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", init_in_fp32: bool = False
    ):
        # Load Whisper config from subfolder (authoritative source)
        # Kimi-Audio stores Whisper config in whisper-large-v3/config.json
        model_path = vllm_config.model_config.model

        # Load WhisperConfig from the subfolder
        whisper_config = HFWhisperConfig.from_pretrained(
            model_path,
            subfolder=KIMIA_WHISPER_SUBFOLDER,
        )

        super().__init__(
            vllm_config=vllm_config.with_hf_config(whisper_config),
            prefix=prefix,
            init_in_fp32=init_in_fp32,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


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
            ]
            * num_audios
        )

        return ProcessorInputs(prompt=dummy_tokens, mm_data_items=dummy_mm_items)


# Field config for Kimi-Audio multimodal data
_KIMIAUDIO_FIELD_CONFIG = {
    "whisper_input_features": MultiModalFieldConfig.batched("audio"),
    "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
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

        # Use the context's call_hf_processor for proper handling
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
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
        # Get audio feature lengths from processed output
        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")

        if feature_attention_mask is not None:
            audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            audio_output_lengths = audio_output_lens.tolist()
        else:
            audio_output_lengths = []

        def get_replacement_kimiaudio(item_idx: int):
            num_features = (
                audio_output_lengths[item_idx]
                if item_idx < len(audio_output_lengths)
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
        prefix: str = "",
    ):
        super().__init__()
        self.whisper_dim = whisper_dim
        self.llm_dim = llm_dim

        # VQ-Adaptor layers (exact checkpoint structure)
        # layers.0: Linear[5120 → 3584]
        self.vq_adaptor_layers_0 = nn.Linear(whisper_dim, llm_dim)
        # layers.3: Linear[3584 → 3584]
        self.vq_adaptor_layers_3 = nn.Linear(llm_dim, llm_dim)
        # layers.4: LayerNorm[3584]
        self.vq_adaptor_layers_4 = nn.LayerNorm(llm_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        # Project: [B, T, 5120] → [B, T, 3584]
        hidden = self.vq_adaptor_layers_0(audio_features)
        hidden = torch.nn.functional.gelu(hidden)
        hidden = self.vq_adaptor_layers_3(hidden)
        hidden = self.vq_adaptor_layers_4(hidden)
        return hidden


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
            "lm_head.": "language_model.lm_head.",
        },
        orig_to_new_substr={
            ".fc1.": ".mlp.fc1.",
            ".fc2.": ".mlp.fc2.",
        },
    )

    # Audio placeholder token sequence
    AUDIO_PLACEHOLDER = "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"

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

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = KimiAudioWhisperEncoder(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "audio_tower"),
            )
            self.multi_modal_projector = KimiAudioMultiModalProjector(
                whisper_dim=getattr(self.config, "kimia_adaptor_input_dim", 5120),
                llm_dim=self.config.hidden_size,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config.with_hf_config(
                    self.config, architectures=["Qwen2ForCausalLM"]
                ),
                prefix=maybe_prefix(prefix, "language_model"),
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

        return {"whisper_input_features": whisper_input_features}

    def _process_audio_input(
        self, audio_input: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        input_features = audio_input["whisper_input_features"]

        # KimiAudioWhisperEncoder expects list of tensors
        if input_features.dim() == 3:
            input_features = input_features.unbind(dim=0)

        # Run through Whisper encoder
        audio_features = self.audio_tower(input_features)

        # Reshape for 4x downsampling (Whisper outputs at 50Hz, need 12.5Hz)
        B, T, D = audio_features.shape
        if T % 4 != 0:
            pad_len = 4 - (T % 4)
            audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad_len))
            T = audio_features.shape[1]  # Update T after padding

        audio_features = audio_features.reshape(B, T // 4, D * 4)

        # Project to LLM dimension
        audio_embeds = self.multi_modal_projector(audio_features)
        return audio_embeds

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor] | None:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        audio_embeds = self._process_audio_input(audio_input)

        # audio_embeds shape: [batch_size, seq_len, hidden_dim]
        # Return as list of 2D tensors, one per batch item
        if audio_embeds.dim() == 3:
            # Unbind batch dimension: [B, T, D] -> list of B tensors [T, D]
            return list(audio_embeds.unbind(dim=0))
        else:
            # Single sample: [T, D] -> wrap in list
            return [audio_embeds]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: tuple[torch.Tensor, ...] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed input IDs and fuse with audio embeddings.

        Kimi-Audio fusion: inputs_embeds = (text_emb + audio_emb) × √2

        For PP compatibility, we use the is_multimodal mask from vLLM engine
        which is correctly computed per pipeline stage.
        """
        # Get text embeddings
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        # is_multimodal must be provided for PP to work correctly
        if is_multimodal is None or not is_multimodal.any():
            return inputs_embeds

        # multimodal_embeddings[0] contains audio embeddings
        audio_embeds = multimodal_embeddings[0]

        # Handle different tensor structures
        if isinstance(audio_embeds, (list, tuple)):
            audio_embeds = torch.cat(audio_embeds, dim=0)
        elif audio_embeds.dim() == 3:
            audio_embeds = audio_embeds.reshape(-1, audio_embeds.shape[-1])

        # In PP, audio_embeds count should match is_multimodal.sum()
        # For now, use embeddings sequentially
        # (works for non-PP, PP needs vLLM infra fix)
        num_mm_tokens = is_multimodal.sum().item()
        num_audio_embeds = audio_embeds.shape[0]

        # Use the minimum of available embeddings and positions
        # This ensures we don't access out-of-bounds
        num_to_use = min(num_audio_embeds, num_mm_tokens)

        # Get positions for the tokens we'll actually process
        mm_positions = is_multimodal.nonzero(as_tuple=True)[0]
        actual_mm_mask = torch.zeros_like(is_multimodal)
        actual_mm_mask[mm_positions[:num_to_use]] = True

        # Use corresponding embeddings
        used_audio_embeds = audio_embeds[:num_to_use]

        # Save text embeddings at multimodal positions
        text_at_mm_positions = inputs_embeds[actual_mm_mask].clone()

        # Replace text with audio at multimodal positions
        inputs_embeds[actual_mm_mask] = used_audio_embeds.to(dtype=inputs_embeds.dtype)

        # Apply Kimi-Audio's unique fusion formula: (text + audio) × √2
        inputs_embeds[actual_mm_mask] = (
            inputs_embeds[actual_mm_mask] + text_at_mm_positions
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
        """Load weights, skipping MIMO layers (TTS-only) for ASR."""
        # Filter out MIMO/TTS weights since we only do ASR (speech-to-text)
        skipped_patterns = [
            # Audio tower
            "model.",
            # MIMO/TTS
            "mimo_layers.",
            "mimo_output.",
            "mimo_norm.",
        ]

        # Load main model weights (LLM + projector) with mapper
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
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        audio = stt_params.audio
        model_config = stt_params.model_config
        task_type = stt_params.task_type
        request_prompt = stt_params.request_prompt

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

        # Incorporate request_prompt as context/instruction if provided
        user_content = (
            f"{request_prompt}\n{cls.AUDIO_PLACEHOLDER}"
            if request_prompt
            else cls.AUDIO_PLACEHOLDER
        )

        prompt = (
            f"<|im_kimia_user_msg_start|>{user_content}"
            f"<|im_msg_end|><|im_kimia_assistant_msg_start|>"
        )

        prompt_token_ids = tokenizer.encode(prompt)

        return TokensPrompt(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data={"audio": audio},
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        if not text:
            return ""
        return text.strip()
