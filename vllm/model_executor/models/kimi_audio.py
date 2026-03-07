# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import (
    AddedToken,
    AutoFeatureExtractor,
    BatchFeature,
    WhisperConfig,
    WhisperFeatureExtractor,
)

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)
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
from vllm.model_executor.models.whisper import (
    WhisperEncoder,
    _create_fake_bias_for_k_proj,
)
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
from vllm.multimodal.processing.processor import BaseMultiModalProcessor
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tokenizers.kimi_audio import KimiAudioTokenizer
from vllm.transformers_utils.processors.kimi_audio import (
    KimiAudioProcessor,
    _get_feat_extract_output_lengths,
)
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.sample.metadata import SamplingMetadata

logger = init_logger(__name__)


class KimiAudioWhisperEncoder(WhisperEncoder):
    """WhisperEncoder for Kimi-Audio with packed_modules_mapping."""

    # packed_modules_mapping for Q/K/V fusion during weight loading
    packed_modules_mapping = {
        "self_attn.qkv_proj": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ],
    }

    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", init_in_fp32: bool = False
    ):
        # Create a WhisperConfig with Kimi-Audio specific values
        # instead of relying on KimiAudioConfig which lacks Whisper attributes
        config = WhisperConfig(
            d_model=1280,  # Standard Whisper large-v3
            encoder_layers=32,
            encoder_attention_heads=20,
            num_mel_bins=128,
            max_source_positions=1500,
            scale_embedding=True,
        )
        # Store original config and temporarily replace it
        original_config = vllm_config.model_config.hf_config
        vllm_config.model_config.hf_config = config

        super().__init__(
            vllm_config=vllm_config, prefix=prefix, init_in_fp32=init_in_fp32
        )

        # Restore original config
        vllm_config.model_config.hf_config = original_config


class KimiAudioInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - nmb: Number of mel bins (128 for Whisper)
        - t: Time frames
    """

    whisper_input_features: Annotated[
        torch.Tensor | None,
        TensorShape("b", "nmb", "t"),
    ]


# -----------------------------------------------------------------------------
# Processing Info, Dummy Inputs, and MultiModal Processor
# (Following Qwen3ASR pattern - same file as model)
# -----------------------------------------------------------------------------


class KimiAudioProcessingInfo(BaseProcessingInfo):
    """Processing info for vLLM registry."""

    _processor = None

    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_hf_processor(self, **kwargs: object) -> KimiAudioProcessor:
        """Get or create the KimiAudioProcessor."""
        if KimiAudioProcessingInfo._processor is None:
            # Load components directly
            model_path = self.ctx.model_config.model
            trust_remote_code = self.ctx.model_config.trust_remote_code

            # Load feature extractor with subfolder support
            feature_extractor = WhisperFeatureExtractor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                subfolder="whisper-large-v3",
            )

            # Use KimiAudioTokenizer for Kimi-Audio
            tokenizer = KimiAudioTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )

            # Add special tokens to tokenizer's added_tokens_decoder
            special_tokens = {
                "<|im_media_begin|>": KimiAudioProcessor.KIMIA_MEDIA_BEGIN,
                "<|im_media_end|>": KimiAudioProcessor.KIMIA_MEDIA_END,
                "<|im_kimia_text_blank|>": KimiAudioProcessor.KIMIA_TEXT_BLANK,
                "<|im_msg_end|>": 151645,
                "<|im_kimia_user_msg_start|>": 151646,
                "<|im_kimia_assistant_msg_start|>": 151647,
            }
            # Add to tokenizer's added_tokens_decoder
            for token_str, token_id in special_tokens.items():
                tokenizer.added_tokens_decoder[token_id] = AddedToken(
                    token_str, single_word=True, normalized=False, special=True
                )

            KimiAudioProcessingInfo._processor = KimiAudioProcessor(
                feature_extractor=feature_extractor, tokenizer=tokenizer
            )

        return KimiAudioProcessingInfo._processor

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        return hf_processor.feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self) -> "KimiAudioMultiModalDataParser":
        """Get data parser for audio inputs."""
        return KimiAudioMultiModalDataParser(
            target_sr=16000,  # Whisper expects 16kHz audio
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class KimiAudioDummyInputsBuilder(BaseDummyInputsBuilder[KimiAudioProcessingInfo]):
    """Dummy inputs builder for vLLM registry."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> list[int]:
        """Return dummy text as token IDs directly."""
        num_audios = mm_counts.get("audio", 0)
        if num_audios == 0:
            return [198]  # "Transcribe" tokenized
        # Return as token IDs directly to avoid tokenizer issues
        return [
            KimiAudioProcessor.KIMIA_MEDIA_BEGIN,
            KimiAudioProcessor.KIMIA_TEXT_BLANK,
            KimiAudioProcessor.KIMIA_MEDIA_END,
        ] * num_audios

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


def _kimiaudio_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    """Field config for Kimi-Audio multimodal data."""
    return dict(
        whisper_input_features=MultiModalFieldConfig.batched("audio"),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
    )


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
                fields_factory=_kimiaudio_field_config,
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

        # Pass audio as list (our processor handles both single and list)
        if audios:
            mm_data["audio"] = audios

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
        return _kimiaudio_field_config(hf_inputs)

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
                else KimiAudioProcessor.AUDIO_SEQ_LEN
            )
            if num_features == 0:
                num_features = KimiAudioProcessor.AUDIO_SEQ_LEN
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

    supported_languages: ClassVar[Mapping[str, str]] = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ar": "Arabic",
    }
    supports_transcription: ClassVar[Literal[True]] = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Audio projector (VQ-Adaptor)
            # checkpoint uses model.vq_adaptor.layers.0,3,4
            "model.vq_adaptor.layers.0.": "multi_modal_projector.vq_adaptor_layers_0.",
            "model.vq_adaptor.layers.3.": "multi_modal_projector.vq_adaptor_layers_3.",
            "model.vq_adaptor.layers.4.": "multi_modal_projector.vq_adaptor_layers_4.",
            # Language model - checkpoint uses model.layers.*
            # we use language_model.model.layers.*
            "model.layers.": "language_model.model.layers.",
            # Embeddings and output
            "model.embed_tokens.": "language_model.model.embed_tokens.",
            "model.norm.": "language_model.model.norm.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.multimodal_config = multimodal_config

        # Use KimiAudioWhisperEncoder for audio feature extraction
        self.audio_tower = KimiAudioWhisperEncoder(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "audio_tower"),
        )

        self.multi_modal_projector = KimiAudioMultiModalProjector(
            whisper_dim=getattr(
                self.config, "kimia_adaptor_input_dim", 5120
            ),  # Kimi-Audio custom Whisper encoder dim (5120)
            llm_dim=self.config.hidden_size,  # From LLM config
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config.with_hf_config(
                self.config, architectures=["Qwen2ForCausalLM"]
            ),
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            self.config.vocab_size,
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_audio_input(self, **kwargs: object) -> KimiAudioInputs:
        whisper_input_features = kwargs.pop("whisper_input_features", None)
        return KimiAudioInputs(whisper_input_features=whisper_input_features)

    def _process_audio_input(self, audio_input: KimiAudioInputs) -> torch.Tensor:
        input_features = audio_input.whisper_input_features

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
        if audio_input.whisper_input_features is None:
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
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Embed input IDs and fuse with audio embeddings.

        Kimi-Audio fusion: inputs_embeds = text_emb + whisper_emb × √2
        """
        # Get text embeddings
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        # Kimi-Audio uses scaled addition: text + audio * √2
        audio_embeds = multimodal_embeddings[0]
        scale_factor = 2**0.5

        # Find audio placeholder positions
        audio_mask = input_ids == self.config.kimia_media_begin

        if audio_mask.any():
            audio_positions = audio_mask.nonzero(as_tuple=True)[0]

            if audio_positions.numel() > 0:
                begin_pos = audio_positions[0].item()
                pos = begin_pos + 1
                audio_len = audio_embeds.shape[0]
                end_pos = pos + audio_len

                # Fuse: (text_emb + audio_emb) * √2
                text_embeds = inputs_embeds[pos:end_pos, :]
                fused_embeds = (text_embeds + audio_embeds) * scale_factor
                inputs_embeds[pos:end_pos, :] = fused_embeds

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
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(
            self.language_model.lm_head, hidden_states, sampling_metadata
        )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights, skipping MIMO layers (TTS-only) for ASR.

        Also loads Whisper encoder weights from standalone folder.
        """
        # Filter out MIMO/TTS weights since we only do ASR (speech-to-text)
        skipped_patterns = [
            "mimo_layers.",
            "mimo_output.",
            "mimo_norm.",
            "audio_decoder.",
        ]

        # Filter weights
        filtered_weights = [
            (name, param)
            for name, param in weights
            if not any(pattern in name for pattern in skipped_patterns)
        ]

        # Load main model weights (LLM + projector) with mapper
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(filtered_weights, mapper=self.hf_to_vllm_mapper)

        # Load Whisper encoder weights from standalone folder
        # Note: Using manual loading (not DefaultModelLoader.Source) because
        # Kimi-Audio requires custom weight transformations (Q/K/V fusion, fc mapping)
        model_path = self.config._name_or_path
        whisper_path = os.path.join(model_path, "whisper-large-v3", "model.safetensors")

        if os.path.exists(whisper_path):
            whisper_weights = []
            with safe_open(whisper_path, framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    if (
                        key.startswith("model.encoder.")
                        and "embed_positions" not in key
                    ):
                        new_key = key.replace("model.encoder.", "")
                        whisper_weights.append((new_key, f.get_tensor(key)))

            # Apply transformations for Q/K/V fusion
            whisper_weights_iter = _create_fake_bias_for_k_proj(
                whisper_weights, ".k_proj.weight"
            )
            whisper_mapper = WeightsMapper(
                orig_to_new_substr={".fc1.": ".mlp.fc1.", ".fc2.": ".mlp.fc2."}
            )
            whisper_weights_iter = list(whisper_mapper.apply(whisper_weights_iter))

            # Manual Q/K/V fusion
            stacked_params_mapping = [
                (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
                (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
                (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            ]

            params_dict = dict(self.audio_tower.named_parameters())
            whisper_loaded: set[str] = set()

            for name, loaded_weight in whisper_weights_iter:
                fused = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    fused_name = name.replace(weight_name, param_name)
                    if fused_name not in params_dict:
                        continue

                    param = params_dict[fused_name]
                    param.weight_loader(param, loaded_weight, shard_id)
                    whisper_loaded.add(f"audio_tower.{fused_name}")
                    fused = True
                    break

                if not fused:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    whisper_loaded.add(f"audio_tower.{name}")

            loaded.update(whisper_loaded)
        else:
            logger.warning(
                "Whisper encoder weights not found at %s. "
                "Audio transcription may not work correctly.",
                whisper_path,
            )

        loaded.add("audio_tower.embed_positions.weight")
        return loaded

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        """Get speech-to-text config with custom processor."""
        # Load feature extractor from model path
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_config.model, trust_remote_code=model_config.trust_remote_code
        )

        # Get tokenizer (this handles the TikTokenTokenizer properly)
        tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )

        # Manually instantiate processor to validate it works
        _ = KimiAudioProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

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
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_placeholder = cls.get_placeholder_str("audio", 0)

        if task_type not in ("transcribe", "translate"):
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                "Supported task types are 'transcribe' and 'translate'."
            )

        # Incorporate request_prompt as context/instruction if provided
        # This can guide the model's style or provide previous context
        if request_prompt:
            # Insert prompt before audio as instruction/context
            user_content = f"{request_prompt}\n{audio_placeholder}"
        else:
            user_content = audio_placeholder

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
