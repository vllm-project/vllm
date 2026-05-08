# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only NVIDIA Parakeet TDT model."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import ParakeetEncoder
from transformers.feature_extraction_utils import BatchFeature

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import (
    ExplicitEncoderDecoderPrompt,
    MultiModalDataDict,
    PromptType,
    TextPrompt,
    TokensPrompt,
)
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
)
from vllm.model_executor.models.parakeet import ParakeetExtractor
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.parakeet_tdt import ParakeetTDTConfig

logger = init_logger(__name__)

PARAKEET_SUPPORTED_LANGUAGES = {
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "ru": "Russian",
    "uk": "Ukrainian",
}


class ParakeetTDTProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> ParakeetTDTConfig:
        return self.ctx.get_hf_config(ParakeetTDTConfig)

    def get_feature_extractor(self) -> ParakeetExtractor:
        return ParakeetExtractor(self.get_hf_config().encoder_config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=self.get_hf_config().sample_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_num_audio_tokens(self, num_samples: int) -> int:
        return self.get_feature_extractor().audio_token_count(num_samples)


class ParakeetTDTDummyInputsBuilder(BaseDummyInputsBuilder[ParakeetTDTProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        sample_rate = self.info.get_hf_config().sample_rate
        audio_overrides = mm_options.get("audio")

        return {
            "audio": self._get_dummy_audios(
                length=30 * sample_rate,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


class ParakeetTDTMultiModalProcessor(
    EncDecMultiModalProcessor[ParakeetTDTProcessingInfo]
):
    skip_decoder_start_token: bool = True

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [0]

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [self.info.get_hf_config().blank_token_id]

    def _extract_audio_features(
        self,
        audios: Sequence[np.ndarray],
        *,
        device: str = "cpu",
    ) -> BatchFeature:
        extractor = self.info.get_feature_extractor()
        raw_speech = [
            torch.as_tensor(audio, device=device, dtype=torch.float32)
            for audio in audios
        ]

        for i, speech in enumerate(raw_speech):
            if len(speech.shape) > 1:
                logger.warning(
                    "Only mono-channel audio is supported for Parakeet TDT. "
                    "Averaging channels to mono."
                )
                raw_speech[i] = speech.mean(-1)

        audio_lengths = torch.tensor(
            [len(speech) for speech in raw_speech],
            dtype=torch.long,
            device=device,
        )
        max_length = max(len(speech) for speech in raw_speech)
        input_features = extractor._pad_raw_speech(raw_speech, max_length, device)
        input_features = extractor._apply_preemphasis(input_features, audio_lengths)
        input_features = extractor._torch_extract_fbank_features(input_features, device)
        input_features, attention_mask = extractor._normalize_mel_features(
            input_features,
            audio_lengths,
        )

        return BatchFeature(
            data={
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type="pt",
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            return BatchFeature(data={"input_ids": [[0]]}, tensor_type="pt")

        raw_audios = mm_data.get("audios")
        if isinstance(raw_audios, np.ndarray):
            audios = [raw_audios]
        elif isinstance(raw_audios, Sequence):
            audios = list(raw_audios)
        else:
            raise ValueError("Parakeet TDT expects audio inputs.")

        inputs = self._extract_audio_features(audios)
        inputs["input_ids"] = [[0]]
        return inputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            attention_mask=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def get_audio_replacement(item_idx: int) -> list[int]:
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio_len = audios.get_audio_length(item_idx)
            num_tokens = self.info.get_num_audio_tokens(num_samples=audio_len)
            return [0] * num_tokens

        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=get_audio_replacement,
            )
        ]


class ParakeetTDTDecoder(nn.Module):
    def __init__(self, config: ParakeetTDTConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.decoder_hidden_size)
        self.lstm = nn.LSTM(
            input_size=config.decoder_hidden_size,
            hidden_size=config.decoder_hidden_size,
            num_layers=config.num_decoder_layers,
            batch_first=True,
        )
        self.decoder_projector = nn.Linear(
            config.decoder_hidden_size,
            config.decoder_hidden_size,
        )

    def predict(
        self,
        token_id: int,
        state: tuple[torch.Tensor, torch.Tensor] | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        label = torch.tensor([[token_id]], dtype=torch.long, device=device)
        hidden_states = self.embedding(label)
        hidden_states, state = self.lstm(hidden_states, state)
        hidden_states = self.decoder_projector(hidden_states)
        return hidden_states[:, 0, :], state


class ParakeetTDTJoint(nn.Module):
    def __init__(self, config: ParakeetTDTConfig) -> None:
        super().__init__()
        self.activation = nn.ReLU()
        self.head = nn.Linear(
            config.decoder_hidden_size,
            config.vocab_size + len(config.durations),
        )

    def forward(
        self,
        encoder_states: torch.Tensor,
        decoder_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.activation(encoder_states + decoder_states)
        return self.head(hidden_states)


class ParakeetTDTForcedDecoderState:
    """Token sequence produced by Parakeet's TDT decoder for one request."""

    def __init__(self, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id
        self._sequence: list[int] = []

    def set_sequence(self, sequence: Sequence[int]) -> None:
        self._sequence = list(sequence)

    def get_forced_token_ids(
        self,
        *,
        positions: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if positions.ndim == 0:
            positions = positions.reshape(1)
        if positions.ndim > 1:
            positions = positions[0]

        forced_token_ids: list[int] = []
        for position in positions:
            seq_idx = int(position.item())
            if 0 <= seq_idx < len(self._sequence):
                forced_token_ids.append(self._sequence[seq_idx])
            else:
                forced_token_ids.append(self.eos_token_id)

        return torch.tensor(forced_token_ids, dtype=torch.long, device=device)


class ParakeetTDTModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: ParakeetTDTConfig = vllm_config.model_config.hf_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        self.encoder = ParakeetEncoder(config.encoder_config).to(self.dtype)
        self.encoder_projector = nn.Linear(
            config.encoder_config.hidden_size,
            config.decoder_hidden_size,
        )
        self.decoder = ParakeetTDTDecoder(config)
        self.joint = ParakeetTDTJoint(config)

    def get_encoder_outputs(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        encoder_outputs = self.encoder(
            input_features=input_features.to(self.dtype),
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs.last_hidden_state
        output_mask = encoder_outputs.attention_mask

        if output_mask is None:
            return list(hidden_states)

        return [
            hidden_state[mask.to(dtype=torch.bool)]
            for hidden_state, mask in zip(hidden_states, output_mask)
        ]

    def _joint_logits(
        self,
        encoder_state: torch.Tensor,
        decoder_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.joint(encoder_state, decoder_state)

    def greedy_decode(self, encoder_output: torch.Tensor) -> list[int]:
        cfg = self.config
        device = encoder_output.device
        token_ids: list[int] = []
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        last_token: int | None = None
        encoder_projected = self.encoder_projector(encoder_output)

        time_idx = 0
        out_len = int(encoder_output.shape[0])
        while time_idx < out_len:
            encoder_state = encoder_projected[time_idx : time_idx + 1]
            symbols_added = 0
            need_loop = True
            skip = 1

            while need_loop and symbols_added < cfg.max_symbols_per_step:
                label = cfg.blank_token_id if last_token is None else last_token
                pred_state, next_state = self.decoder.predict(label, state, device)
                logits = self._joint_logits(encoder_state, pred_state)[0]

                token_logits = logits[: cfg.vocab_size].float()
                duration_logits = logits[cfg.vocab_size :].float()
                score, token = token_logits.max(0)
                del score

                duration_idx = int(duration_logits.argmax().item())
                skip = cfg.durations[duration_idx]
                token_id = int(token.item())
                if token_id == cfg.blank_token_id and skip == 0:
                    skip = 1

                if token_id != cfg.blank_token_id:
                    token_ids.append(token_id)
                    state = next_state
                    last_token = token_id

                symbols_added += 1
                time_idx += skip
                need_loop = skip == 0

            if need_loop:
                time_idx += 1

        token_ids.append(cfg.eos_token_id)
        return token_ids


@MULTIMODAL_REGISTRY.register_processor(
    ParakeetTDTMultiModalProcessor,
    info=ParakeetTDTProcessingInfo,
    dummy_inputs=ParakeetTDTDummyInputsBuilder,
)
class ParakeetForTDT(nn.Module, SupportsTranscription, SupportsMultiModal):
    supports_transcription_only = True
    supported_languages = PARAKEET_SUPPORTED_LANGUAGES
    no_space_languages: set[str] = set()
    # The config hook marks this as a single-request stateful decoder and
    # enforces eager execution. Future V2 support should move this into a
    # ParakeetModelState using ModelState.prepare_inputs() and
    # prepare_dummy_inputs(), matching the Whisper model-state pattern.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "encoder.": "model.encoder.",
            "encoder_projector.": "model.encoder_projector.",
            "decoder.": "model.decoder.",
            "joint.": "model.joint.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config: ParakeetTDTConfig = vllm_config.model_config.hf_config
        self.dtype = vllm_config.model_config.dtype

        with self._mark_tower_model(vllm_config, "audio"):
            self.model = ParakeetTDTModel(vllm_config=vllm_config, prefix=prefix)

        self._forced_decoder_state = ParakeetTDTForcedDecoderState(
            eos_token_id=self.config.eos_token_id
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<audio>"

        raise ValueError("Only audio modality is supported")

    def get_language_model(self) -> nn.Module:
        return self.model.decoder

    def _parse_and_validate_audio_input(
        self,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_features = kwargs.pop("input_features", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if isinstance(input_features, Sequence):
            input_features = nn.utils.rnn.pad_sequence(
                list(input_features),
                batch_first=True,
            )
        if isinstance(attention_mask, Sequence):
            attention_mask = nn.utils.rnn.pad_sequence(
                list(attention_mask),
                batch_first=True,
            )

        if not isinstance(input_features, torch.Tensor):
            raise ValueError("Parakeet TDT requires input_features.")
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("Parakeet TDT requires attention_mask.")

        return input_features, attention_mask

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features, attention_mask = self._parse_and_validate_audio_input(**kwargs)
        return self.model.get_encoder_outputs(input_features, attention_mask)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.decoder.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del intermediate_tensors, kwargs

        if encoder_outputs:
            if len(encoder_outputs) != 1:
                raise ValueError(
                    "Parakeet TDT currently supports one active encoder output."
                )
            self._forced_decoder_state.set_sequence(
                self.model.greedy_decode(encoder_outputs[0])
            )

        if input_ids is None:
            raise ValueError("Parakeet TDT forward requires input_ids.")

        batch_size = input_ids.shape[0]
        device = input_ids.device
        forced_token_ids = self._forced_decoder_state.get_forced_token_ids(
            positions=positions,
            device=device,
        )

        vocab_size = self.config.vocab_size
        logits = torch.full(
            (batch_size, vocab_size),
            -1.0e9,
            dtype=torch.float32,
            device=device,
        )

        logits.scatter_(1, forced_token_ids.unsqueeze(1), 0.0)

        return logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: str,
    ) -> SpeechToTextConfig:
        hf_config = model_config.hf_config
        return SpeechToTextConfig(
            sample_rate=hf_config.sample_rate,
            max_audio_clip_s=30,
        )

    @classmethod
    def get_generation_prompt(
        cls,
        stt_params: SpeechToTextParams,
    ) -> PromptType:
        audio = stt_params.audio
        sample_rate = stt_params.stt_config.sample_rate
        blank_token_id = stt_params.model_config.hf_config.blank_token_id

        return ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="",
                multi_modal_data={"audio": (audio, sample_rate)},
            ),
            decoder_prompt=TokensPrompt(prompt_token_ids=[blank_token_id]),
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        hf_config = model_config.hf_config.encoder_config
        extractor = ParakeetExtractor(hf_config)
        num_samples = int(audio_duration_s * stt_config.sample_rate)
        return extractor.audio_token_count(num_samples)

    @classmethod
    def post_process_output(cls, text: str) -> str:
        for special_token in ("<blank>", "<|endoftext|>"):
            text = text.replace(special_token, "")
        return text.strip()


__all__ = ["ParakeetForTDT", "ParakeetTDTForcedDecoderState"]
