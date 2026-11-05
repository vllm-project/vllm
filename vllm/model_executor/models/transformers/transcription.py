# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers modeling backend mixin for the transcription API."""

import inspect
from typing import TYPE_CHECKING, Any

from transformers.models.whisper.tokenization_whisper import LANGUAGES

from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.inputs import TextPrompt
from vllm.model_executor.models.interfaces import (
    DiarizedTranscriptionSegment,
    SupportsTranscription,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.processor import cached_processor_from_config

from .base import Base
from .multimodal import MultiModalProcessingInfo

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.speech_to_text import SpeechToTextParams
    from vllm.inputs import PromptType


def _decode(text: str, model_config: "ModelConfig", return_format: str) -> Any:
    """Parse the raw model output with the HF processor, if it knows how to."""
    processor = cached_processor_from_config(model_config)
    if "return_format" not in inspect.signature(processor.decode).parameters:
        return text
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    return processor.decode(ids, return_format=return_format)


class TranscriptionMixin(SupportsTranscription, Base):
    supports_diarized_transcription = True
    # HF processors take no language, so whether one works is up to the model
    supported_languages = LANGUAGES

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip SupportsTranscription.__init__ and call the next class in MRO
        super(SupportsTranscription, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: "ModelConfig", task_type: str
    ) -> SpeechToTextConfig:
        info = MULTIMODAL_REGISTRY.get_processing_info(model_config)
        assert isinstance(info, MultiModalProcessingInfo)
        sample_rate = info._get_audio_sampling_rate()
        max_length = info._get_max_audio_length()
        if max_length is None:
            # Chunking would break speaker IDs and timestamps across the chunks
            return SpeechToTextConfig(
                sample_rate=sample_rate,
                max_audio_clip_s=None,
                min_energy_split_window_size=None,
            )
        return SpeechToTextConfig(
            sample_rate=sample_rate, max_audio_clip_s=int(max_length / sample_rate)
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: "SpeechToTextParams") -> "PromptType":
        Base.check_version("5.23.0.dev0", "the transcription API")
        if stt_params.task_type != "transcribe":
            raise ValueError(f"{cls.__name__} does not support translation.")

        processor = cached_processor_from_config(stt_params.model_config)
        if not hasattr(processor, "apply_transcription_request"):
            raise ValueError(
                f"{type(processor).__name__} does not implement "
                "`apply_transcription_request`, so the transcription API is not "
                "available for this model."
            )

        audio = stt_params.audio
        sample_rate = stt_params.stt_config.sample_rate
        context = "\n".join(
            text.strip()
            for text in (stt_params.request_prompt, stt_params.hotwords)
            if text and text.strip()
        )
        [prompt] = processor.apply_transcription_request(
            audio, prompt=context or None, tokenize=False
        )
        # The HF processor fills this in, but its text doesn't reach the prompt
        if duration_token := getattr(processor, "audio_duration_token", None):
            prompt = prompt.replace(duration_token, f"{len(audio) / sample_rate:.2f}")
        return TextPrompt(
            prompt=prompt, multi_modal_data={"audio": (audio, sample_rate)}
        )

    @classmethod
    def post_process_output(cls, text: str, model_config: "ModelConfig") -> str:
        return _decode(text, model_config, "transcription_only").strip()

    @classmethod
    def parse_diarized_transcript(
        cls, text: str, model_config: "ModelConfig"
    ) -> list[DiarizedTranscriptionSegment]:
        segments = _decode(text, model_config, "parsed")
        # Unparsable output comes back as a string
        if not isinstance(segments, list):
            return []
        return [
            DiarizedTranscriptionSegment(
                start=segment["Start"],
                end=segment["End"],
                speaker=str(segment.get("Speaker", 0)),
                text=segment["Content"].strip(),
            )
            for segment in segments
            if segment["Content"].strip()
        ]
