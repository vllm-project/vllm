# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The Qwen team.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3-ASR realtime model."""

import asyncio
import re
from collections.abc import AsyncGenerator, Mapping
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsRealtime,
)
from vllm.model_executor.models.qwen3_asr import (
    _ASR_TEXT_TAG,
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
    _get_feat_extract_output_lengths,
    _sanitize_transcription_user_text,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import _I, BaseMultiModalProcessorCache
from vllm.multimodal.inputs import MultiModalKwargsOptionalItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config

if TYPE_CHECKING:
    from vllm.entrypoints.speech_to_text.realtime.protocol import (
        RealtimeSessionConfig,
    )

logger = init_logger(__name__)

_PRE_ALLOCATE_BUFFER_SIZE_IN_S = 60

_REALTIME_PREAMBLE = re.compile(r"language ([^<\n]*?)<asr_text>")
_LANG_LEAD = "language "


class Qwen3ASRRealtimeBuffer:
    """Audio buffer for Qwen3-ASR realtime streaming.

    Accumulates audio samples and yields segments when enough
    audio has been buffered for processing.
    """

    def __init__(self, sampling_rate: int, segment_duration_s: float = 5.0):
        self._sampling_rate = sampling_rate
        self._segment_size = int(segment_duration_s * sampling_rate)

        self._buffer_size = _PRE_ALLOCATE_BUFFER_SIZE_IN_S * sampling_rate
        self._buffer: np.ndarray = np.empty(self._buffer_size, dtype=np.float32)
        self._filled_len = 0

    def write_audio(self, audio: np.ndarray) -> None:
        put_end = self._filled_len + len(audio)
        if put_end > self._buffer_size:
            new_size = max(self._buffer_size * 2, put_end)
            new_buffer = np.empty(new_size, dtype=np.float32)
            new_buffer[: self._filled_len] = self._buffer[: self._filled_len]
            self._buffer = new_buffer
            self._buffer_size = new_size

        self._buffer[self._filled_len : put_end] = audio
        self._filled_len = put_end

    def read_audio(self) -> np.ndarray | None:
        if self._filled_len < self._segment_size:
            return None

        segment = self._buffer[: self._segment_size].copy()
        remaining = self._filled_len - self._segment_size
        if remaining > 0:
            self._buffer[:remaining] = self._buffer[
                self._segment_size : self._filled_len
            ]
        self._filled_len = remaining
        return segment

    def flush(self) -> np.ndarray | None:
        if self._filled_len == 0:
            return None
        audio = self._buffer[: self._filled_len].copy()
        self._filled_len = 0
        return audio


class Qwen3ASRRealtimeMultiModalProcessor(Qwen3ASRMultiModalProcessor):
    def __init__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None:
        super().__init__(info, dummy_inputs, cache=None)

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsOptionalItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        audios = mm_kwargs.get("audio", [])
        assert len(audios) == 1, (
            f"Expected only one audio input for realtime, got {len(audios)}"
        )

        audio_data = audios[0]
        audio_feature_lengths = audio_data.get("audio_feature_lengths")
        if audio_feature_lengths is not None:
            if isinstance(audio_feature_lengths.data, torch.Tensor):
                audio_len = _get_feat_extract_output_lengths(
                    audio_feature_lengths.data
                ).item()
            else:
                audio_len = int(
                    _get_feat_extract_output_lengths(
                        torch.tensor(audio_feature_lengths.data)
                    ).item()
                )
        else:
            audio_len = 0

        # Get audio_pad token ID and expand placeholder in prompt_ids
        # so that MRoPE position computation matches seq_len.
        tokenizer = self.info.get_tokenizer()
        audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")

        # Find the audio_pad token position and expand it to audio_len tokens
        expanded_ids = list[int]()
        pad_start_idx = -1
        for i, tid in enumerate(prompt_ids):
            if tid == audio_pad_id and pad_start_idx == -1:
                pad_start_idx = i
                expanded_ids.extend([audio_pad_id] * audio_len)
            else:
                expanded_ids.append(tid)

        if pad_start_idx == -1:
            pad_start_idx = 0

        features_info = PlaceholderFeaturesInfo(
            modality="audio",
            item_idx=0,
            start_idx=pad_start_idx,
            tokens=audio_len * [audio_pad_id],
            is_embed=None,
        )
        return expanded_ids, {"audio": [features_info]}


# NOTE: A separate model class is required here because the multimodal
# processor registry binds one processor per model class. The realtime
# endpoint needs a different processor (Qwen3ASRRealtimeMultiModalProcessor)
# than the base transcription endpoint, so we register it on this subclass.
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRRealtimeMultiModalProcessor,
    info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRRealtimeGeneration(Qwen3ASRForConditionalGeneration, SupportsRealtime):
    realtime_max_tokens = 64

    _name_to_iso_cache: ClassVar[dict[str, str] | None] = None
    _known_names_cache: ClassVar[frozenset[str] | None] = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @classmethod
    def _get_realtime_prompt_template(
        cls,
        audio_placeholder: str | None,
        session_config: "RealtimeSessionConfig | None" = None,
    ) -> str:
        request_prompt = session_config.prompt if session_config is not None else None
        context = _sanitize_transcription_user_text(request_prompt or "")
        system_turn = f"<|im_start|>system\n{context}<|im_end|>\n" if context else ""

        prompt = (
            f"{system_turn}"
            f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        language = session_config.language if session_config is not None else None
        if language:
            full_lang_name = cls.supported_languages.get(language, language)
            prompt += f"language {full_lang_name}{_ASR_TEXT_TAG}"

        return prompt

    @classmethod
    def _name_to_iso(cls) -> dict[str, str]:
        if cls._name_to_iso_cache is None:
            cls._name_to_iso_cache = {
                name: code for code, name in cls.supported_languages.items()
            }
        return cls._name_to_iso_cache

    @classmethod
    def _known_names(cls) -> frozenset[str]:
        if cls._known_names_cache is None:
            cls._known_names_cache = frozenset(cls.supported_languages.values())
        return cls._known_names_cache

    @classmethod
    def _is_pending_preamble(cls, cand: str) -> bool:
        """Whether ``cand`` is a viable, not-yet-complete language preamble."""
        if cand and _LANG_LEAD.startswith(cand):
            return True
        if cand.startswith(_LANG_LEAD):
            rest = cand[len(_LANG_LEAD) :]
            if "<" in rest:
                name, _, tail = rest.partition("<")
                return name in cls._known_names() and _ASR_TEXT_TAG.startswith(
                    "<" + tail
                )
            return any(n.startswith(rest) for n in cls._known_names())
        return False

    @classmethod
    def _strip_pending_preamble(cls, text: str) -> str:
        """Withhold a trailing partial preamble at a segment boundary."""
        boundary = text.rfind("\n") + 1
        if cls._is_pending_preamble(text[boundary:]):
            return text[:boundary]
        return text

    @classmethod
    def clean_realtime_text(cls, raw_text: str) -> tuple[str, str | None]:
        """Strip ``language <name><asr_text>`` preambles and detect language.

        Anchored on the ``<asr_text>`` delimiter so the literal word "language"
        in speech is not mistaken for a preamble. A trailing in-progress
        preamble at a segment boundary is withheld until it completes (or is
        disqualified as ordinary speech).

        Args:
            raw_text: Full accumulated raw model text so far.

        Returns:
            Tuple of ``(clean_text, language_code)``; ``language_code`` is the
            most recently detected segment language as an ISO-639-1 code, or
            ``None`` if no complete preamble has been seen.
        """
        lang: str | None = None
        matches = list(_REALTIME_PREAMBLE.finditer(raw_text))
        if matches:
            lang = cls._name_to_iso().get(matches[-1].group(1).strip())
        clean = _REALTIME_PREAMBLE.sub("", raw_text)
        clean = cls._strip_pending_preamble(clean)
        return clean, lang

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
        session_config: "RealtimeSessionConfig | None" = None,
    ) -> AsyncGenerator[PromptType, None]:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        sampling_rate = feature_extractor.sampling_rate
        tokenizer = cached_tokenizer_from_config(model_config)

        # Use a small segment size for low-latency streaming.
        segment_duration_s = 5.0
        buffer = Qwen3ASRRealtimeBuffer(
            sampling_rate=sampling_rate,
            segment_duration_s=segment_duration_s,
        )

        audio_placeholder = cls.get_placeholder_str("audio", 0)
        prompt_template = cls._get_realtime_prompt_template(
            audio_placeholder,
            session_config,
        )

        prompt_token_ids = tokenizer.encode(prompt_template)

        async for audio_chunk in audio_stream:
            buffer.write_audio(audio_chunk)

            while (segment := buffer.read_audio()) is not None:
                yield TokensPrompt(
                    prompt_token_ids=prompt_token_ids,
                    multi_modal_data={"audio": segment},
                )

        remaining = buffer.flush()
        if remaining is not None and len(remaining) > 0:
            yield TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"audio": remaining},
            )

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        return SpeechToTextConfig(
            max_audio_clip_s=None,
            sample_rate=feature_extractor.sampling_rate,
            min_energy_split_window_size=None,
        )
