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
"""Inference-only Qwen3-ASR realtime model.

Implements SDK-style streaming: each inference step sends *all accumulated
audio* plus a prefix of previously decoded text (with a small rollback to
let the model correct boundary tokens).  This mirrors the approach used in
the official Qwen3-ASR SDK and validated to produce ~90% similarity to
single-shot transcription vs ~69% for independent fixed-size segments.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Mapping

import numpy as np
import torch

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.inputs.data import PromptType, TokensPrompt
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

logger = init_logger(__name__)

_PRE_ALLOCATE_BUFFER_SIZE_IN_S = 60
_DEFAULT_SEGMENT_DURATION_S = 2.0
_DEFAULT_UNFIXED_CHUNKS = 2
_DEFAULT_ROLLBACK_TOKENS = 5
_MAX_AUDIO_ACCUMULATION_S = 300.0
_MAX_PREFIX_TOKENS = 1024


class Qwen3ASRRealtimeBuffer:
    """Audio buffer for Qwen3-ASR realtime streaming.

    Accumulates audio samples and signals when a new chunk has arrived
    that warrants re-inference over the full accumulated audio.
    """

    def __init__(self, sampling_rate: int, segment_duration_s: float = 2.0):
        self._sampling_rate = sampling_rate
        self._segment_size = int(segment_duration_s * sampling_rate)

        self._buffer_size = _PRE_ALLOCATE_BUFFER_SIZE_IN_S * sampling_rate
        self._buffer: np.ndarray = np.empty(self._buffer_size, dtype=np.float32)
        self._filled_len = 0
        self._consumed_len = 0

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

    def has_new_segment(self) -> bool:
        """True when enough new audio has arrived since last read."""
        return (self._filled_len - self._consumed_len) >= self._segment_size

    def read_accumulated(self) -> np.ndarray | None:
        """Return ALL accumulated audio and mark the new chunk as consumed."""
        if not self.has_new_segment():
            return None
        self._consumed_len = self._filled_len
        return self._buffer[: self._filled_len].copy()

    def flush(self) -> np.ndarray | None:
        """Return any remaining accumulated audio (final segment)."""
        if self._filled_len == 0:
            return None
        if self._filled_len == self._consumed_len:
            return None
        self._consumed_len = self._filled_len
        return self._buffer[: self._filled_len].copy()

    @property
    def accumulated_duration_s(self) -> float:
        return self._filled_len / self._sampling_rate

    def trim_to(self, max_seconds: float) -> None:
        """Discard oldest audio to keep total duration under max_seconds."""
        max_samples = int(max_seconds * self._sampling_rate)
        if self._filled_len <= max_samples:
            return
        discard = self._filled_len - max_samples
        self._buffer[:max_samples] = self._buffer[discard : self._filled_len]
        self._filled_len = max_samples
        self._consumed_len = max(0, self._consumed_len - discard)


def _rollback_prefix(raw_decoded: str, tokenizer, rollback_tokens: int) -> str:
    """Tokenize raw_decoded, drop the last N tokens, decode back."""
    if not raw_decoded:
        return ""
    token_ids = tokenizer.encode(raw_decoded)
    end_idx = max(0, len(token_ids) - rollback_tokens)
    if end_idx == 0:
        return ""
    prefix = tokenizer.decode(token_ids[:end_idx])
    while "\ufffd" in prefix and end_idx > 0:
        end_idx -= 1
        prefix = tokenizer.decode(token_ids[:end_idx]) if end_idx > 0 else ""
    return prefix


def _cap_prefix_tokens(prefix: str, tokenizer, max_tokens: int) -> str:
    """Truncate prefix from the left so it stays under max_tokens."""
    if not prefix:
        return ""
    token_ids = tokenizer.encode(prefix)
    if len(token_ids) <= max_tokens:
        return prefix
    return tokenizer.decode(token_ids[-max_tokens:])


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
    realtime_max_tokens = 128

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
        *,
        language: str | None = None,
        prompt: str | None = None,
        segment_duration_s: float = _DEFAULT_SEGMENT_DURATION_S,
        rollback_tokens: int = _DEFAULT_ROLLBACK_TOKENS,
        unfixed_chunks: int = _DEFAULT_UNFIXED_CHUNKS,
        max_audio_s: float = _MAX_AUDIO_ACCUMULATION_S,
        max_prefix_tokens: int = _MAX_PREFIX_TOKENS,
        prefix_texts: deque[str] | None = None,
    ) -> AsyncGenerator[PromptType, None]:
        """SDK-style streaming: growing audio + prefix rollback.

        Matches the official Qwen3-ASR SDK streaming format:
          system: {context}
          user:   {audio}
          assistant: [language {Lang}<asr_text>]{prefix}

        Each yield sends the full accumulated audio along with a text
        prefix from the previous decode (minus a small rollback).  The
        model re-transcribes the entire audio every step, producing
        progressively better output as context grows.

        After each yield, reads from ``input_stream`` to collect the
        model's generated tokens and update ``raw_decoded`` for the
        next step's prefix.  An empty list ``[]`` in the stream signals
        that generation for the current segment is complete.
        """
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        sampling_rate = feature_extractor.sampling_rate
        tokenizer = cached_tokenizer_from_config(model_config)

        buffer = Qwen3ASRRealtimeBuffer(
            sampling_rate=sampling_rate,
            segment_duration_s=segment_duration_s,
        )

        audio_placeholder = cls.get_placeholder_str("audio", 0)

        _chatml_delims = ("<|im_start|>", "<|im_end|>")
        context = prompt or ""
        for d in _chatml_delims:
            context = context.replace(d, "")
        lang_prefix = ""
        if language is not None:
            full_lang = cls.supported_languages.get(language, language)
            for d in _chatml_delims:
                full_lang = full_lang.replace(d, "")
            lang_prefix = f"language {full_lang}{_ASR_TEXT_TAG}"

        prompt_base = (
            f"<|im_start|>system\n{context}<|im_end|>\n"
            f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n{lang_prefix}"
        )

        raw_decoded = ""
        chunk_id = 0

        async for audio_chunk in audio_stream:
            buffer.write_audio(audio_chunk)

            if max_audio_s > 0:
                buffer.trim_to(max_audio_s)

            while (accumulated := buffer.read_accumulated()) is not None:
                if chunk_id < unfixed_chunks:
                    prefix = ""
                else:
                    prefix = _rollback_prefix(raw_decoded, tokenizer, rollback_tokens)
                    prefix = _cap_prefix_tokens(prefix, tokenizer, max_prefix_tokens)

                full_prompt = prompt_base + prefix
                prompt_token_ids = tokenizer.encode(full_prompt)

                if prefix_texts is not None:
                    prefix_texts.append(prefix)
                yield TokensPrompt(
                    prompt_token_ids=prompt_token_ids,
                    multi_modal_data={"audio": accumulated},
                )

                gen_text = await cls._collect_generation(input_stream, tokenizer)
                raw_decoded = (prefix + gen_text).rstrip("\n").rstrip()
                chunk_id += 1

        remaining = buffer.flush()
        if remaining is not None and len(remaining) > 0:
            if chunk_id < unfixed_chunks:
                prefix = ""
            else:
                prefix = _rollback_prefix(raw_decoded, tokenizer, rollback_tokens)
                prefix = _cap_prefix_tokens(prefix, tokenizer, max_prefix_tokens)

            full_prompt = prompt_base + prefix
            prompt_token_ids = tokenizer.encode(full_prompt)

            if prefix_texts is not None:
                prefix_texts.append(prefix)
            yield TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"audio": remaining},
            )

    @staticmethod
    async def _collect_generation(
        input_stream: asyncio.Queue[list[int]],
        tokenizer,
    ) -> str:
        """Read generated token IDs from the engine until segment completes.

        An empty list ``[]`` in the stream signals completion.  Returns
        the decoded text for the entire segment.
        """
        all_ids: list[int] = []
        while True:
            token_ids = await input_stream.get()
            if not token_ids:
                break
            all_ids.extend(token_ids)
        text = tokenizer.decode(all_ids, skip_special_tokens=True) if all_ids else ""
        return text

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
