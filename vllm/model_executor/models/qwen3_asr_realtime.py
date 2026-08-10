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
import hashlib
import os
from collections import deque
from collections.abc import AsyncGenerator, Mapping

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
# Full-prefix mode is the Qwen3-ASR streaming default. ``None`` (or an
# explicit session value of 0) means that confirmed text is never silently
# removed. A positive value remains available for deliberately bounded,
# non-standard long-running sessions.
_MAX_PREFIX_TOKENS: int | None = None


def _normalize_realtime_text(text: str) -> str:
    """Remove protocol markers and formatting artifacts from ASR text."""
    if not text:
        return ""
    text = text.replace("\r", "").replace("\n", "")
    if _ASR_TEXT_TAG in text:
        text = text.rsplit(_ASR_TEXT_TAG, 1)[1]
    elif text.startswith("language "):
        text = ""
    return text.strip()


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
        self._last_new_segment = np.empty(0, dtype=np.float32)
        self._last_active_window_start = 0
        self._last_active_window_complete = False

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
        # Advance by exactly one model segment instead of consuming every
        # transport sample currently buffered. This keeps AuT window boundaries
        # aligned even when WebSocket packet sizes do not divide the segment.
        self._consumed_len += self._segment_size
        return self._buffer[: self._consumed_len].copy()

    def read_new_segment(self) -> np.ndarray | None:
        if not self.has_new_segment():
            return None
        start = self._consumed_len
        self._consumed_len += self._segment_size
        return self._buffer[start:self._consumed_len].copy()

    def read_active_window(self, window_size: int) -> np.ndarray | None:
        """Return the current fixed window after consuming one new segment.

        Completed windows are never returned again.  While a window is
        active, every call returns that window from its fixed left boundary so
        its bidirectional AuT representation can be recomputed exactly.
        """
        if not self.has_new_segment():
            return None
        new_start = self._consumed_len
        self._consumed_len += self._segment_size
        window_start = ((self._consumed_len - 1) // window_size) * window_size
        self._last_new_segment = self._buffer[
            new_start:self._consumed_len
        ].copy()
        self._last_active_window_start = window_start
        self._last_active_window_complete = self._consumed_len % window_size == 0
        return self._buffer[window_start:self._consumed_len].copy()

    def flush(self) -> np.ndarray | None:
        """Return any remaining accumulated audio (final segment)."""
        if self._filled_len == 0:
            return None
        if self._filled_len == self._consumed_len:
            return None
        self._consumed_len = self._filled_len
        return self._buffer[: self._filled_len].copy()

    def flush_new(self) -> np.ndarray | None:
        if self._filled_len == self._consumed_len:
            return None
        start = self._consumed_len
        self._consumed_len = self._filled_len
        tail = self._buffer[start:self._filled_len].copy()
        # The Qwen3-ASR feature extractor cannot process an arbitrarily short
        # standalone waveform.  Append-only mode pads only the final new item
        # to a complete inference window; previously cached audio is untouched.
        if tail.shape[0] < self._segment_size:
            tail = np.pad(tail, (0, self._segment_size - tail.shape[0]))
        return tail

    def flush_active_window(self, window_size: int) -> np.ndarray | None:
        """Return the final partial fixed window and mark it committable."""
        if self._filled_len == self._consumed_len:
            return None
        new_start = self._consumed_len
        self._consumed_len = self._filled_len
        window_start = ((self._consumed_len - 1) // window_size) * window_size
        self._last_new_segment = self._buffer[
            new_start:self._consumed_len
        ].copy()
        self._last_active_window_start = window_start
        self._last_active_window_complete = True
        active = self._buffer[window_start:self._consumed_len].copy()
        # Keep the final real sample count in the trace, but pad a very short
        # active window so the Whisper frontend can process it.
        if active.shape[0] < self._segment_size:
            active = np.pad(active, (0, self._segment_size - active.shape[0]))
        return active

    @property
    def consumed_len(self) -> int:
        return self._consumed_len

    @property
    def last_new_segment(self) -> np.ndarray:
        return self._last_new_segment

    @property
    def last_active_window_start(self) -> int:
        return self._last_active_window_start

    @property
    def last_active_window_complete(self) -> bool:
        return self._last_active_window_complete

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


def _cap_prefix_tokens(
    prefix: str, tokenizer, max_tokens: int | None
) -> str:
    """Apply an explicitly requested prefix cap.

    ``None`` and non-positive values preserve the complete prefix. This makes
    unlimited mode intentional instead of relying on Python's ``[-0:]`` slice
    behaviour.
    """
    if not prefix:
        return ""
    if max_tokens is None or max_tokens <= 0:
        return prefix
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
        aut_window_frames = os.environ.get("QWEN3_ASR_AUT_WINDOW_FRAMES")
        if aut_window_frames is not None:
            frames = int(aut_window_frames)
            if frames <= 0:
                raise ValueError(
                    "QWEN3_ASR_AUT_WINDOW_FRAMES must be a positive integer"
                )
            audio_config = (
                vllm_config.model_config.hf_config.thinker_config.audio_config
            )
            audio_config.n_window_infer = frames
            logger.info(
                "QWEN3_ASR_AUT_WINDOW_OVERRIDE n_window_infer=%d", frames
            )
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
        max_prefix_tokens: int | None = _MAX_PREFIX_TOKENS,
        prefix_texts: deque[str] | None = None,
        request_id: str | None = None,
        segment_traces: deque[dict] | None = None,
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
        if not np.isfinite(segment_duration_s) or segment_duration_s <= 0:
            raise ValueError("realtime segment duration must be a finite positive number")
        logger.info(
            "QWEN3_ASR_REALTIME_SEGMENT_DURATION segment_duration_s=%.6f",
            segment_duration_s,
        )

        buffer = Qwen3ASRRealtimeBuffer(
            sampling_rate=sampling_rate,
            segment_duration_s=segment_duration_s,
        )

        audio_placeholder = cls.get_placeholder_str("audio", 0)
        # The first audio item opens the multimodal span.  In append-only
        # audio-history mode, later updates must extend that span instead of
        # opening a second one with another audio_start marker.
        audio_history_append_placeholder = audio_placeholder.replace(
            "<|audio_start|>", "", 1
        )

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

        stream_mode = os.environ.get(
            "QWEN3_ASR_REALTIME_MODE", "cumulative"
        ).strip().lower()
        if stream_mode not in {
            "cumulative",
            "delta_turn",
            "audio_history_kv",
            "aut_stable_window_kv",
        }:
            raise ValueError(
                "QWEN3_ASR_REALTIME_MODE must be cumulative, delta_turn, "
                "audio_history_kv, or aut_stable_window_kv, "
                f"got {stream_mode!r}"
            )
        aut_stable_window_mode = stream_mode == "aut_stable_window_kv"
        aut_window_s = float(
            os.environ.get("QWEN3_ASR_AUT_STABLE_WINDOW_S", "8")
        )
        if aut_stable_window_mode:
            if not np.isfinite(aut_window_s) or aut_window_s <= 0:
                raise ValueError(
                    "QWEN3_ASR_AUT_STABLE_WINDOW_S must be finite and positive"
                )
            aut_window_samples = int(round(aut_window_s * sampling_rate))
            segment_samples = int(round(segment_duration_s * sampling_rate))
            if aut_window_samples % segment_samples != 0:
                raise ValueError(
                    "AuT stable window must be an integer multiple of the "
                    "realtime segment: "
                    f"window={aut_window_s}s segment={segment_duration_s}s"
                )
            if max_audio_s > 0:
                raise ValueError(
                    "aut_stable_window_kv requires max_audio_s=0 because "
                    "fixed AuT window boundaries cannot move after commit"
                )
            logger.info(
                "QWEN3_ASR_AUT_STABLE_WINDOW_CONFIG window_s=%.6f "
                "window_samples=%d segment_samples=%d",
                aut_window_s,
                aut_window_samples,
                segment_samples,
            )
        else:
            aut_window_samples = 0
        if (
            stream_mode in {"audio_history_kv", "aut_stable_window_kv"}
            and audio_history_append_placeholder == audio_placeholder
        ):
            raise ValueError(
                "audio-history append requires an <|audio_start|> marker in "
                "the model audio placeholder"
            )
        delta_turn_prompt = (
            f"<|im_end|>\n<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        raw_decoded = ""
        chunk_id = 0
        last_emitted_samples = 0

        async for audio_chunk in audio_stream:
            buffer.write_audio(audio_chunk)

            if max_audio_s > 0:
                buffer.trim_to(max_audio_s)

            while (accumulated := (
                buffer.read_active_window(aut_window_samples)
                if aut_stable_window_mode
                else buffer.read_new_segment()
                if stream_mode == "audio_history_kv"
                else buffer.read_accumulated()
            )) is not None:
                if stream_mode == "delta_turn" or chunk_id < unfixed_chunks:
                    prefix = ""
                else:
                    prefix = _rollback_prefix(raw_decoded, tokenizer, rollback_tokens)
                    prefix = _cap_prefix_tokens(prefix, tokenizer, max_prefix_tokens)

                full_prompt = prompt_base + prefix
                if stream_mode == "delta_turn" and chunk_id > 0:
                    full_prompt = delta_turn_prompt
                elif (
                    stream_mode in {"audio_history_kv", "aut_stable_window_kv"}
                    and chunk_id > 0
                ):
                    full_prompt = (
                        f"{audio_history_append_placeholder}<|im_end|>\n"
                        f"<|im_start|>assistant\n{lang_prefix}{prefix}"
                    )
                prompt_token_ids = tokenizer.encode(full_prompt)

                if aut_stable_window_mode:
                    new_audio_start_sample = last_emitted_samples
                    new_audio = buffer.last_new_segment
                    llm_audio = accumulated
                    audio_samples = buffer.consumed_len
                elif stream_mode == "audio_history_kv":
                    # ``accumulated`` is segment-local in append-only mode.
                    # The previous implementation sliced it with a cumulative
                    # offset, making diagnostics report zero new samples and
                    # computing an empty new_audio hash after segment 1.
                    new_audio_start_sample = last_emitted_samples
                    new_audio = accumulated
                    llm_audio = accumulated
                    audio_samples = last_emitted_samples + len(accumulated)
                else:
                    new_audio_start_sample = last_emitted_samples
                    new_audio = accumulated[new_audio_start_sample:]
                    audio_samples = len(accumulated)
                if stream_mode == "delta_turn":
                    llm_audio = new_audio
                else:
                    llm_audio = accumulated if stream_mode != "audio_history_kv" else llm_audio
                audio_bytes = np.ascontiguousarray(llm_audio).view(np.uint8)
                new_audio_bytes = np.ascontiguousarray(new_audio).view(np.uint8)
                audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
                new_audio_sha256 = hashlib.sha256(new_audio_bytes).hexdigest()

                segment_id = chunk_id + 1
                trace = {
                    "segment_id": segment_id,
                    "audio_samples": audio_samples,
                    "audio_duration_s": audio_samples / sampling_rate,
                    "new_audio_samples": audio_samples - last_emitted_samples,
                    "new_audio_start_sample": new_audio_start_sample,
                    "new_audio_end_sample": audio_samples,
                    "emit_reason": "threshold",
                    "stream_mode": stream_mode,
                    "llm_audio_samples": len(llm_audio),
                    "text_prompt_tokens": len(prompt_token_ids),
                    "prefix_tokens": len(tokenizer.encode(prefix)),
                    "prefix_chars": len(prefix),
                    "audio_sha256": audio_sha256,
                    "new_audio_sha256": new_audio_sha256,
                    "audio_rms": float(np.sqrt(np.mean(np.square(accumulated)))),
                    "new_audio_rms": float(np.sqrt(np.mean(np.square(new_audio)))),
                    "aut_window_start_sample": (
                        buffer.last_active_window_start
                        if aut_stable_window_mode
                        else None
                    ),
                    "aut_window_samples": (
                        len(accumulated) if aut_stable_window_mode else None
                    ),
                    "aut_window_commit": (
                        buffer.last_active_window_complete
                        if aut_stable_window_mode
                        else False
                    ),
                }
                last_emitted_samples = audio_samples
                if segment_traces is not None:
                    segment_traces.append(trace)
                logger.debug(
                    "QWEN3_ASR_RT_SEGMENT_INPUT request_id=%s segment_id=%d "
                    "emit_reason=%s audio_samples=%d audio_duration_s=%.6f "
                    "new_audio_samples=%d new_audio_range_samples=[%d,%d) "
                    "new_audio_range_s=[%.6f,%.6f) audio_sha256=%s "
                    "new_audio_sha256=%s audio_rms=%.8f new_audio_rms=%.8f "
                    "text_prompt_tokens=%d prefix_tokens=%d prefix_chars=%d "
                    "prompt_text=%r prompt_token_ids=%s",
                    request_id or "-", segment_id, trace["emit_reason"],
                    audio_samples, trace["audio_duration_s"],
                    trace["new_audio_samples"], trace["new_audio_start_sample"],
                    trace["new_audio_end_sample"], new_audio_start_sample / sampling_rate,
                    audio_samples / sampling_rate, trace["audio_sha256"],
                    trace["new_audio_sha256"], trace["audio_rms"],
                    trace["new_audio_rms"], trace["text_prompt_tokens"],
                    trace["prefix_tokens"], trace["prefix_chars"], full_prompt,
                    prompt_token_ids,
                )
                logger.debug(
                    "QWEN3_ASR_RT_AB_INPUT request_id=%s segment_id=%d "
                    "stream_mode=%s cumulative_audio_samples=%d "
                    "llm_audio_samples=%d prompt_text=%r",
                    request_id or "-", segment_id, stream_mode,
                    audio_samples, len(llm_audio), full_prompt,
                )
                if aut_stable_window_mode:
                    logger.info(
                        "QWEN3_ASR_AUT_STABLE_WINDOW request_id=%s "
                        "segment_id=%d window_start_sample=%d "
                        "active_samples=%d new_samples=%d commit=%s",
                        request_id or "-",
                        segment_id,
                        buffer.last_active_window_start,
                        len(accumulated),
                        len(new_audio),
                        buffer.last_active_window_complete,
                    )

                if prefix_texts is not None:
                    prefix_texts.append(prefix)
                prompt = TokensPrompt(
                    prompt_token_ids=prompt_token_ids,
                    multi_modal_data={"audio": llm_audio},
                )
                yield prompt

                gen_text = await cls._collect_generation(
                    input_stream,
                    tokenizer,
                    request_id=request_id,
                    segment_id=segment_id,
                )
                # Normalize only after the complete candidate is assembled.
                # Stripping a generated fragment first removes the leading
                # separator carried by English continuation tokens, while full
                # candidate normalization remains safe for Chinese text.
                raw_decoded = _normalize_realtime_text(prefix + gen_text)
                chunk_id += 1

        remaining = (
            buffer.flush_active_window(aut_window_samples)
            if aut_stable_window_mode
            else buffer.flush_new()
            if stream_mode == "audio_history_kv"
            else buffer.flush()
        )
        if remaining is not None and len(remaining) > 0:
            if stream_mode == "delta_turn" or chunk_id < unfixed_chunks:
                prefix = ""
            else:
                prefix = _rollback_prefix(raw_decoded, tokenizer, rollback_tokens)
                prefix = _cap_prefix_tokens(prefix, tokenizer, max_prefix_tokens)

            full_prompt = prompt_base + prefix
            if stream_mode == "delta_turn" and chunk_id > 0:
                full_prompt = delta_turn_prompt
            elif (
                stream_mode in {"audio_history_kv", "aut_stable_window_kv"}
                and chunk_id > 0
            ):
                full_prompt = (
                    f"{audio_history_append_placeholder}<|im_end|>\n"
                    f"<|im_start|>assistant\n{lang_prefix}{prefix}"
                )
            prompt_token_ids = tokenizer.encode(full_prompt)

            if aut_stable_window_mode:
                new_audio_start_sample = last_emitted_samples
                new_audio = buffer.last_new_segment
                llm_audio = remaining
                audio_samples = buffer.consumed_len
            elif stream_mode == "audio_history_kv":
                new_audio_start_sample = last_emitted_samples
                new_audio = remaining
                llm_audio = remaining
                audio_samples = last_emitted_samples + len(remaining)
            else:
                new_audio_start_sample = last_emitted_samples
                new_audio = remaining[new_audio_start_sample:]
                audio_samples = len(remaining)
            if stream_mode == "delta_turn":
                llm_audio = new_audio
            else:
                llm_audio = remaining if stream_mode != "audio_history_kv" else llm_audio
            audio_bytes = np.ascontiguousarray(llm_audio).view(np.uint8)
            new_audio_bytes = np.ascontiguousarray(new_audio).view(np.uint8)
            audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
            new_audio_sha256 = hashlib.sha256(new_audio_bytes).hexdigest()

            segment_id = chunk_id + 1
            trace = {
                "segment_id": segment_id,
                "audio_samples": audio_samples,
                "audio_duration_s": audio_samples / sampling_rate,
                "new_audio_samples": audio_samples - last_emitted_samples,
                "new_audio_start_sample": new_audio_start_sample,
                "new_audio_end_sample": audio_samples,
                "emit_reason": "flush",
                "stream_mode": stream_mode,
                "llm_audio_samples": len(llm_audio),
                "text_prompt_tokens": len(prompt_token_ids),
                "prefix_tokens": len(tokenizer.encode(prefix)),
                "prefix_chars": len(prefix),
                "audio_sha256": audio_sha256,
                "new_audio_sha256": new_audio_sha256,
                "audio_rms": float(np.sqrt(np.mean(np.square(remaining)))),
                "new_audio_rms": float(np.sqrt(np.mean(np.square(new_audio)))),
                "aut_window_start_sample": (
                    buffer.last_active_window_start
                    if aut_stable_window_mode
                    else None
                ),
                "aut_window_samples": (
                    len(remaining) if aut_stable_window_mode else None
                ),
                "aut_window_commit": aut_stable_window_mode,
            }
            if segment_traces is not None:
                segment_traces.append(trace)
            logger.debug(
                "QWEN3_ASR_RT_SEGMENT_INPUT request_id=%s segment_id=%d "
                "emit_reason=%s audio_samples=%d audio_duration_s=%.6f "
                "new_audio_samples=%d new_audio_range_samples=[%d,%d) "
                "new_audio_range_s=[%.6f,%.6f) audio_sha256=%s "
                "new_audio_sha256=%s audio_rms=%.8f new_audio_rms=%.8f "
                "text_prompt_tokens=%d prefix_tokens=%d prefix_chars=%d "
                "prompt_text=%r prompt_token_ids=%s",
                request_id or "-", segment_id, trace["emit_reason"],
                audio_samples, trace["audio_duration_s"],
                trace["new_audio_samples"], trace["new_audio_start_sample"],
                trace["new_audio_end_sample"], new_audio_start_sample / sampling_rate,
                audio_samples / sampling_rate, trace["audio_sha256"],
                trace["new_audio_sha256"], trace["audio_rms"],
                trace["new_audio_rms"], trace["text_prompt_tokens"],
                trace["prefix_tokens"], trace["prefix_chars"], full_prompt,
                prompt_token_ids,
            )
            logger.debug(
                "QWEN3_ASR_RT_AB_INPUT request_id=%s segment_id=%d "
                "stream_mode=%s cumulative_audio_samples=%d "
                "llm_audio_samples=%d prompt_text=%r",
                request_id or "-", segment_id, stream_mode,
                audio_samples, len(llm_audio), full_prompt,
            )
            if aut_stable_window_mode:
                logger.info(
                    "QWEN3_ASR_AUT_STABLE_WINDOW request_id=%s "
                    "segment_id=%d window_start_sample=%d "
                    "active_samples=%d new_samples=%d commit=True",
                    request_id or "-",
                    segment_id,
                    buffer.last_active_window_start,
                    len(remaining),
                    len(new_audio),
                )

            if prefix_texts is not None:
                prefix_texts.append(prefix)
            prompt = TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"audio": llm_audio},
            )
            yield prompt

            # The final short segment uses the same token-stream handshake as
            # every full segment. Consume its completion marker so the prompt
            # generator can finish and the realtime connection can send done.
            await cls._collect_generation(
                input_stream,
                tokenizer,
                request_id=request_id,
                segment_id=segment_id,
            )

    @staticmethod
    async def _collect_generation(
        input_stream: asyncio.Queue[list[int]],
        tokenizer,
        request_id: str | None = None,
        segment_id: int | None = None,
    ) -> str:
        """Read generated token IDs from the engine until segment completes.

        An empty list ``[]`` in the stream signals completion.  Returns
        the decoded text for the entire segment and logs the exact token IDs.
        """
        all_ids: list[int] = []
        while True:
            token_ids = await input_stream.get()
            if not token_ids:
                break
            all_ids.extend(token_ids)
        text = tokenizer.decode(all_ids, skip_special_tokens=True) if all_ids else ""
        logger.debug(
            "QWEN3_ASR_RT_SEGMENT_LLM_OUTPUT request_id=%s segment_id=%s "
            "generated_tokens=%d token_ids=%s generated_text=%r",
            request_id or "-", segment_id if segment_id is not None else "-",
            len(all_ids), all_ids, text,
        )
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
