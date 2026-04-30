# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.config.utils import config

if TYPE_CHECKING:
    import numpy as np

    from vllm.config.model import ModelConfig


@dataclass
class SpeechToTextParams:
    """All parameters consumed by ``get_generation_prompt()``.

    ``TranscriptionRequest.build_stt_params()`` constructs this object,
    mapping API-level fields into typed attributes.  Models only receive
    this object, so new parameters can be added here without changing the
    ``get_generation_prompt`` signature.
    """

    audio: np.ndarray
    """Resampled audio waveform for a single chunk."""

    stt_config: SpeechToTextConfig
    """Server-level speech-to-text configuration."""

    model_config: ModelConfig
    """Model configuration."""

    language: str | None = None
    """ISO 639-1 language code (validated / auto-detected)."""

    hotwords: str | None = None
    """
    hotwords refers to a list of important words or phrases that the model
    should pay extra attention to during transcription.
    """

    task_type: str = "transcribe"
    """``"transcribe"`` or ``"translate"``."""

    request_prompt: str = ""
    """Optional text prompt to guide the model."""

    to_language: str | None = None
    """Target language for translation (model-dependent)."""


@config
class SpeechToTextConfig:
    """Configuration for speech-to-text models."""

    sample_rate: float = 16_000
    """Sample rate (Hz) to resample input audio to. Most speech models expect
    16kHz audio input. The input audio will be automatically resampled to this
    rate before processing."""

    max_audio_clip_s: int | None = 30
    """Maximum duration in seconds for a single audio clip without chunking.
    Audio longer than this will be split into smaller chunks if
    `allow_audio_chunking` evaluates to True, otherwise it will be rejected. 
    `None` means audio duration can be unlimited and won't be chunked."""

    overlap_chunk_second: int = 1
    """Overlap duration in seconds between consecutive audio chunks when
    splitting long audio. This helps maintain context across chunk boundaries
    and improves transcription quality at split points."""

    min_energy_split_window_size: int | None = 1600
    """Window size in samples for finding low-energy (quiet) regions to split
    audio chunks. The algorithm looks for the quietest moment within this
    window to minimize cutting through speech. Default 1600 samples ≈ 100ms
    at 16kHz. If None, no chunking will be done."""

    @property
    def allow_audio_chunking(self) -> bool:
        return (
            self.min_energy_split_window_size is not None
            and self.max_audio_clip_s is not None
        )
