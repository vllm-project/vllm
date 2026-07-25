# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vllm.config.utils import config

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig


@dataclass
class TextToSpeechParams:
    """Parameters consumed by ``generate_audio()``.

    TTS models receive this object instead of raw API fields, so new
    parameters can be added here without changing the
    ``generate_audio`` signature.
    """

    text: str
    """Text to synthesise into speech."""

    tts_config: TextToSpeechConfig
    """Server-level text-to-speech configuration."""

    model_config: ModelConfig
    """Model configuration."""

    language: str | None = None
    """ISO 639-1 language code for the target language."""

    speaker_wav: np.ndarray | None = None
    """Reference audio waveform for voice cloning (6 s recommended)."""

    speaker_wav_sr: int = 24000
    """Sample rate of *speaker_wav*."""

    speed: float = 1.0
    """Speech rate multiplier (1.0 = natural speed)."""

    emotion: str | None = None
    """Optional emotion hint; model-dependent."""


@config
class TextToSpeechConfig:
    """Configuration for text-to-speech models."""

    sample_rate: int = 24_000
    """Output audio sample rate in Hz.  XTTS-v2 natively produces 24 kHz."""

    max_text_tokens: int = 400
    """Maximum number of text tokens accepted in a single request."""

    max_mel_tokens: int = 600
    """Maximum number of mel/audio tokens the model may generate."""

    speaker_embedding_dim: int = 512
    """Dimensionality of the speaker conditioning vector."""

    default_speed: float = 1.0
    """Default speech rate multiplier."""

    languages: list[str] = field(default_factory=lambda: [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs",
        "ar", "zh", "hu", "ko", "ja", "hi",
    ])
    """ISO 639-1 codes for supported languages (XTTS-v2 default set)."""
