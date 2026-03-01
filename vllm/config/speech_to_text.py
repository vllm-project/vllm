# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.config.utils import config


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

    default_sampling_params: dict[str, float | int] | None = None
    """Optional default sampling parameters for speech-to-text requests.
    These values are only used when the request omits the corresponding fields.
    """

    skip_reading_prefix_cache: bool = False
    """Whether to skip reading from the prefix cache for transcription.
    This can be used by specific models that are sensitive to cache reuse.
    """

    @property
    def allow_audio_chunking(self) -> bool:
        return (
            self.min_energy_split_window_size is not None
            and self.max_audio_clip_s is not None
        )
