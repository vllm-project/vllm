# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Silero voice activity detection (VAD) and VAD-based audio chunking."""

import threading

import numpy as np

from vllm.config.speech_to_text import SpeechToTextConfig, VADConfig
from vllm.multimodal.audio import split_audio
from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.torch_utils import set_default_torch_num_threads

try:
    # silero_vad calls torch.set_num_threads(1) when imported.
    with set_default_torch_num_threads():
        import silero_vad
except ImportError:
    silero_vad = PlaceholderModule("silero_vad")  # type: ignore[assignment]


class VAD:
    def __init__(self):
        self.model = None
        self._get_speech_timestamps = None

    def _ensure_loaded(self) -> None:
        if self._get_speech_timestamps is not None and self.model is not None:
            return

        # The sequence model runs one ONNX call per block of frames
        # instead of one per 32ms frame, so the GIL is released for
        # almost all of the VAD compute.
        self.model = silero_vad.load_silero_vad(sequence=True)
        self._get_speech_timestamps = silero_vad.get_speech_timestamps_sequence

    def ensure_loaded(self) -> None:
        self._ensure_loaded()

    def get_speech_timestamps(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        vad_config: VADConfig,
    ) -> list[dict]:
        if not vad_config.enabled:
            raise RuntimeError("VAD is not enabled.")

        self._ensure_loaded()
        if self._get_speech_timestamps is None or self.model is None:
            raise RuntimeError("VAD is not enabled.")

        return self._get_speech_timestamps(
            audio_data,
            self.model,
            sampling_rate=sample_rate,
            threshold=vad_config.threshold,
            neg_threshold=vad_config.neg_threshold,
            min_speech_duration_ms=vad_config.min_speech_duration_ms,
            max_speech_duration_s=vad_config.max_speech_duration_s,
            min_silence_duration_ms=vad_config.min_silence_duration_ms,
            speech_pad_ms=vad_config.speech_pad_ms,
            min_silence_at_max_speech=vad_config.min_silence_at_max_speech_ms,
            use_max_poss_sil_at_max_speech=(vad_config.use_max_poss_sil_at_max_speech),
        )


class ThreadLocalVADProvider:
    """Create and cache one loaded `VAD` instance per worker thread.

    Each worker thread gets its own ONNX Runtime session so concurrent
    requests run VAD in parallel without sharing a session or serializing
    behind a lock.
    """

    def __init__(self) -> None:
        self._init_lock = threading.Lock()
        self._thread_local = threading.local()

    def get(self) -> VAD:
        vad = getattr(self._thread_local, "vad", None)
        if vad is None:
            vad = VAD()
            # Loading the silero model is not thread-safe (downloads/caches),
            # so serialize only the one-time load, not inference.
            with self._init_lock:
                vad.ensure_loaded()
            self._thread_local.vad = vad

        return vad


def _join_pieces(audio_data: np.ndarray, pieces: list[tuple[int, int]]) -> np.ndarray:
    """Join `(start, end)` sample ranges of the original audio into one chunk."""
    return np.concatenate(
        [audio_data[..., start:end] for start, end in pieces], axis=-1
    )


def _pack_pieces(
    audio_data: np.ndarray,
    pieces: list[tuple[int, int]],
    max_chunk_samples: int,
) -> list[np.ndarray]:
    """Greedily pack consecutive pieces into chunks of at most
    `max_chunk_samples`, dropping the audio between pieces."""
    chunks: list[np.ndarray] = []
    current: list[tuple[int, int]] = []
    current_samples = 0
    for start, end in pieces:
        if current and current_samples + (end - start) > max_chunk_samples:
            chunks.append(_join_pieces(audio_data, current))
            current, current_samples = [], 0
        current.append((start, end))
        current_samples += end - start
    if current:
        chunks.append(_join_pieces(audio_data, current))
    return chunks


def split_audio_with_vad(
    duration: float,
    asr_config: SpeechToTextConfig,
    vad: VAD | None,
    vad_config: VADConfig,
    audio_data: np.ndarray,
    sample_rate: int,
) -> list[np.ndarray]:
    """Split audio according to the configured VAD and chunking policy.
    * Silero VAD improves noise robustness and hallucination robustness.
    * RMS split is used to adhere to the max_clip_duration_s limit.
    Using Silero VAD for satisfying max_clip_duration_s limit didnt work well
    hence we have to use 2 stage chunking for best quality.

    Behavior depends on whether chunking is enabled via
    `SpeechToTextConfig.allow_audio_chunking` and whether VAD is enabled:

    * VAD + RMS: run Silero VAD first, then further split only those
      individual speech segments that exceed `max_audio_clip_s`, and pack
      consecutive speech into chunks of up to `max_audio_clip_s` with the
      non-speech between them removed. Packing keeps the number of chunks
      bounded by the amount of speech, whatever the VAD parameters. If no
      speech is detected, return an empty list.
    * RMS only: run `split_audio()` on the full waveform when the
      overall clip duration exceeds `max_audio_clip_s`.
    * VAD only: remove non-speech by concatenating detected speech spans into
      a single trimmed chunk, since these models expect the whole clip in one
      request. If no speech is detected, return an empty list.
    * no chunking: return the original audio as a single chunk.

    An empty list means the caller should return an empty transcript.
    """
    if asr_config.allow_audio_chunking:
        assert asr_config.max_audio_clip_s is not None
        assert asr_config.min_energy_split_window_size is not None

        if vad_config.enabled:
            # silero VAD + RMS split
            assert vad is not None

            speech_timestamps = vad.get_speech_timestamps(
                audio_data, sample_rate, vad_config
            )
            pieces: list[tuple[int, int]] = []
            for timestamp in speech_timestamps:
                start = timestamp["start"]
                end = timestamp["end"]
                vad_duration_s = (end - start) / sample_rate
                if vad_duration_s > asr_config.max_audio_clip_s:
                    partial_chunk_output = split_audio(
                        audio_data[start:end],
                        sample_rate,
                        asr_config.max_audio_clip_s,
                        asr_config.overlap_chunk_second,
                        asr_config.min_energy_split_window_size,
                    )
                    offset = start
                    for partial_chunk in partial_chunk_output:
                        pieces.append((offset, offset + partial_chunk.shape[-1]))
                        offset += partial_chunk.shape[-1]
                else:
                    pieces.append((start, end))
            return _pack_pieces(
                audio_data,
                pieces,
                int(asr_config.max_audio_clip_s * sample_rate),
            )
        else:
            if duration > asr_config.max_audio_clip_s:
                # RMS split only
                partial_chunk_output = split_audio(
                    audio_data,
                    sample_rate,
                    asr_config.max_audio_clip_s,
                    asr_config.overlap_chunk_second,
                    asr_config.min_energy_split_window_size,
                )
                return partial_chunk_output
            else:
                return [audio_data]
    else:
        if vad_config.enabled:
            # silero VAD only
            assert vad is not None

            speech_timestamps = vad.get_speech_timestamps(
                audio_data, sample_rate, vad_config
            )
            if len(speech_timestamps) > 0:
                speech_pieces = [
                    (timestamp["start"], timestamp["end"])
                    for timestamp in speech_timestamps
                ]
                return [_join_pieces(audio_data, speech_pieces)]
            else:
                return []
        else:
            return [audio_data]
