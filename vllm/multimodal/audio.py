# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import threading
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch

from vllm.config.speech_to_text import SpeechToTextConfig, VADConfig
from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.torch_utils import set_default_torch_num_threads

try:
    import av as av
except ImportError:
    av = PlaceholderModule("av")  # type: ignore[assignment]

try:
    import scipy.signal as scipy_signal
except ImportError:
    scipy_signal = PlaceholderModule("scipy").placeholder_attr("signal")  # type: ignore[assignment]

try:
    import soxr as soxr
except ImportError:
    soxr = PlaceholderModule("soxr")  # type: ignore[assignment]

try:
    import torchaudio
except ImportError:
    torchaudio = PlaceholderModule("torchaudio")  # type: ignore[assignment]


# ============================================================
# Aligned with `librosa.get_duration` function
def get_audio_duration(*, y: npt.NDArray[np.floating], sr: float = 22050) -> float:
    """Get the duration of an audio array in seconds.

    Args:
        y: Audio time series. Can be 1D (samples,) or 2D (channels, samples).
        sr: Sample rate of the audio in Hz.

    Returns:
        Duration of the audio in seconds.

    """
    n_samples = y.shape[-1]
    return float(n_samples) / sr


class ChannelReduction(str, Enum):
    """Method to reduce multi-channel audio to target channels."""

    MEAN = "mean"  # Average across channels (default, preserves energy balance)
    FIRST = "first"  # Take first channel only
    MAX = "max"  # Take max value across channels
    SUM = "sum"  # Sum across channels


@dataclass
class AudioSpec:
    """Specification for target audio format.

    This dataclass defines the expected audio format for a model's feature
    extractor. It is used to normalize audio data before processing.

    Attributes:
        target_channels: Number of output channels. None means passthrough
            (no normalization). 1 = mono, 2 = stereo, etc.
        channel_reduction: Method to reduce channels when input has more
            channels than target. Only used when reducing channels.

    """

    target_channels: int | None = 1
    channel_reduction: ChannelReduction = ChannelReduction.MEAN

    @property
    def needs_normalization(self) -> bool:
        """Whether audio normalization is needed."""
        return self.target_channels is not None

    def __repr__(self) -> str:
        if self.target_channels is None:
            return "AudioSpec(passthrough)"
        return (
            f"AudioSpec(channels={self.target_channels}, "
            f"reduction={self.channel_reduction.value})"
        )


# Pre-defined specs for common use cases
MONO_AUDIO_SPEC = AudioSpec(target_channels=1, channel_reduction=ChannelReduction.MEAN)
PASSTHROUGH_AUDIO_SPEC = AudioSpec(target_channels=None)


def normalize_audio(
    audio: npt.NDArray[np.floating] | torch.Tensor,
    spec: AudioSpec,
) -> npt.NDArray[np.floating] | torch.Tensor:
    """Normalize audio to the specified format.

    This function handles channel reduction for multi-channel audio,
    supporting both numpy arrays and torch tensors.

    Args:
        audio: Input audio data. Can be:
            - 1D array/tensor: (time,) - already mono
            - 2D array/tensor: (channels, time) - standard format from torchaudio
            - 2D array/tensor: (time, channels) - format from soundfile
              (will be auto-detected and transposed if time > channels)
        spec: AudioSpec defining the target format.

    Returns:
        Normalized audio in the same type as input (numpy or torch).
        For mono output (target_channels=1), returns 1D array/tensor.

    Raises:
        ValueError: If audio has unsupported dimensions or channel expansion
            is requested (e.g., mono to stereo).

    """
    if not spec.needs_normalization:
        return audio

    # Handle 1D audio (already mono)
    if audio.ndim == 1:
        if spec.target_channels == 1:
            return audio
        raise ValueError(f"Cannot expand mono audio to {spec.target_channels} channels")

    # Handle 2D audio
    if audio.ndim != 2:
        raise ValueError(f"Unsupported audio shape: {audio.shape}. Expected 1D or 2D.")

    # Auto-detect format: if shape[0] > shape[1], assume (time, channels)
    # This handles soundfile format where time dimension is typically much larger
    if audio.shape[0] > audio.shape[1]:
        # Transpose from (time, channels) to (channels, time)
        audio = audio.T if isinstance(audio, np.ndarray) else audio.T

    num_channels = audio.shape[0]

    # No reduction needed if already at target
    if num_channels == spec.target_channels:
        return audio[0] if num_channels == 1 else audio

    # Cannot expand channels
    if num_channels < spec.target_channels:
        raise ValueError(
            f"Cannot expand {num_channels} channels to {spec.target_channels}"
        )

    # Reduce channels
    is_numpy = isinstance(audio, np.ndarray)

    if spec.target_channels == 1:
        # Reduce to mono
        if spec.channel_reduction == ChannelReduction.MEAN:
            result = np.mean(audio, axis=0) if is_numpy else audio.mean(dim=0)
        elif spec.channel_reduction == ChannelReduction.FIRST:
            result = audio[0]
        elif spec.channel_reduction == ChannelReduction.MAX:
            result = np.max(audio, axis=0) if is_numpy else audio.max(dim=0).values
        elif spec.channel_reduction == ChannelReduction.SUM:
            result = np.sum(audio, axis=0) if is_numpy else audio.sum(dim=0)
        else:
            raise ValueError(f"Unknown reduction method: {spec.channel_reduction}")
        return result
    else:
        # Reduce to N channels (take first N and apply reduction if needed)
        # For now, just take first N channels
        return audio[: spec.target_channels]


# ============================================================
# Audio Resampling
# ============================================================


def resample_audio_pyav(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    """Resample audio using PyAV (libswresample via FFmpeg).

    Args:
        audio: Input audio. Can be:
            - 1D array ``(samples,)``: mono audio
            - 2D array ``(channels, samples)``: stereo audio
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz.

    Returns:
        Resampled audio with the same shape as the input (1D → 1D, 2D → 2D).

    """
    orig_sr_int = int(round(orig_sr))
    target_sr_int = int(round(target_sr))

    if orig_sr_int == target_sr_int:
        return audio

    if audio.ndim == 2:
        # Resample each channel independently and re-stack.
        return np.stack(
            [
                resample_audio_pyav(ch, orig_sr=orig_sr, target_sr=target_sr)
                for ch in audio
            ],
            axis=0,
        )

    expected_len = int(math.ceil(audio.shape[-1] * target_sr_int / orig_sr_int))

    # from_ndarray expects shape (channels, samples) for planar formats.
    # libswresample requires a minimum number of input samples to produce
    # output frames; pad short inputs with zeros so we always get output,
    # then trim to the expected output length.
    _MIN_SAMPLES = 1024
    audio_f32 = np.asarray(audio, dtype=np.float32)
    if len(audio_f32) < _MIN_SAMPLES:
        audio_f32 = np.pad(audio_f32, (0, _MIN_SAMPLES - len(audio_f32)))
    audio_f32 = audio_f32.reshape(1, -1)

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr_int)

    frame = av.AudioFrame.from_ndarray(audio_f32, format="fltp", layout="mono")
    frame.sample_rate = orig_sr_int

    out_frames = resampler.resample(frame)
    out_frames.extend(resampler.resample(None))  # flush buffered samples

    result = np.concatenate([f.to_ndarray() for f in out_frames], axis=1).squeeze(0)
    return result[:expected_len]


def resample_audio_scipy(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    orig_sr_int = int(round(orig_sr))
    target_sr_int = int(round(target_sr))

    if orig_sr_int == target_sr_int:
        return audio

    gcd = math.gcd(orig_sr_int, target_sr_int)
    return scipy_signal.resample_poly(
        audio,
        target_sr_int // gcd,
        orig_sr_int // gcd,
        axis=-1,
    )


def resample_audio_soxr(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    orig_sr_int = int(round(orig_sr))
    target_sr_int = int(round(target_sr))

    if orig_sr_int == target_sr_int:
        return audio

    if audio.ndim == 2:
        return np.stack(
            [
                resample_audio_soxr(ch, orig_sr=orig_sr, target_sr=target_sr)
                for ch in audio
            ],
            axis=0,
        )

    return soxr.resample(audio, orig_sr_int, target_sr_int)


@lru_cache(maxsize=32)
def _get_torchaudio_resampler(
    orig_sr: int, target_sr: int
) -> "torchaudio.transforms.Resample":
    # `torchaudio.transforms.Resample` precomputes its kernel for a fixed
    # (orig_sr, target_sr) pair; cache instances so repeated requests at a
    # common input rate skip the kernel rebuild.
    return torchaudio.transforms.Resample(orig_sr, target_sr)


def resample_audio_torchaudio(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    """Resample audio using torchaudio's bandlimited sinc interpolation.

    Unlike the PyAV resampler, this handles any input length without padding
    and applies the kernel over the trailing axis, so 2D ``(channels,
    samples)`` input needs no per-channel loop.

    Args:
        audio: Input audio. Can be:
            - 1D array ``(samples,)``: mono audio
            - 2D array ``(channels, samples)``: stereo audio
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz.

    Returns:
        Resampled audio with the same shape as the input (1D → 1D, 2D → 2D).

    """
    orig_sr_int = int(round(orig_sr))
    target_sr_int = int(round(target_sr))

    if orig_sr_int == target_sr_int:
        return audio

    # The kernel is float32; cast the input to match (same coercion as the
    # PyAV path).
    tensor = torch.as_tensor(audio, dtype=torch.float32)
    # Resampling runs in the API/server parent process. Keep it from
    # touching OpenMP or oneDNN thread state: both poison subsequently
    # forked engine-core processes, which then segfault on their first
    # parallel CPU op.
    with set_default_torch_num_threads(1), torch.backends.mkldnn.flags(enabled=False):
        resampler = _get_torchaudio_resampler(orig_sr_int, target_sr_int)
        return resampler(tensor).numpy()


class AudioResampler:
    """Resample audio data to a target sample rate."""

    _METHODS = ("pyav", "scipy", "soxr", "torchaudio")

    def __init__(
        self,
        target_sr: float | None = None,
        method: Literal["pyav", "scipy", "soxr", "torchaudio"] = "torchaudio",
    ):
        # Eager validation so a bad method fails at construction rather than
        # on the first audio request.
        if method not in self._METHODS:
            raise ValueError(
                f"Invalid resampling method: {method!r}. "
                f"Supported methods: {list(self._METHODS)}."
            )
        self.target_sr = target_sr
        self.method = method

    def resample(
        self,
        audio: npt.NDArray[np.floating],
        *,
        orig_sr: float,
    ) -> npt.NDArray[np.floating]:
        if self.target_sr is None:
            raise RuntimeError(
                "Audio resampling is not supported when `target_sr` is not provided"
            )
        if math.isclose(
            float(orig_sr),
            float(self.target_sr),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            return audio
        if self.method == "pyav":
            return resample_audio_pyav(audio, orig_sr=orig_sr, target_sr=self.target_sr)
        elif self.method == "scipy":
            return resample_audio_scipy(
                audio, orig_sr=orig_sr, target_sr=self.target_sr
            )
        elif self.method == "soxr":
            return resample_audio_soxr(audio, orig_sr=orig_sr, target_sr=self.target_sr)
        elif self.method == "torchaudio":
            return resample_audio_torchaudio(
                audio, orig_sr=orig_sr, target_sr=self.target_sr
            )
        else:
            raise ValueError(
                f"Invalid resampling method: {self.method}. "
                f"Supported methods are {list(self._METHODS)}."
            )


# ============================================================
# Audio Chunking / Splitting
# ============================================================


class VAD:
    def __init__(self):
        self.model = None
        self._get_speech_timestamps = None

    def _ensure_loaded(self) -> None:
        if self._get_speech_timestamps is not None and self.model is not None:
            return

        try:
            from silero_vad import get_speech_timestamps_sequence, load_silero_vad

            # The sequence model runs one ONNX call per block of frames
            # instead of one per 32ms frame, so the GIL is released for
            # almost all of the VAD compute.
            self.model = load_silero_vad(sequence=True)
            self._get_speech_timestamps = get_speech_timestamps_sequence
        except ImportError as exc:
            raise ImportError(
                "Silero VAD is not installed. Please install it using "
                "`pip install silero-vad` and `pip install onnxruntime`."
            ) from exc

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
    """Create and cache one loaded ``VAD`` instance per worker thread.

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


@dataclass
class TimedAudioChunk:
    """An audio chunk and its start time in the original audio."""

    audio: np.ndarray
    start_s: float


def _timed_chunks(
    chunks: list[np.ndarray], start_sample: int, sample_rate: int
) -> list[TimedAudioChunk]:
    """Attach start times to contiguous chunks that begin at ``start_sample``."""
    timed_chunks = []
    offset = start_sample
    for chunk in chunks:
        timed_chunks.append(TimedAudioChunk(audio=chunk, start_s=offset / sample_rate))
        offset += chunk.shape[-1]
    return timed_chunks


def split_audio_with_vad(
    duration: float,
    asr_config: SpeechToTextConfig,
    vad: VAD | None,
    vad_config: VADConfig,
    audio_data: np.ndarray,
    sample_rate: int,
) -> list[TimedAudioChunk]:
    """Split audio according to the configured VAD and chunking policy.
    * Silero VAD improves noise robustness and hallucination robustness.
    * RMS split is used to adhere to the max_clip_duration_s limit.
    Using Silero VAD for satisfying max_clip_duration_s limit didnt work well
    hence we have to use 2 stage chunking for best quality.

    Behavior depends on whether chunking is enabled via
    ``SpeechToTextConfig.allow_audio_chunking`` and whether VAD is enabled:

    * VAD + RMS: run Silero VAD first, then further split only those
      individual speech segments that exceed ``max_audio_clip_s``. If no
      speech is detected, return an empty list.
    * RMS only: run ``split_audio()`` on the full waveform when the
      overall clip duration exceeds ``max_audio_clip_s``.
    * VAD only: remove non-speech by concatenating detected speech spans into
      a single trimmed chunk, since these models expect the whole clip in one
      request. If no speech is detected, return an empty list.
    * no chunking: return the original audio as a single chunk.

    Each chunk carries its start time in the original audio, so segment
    timestamps stay aligned with the input even when VAD drops non-speech
    audio between chunks. In VAD-only mode the trimmed chunk starts at the
    first speech span, and times after a removed gap are not remapped. An
    empty list means the caller should return an empty transcript.
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
            chunks: list[TimedAudioChunk] = []
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
                    chunks.extend(
                        _timed_chunks(partial_chunk_output, start, sample_rate)
                    )
                else:
                    chunks.append(
                        TimedAudioChunk(
                            audio=audio_data[start:end], start_s=start / sample_rate
                        )
                    )
            return chunks
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
                return _timed_chunks(partial_chunk_output, 0, sample_rate)
            else:
                return [TimedAudioChunk(audio=audio_data, start_s=0.0)]
    else:
        if vad_config.enabled:
            # silero VAD only
            assert vad is not None

            speech_timestamps = vad.get_speech_timestamps(
                audio_data, sample_rate, vad_config
            )
            if len(speech_timestamps) > 0:
                speech_spans = [
                    audio_data[..., timestamp["start"] : timestamp["end"]]
                    for timestamp in speech_timestamps
                ]
                trimmed_audio_data = np.concatenate(speech_spans, axis=-1)
                return [
                    TimedAudioChunk(
                        audio=trimmed_audio_data,
                        start_s=speech_timestamps[0]["start"] / sample_rate,
                    )
                ]
            else:
                return []
        else:
            return [TimedAudioChunk(audio=audio_data, start_s=0.0)]


def split_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    max_clip_duration_s: float,
    overlap_duration_s: float,
    min_energy_window_size: int,
) -> list[np.ndarray]:
    """Split audio into chunks with intelligent split points.

    Splits long audio into smaller chunks at low-energy regions to minimize
    cutting through speech. Uses overlapping windows to find quiet moments
    for splitting.

    Args:
        audio_data: 1D mono audio array to split. ASR models consume mono, so
                   callers must downmix before chunking.
        sample_rate: Sample rate of the audio in Hz.
        max_clip_duration_s: Maximum duration of each chunk in seconds.
        overlap_duration_s: Overlap duration in seconds between consecutive chunks.
                           Used to search for optimal split points.
        min_energy_window_size: Window size in samples for finding low-energy regions.

    Returns:
        List of 1D audio chunks.

    Raises:
        AssertionError: If ``audio_data`` is not 1D.

    Example:
        >>> audio = np.random.randn(1040000)  # 65 seconds at 16kHz
        >>> chunks = split_audio(
        ...     audio_data=audio,
        ...     sample_rate=16000,
        ...     max_clip_duration_s=30.0,
        ...     overlap_duration_s=1.0,
        ...     min_energy_window_size=1600,
        ... )
        >>> len(chunks)
        3

    """
    if audio_data.ndim > 1:
        raise ValueError(
            f"split_audio expects mono audio, got shape {audio_data.shape}"
        )

    chunk_size = int(sample_rate * max_clip_duration_s)
    overlap_size = int(sample_rate * overlap_duration_s)
    chunks = []
    i = 0

    while i < audio_data.shape[-1]:
        if i + chunk_size >= audio_data.shape[-1]:
            # Handle last chunk - take everything remaining
            chunks.append(audio_data[..., i:])
            break

        # Find the best split point in the overlap region
        search_start = i + chunk_size - overlap_size
        search_end = min(i + chunk_size, audio_data.shape[-1])
        split_point = find_split_point(
            audio_data, search_start, search_end, min_energy_window_size
        )

        # Guarantee forward progress: if split_point didn't advance,
        # fall back to the hard chunk boundary.
        if split_point <= i:
            split_point = min(i + chunk_size, audio_data.shape[-1])

        # Extract chunk up to the split point
        chunks.append(audio_data[..., i:split_point])
        i = split_point

    return chunks


def find_split_point(
    wav: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_energy_window: int,
) -> int:
    """Find the best point to split audio by looking for silence or low amplitude.

    Searches for the quietest region within a specified range by calculating
    RMS energy in sliding windows.

    Args:
        wav: 1D mono audio array.
        start_idx: Start index of search region (inclusive).
        end_idx: End index of search region (exclusive).
        min_energy_window: Window size in samples for energy calculation.

    Returns:
        Index of the quietest point within the search region. This is the
        recommended split point to minimize audio artifacts.

    Example:
        >>> audio = np.random.randn(32000)
        >>> # Insert quiet region
        >>> audio[16000:17600] = 0.01
        >>> split_idx = find_split_point(
        ...     wav=audio,
        ...     start_idx=0,
        ...     end_idx=32000,
        ...     min_energy_window=1600,
        ... )
        >>> 16000 <= split_idx <= 17600
        True

    """
    segment = wav[start_idx:end_idx]

    # Calculate RMS energy in small windows
    min_energy = math.inf
    quietest_idx = start_idx

    for i in range(0, len(segment) - min_energy_window, min_energy_window):
        window = segment[i : i + min_energy_window]
        energy = (window**2).mean() ** 0.5
        if not math.isnan(energy) and energy < min_energy:
            quietest_idx = i + start_idx
            min_energy = energy

    return quietest_idx
