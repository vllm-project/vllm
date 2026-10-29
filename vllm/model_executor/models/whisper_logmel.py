# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whisper log-mel used by the GPU worker.

Mirrors ``WhisperFeatureExtractor._torch_extract_fbank_features`` from
transformers 5.16 and 5.19. This module does not import vLLM so the numeric
check can run without initializing the package.
"""

from collections.abc import Sequence

import torch


def _as_mono_clips(
    waveform: torch.Tensor | Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    if isinstance(waveform, torch.Tensor):
        if waveform.ndim == 1:
            return [waveform]
        if waveform.ndim != 2:
            raise ValueError(
                "Whisper audio_pcm must have shape (time,) or (batch, time), "
                f"got {tuple(waveform.shape)}"
            )
        clips = list(waveform.unbind(0))
        if not clips:
            raise ValueError("Whisper audio_pcm batch is empty")
        return clips

    clips = []
    for clip in waveform:
        if not isinstance(clip, torch.Tensor):
            raise TypeError(
                f"Whisper audio_pcm clips must be tensors, got {type(clip).__name__}"
            )
        if clip.ndim == 2 and (clip.shape[0] == 1 or clip.shape[-1] == 1):
            clip = clip.reshape(-1)
        if clip.ndim != 1:
            raise ValueError(
                f"Whisper audio_pcm clip must be mono, got shape {tuple(clip.shape)}"
            )
        clips.append(clip)
    if not clips:
        raise ValueError("Whisper audio_pcm batch is empty")
    return clips


def _fit_n_samples(
    clip: torch.Tensor,
    n_samples: int,
    padding_value: float,
) -> torch.Tensor:
    length = clip.shape[0]
    if length == n_samples:
        return clip
    if length > n_samples:
        return clip[:n_samples]
    return torch.nn.functional.pad(
        clip,
        (0, n_samples - length),
        value=padding_value,
    )


def whisper_log_mel_spectrogram(
    waveform: torch.Tensor | Sequence[torch.Tensor],
    *,
    n_fft: int,
    hop_length: int,
    mel_filters: torch.Tensor,
    n_samples: int,
    padding_value: float = 0.0,
    dither: float = 0.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Log-mel for mono PCM, matching the Hugging Face Whisper extractor.

    Each clip is padded or truncated to ``n_samples``. ``do_normalize`` stays
    off. The result is float32 ``(batch, mel_bins, n_samples // hop_length)``.
    Per-clip max is taken over time, then over bins.
    """
    clips = _as_mono_clips(waveform)
    target = torch.device(device) if device is not None else clips[0].device
    batch = torch.stack(
        [
            _fit_n_samples(
                clip.to(device=target, dtype=torch.float32),
                n_samples,
                padding_value,
            )
            for clip in clips
        ],
        dim=0,
    )
    if dither != 0.0:
        batch = batch + dither * torch.randn(
            batch.shape,
            dtype=batch.dtype,
            device=batch.device,
        )

    window = torch.hann_window(n_fft, device=batch.device, dtype=torch.float32)
    # Defaults stay: center=True, pad_mode="reflect", periodic Hann, onesided.
    stft = torch.stft(
        batch,
        n_fft,
        hop_length,
        window=window,
        return_complex=True,
    )
    # transformers 5.16+ drops the extra center frame. contiguous() does not
    # change values.
    magnitudes = (stft[..., :-1].abs() ** 2).contiguous()
    filters = mel_filters.to(device=batch.device, dtype=torch.float32)
    log_spec = torch.clamp(filters.T @ magnitudes, min=1e-10).log10()
    max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    log_spec = torch.maximum(log_spec, max_val - 8.0)
    return (log_spec + 4.0) / 4.0
