# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from transformers import BatchFeature
from transformers.audio_utils import mel_filter_bank
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOG_ZERO_GUARD_VALUE = 2**-24


class NemotronAsrStreamingFeatureExtractor(SequenceFeatureExtractor):
    """Extract Nemotron's log-mel features without a librosa dependency."""

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        n_fft: int = 512,
        win_length: int = 400,
        preemphasis: float = 0.97,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = preemphasis

        mel_filters = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.mel_filters = torch.from_numpy(mel_filters.T).to(torch.float32)

    def _torch_extract_fbank_features(
        self,
        waveform: torch.Tensor,
        *,
        device: torch.device,
        center: bool,
    ) -> torch.Tensor:
        waveform = waveform.to(device)
        window = torch.hann_window(self.win_length, periodic=False, device=device)
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
            center=center,
        )
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1)).pow(2)
        mel_spec = self.mel_filters.to(device) @ magnitudes
        return torch.log(mel_spec + _LOG_ZERO_GUARD_VALUE).permute(0, 2, 1)

    @staticmethod
    def _as_mono_tensor(audio: object) -> torch.Tensor:
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        if waveform.ndim == 0:
            waveform = waveform.reshape(1)
        if waveform.ndim > 1:
            logger.warning(
                "Only mono-channel audio is supported; averaging audio channels."
            )
            waveform = waveform.mean(-1)
        return waveform

    def __call__(
        self,
        raw_speech: object,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | None = None,
        return_attention_mask: bool | None = None,
        padding: str | bool | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        device: str | torch.device = "cpu",
        center: bool = True,
        **kwargs,
    ) -> BatchFeature:
        del return_attention_mask, kwargs
        if not center:
            raise ValueError(
                "Nemotron 3.5 ASR feature extraction requires center=True."
            )
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling rate {self.sampling_rate}, got {sampling_rate}."
            )
        if sampling_rate is None:
            logger.warning(
                "Audio sampling_rate was not provided; assuming %s.",
                self.sampling_rate,
            )

        if (
            isinstance(raw_speech, (list, tuple))
            and raw_speech
            and np.isscalar(raw_speech[0])
        ):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        if isinstance(raw_speech, (np.ndarray, torch.Tensor)) and raw_speech.ndim <= 1:
            audios = [raw_speech]
        elif isinstance(raw_speech, (Sequence, np.ndarray, torch.Tensor)):
            audios = list(raw_speech)
        else:
            audios = [raw_speech]
        waveforms = [self._as_mono_tensor(audio) for audio in audios]
        if not waveforms:
            raise ValueError("At least one audio waveform is required.")

        lengths = torch.tensor([audio.numel() for audio in waveforms], dtype=torch.long)
        target_length = int(lengths.max().item())
        if isinstance(padding, str) and padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required when padding='max_length'.")
            target_length = max(target_length, max_length)
        elif padding is False or padding is None:
            if len(waveforms) > 1 and len(set(lengths.tolist())) != 1:
                raise ValueError("Variable-length audio requires padding.")
        if truncation and max_length is not None:
            target_length = min(target_length, max_length)
        if pad_to_multiple_of:
            target_length = (
                (target_length + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of

        target_length = max(target_length, 1)
        padded = torch.full(
            (len(waveforms), target_length),
            self.padding_value,
            dtype=torch.float32,
        )
        for index, waveform in enumerate(waveforms):
            length = min(waveform.numel(), target_length)
            padded[index, :length] = waveform[:length]
            lengths[index] = length

        if self.preemphasis is not None:
            time_mask = torch.arange(target_length)[None, :] < lengths[:, None]
            padded = torch.cat(
                [padded[:, :1], padded[:, 1:] - self.preemphasis * padded[:, :-1]],
                dim=1,
            )
            padded = padded.masked_fill(~time_mask, 0.0)

        extract_device = torch.device(device)
        input_features = self._torch_extract_fbank_features(
            padded,
            device=extract_device,
            center=True,
        )
        feature_lengths = torch.div(
            lengths,
            self.hop_length,
            rounding_mode="floor",
        )
        attention_mask = (
            torch.arange(input_features.shape[1], device=extract_device)[None, :]
            < feature_lengths.to(extract_device)[:, None]
        )
        input_features = input_features.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        return BatchFeature(
            {
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )


__all__ = ["NemotronAsrStreamingFeatureExtractor"]
