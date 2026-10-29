# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whisper feature extraction with Ming's mel coefficient precision."""

import numpy as np
from transformers import WhisperFeatureExtractor
from transformers.audio_utils import hertz_to_mel, mel_filter_bank, mel_to_hertz


class MingASRFeatureExtractor(WhisperFeatureExtractor):
    def __init__(self, feature_size: int = 128) -> None:
        if feature_size not in (80, 128):
            raise ValueError("Ming audio requires 80 or 128 mel bins")
        super().__init__(feature_size=feature_size)
        max_frequency = self.sampling_rate / 2
        filters = mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=max_frequency,
            sampling_rate=self.sampling_rate,
            norm=None,
            mel_scale="slaney",
        )
        centers = mel_to_hertz(
            np.linspace(
                0.0,
                hertz_to_mel(max_frequency, mel_scale="slaney"),
                feature_size + 2,
            ),
            mel_scale="slaney",
        )
        # Round before Slaney scaling and retain contiguous mel-major storage.
        self.mel_filters = np.asfortranarray(filters, dtype=np.float32)
        self.mel_filters *= (2.0 / (centers[2:] - centers[:-2]))[None, :]
