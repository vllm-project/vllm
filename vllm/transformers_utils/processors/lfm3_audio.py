# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch
from transformers import AutoTokenizer, BatchFeature
from transformers.audio_utils import mel_filter_bank
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.processing_utils import ProcessorMixin


class Lfm3AudioFeatureExtractor(SequenceFeatureExtractor):
    """Log-mel spectrogram without centering, right-padded to whole frames;
    every `encoder_stride` frames become one encoder token."""

    model_input_names = ["input_features", "feature_attention_mask"]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        log_floor: float = -12.0,
        normalization_offset: float = 4.0,
        normalization_divisor: float = 4.0,
        encoder_stride: int = 8,
        padding_value: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.log_floor = log_floor
        self.normalization_offset = normalization_offset
        self.normalization_divisor = normalization_divisor
        self.encoder_stride = encoder_stride
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def num_frames(self, num_samples: int) -> int:
        return 1 + (self._padded_length(num_samples) - self.n_fft) // self.hop_length

    def num_encoder_tokens(self, num_frames: int) -> int:
        return -(num_frames // -self.encoder_stride)

    def _padded_length(self, num_samples: int) -> int:
        if num_samples < self.n_fft:
            return self.n_fft
        return num_samples + (-(num_samples - self.n_fft) % self.hop_length)

    def _log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = torch.nn.functional.pad(
            waveform, (0, self._padded_length(len(waveform)) - len(waveform))
        )
        spectrogram = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length),
            center=False,
            return_complex=True,
        )
        power = spectrogram.abs().square()
        mel_filters = torch.from_numpy(self.mel_filters).to(power.dtype)
        log_mel = (mel_filters.T @ power).log10().clamp(min=self.log_floor)
        return (log_mel + self.normalization_offset) / self.normalization_divisor

    def __call__(
        self,
        raw_speech: np.ndarray | list[np.ndarray],
        sampling_rate: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
            raw_speech = [raw_speech]
        features = [
            self._log_mel(torch.as_tensor(np.asarray(clip, dtype=np.float32)))
            for clip in raw_speech
        ]
        max_length = max(feature.shape[-1] for feature in features)
        input_features = torch.full(
            (len(features), self.feature_size, max_length), self.padding_value
        )
        feature_attention_mask = torch.zeros(
            (len(features), max_length), dtype=torch.long
        )
        for index, feature in enumerate(features):
            input_features[index, :, : feature.shape[-1]] = feature
            feature_attention_mask[index, : feature.shape[-1]] = 1
        return BatchFeature(
            {
                "input_features": input_features,
                "feature_attention_mask": feature_attention_mask,
            },
            tensor_type=return_tensors,
        )


class Lfm3AudioProcessor(ProcessorMixin):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # The hub processor also loads the speech codec, which only speech
        # output needs.
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        feature_extractor = Lfm3AudioFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __init__(self, feature_extractor, tokenizer, audio_token: str = "<|audio|>"):
        super().__init__(feature_extractor, tokenizer)
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)

    def __call__(
        self,
        text: str | list[str] | None = None,
        audio: np.ndarray | list[np.ndarray] | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        # vLLM expands the audio placeholders itself, so text is only tokenized
        data = {}
        if audio is not None:
            data.update(self.feature_extractor(audio, return_tensors=return_tensors))
        if text is not None:
            data.update(self.tokenizer(text, return_tensors=return_tensors, **kwargs))
        return BatchFeature(data, tensor_type=return_tensors)
