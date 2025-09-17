# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BatchFeature, PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class StepAudio2Processor(ProcessorMixin):
    """
    Processor for Step-Audio2 model that handles audio processing and tokenization.
    Based on the Step1fProcessor from the Step-Audio2 vLLM fork.
    """

    tokenizer_class = ("Qwen2TokenizerFast", "Qwen2Tokenizer")
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[PretrainedConfig] = None,
        **kwargs
    ) -> None:
        super().__init__(tokenizer=tokenizer, **kwargs)

        self.config = config
        self.tokenizer = tokenizer

        # Audio processing parameters
        self.audio_token = "<audio_patch>"
        self.n_mels = 128
        self.max_chunk_size = 29  # ~30s audio from position embedding length 1500
        self.sampling_rate = 16000

        # Pre-compute mel filters
        self._mel_filters = torch.from_numpy(
            librosa.filters.mel(
                sr=self.sampling_rate,
                n_fft=400,
                n_mels=self.n_mels
            )
        )

    @property
    def audio_token_id(self) -> int:
        """Get the token ID for audio patch token."""
        vocab = self.tokenizer.get_vocab()
        return vocab.get(self.audio_token, vocab.get("<audio_patch>", 151690))

    def _log_mel_spectrogram(
        self,
        audio: np.ndarray,
        padding: int = 0,
    ) -> torch.Tensor:
        """
        Convert audio to log mel spectrogram.

        Args:
            audio: Audio array
            padding: Padding to add to audio

        Returns:
            Log mel spectrogram tensor
        """
        audio = F.pad(torch.from_numpy(audio.astype(np.float32)), (0, padding))
        window = torch.hann_window(400).to(audio.device)
        stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs()**2
        filters = self._mel_filters.to(audio.device)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.t()

    def preprocess_audio(self, audio_tensor: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio for the model.

        Args:
            audio_tensor: Raw audio array

        Returns:
            Preprocessed audio features
        """
        return self._log_mel_spectrogram(audio_tensor, padding=479)

    def get_num_audio_tokens(self, max_feature_len: int) -> int:
        """
        Calculate number of audio tokens for given feature length.

        Args:
            max_feature_len: Maximum feature length

        Returns:
            Number of audio tokens
        """
        encoder_output_dim = (max_feature_len + 1) // 2 // 2
        padding = 1
        kernel_size = 3
        stride = 2
        adapter_output_dim = (encoder_output_dim + 2 * padding - kernel_size) // stride + 1
        return adapter_output_dim

    def _get_audio_repl(self, max_feature_len: int) -> tuple[int, list[int]]:
        """
        Get audio replacement tokens.

        Args:
            max_feature_len: Maximum feature length

        Returns:
            Tuple of (number of tokens, token IDs)
        """
        num_audio_tokens = self.get_num_audio_tokens(max_feature_len)

        # Get special tokens
        vocab = self.tokenizer.get_vocab()
        audio_start = vocab.get("<audio_start>", 151688)
        audio_end = vocab.get("<audio_end>", 151689)
        audio_patch = self.audio_token_id

        # Create replacement sequence
        audio_repl_ids = [audio_start] + [audio_patch] * num_audio_tokens + [audio_end]

        return num_audio_tokens, audio_repl_ids

    def __call__(
        self,
        audio: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        audios: Optional[list[np.ndarray]] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Process audio and text inputs.

        Args:
            audio: Single audio array
            text: Input text
            audios: List of audio arrays
            return_tensors: Format for returned tensors
            **kwargs: Additional arguments

        Returns:
            Processed features
        """
        data = {}

        # Handle audio inputs
        if audio is not None:
            audios = [audio]

        if audios is not None:
            audio_mels = []
            audio_lens = []

            for audio_array in audios:
                mel_features = self.preprocess_audio(audio_array)
                audio_mels.append(mel_features)
                audio_lens.append(mel_features.shape[0])

            # Concatenate all audio features
            if audio_mels:
                data["audio_mels"] = torch.cat(audio_mels, dim=0) if len(audio_mels) > 1 else audio_mels[0]
                data["audio_lens"] = torch.tensor(audio_lens, dtype=torch.long)

        # Handle text input
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                **kwargs
            )
            data.update(text_inputs)

        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """Delegate batch decoding to tokenizer."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Delegate decoding to tokenizer."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """Model input names."""
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_input_names = ["audio_mels", "audio_lens"]
        return list(set(tokenizer_input_names + audio_input_names))