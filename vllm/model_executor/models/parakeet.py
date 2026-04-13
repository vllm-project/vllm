# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Modules below used for the audio encoder component in: models/nano_nemotron_vl.py
"""

from collections.abc import Iterable
from functools import cache
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import ParakeetEncoder as HFParakeetEncoder
from transformers import PretrainedConfig
from transformers.audio_utils import mel_filter_bank

from vllm.logger import init_logger
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.configs.parakeet import ExtractorConfig, ParakeetConfig

logger = init_logger(__name__)


class ParakeetProjection(nn.Module):
    def __init__(self, config: ParakeetConfig) -> None:
        super().__init__()
        sound_hidden_size = config.hidden_size
        proj_hidden_size = config.projection_hidden_size
        llm_hidden_size = config.llm_hidden_size
        bias = config.projection_bias

        self.norm = RMSNorm(sound_hidden_size, eps=config.projection_eps)
        self.linear1 = nn.Linear(sound_hidden_size, proj_hidden_size, bias=bias)
        self.activation = ReLUSquaredActivation()
        self.linear2 = nn.Linear(proj_hidden_size, llm_hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class ProjectedParakeet(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        dtype: torch.dtype,
        llm_hidden_size: int,
        max_model_len: int,
    ) -> None:
        super().__init__()
        self.config = ParakeetConfig.from_hf_config(
            config, llm_hidden_size=llm_hidden_size, max_model_len=max_model_len
        )
        self.encoder = HFParakeetEncoder(self.config)
        self.encoder = self.encoder.to(dtype)
        self.projection = ParakeetProjection(self.config)
        self.projection = self.projection.to(dtype)

    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_features=input_features, attention_mask=attention_mask
        )
        outputs = outputs.last_hidden_state
        outputs = outputs.to(dtype=torch.bfloat16)
        outputs = self.projection(outputs)
        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        if isinstance(weights, dict):
            weights_list = list(weights.items())
        else:
            weights_list = list(weights)

        for name, weight in weights_list:
            if name.startswith("sound_encoder.encoder.feature_extractor."):
                # Feature extractor buffers are handled outside the encoder.
                continue
            if name.startswith("sound_encoder."):
                target_name = name[len("sound_encoder.") :]
            elif name.startswith("sound_projection."):
                target_name = f"projection.{name[len('sound_projection.') :]}"
            else:
                continue

            target = params_dict.get(target_name)
            if target is None:
                target = buffers_dict.get(target_name)
            if target is None:
                raise ValueError(f"Unknown weight: {name}")
            weight_loader = getattr(target, "weight_loader", default_weight_loader)
            with torch.no_grad():
                weight_loader(target, weight)
            loaded_params.add(target_name)

        return loaded_params


EPSILON = 1e-5
LOG_ZERO_GUARD_VALUE = 2**-24


class ParakeetExtractor:
    def __init__(self, config: PretrainedConfig) -> None:
        self.config = ExtractorConfig.from_hf_config(config)
        """`config` is named *exactly* for `._get_subsampling_output_length` below"""
        self._clip_target_samples = int(
            round(self.config.clip_duration_s * self.config.sampling_rate)
        )
        self._tail_min_samples = int(
            round(self.config.clip_min_duration_s * self.config.sampling_rate)
        )

    @staticmethod
    @cache
    def _get_window(win_length: int, device: str) -> torch.Tensor:
        return torch.hann_window(win_length, periodic=False, device=device)

    @staticmethod
    @cache
    def _get_mel_filters(
        feature_size: int, sampling_rate: int, n_fft: int, device: str
    ) -> torch.Tensor:
        filter_bank = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        return torch.from_numpy(filter_bank.T).to(device=device, dtype=torch.float32)

    def _torch_extract_fbank_features(self, waveform: torch.Tensor, device: str):
        # spectrogram
        device = str(torch.device(device))
        cfg = self.config
        window = self._get_window(cfg.win_length, device)
        stft = torch.stft(
            waveform,
            self.config.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        mel_filters = self._get_mel_filters(
            cfg.feature_size, cfg.sampling_rate, cfg.n_fft, device
        )
        return self._apply_mel_filters(stft, mel_filters)

    @torch.compile(dynamic=True)
    def _apply_mel_filters(
        self, stft_output: torch.Tensor, mel_filters: torch.Tensor
    ) -> torch.Tensor:
        magnitudes = stft_output.real.square() + stft_output.imag.square()
        mel_spec = mel_filters @ magnitudes
        mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD_VALUE)
        return mel_spec.permute(0, 2, 1)

    @torch.compile(dynamic=True)
    def _apply_preemphasis(
        self, input_features: torch.Tensor, audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        timemask = torch.arange(
            input_features.shape[1], device=input_features.device
        ).unsqueeze(0) < audio_lengths.unsqueeze(1)
        input_features = torch.cat(
            [
                input_features[:, :1],
                input_features[:, 1:]
                - self.config.preemphasis * input_features[:, :-1],
            ],
            dim=1,
        )
        input_features = input_features.masked_fill(~timemask, 0.0)
        return input_features

    @torch.compile(dynamic=True)
    def _normalize_mel_features(
        self, mel_features: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features_lengths = torch.floor_divide(
            audio_lengths + self.config.n_fft // 2 * 2 - self.config.n_fft,
            self.config.hop_length,
        )
        attention_mask = (
            torch.arange(mel_features.shape[1], device=mel_features.device)[None, :]
            < features_lengths[:, None]
        )
        mask = attention_mask.unsqueeze(-1)
        lengths = attention_mask.sum(dim=1)
        mel_features_masked = mel_features * mask
        mean = (mel_features_masked.sum(dim=1) / lengths.unsqueeze(-1)).unsqueeze(1)
        variance = ((mel_features_masked - mean) ** 2 * mask).sum(dim=1) / (
            lengths - 1
        ).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        return (mel_features - mean) / (std + EPSILON) * mask, attention_mask

    def _pad_raw_speech(
        self, raw_speech: list[torch.Tensor], max_len: int, device: str
    ) -> torch.Tensor:
        output = torch.full(
            (len(raw_speech), max_len),
            self.config.padding_value,
            device=device,
            dtype=torch.float32,
        )
        dsts = [output[i, : raw_speech[i].shape[0]] for i in range(len(raw_speech))]
        srcs = [s.squeeze(-1) for s in raw_speech]
        # single kernel horizontal fusion
        torch._foreach_copy_(dsts, srcs)
        return output

    def _clip_sizes(self, audio_len: int) -> list[int]:
        audio_len = max(audio_len, self._tail_min_samples)
        num_full_clips, remainder = divmod(audio_len, self._clip_target_samples)
        clip_sizes = [self._clip_target_samples] * num_full_clips
        if remainder > 0:
            clip_sizes.append(max(remainder, self._tail_min_samples))
        return clip_sizes

    def audio_token_count(self, audio_len: int) -> int:
        total_tokens = 0
        for clip_size in self._clip_sizes(audio_len):
            num_frames = clip_size // self.config.hop_length
            n_tokens = HFParakeetEncoder._get_subsampling_output_length(
                self, torch.tensor([num_frames], dtype=torch.float)
            )
            total_tokens += int(n_tokens.item())
        return max(1, total_tokens)

    def split_audio_into_clips(self, audio: torch.Tensor) -> list[torch.Tensor]:
        assert audio.ndim == 1
        audio_len = int(audio.shape[0])
        clip_sizes = self._clip_sizes(audio_len)
        target_len = sum(clip_sizes)
        if audio_len < target_len:
            audio = torch.nn.functional.pad(audio, (0, target_len - audio_len))

        clips = list[torch.Tensor]()
        offset = 0
        for clip_size in clip_sizes:
            clips.append(audio[offset : offset + clip_size])
            offset += clip_size
        return clips

    def __call__(
        self,
        raw_speech: list[np.ndarray],
        *,
        device: str = "cpu",
    ) -> dict[str, Any]:
        raw_speech = [
            torch.as_tensor(speech, device=device, dtype=torch.float32)
            for speech in raw_speech
        ]

        for i, speech in enumerate(raw_speech):
            if len(speech.shape) > 1:
                logger.warning(
                    "Only mono-channel audio is supported for input to %s. "
                    "We will take the mean of the channels to convert to mono.",
                    self.__class__.__name__,
                )
                raw_speech[i] = speech.mean(-1)

        audio_clips = list[torch.Tensor]()
        audio_num_clips = list[int]()
        for audio in raw_speech:
            clips = self.split_audio_into_clips(audio)
            audio_clips.extend(clips)
            audio_num_clips.append(len(clips))
        raw_speech = audio_clips

        audio_lengths = torch.tensor(
            [len(speech) for speech in raw_speech], dtype=torch.long, device=device
        )

        max_length = max(len(speech) for speech in raw_speech)
        input_features = self._pad_raw_speech(raw_speech, max_length, device)
        input_features = self._apply_preemphasis(input_features, audio_lengths)
        input_features = self._torch_extract_fbank_features(input_features, device)
        input_features, attention_mask = self._normalize_mel_features(
            input_features, audio_lengths
        )

        return {
            "input_audio_features": input_features,
            "feature_attention_mask": attention_mask,
            "audio_num_clips": audio_num_clips,
        }

    @staticmethod
    def audio_length(raw_config: PretrainedConfig, audio_tokens: int) -> int:
        config = ExtractorConfig.from_hf_config(raw_config)
        return int(audio_tokens * config.subsampling_factor * config.hop_length)
