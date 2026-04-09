# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Modules below used for the audio encoder component in: models/nano_nemotron_vl.py
"""

from collections.abc import Iterable
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from transformers import ParakeetEncoder as HFParakeetEncoder
from transformers import ParakeetFeatureExtractor, PretrainedConfig

from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.configs.parakeet import ExtractorConfig, ParakeetConfig


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


class ParakeetExtractor(ParakeetFeatureExtractor):
    def __init__(self, config: PretrainedConfig) -> None:
        self.config = ExtractorConfig.from_hf_config(config)
        super().__init__(**asdict(self.config))
        self._clip_target_samples = int(
            round(self.config.clip_duration_s * self.sampling_rate)
        )
        self._tail_min_samples = int(
            round(self.config.clip_min_duration_s * self.sampling_rate)
        )

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
            num_frames = clip_size // self.hop_length
            n_tokens = HFParakeetEncoder._get_subsampling_output_length(
                self, torch.tensor([num_frames], dtype=torch.float)
            )
            total_tokens += int(n_tokens.item())
        return max(1, total_tokens)

    def split_audio_into_clips(self, audio: np.ndarray) -> list[np.ndarray]:
        assert audio.ndim == 1
        audio_len = int(audio.shape[0])
        clip_sizes = self._clip_sizes(audio_len)
        target_len = sum(clip_sizes)
        if audio_len < target_len:
            audio = np.pad(audio, (0, target_len - audio_len))

        clips = list[np.ndarray]()
        offset = 0
        for clip_size in clip_sizes:
            clips.append(audio[offset : offset + clip_size])
            offset += clip_size
        return clips

    def __call__(self, raw_speech: list[np.ndarray], *args, **kwargs):
        audio_clips = list[np.ndarray]()
        audio_num_clips = list[int]()
        for audio in raw_speech:
            clips = self.split_audio_into_clips(audio)
            audio_clips.extend(clips)
            audio_num_clips.append(len(clips))

        outputs = super().__call__(audio_clips, *args, **kwargs)
        outputs["audio_num_clips"] = audio_num_clips
        return outputs

    @staticmethod
    def audio_length(raw_config: PretrainedConfig, audio_tokens: int) -> int:
        config = ExtractorConfig.from_hf_config(raw_config)
        return int(audio_tokens * config.subsampling_factor * config.hop_length)
