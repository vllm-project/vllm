# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from transformers import ParakeetEncoder as HFParakeetEncoder
from transformers import (
    ParakeetEncoderConfig,
    ParakeetFeatureExtractor,
    PretrainedConfig,
)

from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class ParakeetConfig(ParakeetEncoderConfig):
    llm_hidden_size: int
    projection_hidden_size: int
    projection_bias: bool
    projection_eps: float = 1e-5
    sampling_rate: int

    @staticmethod
    def from_hf_config(
        config: PretrainedConfig, *, llm_hidden_size: int, max_model_len: int
    ) -> "ParakeetConfig":
        assert isinstance(config, PretrainedConfig)
        return ParakeetConfig(
            **config.to_dict(),
            scale_input=False,
            attention_bias=False,
            llm_hidden_size=llm_hidden_size,
            max_position_embeddings=max_model_len
            + 1,  # + 1 because it seems like max_model_len+1 can be passed
        )


class ParakeetProjection(nn.Module):
    def __init__(self, config: ParakeetConfig) -> None:
        super().__init__()
        sound_hidden_size = config.hidden_size
        proj_hidden_size = config.projection_hidden_size
        llm_hidden_size = config.llm_hidden_size
        bias = config.projection_bias

        self.norm = nn.LayerNorm(sound_hidden_size, eps=config.projection_eps)
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


@dataclass(kw_only=True, frozen=True)
class ExtractorConfig:
    feature_size: int
    sampling_rate: int
    subsampling_factor: int
    subsampling_conv_kernel_size: int
    subsampling_conv_stride: int

    @staticmethod
    def from_hf_config(config: PretrainedConfig) -> "ExtractorConfig":
        assert isinstance(config, PretrainedConfig)
        return ExtractorConfig(
            feature_size=config.num_mel_bins,
            sampling_rate=config.sampling_rate,
            subsampling_factor=config.subsampling_factor,
            subsampling_conv_kernel_size=config.subsampling_conv_kernel_size,
            subsampling_conv_stride=config.subsampling_conv_stride,
        )


class ParakeetExtractor(ParakeetFeatureExtractor):
    def __init__(self, config: PretrainedConfig) -> None:
        self.config = ExtractorConfig.from_hf_config(config)
        super().__init__(**asdict(self.config))

    def audio_token_count(self, audio_len: int) -> int:
        num_frames = audio_len // self.hop_length
        n_tokens = HFParakeetEncoder._get_subsampling_output_length(
            self, torch.tensor([num_frames], dtype=torch.float)
        )
        return max(1, n_tokens.item())

    def audio_length(self, audio_tokens: int) -> int:
        return int(audio_tokens * self.config.subsampling_factor * self.hop_length)
