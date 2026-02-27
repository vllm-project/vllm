# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from transformers import ParakeetEncoderConfig, PretrainedConfig


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


@dataclass(kw_only=True, frozen=True)
class ExtractorConfig:
    feature_size: int
    sampling_rate: int
    subsampling_factor: int
    subsampling_conv_kernel_size: int
    subsampling_conv_stride: int
    clip_duration_s: int = 30
    clip_min_duration_s: float = 0.1

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
