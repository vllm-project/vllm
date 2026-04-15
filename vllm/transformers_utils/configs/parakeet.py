# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from transformers import ParakeetEncoderConfig, PretrainedConfig


class ParakeetConfig(ParakeetEncoderConfig):
    def __init__(
        self,
        llm_hidden_size: int,
        projection_hidden_size: int,
        projection_bias: bool,
        sampling_rate: int,
        projection_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_hidden_size = llm_hidden_size
        self.projection_hidden_size = projection_hidden_size
        self.projection_bias = projection_bias
        self.sampling_rate = sampling_rate
        self.projection_eps = projection_eps

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
    hop_length: int = 160
    """Default `160`: Matches HF default"""
    clip_duration_s: int = 30
    clip_min_duration_s: float = 0.1

    win_length: int = 400
    preemphasis: float = 0.97
    n_fft: int = 512
    padding_value: float = 0.0

    @classmethod
    def from_hf_config(cls, config: PretrainedConfig) -> "ExtractorConfig":
        assert isinstance(config, PretrainedConfig)
        defaults = ("hop_length", "win_length", "preemphasis", "n_fft", "padding_value")
        optional_kwargs = {
            name: getattr(config, name) for name in defaults if hasattr(config, name)
        }

        return cls(
            feature_size=config.num_mel_bins,
            sampling_rate=config.sampling_rate,
            subsampling_factor=config.subsampling_factor,
            subsampling_conv_kernel_size=config.subsampling_conv_kernel_size,
            subsampling_conv_stride=config.subsampling_conv_stride,
            **optional_kwargs,
        )
