# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Horizon team, Xiaomi MiLM Plus.
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import PretrainedConfig
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniTextConfig,
)


class DashengConfig(PretrainedConfig):
    model_type = "midashenglm_dasheng_encoder"

    def __init__(
        self,
        embed_dim: int = 768,
        outputdim: int = 527,
        patch_size: int | tuple[int, int] = 16,
        patch_stride: int | tuple[int, int] = 16,
        input_channels: int = 1,
        target_length: int = 1012,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        center: bool = True,
        win_length: int = 512,
        hop_length: int = 160,
        sample_rate: int = 16000,
        n_fft: int = 512,
        n_mels: int = 64,
        **kwargs,
    ):
        self.embed_dim = embed_dim
        self.outputdim = outputdim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.input_channels = input_channels
        self.target_length = target_length
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.init_values = init_values
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.f_min = f_min
        self.f_max = f_max
        self.center = center
        self.win_length = win_length
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        super().__init__(**kwargs)


class MiDashengLMConfig(PretrainedConfig):
    model_type = "midashenglm"

    def __init__(
        self,
        audio_encoder_config: dict | None = None,
        subsample_factor: int = 5,
        text_config: dict | None = None,
        audio_token_id: int | None = None,
        **kwargs,
    ):
        self.audio_encoder_config = DashengConfig(**(audio_encoder_config or {}))
        self.subsample_factor = subsample_factor
        self.text_config = (
            Qwen2_5OmniTextConfig(**text_config)
            if text_config
            else Qwen2_5OmniTextConfig()
        )
        self.text_config.rope_parameters = None  # uses_mrope is false
        self.audio_token_id = audio_token_id
        super().__init__(**kwargs)
