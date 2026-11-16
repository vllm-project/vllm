# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny/blob/ac95e7b1678c08ea6f4a2581a051e432ed63e9b7/configuration_ming_v3_asr.py
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
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
"""Typed configuration for Ming V3 ASR and its reusable audio tower."""

from typing import Any

from transformers import PreTrainedConfig

from .bailing_moe_v3_vl import BailingMoeV3TextConfig


class MingAudioConfig(PreTrainedConfig):
    model_type = "ming_audio_encoder"

    def __init__(
        self,
        n_mels: int = 128,
        n_audio_ctx: int = 3000,
        n_audio_state: int = 1280,
        n_audio_head: int = 20,
        n_audio_layer: int = 32,
        downsampler_kernel_size_list: list[int] | None = None,
        downsampler_stride_list: list[int] | None = None,
        downsampler_padding_list: list[int] | None = None,
        chunk_frames: int = 3000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.downsampler_kernel_size_list = downsampler_kernel_size_list or [3]
        self.downsampler_stride_list = downsampler_stride_list or [2]
        self.downsampler_padding_list = (
            downsampler_padding_list
            if downsampler_padding_list is not None
            else [kernel // 2 for kernel in self.downsampler_kernel_size_list]
        )
        self.chunk_frames = chunk_frames


class MingV3ASRConfig(PreTrainedConfig):
    model_type = "ming_v3_asr"
    sub_configs = {
        "text_config": BailingMoeV3TextConfig,
        "audio_config": MingAudioConfig,
    }

    def __init__(
        self,
        text_config: dict | BailingMoeV3TextConfig | None = None,
        audio_config: dict | MingAudioConfig | None = None,
        audio_token_id: int = 157168,
        **kwargs: Any,
    ) -> None:
        self.text_config = (
            BailingMoeV3TextConfig(**text_config)
            if isinstance(text_config, dict)
            else text_config or BailingMoeV3TextConfig()
        )
        self.audio_config = (
            MingAudioConfig(**audio_config)
            if isinstance(audio_config, dict)
            else audio_config or MingAudioConfig()
        )
        self.audio_token_id = audio_token_id
        super().__init__(**kwargs)
