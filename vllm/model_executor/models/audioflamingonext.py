# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 The vLLM team.
# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

from collections.abc import Mapping

from transformers.models.audioflamingonext import (
    AudioFlamingoNextConfig,
    AudioFlamingoNextProcessor,
)

from vllm.multimodal import MULTIMODAL_REGISTRY

from .musicflamingo import (
    MusicFlamingoDummyInputsBuilder,
    MusicFlamingoEmbeddingInputs,
    MusicFlamingoEncoder,
    MusicFlamingoFeatureInputs,
    MusicFlamingoInputs,
    MusicFlamingoMultiModalProcessor,
    MusicFlamingoMultiModalProjector,
    MusicFlamingoProcessingInfo,
    MusicFlamingoRotaryEmbedding,
)
from .musicflamingo import (
    MusicFlamingoForConditionalGeneration as _MusicFlamingoForConditionalGeneration,
)

AudioFlamingoNextFeatureInputs = MusicFlamingoFeatureInputs
AudioFlamingoNextEmbeddingInputs = MusicFlamingoEmbeddingInputs
AudioFlamingoNextInputs = MusicFlamingoInputs


class AudioFlamingoNextEncoder(MusicFlamingoEncoder):
    pass


class AudioFlamingoNextRotaryEmbedding(MusicFlamingoRotaryEmbedding):
    pass


class AudioFlamingoNextMultiModalProjector(MusicFlamingoMultiModalProjector):
    pass


class AudioFlamingoNextProcessingInfo(MusicFlamingoProcessingInfo):
    def get_hf_config(self) -> AudioFlamingoNextConfig:
        return self.ctx.get_hf_config(AudioFlamingoNextConfig)

    def get_hf_processor(self, **kwargs: object) -> AudioFlamingoNextProcessor:
        return self.ctx.get_hf_processor(AudioFlamingoNextProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}


class AudioFlamingoNextDummyInputsBuilder(MusicFlamingoDummyInputsBuilder):
    pass


class AudioFlamingoNextMultiModalProcessor(MusicFlamingoMultiModalProcessor):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    AudioFlamingoNextMultiModalProcessor,
    info=AudioFlamingoNextProcessingInfo,
    dummy_inputs=AudioFlamingoNextDummyInputsBuilder,
)
class AudioFlamingoNextForConditionalGeneration(_MusicFlamingoForConditionalGeneration):
    """vLLM AudioFlamingoNext model aligned with HF modular_audioflamingonext."""

    pass
