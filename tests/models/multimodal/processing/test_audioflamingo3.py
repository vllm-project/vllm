# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
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

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig

from tests.models.registry import HF_EXAMPLE_MODELS


class MockAudioFlamingo3Config(PretrainedConfig):
    model_type = "audioflamingo3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_config = PretrainedConfig()
        self.text_config = PretrainedConfig()


class MockAudioFlamingo3Processor:
    def __init__(self):
        self.audio_token = "<sound>"
        self.audio_token_id = 12345
        self.feature_extractor = MockFeatureExtractor()

    def __call__(self, text=None, audios=None, **kwargs):
        return {"input_ids": [1, 2, 3], "input_features": [np.zeros((3000, 80))]}


class MockFeatureExtractor:
    def __init__(self):
        self.sampling_rate = 16000
        self.chunk_length = 30


@pytest.fixture
def mock_ctx():
    config = MockAudioFlamingo3Config()

    ctx = MagicMock()
    ctx.get_hf_config.return_value = config
    ctx.get_hf_processor.return_value = MockAudioFlamingo3Processor()
    ctx.model_config.hf_config = config
    return ctx


@pytest.fixture(autouse=True)
def check_transformers_version():
    # Check if the model is supported by the current transformers version
    model_info = HF_EXAMPLE_MODELS.get_hf_info("AudioFlamingo3ForConditionalGeneration")
    model_info.check_transformers_version(on_fail="skip")


def test_audio_chunk_counting(mock_ctx):
    from vllm.model_executor.models.audioflamingo3 import (
        AudioFlamingo3DummyInputsBuilder,
        AudioFlamingo3MultiModalProcessor,
        AudioFlamingo3ProcessingInfo,
    )

    info = AudioFlamingo3ProcessingInfo(mock_ctx)
    processor = AudioFlamingo3MultiModalProcessor(
        info, AudioFlamingo3DummyInputsBuilder(info)
    )

    sr = 16000
    audio_1 = np.zeros(30 * sr)
    audio_2 = np.zeros(45 * sr)

    mm_data = {"audio": [audio_1, audio_2]}
    prompt = "<|user|>Listen.<|end|>"

    from vllm.multimodal.processing import BaseMultiModalProcessor

    def mock_base_call(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        return {"input_ids": [1, 2, 3], "input_features": torch.randn(1, 80, 3000)}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BaseMultiModalProcessor, "_call_hf_processor", mock_base_call)

        processed = processor._call_hf_processor(prompt, mm_data, {}, {})

        chunk_counts = processed["chunk_counts"]

        assert chunk_counts[0].item() == 1
        assert chunk_counts[1].item() == 2
        assert len(chunk_counts) == 2


def test_dummy_data_generation(mock_ctx):
    from vllm.model_executor.models.audioflamingo3 import (
        AudioFlamingo3DummyInputsBuilder,
        AudioFlamingo3ProcessingInfo,
    )

    info = AudioFlamingo3ProcessingInfo(mock_ctx)
    builder = AudioFlamingo3DummyInputsBuilder(info)

    mm_counts = {"audio": 2}
    dummy_data = builder.get_dummy_mm_data(100, mm_counts, None)

    assert "audio" in dummy_data
    assert len(dummy_data["audio"]) == 2

    expected_len = 600 * 16000
    assert len(dummy_data["audio"][0]) == expected_len
