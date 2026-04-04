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

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig

from tests.models.registry import HF_EXAMPLE_MODELS


class MockMusicFlamingoConfig(PretrainedConfig):
    model_type = "musicflamingo"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_config = PretrainedConfig()
        self.text_config = PretrainedConfig()


class MockMusicFlamingoProcessor:
    def __init__(self):
        self.audio_token = "<sound>"
        self.audio_token_id = 12345
        self.audio_bos_token = "<|sound_bos|>"
        self.audio_bos_token_id = 12346
        self.audio_eos_token = "<|sound_eos|>"
        self.audio_eos_token_id = 12347
        self.max_audio_len = 1200
        self.feature_extractor = MockFeatureExtractor()


class MockFeatureExtractor:
    def __init__(self):
        self.sampling_rate = 16000
        self.chunk_length = 30


@pytest.fixture
def mock_ctx():
    config = MockMusicFlamingoConfig()

    ctx = MagicMock()
    ctx.get_hf_config.return_value = config
    ctx.get_hf_processor.return_value = MockMusicFlamingoProcessor()
    ctx.model_config.hf_config = config
    return ctx


@pytest.fixture(autouse=True)
def check_transformers_version():
    model_info = HF_EXAMPLE_MODELS.get_hf_info("MusicFlamingoForConditionalGeneration")
    model_info.check_transformers_version(on_fail="skip")


def test_musicflamingo_chunk_counting(mock_ctx, monkeypatch):
    from vllm.model_executor.models.musicflamingo import (
        MusicFlamingoDummyInputsBuilder,
        MusicFlamingoMultiModalProcessor,
        MusicFlamingoProcessingInfo,
    )

    info = MusicFlamingoProcessingInfo(mock_ctx)
    processor = MusicFlamingoMultiModalProcessor(
        info, MusicFlamingoDummyInputsBuilder(info)
    )

    sr = 16000
    audio_1 = np.zeros(30 * sr)
    audio_2 = np.zeros(45 * sr)

    mm_data = {"audio": [audio_1, audio_2]}
    prompt = "<|user|>Listen.<|end|>"

    from vllm.multimodal.processing import BaseMultiModalProcessor

    def mock_base_call(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        del self, prompt, mm_data, mm_kwargs, tok_kwargs
        return {
            "input_ids": [1, 2, 3],
            "input_features": torch.randn(3, 80, 3000),
        }

    monkeypatch.setattr(BaseMultiModalProcessor, "_call_hf_processor", mock_base_call)

    processed = processor._call_hf_processor(prompt, mm_data, {}, {})

    chunk_counts = processed["chunk_counts"]

    assert chunk_counts.tolist() == [1, 2]
    assert "rote_timestamps" not in processed


def test_musicflamingo_dummy_text_uses_plain_audio_tokens(mock_ctx):
    from vllm.model_executor.models.musicflamingo import (
        MusicFlamingoDummyInputsBuilder,
        MusicFlamingoProcessingInfo,
    )

    info = MusicFlamingoProcessingInfo(mock_ctx)
    builder = MusicFlamingoDummyInputsBuilder(info)

    assert builder.get_dummy_text({"audio": 2}) == "<sound><sound>"


def test_musicflamingo_audio_feature_pipeline_matches_hf_small_config():
    from transformers.models.musicflamingo import (
        modeling_musicflamingo as hf_musicflamingo_modeling,
    )
    from transformers.models.musicflamingo.configuration_musicflamingo import (
        MusicFlamingoConfig,
    )

    from vllm.model_executor.models.audioflamingo3 import (
        _build_audio_encoder_attention_mask,
        _flatten_valid_audio_embeddings,
    )
    from vllm.model_executor.models.musicflamingo import (
        MusicFlamingoEncoder,
        MusicFlamingoMultiModalProjector,
        MusicFlamingoRotaryEmbedding,
        apply_rotary_time_emb,
    )

    text_config = {
        "model_type": "qwen2",
        "intermediate_size": 64,
        "initializer_range": 0.02,
        "hidden_size": 32,
        "max_position_embeddings": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "pad_token_id": 1,
        "use_mrope": False,
    }
    audio_config = {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_mel_bins": 80,
        "max_source_positions": 1500,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.0,
    }

    torch.manual_seed(0)
    config = MusicFlamingoConfig(
        text_config=text_config,
        audio_config=audio_config,
        audio_token_id=0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 2048,
            "partial_rotary_factor": 0.5,
        },
    )
    hf_model = hf_musicflamingo_modeling.MusicFlamingoForConditionalGeneration(
        config
    ).eval()

    vllm_encoder = MusicFlamingoEncoder(config.audio_config).eval()
    vllm_encoder.load_state_dict(hf_model.audio_tower.state_dict())

    vllm_projector = MusicFlamingoMultiModalProjector(config).eval()
    vllm_projector.load_state_dict(hf_model.multi_modal_projector.state_dict())

    vllm_rope = MusicFlamingoRotaryEmbedding(config).eval()
    vllm_rope.load_state_dict(hf_model.pos_emb.state_dict(), strict=False)

    input_features = torch.randn(3, 80, 3000)
    feature_attention_mask = torch.zeros(3, 3000, dtype=torch.bool)
    feature_attention_mask[0, :3000] = True
    feature_attention_mask[1, :2500] = True
    feature_attention_mask[2, :1500] = True
    chunk_counts = [2, 1]
    post_lengths = [750 + 625, 375]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.zeros(length, dtype=torch.long) for length in post_lengths],
        batch_first=True,
        padding_value=1,
    )

    hf_output = hf_model.get_audio_features(
        input_features,
        feature_attention_mask,
        input_ids=input_ids,
        return_dict=True,
    ).pooler_output
    vllm_attention_mask = _build_audio_encoder_attention_mask(
        feature_attention_mask,
        dtype=vllm_encoder.conv1.weight.dtype,
        device=vllm_encoder.conv1.weight.device,
    )
    vllm_hidden_states = vllm_encoder(
        input_features,
        attention_mask=vllm_attention_mask,
    )
    from vllm.model_executor.models.musicflamingo import _build_audio_timestamps

    audio_timestamps = _build_audio_timestamps(
        feature_attention_mask,
        chunk_counts,
        vllm_hidden_states.shape[-2],
        config.audio_frame_step,
    )
    cos, sin = vllm_rope(audio_timestamps, seq_len=vllm_hidden_states.shape[-2])
    vllm_hidden_states = apply_rotary_time_emb(vllm_hidden_states, cos, sin)
    vllm_output, _ = _flatten_valid_audio_embeddings(
        vllm_projector(vllm_hidden_states),
        feature_attention_mask,
    )

    torch.testing.assert_close(vllm_output, hf_output)
