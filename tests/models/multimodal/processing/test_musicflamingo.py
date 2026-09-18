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
        self.tokenizer = self._tokenize

    def __call__(self, text=None, audio=None, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "input_features": torch.zeros((3, 80, 3000)),
            "input_features_mask": torch.ones((3, 3000), dtype=torch.long),
        }

    def _tokenize(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


class MockFeatureExtractor:
    def __init__(self):
        self.sampling_rate = 16000
        self.chunk_length = 30
        self.hop_length = 160

    def __call__(self, audios, **kwargs):
        return {
            "input_features": torch.zeros((len(audios), 80, 3000)),
            "attention_mask": torch.ones((len(audios), 3000), dtype=torch.long),
        }


@pytest.fixture
def mock_ctx():
    config = MockMusicFlamingoConfig()

    ctx = MagicMock()
    ctx.get_hf_config.return_value = config
    ctx.get_hf_processor.return_value = MockMusicFlamingoProcessor()
    ctx.call_hf_processor.side_effect = lambda processor, data, kwargs: processor(
        **data, **kwargs
    )
    ctx.model_config.hf_config = config
    return ctx


@pytest.fixture(autouse=True)
def check_transformers_version():
    model_info = HF_EXAMPLE_MODELS.get_hf_info("MusicFlamingoForConditionalGeneration")
    model_info.check_transformers_version(on_fail="skip")


def test_musicflamingo_chunk_counting_without_rote_timestamps(mock_ctx):
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

    mm_items = processor.info.parse_mm_data(mm_data, validate=False)
    processed = processor._apply_hf_processor_main(mm_items, {})

    chunk_counts = processed["chunk_counts"]

    assert chunk_counts.tolist() == [1, 2]
    assert "rote_timestamps" not in processed
    assert processed["feature_attention_mask"].shape == (3, 3000)


def test_musicflamingo_dummy_text_uses_plain_audio_tokens(mock_ctx):
    from vllm.model_executor.models.musicflamingo import (
        MusicFlamingoDummyInputsBuilder,
        MusicFlamingoProcessingInfo,
    )

    info = MusicFlamingoProcessingInfo(mock_ctx)
    builder = MusicFlamingoDummyInputsBuilder(info)

    assert builder.get_dummy_text({"audio": 2}) == "<sound><sound>"


def test_musicflamingo_audio_features_match_hf_small_config(default_vllm_config):
    """Chunk-derived RoTE timestamps must match HF's input_ids-derived ones.

    HF reads each chunk's index within its own audio off the runs of audio tokens in
    `input_ids`. vLLM encodes audio without the text tokens, so it derives the same
    index from `chunk_counts`; the two encoders must still agree exactly.
    """
    from transformers.models.musicflamingo import (
        MusicFlamingoConfig,
        MusicFlamingoForConditionalGeneration,
    )

    from vllm.model_executor.models.audioflamingo3 import (
        _build_audio_encoder_attention_mask,
        _flatten_valid_audio_embeddings,
        _get_audio_post_pool_output_lengths,
    )
    from vllm.model_executor.models.musicflamingo import (
        MusicFlamingoEncoder,
        MusicFlamingoMultiModalProjector,
        MusicFlamingoRotaryEmbedding,
        _build_audio_timestamps,
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
    }
    audio_config = {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "model_type": "audioflamingo3_encoder",
        "num_mel_bins": 80,
        "max_source_positions": 1500,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
    }

    torch.manual_seed(0)
    audio_token_id = 0
    config = MusicFlamingoConfig(
        text_config=text_config,
        audio_config=audio_config,
        audio_token_id=audio_token_id,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 2048,
            "partial_rotary_factor": 0.2,
        },
    )
    hf_model = MusicFlamingoForConditionalGeneration(config).eval()

    # Two audios, the second split across two 30s chunks.
    chunk_counts = [1, 2]
    input_features = torch.randn(3, 80, 3000)
    feature_attention_mask = torch.zeros(3, 3000, dtype=torch.bool)
    feature_attention_mask[0, :3000] = True
    feature_attention_mask[1, :3000] = True
    feature_attention_mask[2, :1500] = True

    post_lengths = _get_audio_post_pool_output_lengths(
        feature_attention_mask.sum(-1).to(torch.long)
    )
    # One run of audio tokens per audio, so HF sees the same chunk grouping.
    input_ids = []
    current_chunk = 0
    for count in chunk_counts:
        num_tokens = int(post_lengths[current_chunk : current_chunk + count].sum())
        input_ids.extend([audio_token_id] * num_tokens)
        input_ids.append(audio_token_id + 1)
        current_chunk += count
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        hf_output = hf_model.get_audio_features(
            input_features,
            feature_attention_mask,
            input_ids,
            return_dict=True,
        ).pooler_output

    vllm_encoder = MusicFlamingoEncoder(config.audio_config).eval()
    vllm_encoder.load_state_dict(hf_model.model.audio_tower.state_dict())
    vllm_projector = MusicFlamingoMultiModalProjector(config).eval()
    vllm_projector.load_state_dict(hf_model.model.multi_modal_projector.state_dict())
    vllm_rope = MusicFlamingoRotaryEmbedding(config).eval()

    with torch.no_grad():
        vllm_hidden_states = vllm_encoder(
            input_features,
            attention_mask=_build_audio_encoder_attention_mask(
                feature_attention_mask,
                dtype=vllm_encoder.conv1.weight.dtype,
                device=vllm_encoder.conv1.weight.device,
            ),
        )
        seq_len = vllm_hidden_states.shape[-2]
        rote_timestamps = _build_audio_timestamps(
            chunk_counts,
            seq_len=seq_len,
            audio_frame_step=config.audio_frame_step,
            device=vllm_hidden_states.device,
        )
        cos, sin = vllm_rope(rote_timestamps, seq_len=seq_len)
        vllm_output, _ = _flatten_valid_audio_embeddings(
            vllm_projector(apply_rotary_time_emb(vllm_hidden_states, cos, sin)),
            feature_attention_mask,
        )

    torch.testing.assert_close(vllm_output, hf_output)
