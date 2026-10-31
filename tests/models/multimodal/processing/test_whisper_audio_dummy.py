# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dummy audio and token budgets must use the processor's chunk_length."""

from unittest.mock import MagicMock

from vllm.model_executor.models.kimi_audio import KimiAudioDummyInputsBuilder
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
)
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioDummyInputsBuilder,
    Qwen2AudioProcessingInfo,
)
from vllm.model_executor.models.qwen3_asr import Qwen3ASRDummyInputsBuilder
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
)


class _FeatureExtractor:
    def __init__(
        self,
        chunk_length: int,
        sampling_rate: int = 16000,
        hop_length: int = 160,
    ):
        self.chunk_length = chunk_length
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length


def _expected_audio_tokens(
    chunk_length: int, sampling_rate: int = 16000, hop_length: int = 160
) -> int:
    audio_len = int(chunk_length * sampling_rate)
    max_mel_seq_len = audio_len // hop_length
    feat_lengths = (max_mel_seq_len - 1) // 2 + 1
    return (feat_lengths - 2) // 2 + 1


def _processing_info(extractor: _FeatureExtractor) -> Qwen2AudioProcessingInfo:
    info = Qwen2AudioProcessingInfo(ctx=MagicMock())
    info.get_feature_extractor = MagicMock(return_value=extractor)
    return info


def test_qwen2_audio_max_tokens_uses_processor_chunk_length():
    info = _processing_info(_FeatureExtractor(chunk_length=300))
    tokens = info.get_mm_max_tokens_per_item(
        seq_len=32768,
        mm_counts={"audio": 1},
    )
    assert tokens == {"audio": _expected_audio_tokens(300)}
    assert tokens["audio"] == 7500


def test_qwen2_audio_max_tokens_matches_thirty_second_processor():
    info = _processing_info(_FeatureExtractor(chunk_length=30))
    tokens = info.get_mm_max_tokens_per_item(
        seq_len=32768,
        mm_counts={"audio": 1},
    )
    assert tokens == {"audio": _expected_audio_tokens(30)}
    assert tokens["audio"] == 750


def _dummy_info(extractor: _FeatureExtractor) -> MagicMock:
    info = MagicMock()
    info.get_feature_extractor.return_value = extractor
    info.get_image_size_with_most_features.return_value = (32, 32)
    info.get_num_frames_with_most_features.return_value = 2
    return info


def test_qwen2_audio_dummy_uses_processor_chunk_length():
    builder = Qwen2AudioDummyInputsBuilder(_dummy_info(_FeatureExtractor(300)))
    mm_data = builder.get_dummy_mm_data(
        seq_len=8192,
        mm_counts={"audio": 1},
        mm_options={},
    )
    assert mm_data["audio"][0].shape[0] == 300 * 16000


def test_qwen2_5_omni_dummy_uses_processor_chunk_length():
    builder = Qwen2_5OmniThinkerDummyInputsBuilder(_dummy_info(_FeatureExtractor(300)))
    mm_data = builder.get_dummy_mm_data(
        seq_len=8192,
        mm_counts={"audio": 1, "image": 0, "video": 0},
        mm_options={},
    )
    assert mm_data["audio"][0].shape[0] == 300 * 16000


def test_qwen3_omni_dummy_builder_is_qwen2_5_omni_builder():
    assert Qwen3OmniMoeThinkerDummyInputsBuilder is (
        Qwen2_5OmniThinkerDummyInputsBuilder
    )


def test_kimi_audio_dummy_uses_processor_chunk_length():
    builder = KimiAudioDummyInputsBuilder(_dummy_info(_FeatureExtractor(300)))
    mm_data = builder.get_dummy_mm_data(
        seq_len=8192,
        mm_counts={"audio": 1},
        mm_options={},
    )
    assert mm_data["audio"][0].shape[0] == 300 * 16000


def test_qwen3_asr_dummy_uses_processor_chunk_length():
    builder = Qwen3ASRDummyInputsBuilder(_dummy_info(_FeatureExtractor(300)))
    mm_data = builder.get_dummy_mm_data(
        seq_len=8192,
        mm_counts={"audio": 1},
        mm_options={},
    )
    assert mm_data["audio"][0].shape[0] == 300 * 16000
