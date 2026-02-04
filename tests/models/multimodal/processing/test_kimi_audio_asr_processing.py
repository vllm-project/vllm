# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.kimi_audio_asr import (
    KimiAudioASRMultiModalDataParser,
    KimiAudioASRMultiModalProcessor,
)


def test_kimi_audio_uses_processing_info_data_parser_api():
    # BaseMultiModalProcessor prefers `info.get_data_parser()` unless a subclass
    # defines the deprecated `_get_data_parser` hook.
    assert not hasattr(KimiAudioASRMultiModalProcessor, "_get_data_parser")


def test_kimi_audio_data_parser_accepts_dict_of_tensors():
    parser = KimiAudioASRMultiModalDataParser()

    data = {
        "whisper_input_features": torch.zeros((1, 1, 5120), dtype=torch.float16),
        "is_continuous_mask": torch.zeros((1, 1), dtype=torch.bool),
        "text_input_ids": torch.zeros((1, 1), dtype=torch.long),
        "audio_input_ids": torch.zeros((1, 1), dtype=torch.long),
    }

    # The Kimi parser should accept dict-of-tensors (not waveform AudioItem).
    items = parser._parse_audio_data(data)
    assert items is not None
