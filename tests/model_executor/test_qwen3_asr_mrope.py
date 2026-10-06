# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange


def test_mrope_positions_with_stripped_audio_data() -> None:
    input_tokens = list(range(7))
    feature = MultiModalFeatureSpec(
        data=None,
        modality="audio",
        identifier="audio",
        mm_position=PlaceholderRange(offset=2, length=3),
    )

    positions, delta = Qwen3ASRForConditionalGeneration.get_mrope_input_positions(
        None, input_tokens, [feature]
    )

    expected = torch.arange(len(input_tokens), dtype=torch.long).expand(3, -1)
    torch.testing.assert_close(positions, expected)
    assert delta == 0
