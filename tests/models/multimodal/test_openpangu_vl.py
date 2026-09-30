# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.openpangu_vl import (
    OpenPanguVLForConditionalGeneration,
)


class _LanguageModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.float().unsqueeze(-1).expand(-1, self.hidden_size).clone()


class _OpenPanguVL(OpenPanguVLForConditionalGeneration):
    def __init__(self, hidden_size: int) -> None:
        nn.Module.__init__(self)
        self.config = SimpleNamespace(image_token_id=41, video_token_id=42)
        self.language_model = _LanguageModel(hidden_size)

    def get_language_model(self) -> _LanguageModel:
        return self.language_model


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_get_input_embeddings_merges_image_and_video_embeddings(
    device_type: str,
) -> None:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device(device_type)
    hidden_size = 4
    model = _OpenPanguVL(hidden_size).to(device)
    input_ids = torch.tensor([7, 41, 41, 8, 42, 9], device=device)
    multimodal_embeddings = (
        torch.full((2, hidden_size), -1.0, device=device),
        torch.full((1, hidden_size), -2.0, device=device),
    )

    actual = model.get_input_embeddings(input_ids, multimodal_embeddings)

    expected = model.language_model.embed_input_ids(input_ids)
    expected[1:3] = -1.0
    expected[4] = -2.0
    torch.testing.assert_close(actual, expected)
