# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.kimi_audio_asr import KimiAudioForConditionalGeneration


class _DummyEmbedTokens:
    def __init__(self, hidden: int):
        self.embedding_dim = hidden
        self.weight = torch.zeros((1, hidden), dtype=torch.float16)

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [S]
        s = int(ids.numel())
        return torch.zeros((s, self.embedding_dim), dtype=self.weight.dtype)


class _DummyVQAdaptor(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [S, B, F] and return [S, B, H]
        assert x.dim() == 3
        s, b = int(x.shape[0]), int(x.shape[1])
        return torch.zeros((s, b, self.hidden), dtype=torch.float16)


class _DummyModel:
    def __init__(self, hidden: int):
        self.embed_tokens = _DummyEmbedTokens(hidden)
        self.vq_adaptor = _DummyVQAdaptor(hidden)


def test_embed_input_ids_smoke_no_shape_errors():
    # This is a lightweight unit test to ensure the embed_input_ids mixing path
    # works with flattened token shapes (vLLM V1 convention) and does not crash
    # on common [S] / [S, F] inputs.
    hidden = 3584
    s = 8

    dummy_self = type("_Dummy", (), {})()
    dummy_self.model = _DummyModel(hidden)

    input_ids = torch.zeros((s,), dtype=torch.long)
    audio_input_ids = torch.ones((s,), dtype=torch.long)
    is_continuous_mask = torch.ones((s,), dtype=torch.bool)
    text_input_ids = torch.zeros((s,), dtype=torch.long)

    # Case 1: raw whisper feature dim (needs vq_adaptor)
    out_raw = KimiAudioForConditionalGeneration.embed_input_ids(
        dummy_self,
        input_ids,
        whisper_input_features=torch.zeros((s, 5120), dtype=torch.float16),
        is_continuous_mask=is_continuous_mask,
        text_input_ids=text_input_ids,
        audio_input_ids=audio_input_ids,
    )
    assert isinstance(out_raw, torch.Tensor)
    assert out_raw.shape == (s, hidden)

    # Case 2: already-projected whisper embeddings (hidden_size), should bypass
    # adaptor and still succeed.
    out_proj = KimiAudioForConditionalGeneration.embed_input_ids(
        dummy_self,
        input_ids,
        whisper_input_features=torch.zeros((s, hidden), dtype=torch.float16),
        is_continuous_mask=is_continuous_mask,
        text_input_ids=text_input_ids,
        audio_input_ids=audio_input_ids,
    )
    assert isinstance(out_proj, torch.Tensor)
    assert out_proj.shape == (s, hidden)
