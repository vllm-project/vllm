# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import SpeechToTextConfig
from vllm.model_executor.models.cohere_asr import (
    CohereAsrForConditionalGeneration,
    RelPositionMultiHeadAttention,
)


class _ReferenceRelPositionMultiHeadAttention(RelPositionMultiHeadAttention):
    """Relative attention implementation before score accumulation is fused."""

    def forward(self, query, key, value, mask, pos_emb=None):
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)

        assert pos_emb is not None
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) / self.s_d_k
        return self.forward_attention(v, scores, mask)


@pytest.mark.parametrize(("query_len", "key_len"), [(2, 2), (3, 5)])
def test_relative_attention_fuses_score_accumulation(monkeypatch, query_len, key_len):
    torch.manual_seed(7)
    batch, heads, model_dim = 2, 4, 32
    optimized = RelPositionMultiHeadAttention(heads, model_dim, None, None)
    reference = _ReferenceRelPositionMultiHeadAttention(heads, model_dim, None, None)
    reference.load_state_dict(optimized.state_dict())

    query = torch.randn(batch, query_len, model_dim)
    key = torch.randn(batch, key_len, model_dim)
    value = torch.randn(batch, key_len, model_dim)
    pos_emb = torch.randn(1, query_len + key_len - 1, model_dim)
    expected = reference(query, key, value, None, pos_emb)

    original_baddbmm = torch.baddbmm
    calls = 0

    def tracked_baddbmm(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_baddbmm(*args, **kwargs)

    monkeypatch.setattr(torch, "baddbmm", tracked_baddbmm)
    actual = optimized(query, key, value, None, pos_emb)

    torch.testing.assert_close(actual, expected)
    assert calls == 1


@pytest.mark.parametrize(
    ("audio_duration_s", "expected_tokens"),
    # window_stride=0.01s @ 16kHz -> 160-sample hop; subsampling_factor=8.
    # frames = floor(duration * 16000 / 160); tokens = ceil(frames / 8).
    [(1.0, 13), (10.0, 125), (30.0, 375)],
)
def test_get_num_audio_tokens_streaming_estimate(audio_duration_s, expected_tokens):
    """The duration-based estimate must convert ``window_stride`` (seconds) to a
    sample hop and divide by the encoder subsampling factor. Values are pinned to
    concrete numbers so a rounding regression is actually caught."""
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            preprocessor={"window_stride": 0.01, "sample_rate": 16000},
            encoder={"subsampling_factor": 8},
        )
    )
    stt_config = SpeechToTextConfig(sample_rate=16000)

    got = CohereAsrForConditionalGeneration.get_num_audio_tokens(
        audio_duration_s, stt_config, model_config
    )

    assert got == expected_tokens
