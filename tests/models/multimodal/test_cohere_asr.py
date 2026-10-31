# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import SpeechToTextConfig
from vllm.model_executor.models.cohere_asr import (
    INF_VAL,
    CohereAsrForConditionalGeneration,
    RelPositionMultiHeadAttention,
    cohere_chunk_attention_mask,
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


def _track_score_tiles(monkeypatch) -> dict[str, int]:
    """Record 4D score matmuls and reject a query tile longer than 128."""
    original = torch.matmul
    seen = {"tiles": 0}

    def tracked(left, right, *args, **kwargs):
        if torch.is_tensor(left) and left.dim() == 4:
            query_tile = left.size(-2)
            if query_tile > 128:
                raise RuntimeError(f"score tile has {query_tile} queries")
            seen["tiles"] += 1
        return original(left, right, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", tracked)
    return seen


def legacy_cohere_att_mask(
    padding_length: torch.Tensor,
    att_context_size: list[int],
    style: str,
    length: int,
) -> torch.Tensor:
    """Full-sequence mask from the pre-chunk ``_create_masks`` pairs."""
    device = padding_length.device
    att_mask = torch.ones(1, length, length, dtype=torch.bool, device=device)
    left, right = int(att_context_size[0]), int(att_context_size[1])
    if style == "regular":
        if left >= 0:
            att_mask = att_mask.triu(diagonal=-left)
        if right >= 0:
            att_mask = att_mask.tril(diagonal=right)
    elif style == "chunked_limited":
        if right == -1:
            if left >= 0:
                att_mask = att_mask.triu(diagonal=-left)
        else:
            chunk_size = right + 1
            left_chunks = left // chunk_size if left >= 0 else 10000
            chunk_idx = torch.arange(0, length, dtype=torch.int, device=device)
            chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
            diff = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
            limited = torch.logical_and(diff <= left_chunks, diff >= 0)
            att_mask = torch.logical_and(att_mask, limited.unsqueeze(0))
    valid = torch.arange(0, length, device=device).expand(
        padding_length.shape[0], -1
    ) < padding_length.unsqueeze(-1)
    valid_pair = valid.unsqueeze(1).repeat(1, length, 1)
    valid_pair = torch.logical_and(valid_pair, valid_pair.transpose(1, 2))
    return ~torch.logical_and(valid_pair, att_mask)


@pytest.mark.parametrize(
    ("query_len", "key_len"), [(2, 2), (3, 5), (130, 130), (130, 64)]
)
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

    seen = _track_score_tiles(monkeypatch)
    actual = optimized(query, key, value, None, pos_emb)

    torch.testing.assert_close(actual, expected)
    assert seen["tiles"] >= 1


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("style", "context"),
    [
        ("regular", [-1, -1]),
        ("regular", [4, 6]),
        ("chunked_limited", [8, 3]),
        ("chunked_limited", [5, -1]),
    ],
)
def test_cohere_chunk_mask_matches_full_mask(style, context):
    length = 130
    padding = torch.tensor([length, 7, 0])
    expected = legacy_cohere_att_mask(padding, context, style, length)
    parts = []
    for row_start in range(0, length, 128):
        row_end = min(row_start + 128, length)
        parts.append(
            cohere_chunk_attention_mask(
                row_start, row_end, length, padding, context, style
            )
        )
    actual = torch.cat(parts, dim=1)
    assert torch.equal(actual, expected)
    # Local row 0 of the second tile is absolute row 128.
    second = cohere_chunk_attention_mask(128, length, length, padding, context, style)
    assert torch.equal(second[:, 0], expected[:, 128])


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("style", "context"),
    [("regular", [4, 6]), ("chunked_limited", [8, 3])],
)
def test_cohere_fully_masked_row_is_zero_after_fill(monkeypatch, style, context):
    torch.manual_seed(11)
    batch, heads, model_dim, length = 2, 4, 32, 130
    optimized = RelPositionMultiHeadAttention(
        heads,
        model_dim,
        None,
        None,
        att_context_size=context,
        att_context_style=style,
    )
    reference = _ReferenceRelPositionMultiHeadAttention(
        heads,
        model_dim,
        None,
        None,
        att_context_size=context,
        att_context_style=style,
    )
    reference.load_state_dict(optimized.state_dict())

    query = torch.randn(batch, length, model_dim)
    key = torch.randn(batch, length, model_dim)
    value = torch.randn(batch, length, model_dim)
    pos_emb = torch.randn(1, 2 * length - 1, model_dim)
    padding = torch.tensor([length, 0])
    legacy_mask = legacy_cohere_att_mask(padding, context, style, length)
    expected = reference(query, key, value, legacy_mask, pos_emb)

    fill_values: list[float] = []
    original_fill = torch.Tensor.masked_fill

    def tracked_fill(tensor, mask, value):
        if isinstance(value, (int, float)):
            fill_values.append(float(value))
        return original_fill(tensor, mask, value)

    monkeypatch.setattr(torch.Tensor, "masked_fill", tracked_fill)
    seen = _track_score_tiles(monkeypatch)
    output, probs = optimized(
        query,
        key,
        value,
        None,
        pos_emb,
        padding_length=padding,
        need_weights=True,
    )

    torch.testing.assert_close(output, expected)
    assert probs.shape == (batch, heads, length, length)
    assert torch.isfinite(probs).all()
    assert torch.count_nonzero(probs[1]) == 0
    torch.testing.assert_close(probs[0].sum(dim=-1), torch.ones(heads, length))
    assert -INF_VAL in fill_values
    assert 0.0 in fill_values
    assert seen["tiles"] >= 1


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
