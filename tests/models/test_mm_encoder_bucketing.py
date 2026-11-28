# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for encoder length bucketing (`mm_encoder_bucketing`).

The bucketer decides, per batch, how many rungs to run and which rows go in
each. Tests exercise that decision plus the two properties the feature lives or
dies by: each row is cropped to its own rung (no content loss) and a mask-aware
encoder's output is unchanged by the crop.
"""

import pytest
import torch
from torch import nn

from vllm.model_executor.models.mm_encoder_bucketing import (
    EncoderBucketing,
    EncoderBucketingSpec,
    LengthBucketingConfig,
    _compute_budget,
    _ladder_seconds,
    _rungs_from_seconds,
)


class _Spec(EncoderBucketingSpec):
    """Mirrors `Qwen2AudioBucketingSpec`: `[B][mel][T]` + mask, crop on time."""

    frames_per_second = 100
    context_margin = 0

    def item_lengths(self, mm_kwargs_batch):
        mask = mm_kwargs_batch.get("feature_attention_mask")
        return mask.sum(-1).tolist() if mask is not None else None

    def padded_time(self, mm_kwargs_batch):
        return int(mm_kwargs_batch["input_features"].shape[-1])

    def crop(self, mm_kwargs_batch, indices, rung):
        rows = torch.tensor(indices, dtype=torch.long)
        return {
            "input_features": mm_kwargs_batch["input_features"][rows, :, :rung],
            "feature_attention_mask": mm_kwargs_batch["feature_attention_mask"][
                rows, :rung
            ],
        }


class _MarginSpec(_Spec):
    context_margin = 4


class _MaskAwareToy(nn.Module):
    """Mean over valid frames; padding cannot reach the output."""

    def forward(self, input_features, feature_attention_mask):
        lengths = feature_attention_mask.sum(-1).tolist()
        outs = [
            input_features[i, :, :length].mean(-1) for i, length in enumerate(lengths)
        ]
        return torch.stack(outs, dim=0)


def _batch(lengths, time_extent, mel=4):
    feats = torch.randn(len(lengths), mel, time_extent)
    mask = torch.zeros(len(lengths), time_extent, dtype=torch.long)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return {"input_features": feats, "feature_attention_mask": mask}


def test_rung_ladder_math():
    assert _rungs_from_seconds((2.0, 4.0), 10) == (20, 40)
    assert _rungs_from_seconds((0.05,), 10) == (1,)  # never a zero-width rung
    assert _rungs_from_seconds((2.0, 4.0), 10, align_to=8) == (24, 40)

    ladder = LengthBucketingConfig(buckets=(200, 400, 3000))
    assert ladder.padded_length(150) == 200
    assert ladder.padded_length(3000) == 3000
    assert ladder.padded_length(9999) is None  # bucket() falls back, never clamps

    assert _ladder_seconds({"audio_bucket_seconds": [4, 2, 4]}) == (2.0, 4.0)
    assert _ladder_seconds(None) is None
    assert _ladder_seconds({}) is None
    for bad in ({"audio_bucket_seconds": 4}, {"audio_bucket_seconds": [0]}):
        with pytest.raises(ValueError):
            _ladder_seconds(bad)


def test_bucket_groups_items_into_rungs():
    batch = _batch([5, 30, 12], time_extent=32)
    bucketer = EncoderBucketing(_Spec(), LengthBucketingConfig((8, 16, 32)))
    rungs = bucketer.bucket(batch, 3)
    got = {kw["input_features"].shape[-1]: indices for indices, kw in rungs}
    assert got == {8: [0], 32: [1], 16: [2]}
    for indices, kw in rungs:
        window = kw["input_features"].shape[-1]
        assert kw["input_features"].shape == (len(indices), 4, window)
        assert kw["feature_attention_mask"].shape == (len(indices), window)

    # Context margin: length 5 + 4 = 9 selects rung 16 so content [0, 5) fits.
    batch = _batch([5], time_extent=32)
    margin_rungs = EncoderBucketing(
        _MarginSpec(), LengthBucketingConfig((8, 16, 32))
    ).bucket(batch, 1)
    assert margin_rungs[0][0] == [0]
    assert margin_rungs[0][1]["input_features"].shape[-1] == 16


def test_bucket_fallbacks():
    # Waste gate: uniform near-full items compute no less than the batch.
    batch = _batch([28, 28], time_extent=32)
    rungs = EncoderBucketing(
        _Spec(), LengthBucketingConfig((8, 16, 32))
    ).bucket(batch, 2)
    assert len(rungs) == 1 and rungs[0][1] is batch

    # Too long for the ladder: never truncate content.
    batch = _batch([30], time_extent=32)
    rungs = EncoderBucketing(_Spec(), LengthBucketingConfig((8, 16))).bucket(batch, 1)
    assert len(rungs) == 1 and rungs[0][1] is batch

    # No ladder: bucketing is inactive.
    batch = _batch([5, 30], time_extent=32)
    bucketer = EncoderBucketing(_Spec(), None)
    assert not bucketer.active
    rungs = bucketer.bucket(batch, 2)
    assert len(rungs) == 1 and rungs[0][0] == [0, 1] and rungs[0][1] is batch


def test_mask_aware_shrink_is_bit_identical():
    torch.manual_seed(0)
    batch = _batch([5, 30, 20], time_extent=32)
    encoder = _MaskAwareToy()
    full_out = encoder(**batch)

    bucketer = EncoderBucketing(_Spec(), LengthBucketingConfig((8, 16, 32)))
    bucketed = torch.empty_like(full_out)
    for indices, rung_kwargs in bucketer.bucket(batch, 3):
        out = encoder(**rung_kwargs)
        for position, row in zip(indices, out):
            bucketed[position] = row

    torch.testing.assert_close(bucketed, full_out, rtol=0, atol=0)


def test_budget_merging():
    assert _compute_budget(None) is None
    assert _compute_budget({"audio_bucket_budget": 48}) == 48
    # The retired fill-floor knob is ignored; the merge trigger is derived.
    retired = {"audio_bucket_budget": 48, "audio_bucket_fill_floor": 12}
    assert _compute_budget(retired) == 48
    for bad in (
        {"audio_bucket_budget": 0},
        {"audio_bucket_budget": -4},
        {"audio_bucket_budget": True},
        {"audio_bucket_budget": 4.5},
    ):
        with pytest.raises(ValueError):
            _compute_budget(bad)

    # Sparse group (8 < floor 64 // 6 = 10) folds into rung 16 (32 <= 64).
    batch = _batch([5, 30, 12], time_extent=32)
    got = {
        kw["input_features"].shape[-1]: indices
        for indices, kw in EncoderBucketing(
            _Spec(), LengthBucketingConfig((8, 16, 32)), compute_budget=64
        ).bucket(batch, 3)
    }
    assert got == {16: [0, 2], 32: [1]}

    # Oversized group (640 > 512) splits into balanced 10 + 10 calls.
    batch = _batch([1] + [30] * 20, time_extent=32)
    rungs = EncoderBucketing(
        _Spec(), LengthBucketingConfig((8, 16, 32)), compute_budget=512
    ).bucket(batch, 21)
    assert [len(indices) for indices, _ in rungs] == [1, 10, 10]

    # A batch that fits the budget runs as one call at its top rung.
    batch = _batch([5, 30, 12], time_extent=32)
    rungs = EncoderBucketing(
        _Spec(), LengthBucketingConfig((8, 16, 32)), compute_budget=96
    ).bucket(batch, 3)
    assert len(rungs) == 1
    indices, kw = rungs[0]
    assert sorted(indices) == [0, 1, 2]
    assert kw["input_features"].shape[-1] == 32


def test_full_window_shim_is_byte_identical_to_original_forward():
    from transformers.models.qwen2_audio import (
        Qwen2AudioEncoder,
        Qwen2AudioEncoderConfig,
    )

    from vllm.model_executor.models.qwen2_audio import _qwen2audio_varlen_shim

    # Full window = 8 positions * conv strides (1 * 2) = 16 mel frames.
    config = Qwen2AudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=64,
        d_model=32,
        dropout=0.0,
        max_source_positions=8,
    )
    encoder = Qwen2AudioEncoder(config).eval()
    torch.manual_seed(0)
    input_features = torch.randn(2, 8, 16)
    original = encoder(input_features)
    shimmed = _qwen2audio_varlen_shim(encoder, input_features, None)
    torch.testing.assert_close(
        shimmed.last_hidden_state, original.last_hidden_state, rtol=0, atol=0
    )