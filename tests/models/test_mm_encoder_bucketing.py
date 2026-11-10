# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for encoder length bucketing (``mm_encoder_bucketing``).

Bucketing runs at the batch level in ``EncoderRunner.execute_mm_encoder``, before
the CUDA-graph-or-eager split, so it is exercised here through the bucketer's own
``bucket`` -- the step that decides the rungs and crops each one. The two
properties the feature lives or dies by:

1. Bucketing crops each row to its own length rung and re-assembles rows in batch
   order (no content is lost, only padding).
2. For a mask-aware encoder the crop is lossless: the mask excludes padding from
   the function, so shrinking the padding cannot change the output at the valid
   frames -- bit-identical.

Host-side rung math (ladder, seconds conversion, waste gate, too-long fallback)
is covered directly; the variable-length shim is covered for install/uninstall.
"""

import functools

import torch
from torch import nn

from vllm.model_executor.models.mm_encoder_bucketing import (
    EncoderBucketing,
    EncoderBucketingSpec,
    LengthBucketingConfig,
    _FORWARD_SHIMS,
    _ladder_seconds,
    _rungs_from_seconds,
    install_encoder_bucketing,
    uninstall_encoder_bucketing,
)


class _Spec(EncoderBucketingSpec):
    """Mel-frame bucketing semantics for a ``[B][mel][T]`` + ``[B][T]`` batch.

    Mirrors `Qwen2AudioBucketingSpec`: per-item length is the valid mel-frame
    count, both fields crop on their mel time axis to the rung.
    """

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
    """Same semantics with a context margin past the published length."""

    context_margin = 4


class _MaskAwareToy(nn.Module):
    """Toy mask-aware encoder: mean over valid frames, padding excluded.

    The output depends only on the valid frames (``[:length]``), so shrinking
    padding while keeping content cannot change it.
    """

    def forward(self, input_features, feature_attention_mask):
        lengths = feature_attention_mask.sum(-1).tolist()
        outs = [
            input_features[i, :, :length].mean(-1)
            for i, length in enumerate(lengths)
        ]
        return torch.stack(outs, dim=0)


class Qwen2AudioEncoder(nn.Module):
    """Named to match the shim key so install wires the variable-length recipe."""

    def forward(self, input_features, attention_mask=None):
        raise AssertionError("original forward runs when the feature is off")


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_tower = Qwen2AudioEncoder()


def _batch(lengths, time_extent, mel=4):
    feats = torch.randn(len(lengths), mel, time_extent)
    mask = torch.zeros(len(lengths), time_extent, dtype=torch.long)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return {"input_features": feats, "feature_attention_mask": mask}


def test_rungs_from_seconds_scale_by_frame_rate():
    assert _rungs_from_seconds((2.0, 4.0), 10) == (20, 40)
    assert _rungs_from_seconds((0.05,), 10) == (1,)  # never a zero-width rung


def test_ladder_seconds_parses_and_rejects_bad_values():
    assert _ladder_seconds({"audio_bucket_seconds": [4, 2, 4]}) == (2.0, 4.0)
    assert _ladder_seconds({"audio_bucket_seconds": [2]}) == (2.0,)
    assert _ladder_seconds(None) is None
    assert _ladder_seconds({}) is None
    for bad in ({"audio_bucket_seconds": 4}, {"audio_bucket_seconds": [0]}):
        try:
            _ladder_seconds(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_padded_length_picks_smallest_fitting_rung():
    ladder = LengthBucketingConfig(buckets=(200, 400, 3000))
    assert ladder.padded_length(150) == 200
    assert ladder.padded_length(400) == 400
    assert ladder.padded_length(9999) == 3000  # fall back to the largest rung


def test_bucket_groups_items_into_fixed_length_rungs():
    batch = _batch([5, 30, 12], time_extent=32)
    bucketer = EncoderBucketing(_Spec(), LengthBucketingConfig((8, 16, 32)))
    rungs = bucketer.bucket(batch, 3)

    # index each returned sub-batch by its window length -> row indices
    got = {
        rung_kwargs["input_features"].shape[-1]: indices
        for indices, rung_kwargs in rungs
    }
    assert got == {8: [0], 32: [1], 16: [2]}
    # each sub-batch is a fixed-shape crop: selected rows, truncated to the rung
    for indices, rung_kwargs in rungs:
        rung = rung_kwargs["input_features"].shape[-1]
        assert rung_kwargs["input_features"].shape == (len(indices), 4, rung)
        assert rung_kwargs["feature_attention_mask"].shape == (len(indices), rung)


def test_bucket_margin_selects_a_rung_that_never_clips_content():
    batch = _batch([5], time_extent=32)
    bucketer = EncoderBucketing(_MarginSpec(), LengthBucketingConfig((8, 16, 32)))
    rungs = bucketer.bucket(batch, 1)
    # length 5 + margin 4 = 9 -> smallest rung >= 9 is 16, so content [0, 5) fits
    indices, rung_kwargs = rungs[0]
    assert indices == [0]
    assert rung_kwargs["input_features"].shape[-1] == 16


def test_bucket_waste_gate_keeps_full_batch():
    # Uniform near-full items: capped windows compute no less than the batch.
    batch = _batch([28, 28], time_extent=32)
    bucketer = EncoderBucketing(_Spec(), LengthBucketingConfig((8, 16, 32)))
    rungs = bucketer.bucket(batch, 2)
    assert len(rungs) == 1 and rungs[0][0] == [0, 1] and rungs[0][1] is batch


def test_bucket_too_long_item_keeps_full_batch():
    # An item longer than the ladder has no safe rung -- never truncate content.
    batch = _batch([30], time_extent=32)
    bucketer = EncoderBucketing(_Spec(), LengthBucketingConfig((8, 16)))
    rungs = bucketer.bucket(batch, 1)
    assert len(rungs) == 1 and rungs[0][0] == [0] and rungs[0][1] is batch


def test_bucket_without_ladder_is_passthrough():
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


def test_install_wires_shim_and_uninstall_restores_forward():
    # Fixed-window shims are registered per model (see qwen2_audio.py); register
    # a dummy here to exercise the install/uninstall mechanism itself.
    def dummy_shim(module, input_features, attention_mask=None, **kwargs):
        return input_features

    _FORWARD_SHIMS["Qwen2AudioEncoder"] = dummy_shim
    try:
        model = _Model()
        tower = model.audio_tower
        original_forward = tower.forward
        labels = install_encoder_bucketing(
            model,
            {"audio": tower},
            _Spec(),
            model_type="qwen2_audio",
            additional_config={"audio_bucket_seconds": [1, 2]},
        )
        assert labels == ["audio"]
        assert model._encoder_bucketing.active
        assert isinstance(tower.forward, functools.partial)  # variable-length shim

        restored = uninstall_encoder_bucketing(model)
        assert restored == ["Qwen2AudioEncoder"]
        assert tower.forward == original_forward  # original forward restored
        assert getattr(model, "_encoder_bucketing", None) is None
    finally:
        del _FORWARD_SHIMS["Qwen2AudioEncoder"]


def test_install_disabled_keeps_original_forward():
    model = _Model()
    tower = model.audio_tower
    original_forward = tower.forward
    install_encoder_bucketing(
        model,
        {"audio": tower},
        _Spec(),
        model_type="qwen2_audio",
        additional_config=None,
    )
    assert not model._encoder_bucketing.active
    assert tower.forward == original_forward  # no shim when the feature is off


def test_install_unverified_model_type_is_blocked():
    # The whitelist restricts usage: a model not on it never buckets, even with
    # the config on -- it may still need adaptation before the crop is lossless.
    model = _Model()
    tower = model.audio_tower
    original_forward = tower.forward
    install_encoder_bucketing(
        model,
        {"audio": tower},
        _Spec(),
        model_type="some_unadapted_model",
        additional_config={"audio_bucket_seconds": [1, 2]},
    )
    assert not model._encoder_bucketing.active
    assert tower.forward == original_forward  # no shim, no bucketing


def test_qwen2_audio_inputs_schema_contract():
    # Regression: bucketing crops the mel window to a rung, which the fixed
    # 3000-frame input schema would reject. The model skips the fixed-size
    # check only while bucketing is active (validate=False), and keeps the
    # original 3000 check when bucketing is off.
    from vllm.model_executor.models.qwen2_audio import Qwen2AudioFeatureInputs

    feats = torch.randn(2, 128, 800)
    mask = torch.ones(2, 800, dtype=torch.int32)
    # bucketing off (default validate=True): non-3000 window is rejected
    try:
        Qwen2AudioFeatureInputs(
            type="audio_features",
            input_features=feats,
            feature_attention_mask=mask,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-3000 window must be rejected when bucketing is off")
    # bucketing on: the model passes validate=False, accepting the rung window
    Qwen2AudioFeatureInputs(
        type="audio_features",
        input_features=feats,
        feature_attention_mask=mask,
        validate=False,
    )


def test_rungs_from_seconds_aligns_to_multiple():
    assert _rungs_from_seconds((2.0, 4.0), 10, align_to=8) == (24, 40)
    assert _rungs_from_seconds((2.0,), 10) == (20,)  # align_to=1 disables it


def test_whitelist_carries_lossless_level():
    from vllm.model_executor.models.mm_encoder_bucketing import (
        LOSSLESS_BIT_EXACT,
        _VERIFIED_FOR_BUCKETING,
    )

    assert _VERIFIED_FOR_BUCKETING["qwen2_audio"] == LOSSLESS_BIT_EXACT


def test_bucket_accepts_list_shaped_crop():
    # Canary-style: features are per-item lists, so crop selects rows and
    # truncates each row's own leading time axis rather than indexing a tensor.
    class ListSpec(EncoderBucketingSpec):
        frames_per_second = 100
        context_margin = 0

        def item_lengths(self, mm_kwargs_batch):
            mask = mm_kwargs_batch.get("feature_attention_mask")
            return [int(m.sum()) for m in mask] if mask is not None else None

        def padded_time(self, mm_kwargs_batch):
            return max(int(m.numel()) for m in mm_kwargs_batch["feature_attention_mask"])

        def crop(self, mm_kwargs_batch, indices, rung):
            feats = mm_kwargs_batch["input_features"]
            mask = mm_kwargs_batch["feature_attention_mask"]
            return {
                "input_features": [feats[i][:, :rung] for i in indices],
                "feature_attention_mask": [mask[i][:rung] for i in indices],
            }

    # three items padded to 40 frames; true lengths 5 / 30 / 12
    feats = [torch.randn(4, 40) for _ in range(3)]
    mask = [
        torch.cat([torch.ones(5), torch.zeros(35)]),
        torch.cat([torch.ones(30), torch.zeros(10)]),
        torch.cat([torch.ones(12), torch.zeros(28)]),
    ]
    batch = {"input_features": feats, "feature_attention_mask": mask}
    bucketer = EncoderBucketing(ListSpec(), LengthBucketingConfig((8, 16, 32)))
    rungs = bucketer.bucket(batch, 3)
    got = {r[1]["input_features"][0].shape[1]: r[0] for r in rungs}
    assert got == {8: [0], 32: [1], 16: [2]}


def test_canary_spec_buckets_list_shaped_features():
    from vllm.model_executor.models.canary import CanaryBucketingSpec

    feats = [torch.randn(5, 80), torch.randn(30, 80), torch.randn(12, 80)]
    lengths = torch.tensor([5, 30, 12])
    batch = {"input_features": feats, "feature_lengths": lengths}
    bucketer = EncoderBucketing(
        CanaryBucketingSpec(), LengthBucketingConfig((8, 16, 32))
    )
    rungs = bucketer.bucket(batch, 3)

    # each item lands in its own rung; crop selects the row and keeps each list
    # feature at its true length (list features are padded later in
    # embed_multimodal, not in crop)
    by_len = {}
    all_lengths = []
    for indices, kw in rungs:
        assert isinstance(kw["input_features"], list)
        assert len(indices) == 1
        by_len[indices[0]] = kw["input_features"][0].shape[0]
        all_lengths.append(kw["feature_lengths"].item())
    assert by_len == {0: 5, 1: 30, 2: 12}
    assert sorted(all_lengths) == [5, 12, 30]


def test_firered_spec_buckets_tensor_shaped_features():
    from vllm.model_executor.models.fireredasr2 import FireRedASR2BucketingSpec

    feats = torch.randn(3, 3000, 80)  # [B, T, n_mel], time on dim 1
    speech_lengths = torch.tensor([500, 2000, 1200])
    fake_token_lengths = torch.tensor([124, 499, 299])
    batch = {
        "input_features": feats,
        "speech_lengths": speech_lengths,
        "fake_token_lengths": fake_token_lengths,
    }
    bucketer = EncoderBucketing(
        FireRedASR2BucketingSpec(), LengthBucketingConfig((800, 1600, 3000))
    )
    rungs = bucketer.bucket(batch, 3)
    # windows (margin 7): 500->800, 2000->3000, 1200->1600
    got = {r[1]["input_features"].shape[1]: r[0] for r in rungs}
    assert got == {800: [0], 3000: [1], 1600: [2]}
    # crop selects rows and keeps the length tensors aligned
    for indices, kw in rungs:
        assert kw["speech_lengths"].shape == (len(indices),)
        assert kw["fake_token_lengths"].shape == (len(indices),)


def test_cohere_spec_buckets_on_dim2_time_axis():
    from vllm.model_executor.models.cohere_asr import CohereASRBucketingSpec

    feats = torch.randn(3, 80, 3000)  # [B, mel, T], time on dim 2
    length = torch.tensor([500, 2000, 1200])
    batch = {"input_features": feats, "length": length}
    bucketer = EncoderBucketing(
        CohereASRBucketingSpec(), LengthBucketingConfig((800, 1600, 3000))
    )
    rungs = bucketer.bucket(batch, 3)
    # windows (margin 8): 500->800, 2000->3000, 1200->1600
    got = {r[1]["input_features"].shape[-1]: r[0] for r in rungs}
    assert got == {800: [0], 3000: [1], 1600: [2]}
    for indices, kw in rungs:
        assert kw["length"].shape == (len(indices),)
