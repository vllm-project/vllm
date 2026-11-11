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
import math
from contextlib import contextmanager

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


@contextmanager
def _gpu_worker():
    """Force the GPU-only install gate open so tests are host-independent.

    Bucketing is GPU-only (the bucketer's sole consumer is the GPU encoder
    runner); the gate is tested separately by `test_install_gpu_only_gate_blocks_non_gpu_worker`.
    """
    from vllm.model_executor.models import mm_encoder_bucketing as meb

    original = meb._gpu_execution_supported
    meb._gpu_execution_supported = lambda: True
    try:
        yield
    finally:
        meb._gpu_execution_supported = original


def _rungs_via_runner(model, batch, num_items):
    """Split a batch through the runner's duck-typed hook.

    This is the observable behavior of "bucketing engages" -- exactly what the
    GPU encoder runner does at runtime -- so tests assert on it instead of the
    bucketer's private state.
    """
    from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

    class _StubRunner:
        def __init__(self):
            self.model = model

    return EncoderRunner._bucket_mm_batch(_StubRunner(), batch, num_items)


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
    assert ladder.padded_length(3000) == 3000  # the largest rung still fits
    # Nothing fits: signal it instead of silently clamping to the largest rung
    # (a clamp would truncate content; bucket() falls back to the full batch).
    assert ladder.padded_length(9999) is None


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
        with _gpu_worker():
            install_encoder_bucketing(
                model,
                {"audio": tower},
                _Spec(),
                model_type="qwen2_audio",
                additional_config={"audio_bucket_seconds": [1, 2]},
            )
        assert isinstance(tower.forward, functools.partial)  # variable-length shim
        # Observable through the runner's duck-typed hook: the batch splits.
        rungs = _rungs_via_runner(model, _batch([50, 200], 250), 2)
        assert len(rungs) == 2  # bucketing engages

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
    batch = _batch([50, 200], 250)
    rungs = _rungs_via_runner(model, batch, 2)
    assert len(rungs) == 1 and rungs[0][1] is batch  # no bucketing engages
    assert tower.forward == original_forward  # no shim when the feature is off


def test_install_unverified_model_type_is_blocked():
    # The whitelist restricts usage: a model not on it never buckets, even with
    # the config on -- it may still need adaptation before the crop is lossless.
    model = _Model()
    tower = model.audio_tower
    original_forward = tower.forward
    with _gpu_worker():
        install_encoder_bucketing(
            model,
            {"audio": tower},
            _Spec(),
            model_type="some_unadapted_model",
            additional_config={"audio_bucket_seconds": [1, 2]},
        )
    batch = _batch([50, 200], 250)
    rungs = _rungs_via_runner(model, batch, 2)
    assert len(rungs) == 1 and rungs[0][1] is batch  # no bucketing engages
    assert tower.forward == original_forward  # no shim, no bucketing


def test_install_gpu_only_gate_blocks_non_gpu_worker():
    # GPU-only: the bucketer's sole consumer is the GPU encoder runner, so on
    # any other execution path install must leave the model completely
    # untouched -- no shim, no active bucketer -- instead of installing a shim
    # nothing consumes.
    from vllm.model_executor.models import mm_encoder_bucketing as meb

    model = _Model()
    tower = model.audio_tower
    original_forward = tower.forward
    original = meb._gpu_execution_supported
    meb._gpu_execution_supported = lambda: False
    try:
        install_encoder_bucketing(
            model,
            {"audio": tower},
            _Spec(),
            model_type="qwen2_audio",
            additional_config={"audio_bucket_seconds": [1, 2]},
        )
    finally:
        meb._gpu_execution_supported = original

    batch = _batch([50, 200], 250)
    rungs = _rungs_via_runner(model, batch, 2)
    assert len(rungs) == 1 and rungs[0][1] is batch  # no bucketing engages
    assert tower.forward == original_forward  # no shim outside GPU workers


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
        LOSSLESS_OUTPUT_LE_1FRAME,
        _VERIFIED_FOR_BUCKETING,
    )

    assert _VERIFIED_FOR_BUCKETING["qwen2_audio"] == LOSSLESS_BIT_EXACT
    # GLM-ASR's attention is mask-less: output frame counts are exact but
    # values drift, so it is whitelisted at the output-counts-only level.
    assert _VERIFIED_FOR_BUCKETING["glmasr"] == LOSSLESS_OUTPUT_LE_1FRAME


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


def test_granite_spec_buckets_tensor_features():
    from vllm.model_executor.models.granite_speech import (
        GraniteSpeechBucketingSpec,
    )

    # [B, T, 160] with time on dim -2; 50 fps encoder frames
    feats = torch.randn(3, 300, 160)
    sizes = torch.tensor([21, 66, 30])  # projector tokens
    batch = {"input_features": feats, "audio_embed_sizes": sizes}
    bucketer = EncoderBucketing(
        GraniteSpeechBucketingSpec(), LengthBucketingConfig((100, 200, 400))
    )
    rungs = bucketer.bucket(batch, 3)

    # item lengths are the feature time extents (300 each -> no shrink past
    # the window for any of them once margin is added), so grouping is driven
    # by the rung ladder: 300 + 7 -> 400 for all three.
    assert len(rungs) == 1
    assert rungs[0][0] == [0, 1, 2]
    assert rungs[0][1]["input_features"].shape[-2] == 300

    # crop truncates the time axis and keeps sizes aligned
    cropped = rungs[0][1]
    assert cropped["audio_embed_sizes"].tolist() == [21, 66, 30]


def test_granite_spec_crop_truncates_time_axis():
    from vllm.model_executor.models.granite_speech import (
        GraniteSpeechBucketingSpec,
    )

    feats = [torch.randn(250, 160), torch.randn(80, 160)]
    sizes = [torch.tensor(50), torch.tensor(15)]
    batch = {"input_features": feats, "audio_embed_sizes": sizes}
    spec = GraniteSpeechBucketingSpec()
    assert spec.item_lengths(batch) == [250, 80]
    assert spec.padded_time(batch) == 250

    cropped = spec.crop(batch, [1], 100)
    assert isinstance(cropped["input_features"], list)
    assert cropped["input_features"][0].shape == (80, 160)  # never clips
    assert cropped["audio_embed_sizes"] == [torch.tensor(15)]

    cropped = spec.crop(batch, [0], 100)
    assert cropped["input_features"][0].shape == (100, 160)  # truncates


def test_granite_spec_lengths_from_embed_sizes():
    from vllm.model_executor.models.granite_speech import (
        GraniteSpeechBucketingSpec,
    )

    # tokens = ceil(T/15) * 3 -> upper bound tokens // 3 * 15 covers T
    spec = GraniteSpeechBucketingSpec()
    for t_enc, tokens in ((164, 33), (250, 51), (1500, 300)):
        assert math.ceil(t_enc / 15) * 3 == tokens
        upper = spec._lengths_from_embed_sizes([tokens])[0]
        assert upper >= t_enc and upper - t_enc < 15
    # fallback path: no input_features -> sizes drive the rung choice
    assert spec.item_lengths({"audio_embed_sizes": torch.tensor([33, 66])}) == [
        165,
        330,
    ]


def test_glmasr_spec_buckets_tensor_chunked_features():
    from vllm.model_executor.models.glmasr import GlmAsrBucketingSpec

    # chunked mel input: rows are chunks, chunk_counts maps chunks -> audio
    # audio 0: chunks [1200, 300] frames, audio 1: chunk [2000] frames
    feats = torch.randn(3, 4, 3000)  # [chunks, nmb, chunk_length], time last
    mask = torch.zeros(3, 3000, dtype=torch.long)
    mask[0, :1200] = 1
    mask[1, :300] = 1
    mask[2, :2000] = 1
    counts = torch.tensor([2, 1])
    batch = {
        "input_features": feats,
        "feature_attention_mask": mask,
        "chunk_counts": counts,
    }
    bucketer = EncoderBucketing(
        GlmAsrBucketingSpec(), LengthBucketingConfig((800, 1600, 3000))
    )
    rungs = bucketer.bucket(batch, 2)

    # item length is the audio's longest chunk; windows (margin 4):
    # 1200 -> 1600, 2000 -> 3000
    got = {tuple(r[0]): r[1]["input_features"].shape[-1] for r in rungs}
    assert got == {(0,): 1600, (1,): 3000}
    for indices, kw in rungs:
        # crop expands audio indices to their chunk rows and subsets counts
        expected_rows = sum(counts[i].item() for i in indices)
        assert kw["input_features"].shape == (expected_rows, 4, kw["input_features"].shape[-1])
        assert kw["feature_attention_mask"].shape == (
            expected_rows,
            kw["input_features"].shape[-1],
        )
        assert kw["chunk_counts"].tolist() == [counts[i].item() for i in indices]
    # audio 0's two chunks land together in the 1600 rung
    by_indices = {tuple(r[0]): r[1] for r in rungs}
    assert by_indices[(0,)]["feature_attention_mask"].sum(-1).tolist() == [1200, 300]


def test_glmasr_spec_full_window_chunk_keeps_full_batch():
    # A chunk at the full window has no safe rung once the context margin is
    # counted, so the batch runs unbucketed rather than clip content.
    from vllm.model_executor.models.glmasr import GlmAsrBucketingSpec

    feats = torch.randn(2, 4, 3000)
    mask = torch.zeros(2, 3000, dtype=torch.long)
    mask[0, :3000] = 1
    mask[1, :500] = 1
    counts = torch.tensor([1, 1])
    batch = {
        "input_features": feats,
        "feature_attention_mask": mask,
        "chunk_counts": counts,
    }
    bucketer = EncoderBucketing(
        GlmAsrBucketingSpec(), LengthBucketingConfig((800, 1600, 3000))
    )
    rungs = bucketer.bucket(batch, 2)
    assert len(rungs) == 1
    assert rungs[0][0] == [0, 1]
    assert rungs[0][1] is batch


def test_glmasr_spec_buckets_list_shaped_features():
    from vllm.model_executor.models.glmasr import GlmAsrBucketingSpec

    # list branch: one element per audio (its chunk stack), time last
    feats = [torch.randn(1, 4, 3000), torch.randn(2, 4, 3000)]
    mask = [torch.zeros(1, 3000, dtype=torch.long), torch.zeros(2, 3000, dtype=torch.long)]
    mask[0][0, :500] = 1
    mask[1][0, :1200] = 1
    mask[1][1, :300] = 1
    counts = torch.tensor([1, 2])
    batch = {
        "input_features": feats,
        "feature_attention_mask": mask,
        "chunk_counts": counts,
    }
    bucketer = EncoderBucketing(
        GlmAsrBucketingSpec(), LengthBucketingConfig((800, 1600, 3000))
    )
    rungs = bucketer.bucket(batch, 2)

    # windows (margin 4): 500 -> 800, max(1200, 300) -> 1600
    got = {r[0][0]: r[1]["input_features"][0].shape[-1] for r in rungs}
    assert got == {0: 800, 1: 1600}
    for indices, kw in rungs:
        assert isinstance(kw["input_features"], list)
        assert len(indices) == 1
        i = indices[0]
        assert kw["input_features"][0].shape == feats[i][:, :, : got[i]].shape
        assert kw["chunk_counts"].tolist() == [counts[i].item()]
        # true chunk lengths survive the crop
        assert kw["feature_attention_mask"][0].sum(-1).tolist() == (
            mask[i].sum(-1).tolist()
        )


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


def test_merge_rungs_folds_underfilled_buckets():
    # With min_bucket_size configured, under-filled adjacent rungs are folded
    # into the larger neighbor. Three items land in three rungs (8/16/32),
    # each below the min size, so they collapse into a single rung-32 batch.
    batch = _batch([5, 30, 12], time_extent=32)
    bucketer = EncoderBucketing(
        _Spec(), LengthBucketingConfig((8, 16, 32)), min_bucket_size=4
    )
    rungs = bucketer.bucket(batch, 3)
    assert len(rungs) == 1
    indices, kw = rungs[0]
    assert sorted(indices) == [0, 1, 2]
    assert kw["input_features"].shape[-1] == 32  # merged to the largest rung

    # Without min_bucket_size, the same batch stays split into three rungs.
    bucketer_no_merge = EncoderBucketing(_Spec(), LengthBucketingConfig((8, 16, 32)))
    rungs_no_merge = bucketer_no_merge.bucket(batch, 3)
    assert len(rungs_no_merge) == 3


def test_funasr_spec_buckets_tensor_features():
    from vllm.model_executor.models.funasr import FunASRBucketingSpec

    feats = torch.randn(3, 1000, 560)  # [B, T, feat], time dim 1, fixed window 1000
    speech_lengths = torch.tensor([400, 900, 600])
    fake_token_lengths = torch.tensor([100, 225, 150])
    batch = {
        "input_features": feats,
        "speech_lengths": speech_lengths,
        "fake_token_lengths": fake_token_lengths,
    }
    bucketer = EncoderBucketing(
        FunASRBucketingSpec(), LengthBucketingConfig((500, 800, 1000))
    )
    rungs = bucketer.bucket(batch, 3)
    # windows (margin 8): 400->500, 900->1000, 600->800
    got = {r[1]["input_features"].shape[1]: r[0] for r in rungs}
    assert got == {500: [0], 1000: [1], 800: [2]}
    for indices, kw in rungs:
        assert kw["speech_lengths"].shape == (len(indices),)


def test_firered_spec_buckets_list_shaped_input_features():
    from vllm.model_executor.models.fireredasr2 import FireRedASR2BucketingSpec

    feats = [torch.randn(500, 80), torch.randn(2000, 80), torch.randn(1200, 80)]
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

    # list branch selects rows and truncates each to the rung; the length
    # fields stay aligned with the selected rows
    by_len = {}
    for indices, kw in rungs:
        assert isinstance(kw["input_features"], list)
        assert len(indices) == 1
        by_len[indices[0]] = kw["input_features"][0].shape[0]
        assert kw["speech_lengths"].tolist() == [speech_lengths[indices[0]].item()]
        assert kw["fake_token_lengths"].tolist() == [
            fake_token_lengths[indices[0]].item()
        ]
    # windows (margin 7): 500->800 (keeps 500), 2000->3000 (keeps 2000),
    # 1200->1600 (keeps 1200) -- rung >= true length, so nothing is clipped
    assert by_len == {0: 500, 1: 2000, 2: 1200}


def test_funasr_spec_buckets_list_shaped_input_features():
    from vllm.model_executor.models.funasr import FunASRBucketingSpec

    feats = [torch.randn(400, 560), torch.randn(900, 560), torch.randn(600, 560)]
    speech_lengths = torch.tensor([400, 900, 600])
    fake_token_lengths = torch.tensor([100, 225, 150])
    batch = {
        "input_features": feats,
        "speech_lengths": speech_lengths,
        "fake_token_lengths": fake_token_lengths,
    }
    bucketer = EncoderBucketing(
        FunASRBucketingSpec(), LengthBucketingConfig((500, 800, 1000))
    )
    rungs = bucketer.bucket(batch, 3)

    by_len = {}
    for indices, kw in rungs:
        assert isinstance(kw["input_features"], list)
        assert len(indices) == 1
        by_len[indices[0]] = kw["input_features"][0].shape[0]
        assert kw["speech_lengths"].tolist() == [speech_lengths[indices[0]].item()]
    # windows (margin 8): 400->500, 900->1000, 600->800 -- nothing clipped
    assert by_len == {0: 400, 1: 900, 2: 600}


def test_full_window_shim_is_byte_identical_to_original_forward():
    from transformers.models.qwen2_audio import (
        Qwen2AudioEncoder,
        Qwen2AudioEncoderConfig,
    )

    from vllm.model_executor.models.qwen2_audio import _qwen2audio_varlen_shim

    # Small encoder whose full window is 8 positions * conv1.stride(1) *
    # conv2.stride(2) = 16 mel frames, so a 16-frame input is exactly the
    # fixed-length case the shim must match byte-for-byte.
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
        shimmed.last_hidden_state,
        original.last_hidden_state,
        rtol=0,
        atol=0,
    )


def test_qwen2_audio_spec_is_instantiated_and_crops():
    from vllm.model_executor.models.qwen2_audio import Qwen2AudioBucketingSpec

    spec = Qwen2AudioBucketingSpec()
    assert spec.frames_per_second == 100
    assert spec.context_margin == 4

    batch = _batch([5, 30, 12], time_extent=32)
    cropped = spec.crop(batch, [0, 2], 16)
    assert cropped["input_features"].shape == (2, 4, 16)
    assert cropped["feature_attention_mask"].shape == (2, 16)


def test_bucket_mm_batch_distinguishes_wired_from_silently_dead():
    from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

    batch = _batch([5, 30, 12], time_extent=32)

    class _FakeRunner:
        def __init__(self, model):
            self.model = model

    class _ModelWithBucketing(nn.Module):
        def __init__(self):
            super().__init__()
            self._encoder_bucketing = EncoderBucketing(
                _Spec(), LengthBucketingConfig((8, 16, 32))
            )

    class _ModelWithoutBucketing(nn.Module):
        pass

    wired = EncoderRunner._bucket_mm_batch(
        _FakeRunner(_ModelWithBucketing()), batch, 3
    )
    assert len(wired) == 3  # feature engages: three rungs

    dead = EncoderRunner._bucket_mm_batch(
        _FakeRunner(_ModelWithoutBucketing()), batch, 3
    )
    assert len(dead) == 1 and dead[0][0] == [0, 1, 2]  # single passthrough


def test_cohere_spec_crop_list_branch_pads_up_in_crop():
    from vllm.model_executor.models.cohere_asr import CohereASRBucketingSpec

    # List-shaped input_features: per-item [mel, T_i] (time on dim 1). Crop must
    # select the rows and zero-pad each up to the rung, then stack into a fixed
    # [n, mel, rung] tensor (pad-up-in-crop), instead of indexing a tensor.
    feats = [torch.randn(80, 500), torch.randn(80, 2000), torch.randn(80, 1200)]
    length = torch.tensor([500, 2000, 1200])
    batch = {"input_features": feats, "length": length}
    spec = CohereASRBucketingSpec()

    cropped = spec.crop(batch, [0, 2], 800)
    feats_out = cropped["input_features"]
    assert isinstance(feats_out, torch.Tensor)
    assert feats_out.shape == (2, 80, 800)
    # item 0 (500 < 800) is zero-padded to the rung, item 2 (1200) is cropped
    assert torch.count_nonzero(feats_out[0][:, 500:]).item() == 0
    assert torch.equal(feats_out[1], feats[2][:, :800])
    assert cropped["length"].tolist() == [500, 1200]
