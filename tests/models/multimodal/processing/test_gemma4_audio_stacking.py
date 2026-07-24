# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for Gemma 4 multi-clip audio input stacking.

A chat prompt carrying multiple audio clips of differing MEL-frame length used
to crash in ``_process_audio_input`` with
``AttributeError: 'list' object has no attribute 'squeeze'``: the
``input_features_padded`` field is ``MultiModalFieldConfig.batched("audio")``,
whose re-pad is a no-op when the per-clip frame counts differ, so it arrives as
a list of per-clip tensors rather than one stacked tensor. ``.squeeze(1)`` then
fails on the list. ``stack_audio_input_features`` pads each clip to the batch
maximum and stacks. These tests are pure-tensor and run on CPU.
"""
import torch

from vllm.model_executor.models.gemma4_mm import stack_audio_input_features

_F = 80  # MEL bins


def test_list_of_3d_clips_pads_and_stacks():
    # Two clips of different frame counts, each [1, s, f] (the multi-audio case).
    feats = [torch.randn(1, 50, _F), torch.randn(1, 73, _F)]
    masks = [
        torch.ones(1, 50, dtype=torch.bool),
        torch.ones(1, 73, dtype=torch.bool),
    ]

    out_feats, out_masks = stack_audio_input_features(feats, masks)

    assert out_feats.shape == (2, 73, _F)  # padded to batch-max, stacked
    assert out_masks.shape == (2, 73)
    # Original frames preserved; pad region zeroed and masked out.
    torch.testing.assert_close(out_feats[0, :50], feats[0].squeeze(0))
    torch.testing.assert_close(out_feats[1], feats[1].squeeze(0))
    assert bool(out_masks[0, :50].all()) and not bool(out_masks[0, 50:].any())
    assert bool(out_masks[1].all())
    assert torch.count_nonzero(out_feats[0, 50:]) == 0


def test_list_of_2d_clips_also_stacks():
    feats = [torch.randn(40, _F), torch.randn(64, _F)]
    masks = [torch.ones(40, dtype=torch.bool), torch.ones(64, dtype=torch.bool)]

    out_feats, out_masks = stack_audio_input_features(feats, masks)

    assert out_feats.shape == (2, 64, _F)
    assert out_masks.shape == (2, 64)
    assert not bool(out_masks[0, 40:].any())


def test_single_batched_tensor_fast_path():
    # The original single-clip path: [bn, 1, s, f] -> squeeze(1) -> [bn, s, f].
    feats = torch.randn(3, 1, 55, _F)
    masks = torch.ones(3, 1, 55, dtype=torch.bool)

    out_feats, out_masks = stack_audio_input_features(feats, masks)

    assert out_feats.shape == (3, 55, _F)
    assert out_masks.shape == (3, 55)
