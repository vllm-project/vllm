# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the audio batching helper shared by Gemma3n and Gemma4.

Clips of differing duration reach `_process_audio_input` as a list rather than
a stacked tensor. Treating that list as a tensor used to abort the forward pass
and kill EngineCore, so these tests pin the list handling down.
"""

import torch

from vllm.model_executor.models.audio_utils import batch_audio_features


def test_ragged_clips_are_padded_to_batch_max():
    short = torch.randn(3, 4)
    long = torch.randn(5, 4)
    masks = [torch.ones(3, dtype=torch.bool), torch.ones(5, dtype=torch.bool)]

    features, mask = batch_audio_features([short, long], masks)

    assert features.shape == (2, 5, 4)
    assert mask.shape == (2, 5)
    torch.testing.assert_close(features[0, :3], short)
    torch.testing.assert_close(features[1], long)
    # The frames that only exist to square off the batch carry no signal and
    # are masked out, so the audio tower never attends to them.
    assert torch.all(features[0, 3:] == 0)
    assert mask[0].tolist() == [True, True, True, False, False]
    assert mask[1].all()


def test_padding_preserves_masked_out_frames_inside_a_clip():
    features = [torch.randn(4, 2), torch.randn(2, 2)]
    masks = [
        torch.tensor([True, True, False, False]),
        torch.tensor([True, False]),
    ]

    _, mask = batch_audio_features(features, masks)

    assert mask[0].tolist() == [True, True, False, False]
    assert mask[1].tolist() == [True, False, False, False]


def test_equal_length_clips_are_left_alone():
    features = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)

    batched_features, batched_mask = batch_audio_features(features, mask)

    torch.testing.assert_close(batched_features, features)
    torch.testing.assert_close(batched_mask, mask)


def test_padding_keeps_feature_dtype():
    features = [
        torch.randn(3, 4, dtype=torch.bfloat16),
        torch.randn(5, 4, dtype=torch.bfloat16),
    ]
    masks = [torch.ones(3, dtype=torch.bool), torch.ones(5, dtype=torch.bool)]

    batched_features, batched_mask = batch_audio_features(features, masks)

    assert batched_features.dtype == torch.bfloat16
    assert batched_mask.dtype == torch.bool
