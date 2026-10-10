# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin,
    unpad_and_flat_audio_features,
)
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from vllm.multimodal.inputs import MultiModalBatchedField, MultiModalFieldElem


@pytest.fixture(
    params=[
        Qwen3ASRForConditionalGeneration._parse_and_validate_audio_input,
        Qwen2_5OmniConditionalGenerationMixin._parse_and_validate_audio_input,
    ],
    ids=["qwen3_asr", "qwen_omni"],
)
def parse_audio(request):
    # The parser does not access model state; no weights or model constructor.
    return request.param


def _batch(tensors):
    field = MultiModalBatchedField()
    return field.reduce_data(
        [MultiModalFieldElem(data=t, field=field) for t in tensors],
        device=None,
        pin_memory=False,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "lengths,widths",
    [
        pytest.param([5], [8], id="singleton"),
        pytest.param([5, 5], [8, 8], id="same_length"),
        pytest.param([5, 9], [5, 9], id="ragged"),
        pytest.param([5, 9], [8, 12], id="ragged_with_padding"),
        pytest.param([5, 9], [12, 12], id="shared_padding"),
        pytest.param([2, 7, 11], [4, 9, 13], id="three_ragged_items"),
    ],
)
def test_parse_batched_audio_preserves_valid_frames(
    parse_audio, dtype, lengths, widths
):
    """Cross-request batching may produce a tensor or a ragged tensor list."""
    features = []
    valid_features = []
    masks = []
    for index, (length, width) in enumerate(zip(lengths, widths)):
        # Distinct clip, mel-bin and frame values expose ordering/axis mistakes.
        valid = (
            index * 32 + torch.arange(4).reshape(4, 1) * 4 + torch.arange(length)
        ).to(dtype)
        padded = torch.full((4, width), -100, dtype=dtype)
        padded[:, :length] = valid
        valid_features.append(valid)
        features.append(padded)
        masks.append((torch.arange(width) < length).to(torch.int64))

    snapshots = [t.clone() for t in features]
    batched = _batch(features)
    if len(set(widths)) > 1:
        assert isinstance(batched, list)
    else:
        assert isinstance(batched, torch.Tensor)
        assert batched.ndim == 3

    feature_lengths = torch.tensor(lengths)
    mask = _batch(masks)
    parsed = parse_audio(
        None,
        input_audio_features=batched,
        audio_feature_lengths=feature_lengths,
        feature_attention_mask=mask,
    )

    expected = torch.cat(valid_features, dim=1)
    actual = parsed["input_features"]
    assert actual.shape == (4, sum(lengths))
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert parsed["audio_feature_lengths"] is feature_lengths
    assert parsed["feature_attention_mask"] is mask
    for actual_input, original_input in zip(features, snapshots):
        torch.testing.assert_close(actual_input, original_input, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_parse_flat_audio_keeps_existing_tensor(parse_audio, dtype):
    flat = torch.arange(28).reshape(4, 7).to(dtype)
    parsed = parse_audio(
        None,
        input_audio_features=flat,
        audio_feature_lengths=torch.tensor([3, 4]),
        feature_attention_mask=torch.ones(2, 4, dtype=torch.int64),
    )
    assert parsed["input_features"] is flat


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_parse_ragged_audio_tuple(parse_audio, dtype):
    first = torch.full((4, 3), 1, dtype=dtype)
    second = torch.full((4, 5), 2, dtype=dtype)
    second[:, 4:] = -100
    parsed = parse_audio(
        None,
        input_audio_features=(first, second),
        audio_feature_lengths=torch.tensor([3, 4]),
        feature_attention_mask=[torch.ones(3), torch.tensor([1, 1, 1, 1, 0])],
    )
    expected = torch.tensor([[1, 1, 1, 2, 2, 2, 2]], dtype=dtype).expand(4, -1)
    torch.testing.assert_close(parsed["input_features"], expected, rtol=0, atol=0)


def test_parse_missing_audio_returns_none(parse_audio):
    assert parse_audio(None) is None


@pytest.mark.parametrize(
    "container", [list, tuple, torch.stack], ids=["list", "tuple", "tensor"]
)
def test_unpad_audio_rejects_mismatched_length_count(container):
    features = container([torch.zeros(4, 8), torch.zeros(4, 8)])
    with pytest.raises(ValueError, match="Length of audio_feature_lengths must match"):
        unpad_and_flat_audio_features(features, torch.tensor([5]))
