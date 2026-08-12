# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.v1.core.sched.output import NewRequestData, strip_covered_mm_data


def _create_new_requests_data(prompt_embeds: torch.Tensor | None) -> NewRequestData:
    return NewRequestData(
        req_id="test_req",
        prompt_token_ids=None,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=prompt_embeds,
    )


def test_repr_with_none() -> None:
    """Test repr when prompt_embeds is None."""
    new_requests_data = _create_new_requests_data(None)

    assert "prompt_embeds_shape=None" in repr(new_requests_data)
    assert "prompt_embeds_shape=None" in new_requests_data.anon_repr()


def test_repr_with_multi_element_tensor() -> None:
    """Test repr when prompt_embeds is a multi-element tensor."""
    prompt_embeds = torch.randn(10, 768)
    new_requests_data = _create_new_requests_data(prompt_embeds)

    assert "prompt_embeds_shape=torch.Size([10, 768])" in repr(new_requests_data)
    assert "prompt_embeds_shape=torch.Size([10, 768])" in new_requests_data.anon_repr()


def _mm_feature(offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=offset, length=length),
        identifier=f"hash_{offset}",
        modality="image",
    )


def test_strip_covered_mm_data() -> None:
    """Items fully inside the computed prefix lose their data; items touching
    the uncomputed region keep it; already-None data stays None."""
    from dataclasses import replace

    covered = _mm_feature(offset=0, length=100)
    boundary = _mm_feature(offset=150, length=100)  # ends exactly at 250
    uncovered = _mm_feature(offset=300, length=100)
    already_none = replace(_mm_feature(offset=100, length=50), data=None)

    stripped = strip_covered_mm_data(
        [covered, boundary, uncovered, already_none], num_computed_tokens=250
    )

    assert stripped[0].data is None  # fully covered -> stripped
    assert stripped[1].data is None  # span end == computed -> covered -> stripped
    assert stripped[2].data is not None  # extends past prefix -> kept
    assert stripped[3].data is None  # was already None
    # non-data fields are preserved
    assert stripped[0].identifier == covered.identifier
    assert stripped[0].mm_position == covered.mm_position
    # original list is not mutated
    assert covered.data is not None


def test_strip_covered_mm_data_zero_computed() -> None:
    """With no prefix hit nothing is stripped."""
    features = [_mm_feature(offset=0, length=100)]
    stripped = strip_covered_mm_data(features, num_computed_tokens=0)
    assert stripped[0].data is not None
