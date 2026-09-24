# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
import torch

from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import (
    strip_covered_mm_data,
    strip_covered_mm_data_incremental,
)
from vllm.v1.core.sched.output import NewRequestData


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


def _mm_feature_mixed(offset: int, length: int) -> MultiModalFeatureSpec:
    data = MultiModalKwargsItem(
        {
            "pixel_values": MultiModalFieldElem(
                data=torch.empty(4), field=MultiModalBatchedField()
            ),
            "image_grid_thw": MultiModalFieldElem(
                data=torch.ones(1, 3, dtype=torch.long),
                field=MultiModalBatchedField(keep_on_cpu=True),
            ),
        }
    )
    return MultiModalFeatureSpec(
        data=data,
        mm_position=PlaceholderRange(offset=offset, length=length),
        identifier=f"hash_{offset}",
        modality="image",
    )


def test_strip_covered_mm_data_mrope() -> None:
    """For M-RoPE models, covered items keep their keep_on_cpu metadata fields
    (the worker needs them to compute positions); payload fields are dropped."""
    covered = _mm_feature_mixed(offset=0, length=100)
    uncovered = _mm_feature_mixed(offset=300, length=100)

    stripped = strip_covered_mm_data(
        [covered, uncovered], num_computed_tokens=250, uses_mrope=True
    )

    assert stripped[0].data is not None
    assert list(stripped[0].data.keys()) == ["image_grid_thw"]
    assert stripped[1].data is not None
    assert set(stripped[1].data.keys()) == {"pixel_values", "image_grid_thw"}
    # original list is not mutated
    assert set(covered.data.keys()) == {"pixel_values", "image_grid_thw"}


def test_strip_covered_mm_data_shm_address_item() -> None:
    """SHM address descriptors must survive stripping even when covered: the
    worker needs the address to resolve the payload and to acknowledge the
    sender's writer reference. Stripping it crashed the SHM receiver cache on
    the second identical request (vllm-project/vllm#54994)."""
    address_item = MultiModalKwargsItem(
        {
            "address": MultiModalFieldElem(data=4096, field=MultiModalBatchedField()),
            "monotonic_id": MultiModalFieldElem(data=7, field=MultiModalBatchedField()),
        }
    )
    feature = MultiModalFeatureSpec(
        data=address_item,
        mm_position=PlaceholderRange(offset=0, length=100),
        identifier="shm_item",
        modality="image",
    )

    stripped = strip_covered_mm_data([feature], num_computed_tokens=250)

    assert stripped[0].data is address_item


def _incremental_request(
    features: list[MultiModalFeatureSpec], num_computed_tokens: int
) -> SimpleNamespace:
    """Stand-in for Request exposing the incremental strip state."""
    return SimpleNamespace(
        mm_features=features,
        num_computed_tokens=num_computed_tokens,
        _mm_stripped_mm_features=[],
        _mm_strip_cursor=0,
    )


def test_strip_covered_incremental_matches_pure() -> None:
    """Repeated incremental calls at growing prefixes produce the same result
    as the pure strip_covered_mm_data at every step."""
    from dataclasses import replace

    covered_none = replace(_mm_feature(100, 50), data=None)
    features = [
        _mm_feature(0, 50),
        covered_none,
        _mm_feature(150, 50),
        _mm_feature(300, 50),
    ]

    for computed in range(0, 401, 25):
        req = _incremental_request(features, computed)
        incremental = strip_covered_mm_data_incremental(req)
        pure = strip_covered_mm_data(features, computed)
        assert [f.data for f in incremental] == [f.data for f in pure]
        assert len(incremental) == len(pure)


def test_strip_covered_incremental_advances_only() -> None:
    """Each call only strips features appended since the previous one; the
    stripped objects are reused instead of being rebuilt."""
    features = [_mm_feature(0, 50), _mm_feature(100, 50), _mm_feature(200, 50)]
    req = _incremental_request(features, num_computed_tokens=60)

    out1 = strip_covered_mm_data_incremental(req)
    assert req._mm_strip_cursor == 1
    assert out1[0].data is None
    assert out1[1].data is not None

    req.num_computed_tokens = 160
    out2 = strip_covered_mm_data_incremental(req)
    assert req._mm_strip_cursor == 2
    # Previously stripped object is reused, not rebuilt
    assert out2[0] is out1[0]
    assert out2[1].data is None
    assert out2[2].data is not None


def test_strip_covered_incremental_appended_features() -> None:
    """Simulates a realtime session: features are appended between calls, and
    only the newly appended ones are inspected."""
    features = [_mm_feature(0, 50)]
    req = _incremental_request(features, num_computed_tokens=50)

    out1 = strip_covered_mm_data_incremental(req)
    assert req._mm_strip_cursor == 1
    assert out1[0].data is None

    # New audio chunk arrives at offset 50.
    features.append(_mm_feature(50, 50))
    features.append(_mm_feature(100, 50))
    req.num_computed_tokens = 110

    out2 = strip_covered_mm_data_incremental(req)
    assert req._mm_strip_cursor == 2
    assert out2[0] is out1[0]
    assert out2[1].data is None
    assert out2[2].data is not None
