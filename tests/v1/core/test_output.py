# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import strip_covered_mm_data
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


def test_strip_covered_mm_data_reuses_stripped_objects(monkeypatch) -> None:
    """Repeated stripping must not re-copy already-covered features.

    Realtime sessions re-run this every engine step as features accumulate.
    Memoized stripped copies keep dataclasses.replace O(new features).
    """
    from dataclasses import replace as real_replace

    replace_calls = {"n": 0}

    def counting_replace(*args, **kwargs):
        replace_calls["n"] += 1
        return real_replace(*args, **kwargs)

    monkeypatch.setattr("vllm.multimodal.utils.replace", counting_replace)

    features = [_mm_feature(offset=i * 10, length=10) for i in range(100)]

    first = strip_covered_mm_data(features, num_computed_tokens=500)
    assert replace_calls["n"] == 50
    assert first[0] is not features[0]
    assert first[0].data is None
    assert first[50] is features[50]

    second = strip_covered_mm_data(features, num_computed_tokens=500)
    assert replace_calls["n"] == 50
    assert second[0] is first[0]
    assert second[49] is first[49]
    assert second[50] is features[50]


def test_strip_covered_mm_data_prefix_cache_is_o_new(monkeypatch) -> None:
    """With stripped_prefix, later steps only inspect newly covered features."""
    from dataclasses import replace as real_replace

    from vllm.multimodal.utils import _is_covered_mm_feature

    replace_calls = {"n": 0}
    cover_checks = {"n": 0}

    def counting_replace(*args, **kwargs):
        replace_calls["n"] += 1
        return real_replace(*args, **kwargs)

    def counting_is_covered(f, num_computed_tokens):
        cover_checks["n"] += 1
        return _is_covered_mm_feature(f, num_computed_tokens)

    monkeypatch.setattr("vllm.multimodal.utils.replace", counting_replace)
    monkeypatch.setattr(
        "vllm.multimodal.utils._is_covered_mm_feature", counting_is_covered
    )

    features = [_mm_feature(offset=i * 10, length=10) for i in range(200)]
    prefix: list[MultiModalFeatureSpec] = []

    first = strip_covered_mm_data(
        features, num_computed_tokens=500, stripped_prefix=prefix
    )
    assert replace_calls["n"] == 50
    # 50 covered + the first uncovered that stops the scan.
    assert cover_checks["n"] == 51
    assert len(prefix) == 50

    cover_checks["n"] = 0
    second = strip_covered_mm_data(
        features, num_computed_tokens=500, stripped_prefix=prefix
    )
    assert replace_calls["n"] == 50
    # Only the first still-uncovered feature is examined.
    assert cover_checks["n"] == 1
    assert all(a is b for a, b in zip(first, second, strict=True))

    cover_checks["n"] = 0
    third = strip_covered_mm_data(
        features, num_computed_tokens=530, stripped_prefix=prefix
    )
    assert replace_calls["n"] == 53
    # Three newly covered + the next uncovered.
    assert cover_checks["n"] == 4
    assert third[0] is first[0]
    assert third[52].data is None
    assert third[53] is features[53]
    assert features[52].data is not None


def test_strip_covered_mm_data_prefix_cache_resets_on_preempt() -> None:
    """num_computed_tokens == 0 (preempt) must drop the prefix cache so a
    later partial prefix-cache hit can re-strip from the originals."""
    features = [_mm_feature(offset=i * 10, length=10) for i in range(20)]
    prefix: list[MultiModalFeatureSpec] = []

    stripped = strip_covered_mm_data(
        features, num_computed_tokens=150, stripped_prefix=prefix
    )
    assert len(prefix) == 15
    assert stripped[0].data is None

    unstripped = strip_covered_mm_data(
        features, num_computed_tokens=0, stripped_prefix=prefix
    )
    assert prefix == []
    assert unstripped is features
    assert features[0].data is not None

    half = strip_covered_mm_data(
        features, num_computed_tokens=80, stripped_prefix=prefix
    )
    assert len(prefix) == 8
    assert half[7].data is None
    assert half[8] is features[8]


def test_new_request_data_from_request_reuses_stripped_mm_features() -> None:
    """NewRequestData.from_request is the realtime hot path: the same Request
    is converted every step and must reuse already-stripped mm features."""
    from types import SimpleNamespace

    features = [_mm_feature(offset=i * 10, length=10) for i in range(40)]
    request = SimpleNamespace(
        request_id="rt",
        prompt_token_ids=list(range(500)),
        sampling_params=None,
        pooling_params=None,
        mm_features=features,
        num_computed_tokens=250,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        replay_start=0,
        _mm_stripped_prefix=[],
    )

    first = NewRequestData.from_request(request, block_ids=([],))
    second = NewRequestData.from_request(request, block_ids=([],))

    assert len(first.mm_features) == 40
    assert first.mm_features[0].data is None
    assert first.mm_features[24].data is None
    assert first.mm_features[25] is features[25]
    assert second.mm_features[0] is first.mm_features[0]
    assert second.mm_features[24] is first.mm_features[24]
    assert features[0].data is not None
    assert len(request._mm_stripped_prefix) == 25
