# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from array import array

import pytest
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


@pytest.mark.parametrize("aliased", [True, False])
def test_new_request_data_packs_token_ids(aliased: bool) -> None:
    prompt = list(range(150_000, 150_300))
    prefill = prompt if aliased else prompt + [5, 6]
    data = NewRequestData(
        req_id="test_req",
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([1, 2, 3],),
        num_computed_tokens=128,
        lora_request=None,
        prefill_token_ids=prefill,
        pack_token_ids=True,
    )
    # What pickle sends to the workers.
    state = data.__getstate__()
    # The scheduler-side object keeps its lists.
    assert data.prompt_token_ids is prompt and data.prefill_token_ids is prefill
    assert isinstance(state["prompt_token_ids"], array)
    assert isinstance(state["prefill_token_ids"], array)
    assert state["prompt_token_ids"].tolist() == prompt
    assert state["prefill_token_ids"].tolist() == prefill
    assert (state["prompt_token_ids"] is state["prefill_token_ids"]) == aliased
    assert state["block_ids"] == ([1, 2, 3],) and state["num_computed_tokens"] == 128

    data.pack_token_ids = False
    state = data.__getstate__()
    assert state["prompt_token_ids"] is prompt
    assert state["prefill_token_ids"] is prefill


def test_new_request_data_uses_prepacked_prompt() -> None:
    prompt = list(range(1000, 1300))
    prepacked = array("i", prompt)
    data = NewRequestData(
        req_id="test_req",
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([1],),
        num_computed_tokens=0,
        lora_request=None,
        prefill_token_ids=prompt,
        pack_token_ids=True,
        packed_prompt_token_ids=prepacked,
    )
    state = data.__getstate__()
    assert state["prompt_token_ids"] is prepacked
    assert state["prefill_token_ids"] is prepacked
    # The pre-packed copy itself is not sent.
    assert "packed_prompt_token_ids" not in state
    # A stale pre-packed copy (prompt extended since) is not used.
    data.packed_prompt_token_ids = array("i", prompt[:10])
    assert data.__getstate__()["prompt_token_ids"].tolist() == prompt
