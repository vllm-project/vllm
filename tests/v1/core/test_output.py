# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import gc

import torch

from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import strip_covered_mm_data
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    NewRequestData,
    UncomputedPromptEmbeds,
    strip_computed_prompt_embeds,
)
from vllm.v1.request import Request


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
        prefill_token_ids=[],
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


def test_unpickled_long_lists_are_untracked() -> None:
    """Workers receive these by pickle (deepcopy takes the same __setstate__
    path); long lists come back hidden from the cyclic GC, short ones (the
    common per-step case) are left alone."""
    long, short = list(range(1000)), [1, 2]
    new = _create_new_requests_data(None)
    new.block_ids = (long, short)
    new = copy.deepcopy(new)
    assert not gc.is_tracked(new.block_ids[0]) and gc.is_tracked(new.block_ids[1])
    assert new.block_ids[0] == long


def _embeds_request(
    prompt_embeds: torch.Tensor,
    num_computed_tokens: int,
    slice_prompt_embeds: bool = True,
) -> NewRequestData:
    request = Request(
        request_id="test_req",
        prompt_token_ids=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        prompt_embeds=prompt_embeds,
    )
    request.num_computed_tokens = num_computed_tokens
    return NewRequestData.from_request(
        request,
        ([],),
        request.prefill_token_ids,
        slice_prompt_embeds=slice_prompt_embeds,
    )


def test_strip_computed_prompt_embeds() -> None:
    """Workers never read prompt-embeds rows below num_computed_tokens (prefix
    cache hits), so they are dropped; mixed prompts drop the mask alike."""
    prompt_embeds = torch.randn(6, 4)
    is_token_ids = [True, False, False, True, False, False]

    uncomputed = strip_computed_prompt_embeds(prompt_embeds, None, 4)
    assert uncomputed.offset == 4 and uncomputed.is_token_ids is None
    torch.testing.assert_close(uncomputed.embeds, prompt_embeds[4:])

    uncomputed = strip_computed_prompt_embeds(prompt_embeds, is_token_ids, 3)
    assert uncomputed.offset == 3
    assert uncomputed.is_token_ids == is_token_ids[3:]
    torch.testing.assert_close(uncomputed.embeds, prompt_embeds[3:])

    uncomputed = strip_computed_prompt_embeds(prompt_embeds, is_token_ids, 0)
    assert uncomputed == UncomputedPromptEmbeds(prompt_embeds, is_token_ids, 0)


def test_strip_computed_prompt_embeds_clamps_offset() -> None:
    """A resumed request may have computed past its prompt."""
    uncomputed = strip_computed_prompt_embeds(torch.randn(6, 4), None, 9)

    assert uncomputed.offset == 6
    assert uncomputed.embeds is not None and len(uncomputed.embeds) == 0


def test_from_request_sends_only_uncached_prompt_embeds() -> None:
    """prompt_len still covers the whole prompt after slicing."""
    prompt_embeds = torch.randn(6, 4)
    new_req = _embeds_request(prompt_embeds, num_computed_tokens=4)

    assert new_req.prompt_embeds_offset == 4
    torch.testing.assert_close(new_req.prompt_embeds, prompt_embeds[4:])
    assert new_req.prompt_len == 6


def test_from_request_keeps_prompt_embeds_unless_sliced() -> None:
    """Schedulers whose failed KV loads may roll back computed tokens send the
    whole tensor."""
    prompt_embeds = torch.randn(6, 4)
    new_req = _embeds_request(prompt_embeds, 4, slice_prompt_embeds=False)

    assert new_req.prompt_embeds_offset == 0
    assert new_req.prompt_embeds is prompt_embeds
