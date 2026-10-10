# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.worker.gpu.attn_utils import compute_mm_prefix_ranges
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

pytestmark = pytest.mark.cpu_test


def _feature(
    identifier: str,
    offset: int,
    length: int,
    *,
    modality: str = "image",
    is_embed: torch.Tensor | None = None,
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality=modality,
        identifier=identifier,
        mm_position=PlaceholderRange(
            offset=offset,
            length=length,
            is_embed=is_embed,
        ),
    )


def _state_input_batch(req_ids: list[str]) -> SimpleNamespace:
    num_reqs = len(req_ids)
    query_start_loc_np = np.arange(num_reqs + 1, dtype=np.int32)
    return SimpleNamespace(
        req_ids=req_ids,
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        query_start_loc_np=query_start_loc_np,
        query_start_loc=torch.from_numpy(query_start_loc_np),
        num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
        num_tokens=num_reqs,
        num_tokens_after_padding=num_reqs,
        seq_lens=torch.ones(num_reqs, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.ones(num_reqs, dtype=torch.int32),
        dcp_local_seq_lens=None,
        dcp_local_seq_lens_cpu_upper_bound=None,
        positions=torch.arange(num_reqs, dtype=torch.int64),
        is_prefilling_np=np.ones(num_reqs, dtype=np.bool_),
        prompt_lens=None,
        idx_mapping_np=np.arange(num_reqs),
        max_query_len=None,
        fast_prefill=None,
    )


def _default_state(
    *,
    supports_mm_inputs: bool = True,
    encoder_cache: EncoderCache | None = None,
    is_mm_prefix_lm: bool = True,
    sliding_window: int | None = None,
) -> DefaultModelState:
    state = DefaultModelState.__new__(DefaultModelState)
    state.supports_mm_inputs = supports_mm_inputs
    state.encoder_cache = encoder_cache
    state.max_model_len = 4096
    state.model_config = SimpleNamespace(
        is_mm_prefix_lm=is_mm_prefix_lm,
        get_sliding_window=lambda: sliding_window,
    )
    return state


def test_compute_mm_prefix_ranges_preserves_range_semantics_and_masks():
    all_true = torch.tensor([True, True, True])
    all_false = torch.tensor([False, False, False])
    disjoint = torch.tensor([True, False, True, True, False, True])
    empty = torch.empty(0, dtype=torch.bool)
    original_masks = [
        all_true.clone(),
        all_false.clone(),
        disjoint.clone(),
        empty.clone(),
    ]

    mm_features = {
        "req0": [
            _feature("image-full", 0, 3),
            _feature("video-all", 5, 3, modality="video", is_embed=all_true),
            _feature("image-none", 10, 3, is_embed=all_false),
            _feature("image-disjoint", 20, 6, is_embed=disjoint),
            _feature("audio-ignored", 30, 4, modality="audio"),
            _feature("image-empty-mask", 40, 0, is_embed=empty),
            _feature("image-zero-len", 50, 0),
        ],
        "empty": [],
        "audio-only": [_feature("audio", 60, 2, modality="audio")],
    }

    ranges = compute_mm_prefix_ranges(
        ["missing", "req0", "empty", "audio-only"], mm_features
    )

    assert ranges == {
        0: [],
        1: [(0, 2), (5, 7), (20, 20), (22, 23), (25, 25), (50, 49)],
        2: [],
        3: [],
    }
    for mask, expected in zip([all_true, all_false, disjoint, empty], original_masks):
        assert torch.equal(mask, expected)


def test_compute_mm_prefix_ranges_sliding_window_cutoff_is_inclusive():
    mm_features = {
        "req0": [
            _feature("within", 0, 3),
            _feature("too-long", 10, 4),
            _feature("zero-len", 20, 0),
        ]
    }

    assert compute_mm_prefix_ranges(["req0"], mm_features, sliding_window=3) == {
        0: [(0, 2), (20, 19)]
    }


def test_compute_mm_prefix_ranges_cache_reuses_lists_and_fresh_batch_dict(
    monkeypatch,
):
    original = PlaceholderRange.extract_embeds_range
    extract_calls = 0

    def counted_extract(self: PlaceholderRange):
        nonlocal extract_calls
        extract_calls += 1
        return original(self)

    monkeypatch.setattr(PlaceholderRange, "extract_embeds_range", counted_extract)

    mm_features = {
        "image": [_feature("image", 0, 2)],
        "empty": [],
        "audio": [_feature("audio", 3, 1, modality="audio")],
    }
    cache: dict[int | None, dict[str, list[tuple[int, int]]]] = {}

    first = compute_mm_prefix_ranges(
        ["image", "empty", "missing", "audio"],
        mm_features,
        mm_prefix_ranges_cache=cache,
    )
    second = compute_mm_prefix_ranges(
        ["audio", "missing", "empty", "image"],
        mm_features,
        mm_prefix_ranges_cache=cache,
    )

    assert first == {0: [(0, 1)], 1: [], 2: [], 3: []}
    assert second == {0: [], 1: [], 2: [], 3: [(0, 1)]}
    assert first is not second
    assert first[0] is second[3]
    assert first[1] is second[2]
    assert first[3] is second[0]
    assert first[2] is not second[1]
    assert "missing" not in cache[None]
    assert extract_calls == 1


def test_encoder_cache_invalidates_prefix_range_memos_without_mutating_old_metadata():
    mask = torch.tensor([True, True, True])
    feature = _feature("image", 0, 3, is_embed=mask)
    cache = EncoderCache()

    cache.add_request("req0", [feature])
    first = compute_mm_prefix_ranges(
        ["req0"],
        cache.mm_features,
        mm_prefix_ranges_cache=cache.mm_prefix_ranges_cache,
    )
    assert first == {0: [(0, 2)]}
    first_ranges = first[0]

    mask[1] = False
    cache.remove_request("req0")
    cache.add_request("req0", [feature])
    second = compute_mm_prefix_ranges(
        ["req0"],
        cache.mm_features,
        mm_prefix_ranges_cache=cache.mm_prefix_ranges_cache,
    )

    assert first == {0: [(0, 2)]}
    assert second == {0: [(0, 0), (2, 2)]}
    assert first_ranges is not second[0]


def test_encoder_cache_replacement_invalidates_all_sliding_window_contexts():
    cache = EncoderCache()
    cache.add_request("req0", [_feature("long", 0, 4)])

    full = compute_mm_prefix_ranges(
        ["req0"],
        cache.mm_features,
        sliding_window=None,
        mm_prefix_ranges_cache=cache.mm_prefix_ranges_cache,
    )
    windowed = compute_mm_prefix_ranges(
        ["req0"],
        cache.mm_features,
        sliding_window=2,
        mm_prefix_ranges_cache=cache.mm_prefix_ranges_cache,
    )
    assert full == {0: [(0, 3)]}
    assert windowed == {0: []}
    assert set(cache.mm_prefix_ranges_cache) == {None, 2}

    cache.add_request("req0", [_feature("short", 10, 2)])

    assert "req0" not in cache.mm_prefix_ranges_cache[None]
    assert "req0" not in cache.mm_prefix_ranges_cache[2]
    assert compute_mm_prefix_ranges(
        ["req0"],
        cache.mm_features,
        sliding_window=None,
        mm_prefix_ranges_cache=cache.mm_prefix_ranges_cache,
    ) == {0: [(10, 11)]}
    assert compute_mm_prefix_ranges(
        ["req0"],
        cache.mm_features,
        sliding_window=2,
        mm_prefix_ranges_cache=cache.mm_prefix_ranges_cache,
    ) == {0: [(10, 11)]}


def test_default_model_state_prepare_attn_passes_encoder_cache_memo():
    encoder_cache = EncoderCache()
    encoder_cache.add_request("req0", [_feature("image", 0, 2)])
    state = _default_state(
        encoder_cache=encoder_cache,
        is_mm_prefix_lm=True,
        sliding_window=2,
    )

    with patch(
        "vllm.v1.worker.gpu.model_states.default.build_attn_metadata",
        return_value={"metadata": object()},
    ) as build_attn_metadata:
        out = state.prepare_attn(
            _state_input_batch(["req0"]),
            CUDAGraphMode.NONE,
            block_tables=(),
            slot_mappings=torch.empty(0, dtype=torch.int64),
            attn_groups=[],
            kv_cache_config=SimpleNamespace(),
        )

    assert "metadata" in out
    build_kwargs = build_attn_metadata.call_args.kwargs
    assert build_kwargs["mm_req_doc_ranges"] == {0: [(0, 1)]}
    assert (
        build_kwargs["mm_req_doc_ranges"][0]
        is (encoder_cache.mm_prefix_ranges_cache[2]["req0"])
    )


@pytest.mark.parametrize(
    ("supports_mm_inputs", "has_encoder_cache", "is_mm_prefix_lm"),
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_default_model_state_prepare_attn_gates_prefix_range_cache(
    monkeypatch,
    supports_mm_inputs: bool,
    has_encoder_cache: bool,
    is_mm_prefix_lm: bool,
):
    original = PlaceholderRange.extract_embeds_range
    extract_calls = 0

    def counted_extract(self: PlaceholderRange):
        nonlocal extract_calls
        extract_calls += 1
        return original(self)

    monkeypatch.setattr(PlaceholderRange, "extract_embeds_range", counted_extract)

    encoder_cache = EncoderCache() if has_encoder_cache else None
    if encoder_cache is not None:
        encoder_cache.add_request("req0", [_feature("image", 0, 2)])
    state = _default_state(
        supports_mm_inputs=supports_mm_inputs,
        encoder_cache=encoder_cache,
        is_mm_prefix_lm=is_mm_prefix_lm,
        sliding_window=2,
    )

    with patch(
        "vllm.v1.worker.gpu.model_states.default.build_attn_metadata",
        return_value={"metadata": object()},
    ) as build_attn_metadata:
        state.prepare_attn(
            _state_input_batch(["req0"]),
            CUDAGraphMode.NONE,
            block_tables=(),
            slot_mappings=torch.empty(0, dtype=torch.int64),
            attn_groups=[],
            kv_cache_config=SimpleNamespace(),
        )

    assert build_attn_metadata.call_args.kwargs["mm_req_doc_ranges"] is None
    assert extract_calls == 0
    if encoder_cache is not None:
        assert encoder_cache.mm_prefix_ranges_cache == {}
