# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU position replay tests with a linear position provider as protocol input."""

import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.streaming.mrope import (
    StreamingMropeState,
    make_chunk_relative_mm_features,
)

pytestmark = pytest.mark.cpu_test


class LinearPositionProvider:
    def get_mrope_input_positions(self, token_ids, mm_features):
        return torch.arange(len(token_ids)).expand(3, -1).clone(), 0


@pytest.fixture
def position_provider():
    return LinearPositionProvider()


def test_chunk_feature_offsets_are_relative_without_mutating_history():
    feature = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="frame",
        mm_position=PlaceholderRange(offset=12, length=4),
    )
    relative = make_chunk_relative_mm_features([feature], prev_num_tokens=10)
    assert relative[0].mm_position.offset == 2
    assert relative[0].mm_position.length == 4
    assert feature.mm_position.offset == 12


def test_eviction_in_decoded_output_keeps_survivor_positions(position_provider):
    state = StreamingMropeState.from_prompt(position_provider, [1] * 4, [])
    state.record_computed_tokens(12)
    state.update(position_provider, [1] * 10, [], [(6, 10)])

    assert state.positions.tolist() == [[0, 1, 2, 3, 4, 5, 10, 11, 12, 13]] * 3
    assert state.delta == 4
    assert state.computed_watermark(10) == 13


def test_ordered_evictions_use_each_compacted_index_space(position_provider):
    state = StreamingMropeState.from_prompt(position_provider, [1] * 12, [])
    state.update(position_provider, [1] * 8, [], [(2, 4), (4, 6)])

    assert state.positions[0].tolist() == [0, 1, 4, 5, 8, 9, 10, 11]
    assert state.computed_watermark(4) == 5


def test_new_positions_do_not_reuse_evicted_tail_positions(position_provider):
    state = StreamingMropeState.from_prompt(position_provider, [1] * 8, [])
    state.update(position_provider, [1] * 6, [], [(4, 8)])

    assert state.positions[0].tolist() == [0, 1, 2, 3, 8, 9]
    state.record_computed_tokens(8)
    assert state.positions[0].tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
    assert state.computed_watermark(8) == 11


def test_unreported_eviction_cannot_silently_reposition_cached_keys(position_provider):
    state = StreamingMropeState.from_prompt(position_provider, [1] * 8, [])

    with pytest.raises(ValueError, match="shrank"):
        state.update(position_provider, [1] * 4, [], [])
