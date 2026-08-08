# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accounting checks for EAGLE3 aux taps sent directly to the last PP rank.

Each stage produces only its local taps; the last rank gathers them in
pipeline order. These run on CPU with no distributed init.
"""

import pytest

from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.interfaces import EagleModelMixin

# Kimi-K3 / DSpark: 93 layers, target_layer_ids [2, 23, 47, 71, 89].
# get_eagle3_aux_layers_from_config maps target_layer_ids -> +1.
NUM_LAYERS = 93
AUX_IDS = (3, 24, 48, 72, 90)


def _taps_emitted(start_layer, end_layer, aux_ids, is_first_rank):
    return list(
        EagleModelMixin.local_aux_tap_ids(
            start_layer, end_layer, aux_ids, is_first_rank
        )
    )


def _simulate(num_layers, pp_size, aux_ids):
    """Gather local taps from every stage in rank order (direct-to-last)."""
    gathered: list[int] = []
    for rank in range(pp_size):
        start, end = get_pp_indices(num_layers, rank, pp_size)
        gathered.extend(_taps_emitted(start, end, aux_ids, rank == 0))
    return gathered


@pytest.mark.parametrize("pp_size", [1, 2, 3, 4, 6, 8])
def test_drafter_sees_all_taps_in_order(pp_size):
    assert _simulate(NUM_LAYERS, pp_size, AUX_IDS) == list(AUX_IDS)


def test_pp2_split_matches_expected_stages():
    start0, end0 = get_pp_indices(NUM_LAYERS, 0, 2)
    start1, end1 = get_pp_indices(NUM_LAYERS, 1, 2)
    assert _taps_emitted(start0, end0, AUX_IDS, True) == [3, 24]
    assert _taps_emitted(start1, end1, AUX_IDS, False) == [48, 72, 90]


@pytest.mark.parametrize("pp_size", [1, 2, 4])
def test_no_drafter_adds_no_payload_keys(pp_size):
    assert _simulate(NUM_LAYERS, pp_size, ()) == []


def test_tap_on_stage_boundary_is_not_double_counted():
    """A boundary tap is emitted once by the upstream stage that computed it."""
    _, end0 = get_pp_indices(NUM_LAYERS, 0, 2)
    aux_ids = tuple(sorted(AUX_IDS + (end0,)))
    assert _simulate(NUM_LAYERS, 2, aux_ids).count(end0) == 1


def test_middle_stage_sends_only_local_taps():
    """PP>2 middle ranks must not re-send upstream taps."""
    start, end = get_pp_indices(NUM_LAYERS, 1, 4)
    local = _taps_emitted(start, end, AUX_IDS, False)
    upstream = [a for a in AUX_IDS if a <= start]
    assert not set(local) & set(upstream)
    assert local == [a for a in AUX_IDS if start < a <= end]
