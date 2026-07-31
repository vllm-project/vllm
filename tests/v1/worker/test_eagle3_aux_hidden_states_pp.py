# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accounting checks for EAGLE3 aux hidden states carried across PP stages.

Correctness of the PP transport rests on one invariant: the number of aux
tensors stage r puts on the wire must equal the number stage r+1 pre-allocates
recv buffers for, and the concatenated order must match the drafter's
configured tap order. If that holds, the runner's key-wise copy into its
pre-allocated PP buffer cannot KeyError and the drafter cannot be handed a
permuted input.

These run on CPU with no distributed init: get_pp_group is stubbed so the
mixin's rank-dependent branch can be exercised for every rank.

Note the mismatch between what is tested here and what the runner allows. The
accounting below is size-agnostic and is checked up to pp=8, but
GPUModelRunner.load_model currently refuses pipeline_parallel_size > 2 because
only pp=2 has been validated end to end on hardware. The wider parametrization
is deliberate: it is what a future pp>2 enablement would build on, and it pins
the middle-stage case (a stage that both adopts upstream taps and contributes
its own) that pp>2 introduces.
"""

from unittest.mock import patch

import pytest

from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.interfaces import EagleModelMixin

# Kimi-K3 / DSpark: 93 layers, target_layer_ids [2, 23, 47, 71, 89].
# get_eagle3_aux_layers_from_config maps target_layer_ids -> +1.
NUM_LAYERS = 93
AUX_IDS = (3, 24, 48, 72, 90)


class _Stage(EagleModelMixin):
    """Minimal stand-in exposing just what the mixin reads."""

    def __init__(self, start_layer: int, aux_ids: tuple[int, ...]):
        self.start_layer = start_layer
        self.aux_hidden_state_layers = aux_ids


def _num_upstream(start_layer, aux_ids, is_first_rank):
    stage = _Stage(start_layer, aux_ids)
    with patch("vllm.distributed.parallel_state.get_pp_group") as get_group:
        get_group.return_value.is_first_rank = is_first_rank
        return stage.num_upstream_aux_hidden_states()


def _taps_emitted(start_layer, end_layer, aux_ids, is_first_rank):
    """Aux ids a stage appends locally, in append order.

    Mirrors the model forward: on the first rank the `start_layer in aux_ids`
    tap fires (id 0 is the raw embedding); on later ranks it is skipped because
    that id is the upstream stage's end-of-window tap. Then the layer loop taps
    `layer_idx + 1`.
    """
    out = []
    if is_first_rank and start_layer in aux_ids:
        out.append(start_layer)
    for layer_idx in range(start_layer, end_layer):
        if (layer_idx + 1) in aux_ids:
            out.append(layer_idx + 1)
    return out


def _simulate(num_layers, pp_size, aux_ids):
    """Walk the pipeline, returning the aux id list the drafter finally sees."""
    carried: list[int] = []
    for rank in range(pp_size):
        start, end = get_pp_indices(num_layers, rank, pp_size)
        is_first = rank == 0

        # Recv side: the runner sizes its buffer from this, so it must match
        # exactly what upstream sent.
        expected = _num_upstream(start, aux_ids, is_first)
        assert expected == len(carried), (
            f"pp_size={pp_size} rank={rank}: {expected} recv slots but "
            f"{len(carried)} sent by upstream (start_layer={start})"
        )
        carried = carried + _taps_emitted(start, end, aux_ids, is_first)
    return carried


@pytest.mark.parametrize("pp_size", [1, 2, 3, 4, 6, 8])
def test_drafter_sees_all_taps_in_order(pp_size):
    """Send/recv counts agree at every boundary and ordering is preserved."""
    assert _simulate(NUM_LAYERS, pp_size, AUX_IDS) == list(AUX_IDS)


def test_pp2_split_matches_expected_stages():
    """Pin the deployed shape so a get_pp_indices change fails readably."""
    start0, end0 = get_pp_indices(NUM_LAYERS, 0, 2)
    start1, end1 = get_pp_indices(NUM_LAYERS, 1, 2)
    assert _taps_emitted(start0, end0, AUX_IDS, True) == [3, 24]
    assert _taps_emitted(start1, end1, AUX_IDS, False) == [48, 72, 90]


@pytest.mark.parametrize("pp_size", [1, 2, 4])
def test_no_drafter_adds_no_payload_keys(pp_size):
    """Non-speculative deployments must carry zero extra PP payload keys."""
    assert _simulate(NUM_LAYERS, pp_size, ()) == []
    start, _ = get_pp_indices(NUM_LAYERS, pp_size - 1, pp_size)
    assert _num_upstream(start, (), pp_size == 1) == 0


def test_tap_on_stage_boundary_is_not_double_counted():
    """A tap landing exactly on a boundary is the one aliasing hazard.

    It must be emitted once, by the upstream stage that computed it, rather
    than again by the downstream stage whose start_layer equals it.
    """
    _, end0 = get_pp_indices(NUM_LAYERS, 0, 2)
    aux_ids = tuple(sorted(AUX_IDS + (end0,)))
    assert _simulate(NUM_LAYERS, 2, aux_ids).count(end0) == 1


def test_first_rank_has_no_upstream_tensors():
    assert _num_upstream(0, AUX_IDS, True) == 0
