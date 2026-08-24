# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EAGLE3 auxiliary hidden-state ordering across PP stages."""

import pytest

from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.mimo import MiMoModel
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    verify_supports_aux_hidden_states_over_pp,
)


def _local_aux_layer_ids(start_layer, end_layer, aux_ids, is_first_rank):
    return list(
        EagleModelMixin.local_aux_layer_ids(
            start_layer, end_layer, aux_ids, is_first_rank
        )
    )


def _simulate(num_layers, pp_size, aux_ids):
    gathered: list[int] = []
    for rank in range(pp_size):
        start, end = get_pp_indices(num_layers, rank, pp_size)
        gathered.extend(_local_aux_layer_ids(start, end, aux_ids, rank == 0))
    return gathered


@pytest.mark.parametrize("pp_size", [1, 2, 3, 4, 6, 8])
def test_drafter_sees_all_aux_layers_in_order(pp_size):
    assert _simulate(NUM_LAYERS, pp_size, AUX_IDS) == list(AUX_IDS)


def test_pp2_split_matches_expected_stages():
    start0, end0 = get_pp_indices(NUM_LAYERS, 0, 2)
    start1, end1 = get_pp_indices(NUM_LAYERS, 1, 2)
    assert _local_aux_layer_ids(start0, end0, AUX_IDS, True) == [3, 24]
    assert _local_aux_layer_ids(start1, end1, AUX_IDS, False) == [48, 72, 90]


@pytest.mark.parametrize("pp_size", [1, 2, 4])
def test_no_drafter_adds_no_payload_keys(pp_size):
    assert _simulate(NUM_LAYERS, pp_size, ()) == []


def test_aux_layer_on_stage_boundary_is_not_double_counted():
    _, end0 = get_pp_indices(NUM_LAYERS, 0, 2)
    aux_ids = tuple(sorted(AUX_IDS + (end0,)))
    assert _simulate(NUM_LAYERS, 2, aux_ids).count(end0) == 1


def test_middle_stage_produces_only_local_aux_states():
    start, end = get_pp_indices(NUM_LAYERS, 1, 4)
    local = _local_aux_layer_ids(start, end, AUX_IDS, False)
    upstream = [a for a in AUX_IDS if a <= start]
    assert not set(local) & set(upstream)
    assert local == [a for a in AUX_IDS if start < a <= end]


def test_aux_layers_are_sorted():
    model = EagleModelMixin()
    model._set_aux_hidden_state_layers((48, 3, 90, 24))
    assert model.aux_hidden_state_layers == (3, 24, 48, 90)


@pytest.mark.parametrize("model_cls", [LlamaModel, Qwen2Model])
def test_forward_does_not_name_update(model_cls):
    """Packing auxiliary states must not go through dict.update.

    TorchDynamoWrapper.bytecode_hook refuses any compiled forward whose
    bytecode names `update`, so a model that packs its states that way cannot
    start under cudagraphs. The same holds for the other opted-in models.
    """
    assert "update" not in model_cls.forward.__code__.co_names
