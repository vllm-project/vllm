# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    _EAGLE3_AUX_KEY_PREFIX,
)
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
from vllm.sequence import IntermediateTensors


class DummyEagleModel(EagleModelMixin):
    def __init__(self, start_layer: int, hidden_size: int = 4):
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.start_layer = start_layer
        self.aux_hidden_state_layers = ()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], hidden_size
            )
        )


def test_eagle3_aux_empty_intermediate_tensors_include_previous_pp_layers():
    model = DummyEagleModel(start_layer=3)

    model._set_aux_hidden_state_layers((1, 3, 5))
    tensors = model.make_empty_intermediate_tensors(
        batch_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ).tensors

    assert set(tensors) == {
        "hidden_states",
        "residual",
        f"{_EAGLE3_AUX_KEY_PREFIX}0",
        f"{_EAGLE3_AUX_KEY_PREFIX}1",
    }
    assert tensors[f"{_EAGLE3_AUX_KEY_PREFIX}0"].shape == (2, 4)


def test_eagle3_set_aux_layers_refreshes_top_level_empty_tensor_factory():
    class DummyTopLevel:
        def __init__(self):
            self.model = DummyEagleModel(start_layer=3)
            self.make_empty_intermediate_tensors = (
                self.model.make_empty_intermediate_tensors
            )

    model = DummyTopLevel()

    SupportsEagle3.set_aux_hidden_state_layers(model, (1,))
    tensors = model.make_empty_intermediate_tensors(
        batch_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ).tensors

    assert f"{_EAGLE3_AUX_KEY_PREFIX}0" in tensors


def test_eagle3_aux_hidden_states_round_trip_in_intermediate_tensors():
    model = DummyEagleModel(start_layer=3)
    model._set_aux_hidden_state_layers((1, 3))

    previous_aux = torch.ones((2, 4))
    local_hidden = torch.full((2, 4), 2.0)
    local_residual = torch.full((2, 4), 3.0)
    incoming = IntermediateTensors(
        {
            "hidden_states": torch.zeros((2, 4)),
            "residual": torch.zeros((2, 4)),
            f"{_EAGLE3_AUX_KEY_PREFIX}0": previous_aux,
        }
    )

    aux_hidden_states = model._get_eagle3_aux_hidden_states(incoming)
    model._maybe_add_hidden_state(
        aux_hidden_states, 3, local_hidden, local_residual
    )
    output = model._make_intermediate_tensors_with_eagle3_aux(
        local_hidden,
        local_residual,
        aux_hidden_states,
    )

    assert torch.equal(output.tensors[f"{_EAGLE3_AUX_KEY_PREFIX}0"], previous_aux)
    assert torch.equal(
        output.tensors[f"{_EAGLE3_AUX_KEY_PREFIX}1"],
        local_hidden + local_residual,
    )
