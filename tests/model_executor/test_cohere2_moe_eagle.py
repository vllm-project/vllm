# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.models import cohere2_moe as cohere_mod
from vllm.model_executor.models.interfaces import supports_eagle3


class DummyPPGroup:
    is_first_rank = True
    is_last_rank = True


class ParallelLayer(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, positions, hidden_states, residual):
        return hidden_states + self.delta, hidden_states


class IdentityNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states, residual


def make_synthetic_model(start_layer: int = 0) -> cohere_mod.Cohere2MoeModel:
    model = cohere_mod.Cohere2MoeModel.__new__(cohere_mod.Cohere2MoeModel)
    nn.Module.__init__(model)
    model.start_layer = start_layer
    model.end_layer = start_layer + 3
    model.layers = nn.ModuleList(
        [nn.Identity() for _ in range(start_layer)]
        + [ParallelLayer(1.0), ParallelLayer(2.0), ParallelLayer(4.0)]
    )
    model.norm = IdentityNorm()
    model.aux_hidden_state_layers = ()
    model.do_not_compile = True
    return model


def set_aux_hidden_state_layers(
    model: cohere_mod.Cohere2MoeModel, layers: tuple[int, ...]
) -> None:
    target = cohere_mod.Cohere2MoeForCausalLM.__new__(cohere_mod.Cohere2MoeForCausalLM)
    nn.Module.__init__(target)
    target.model = model
    target.set_aux_hidden_state_layers(layers)


def run_model(model: cohere_mod.Cohere2MoeModel, inputs: torch.Tensor):
    input_ids = torch.zeros(inputs.shape[0], dtype=torch.long)
    positions = torch.arange(inputs.shape[0])
    return model(input_ids, positions, inputs_embeds=inputs)


@pytest.fixture(autouse=True)
def patch_pp_group(monkeypatch):
    monkeypatch.setattr(cohere_mod, "get_pp_group", lambda: DummyPPGroup())


@pytest.mark.cpu_test
def test_cohere2_moe_advertises_eagle3_support():
    assert supports_eagle3(cohere_mod.Cohere2MoeForCausalLM)


@pytest.mark.cpu_test
def test_aux_states_use_global_boundaries_and_preserve_output():
    model = make_synthetic_model(start_layer=2)
    inputs = torch.full((2, 4), 10.0)

    baseline = run_model(model, inputs)
    set_aux_hidden_state_layers(model, (5, 3, 2))
    output, aux_hidden_states = run_model(model, inputs)

    assert torch.equal(output, baseline)
    assert [state[0, 0].item() for state in aux_hidden_states] == [10.0, 11.0, 17.0]


@pytest.mark.cpu_test
def test_aux_states_capture_embedding_boundary():
    model = make_synthetic_model()
    inputs = torch.full((1, 2), 10.0)
    set_aux_hidden_state_layers(model, (0,))

    _, aux_hidden_states = run_model(model, inputs)

    assert torch.equal(aux_hidden_states[0], inputs)


@pytest.mark.cpu_test
def test_aux_capture_does_not_add_parallel_residual_twice():
    model = make_synthetic_model()
    inputs = torch.full((1, 2), 10.0)
    set_aux_hidden_state_layers(model, (1,))

    output, aux_hidden_states = run_model(model, inputs)

    assert torch.equal(output, torch.full_like(inputs, 17.0))
    assert torch.equal(aux_hidden_states[0], torch.full_like(inputs, 11.0))
