# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.kimi_k3.amd import linear as kimi_linear
from vllm.models.kimi_k3.amd.linear import KimiLinearModel
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AMD Kimi-K3 requires ROCm",
)


def _model(
    *,
    start_layer: int,
    end_layer: int,
    layer_output: tuple[torch.Tensor, torch.Tensor | None],
    aux_layers: tuple[int, ...],
) -> KimiLinearModel:
    model = object.__new__(KimiLinearModel)
    torch.nn.Module.__init__(model)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(attn_res_block_size=None),
    )
    object.__setattr__(model, "start_layer", start_layer)
    object.__setattr__(model, "end_layer", end_layer)
    layers = [None] * start_layer + [Mock(return_value=layer_output)]
    object.__setattr__(model, "layers", layers)
    object.__setattr__(model, "aux_hidden_state_layers", aux_layers)
    return model


def test_amd_kimi_declares_aux_hidden_state_pp_support():
    assert KimiLinearModel.supports_aux_hidden_states_over_pp
    assert "update" not in KimiLinearModel.forward.__code__.co_names


def test_first_stage_packs_local_aux_hidden_states(monkeypatch):
    initial = torch.tensor([[1.0, 2.0]])
    layer_hidden = torch.tensor([[3.0, 4.0]])
    model = _model(
        start_layer=0,
        end_layer=1,
        layer_output=(layer_hidden, None),
        aux_layers=(0,),
    )
    monkeypatch.setattr(
        kimi_linear,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial,
    )

    assert isinstance(output, IntermediateTensors)
    torch.testing.assert_close(output["aux_hidden_states_0"], initial)


def test_last_stage_prepends_remote_aux_hidden_states(monkeypatch):
    remote = torch.tensor([[1.0, 2.0]])
    incoming = torch.tensor([[3.0, 4.0]])
    local = torch.tensor([[5.0, 6.0]])
    model = _model(
        start_layer=1,
        end_layer=2,
        layer_output=(local, None),
        aux_layers=(0, 2),
    )
    object.__setattr__(model, "_aux_upstream_total_cached", 1)
    monkeypatch.setattr(
        kimi_linear,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    intermediate = IntermediateTensors(
        {
            "hidden_states": incoming,
            "residual": incoming,
            "aux_hidden_states_0": remote,
        }
    )

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=intermediate,
    )

    torch.testing.assert_close(output, local)
    assert len(aux_hidden_states) == 2
    torch.testing.assert_close(aux_hidden_states[0], remote)
    torch.testing.assert_close(aux_hidden_states[1], local)
