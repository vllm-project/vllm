# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models import plamo3 as plamo3_mod
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)


class DummyDecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        del positions
        return hidden_states + 1, hidden_states + 2


class DummyNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


@pytest.mark.cpu_test
def test_plamo3_returns_dflash_selected_auxiliary_hidden_states(monkeypatch):
    decoder = plamo3_mod.Plamo3Decoder.__new__(plamo3_mod.Plamo3Decoder)
    nn.Module.__init__(decoder)
    decoder.start_layer = 0
    decoder.end_layer = 2
    decoder.layers = nn.ModuleList([DummyDecoderLayer() for _ in range(8)])

    model = plamo3_mod.Plamo3Model.__new__(plamo3_mod.Plamo3Model)
    nn.Module.__init__(model)
    model.layers = decoder
    model.norm = DummyNorm()
    model.do_not_compile = True

    target = plamo3_mod.Plamo3ForCausalLM.__new__(plamo3_mod.Plamo3ForCausalLM)
    nn.Module.__init__(target)
    target.model = model

    monkeypatch.setattr(
        plamo3_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    assert supports_eagle3(target)
    assert target.get_eagle3_default_aux_hidden_state_layers() == (2, 4, 5)

    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"target_layer_ids": [0, 1]})
        )
    )
    set_eagle3_aux_hidden_state_layers(target, spec_config)
    assert decoder.aux_hidden_state_layers == (1, 2)

    output, aux_hidden_states = target(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=torch.tensor([[1.0, 2.0]]),
    )

    torch.testing.assert_close(output, torch.tensor([[7.0, 9.0]]))
    assert len(aux_hidden_states) == 2
    torch.testing.assert_close(aux_hidden_states[0], torch.tensor([[5.0, 7.0]]))
    torch.testing.assert_close(aux_hidden_states[1], torch.tensor([[7.0, 9.0]]))
