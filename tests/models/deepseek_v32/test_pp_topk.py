# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v32 import attention as deepseek_v32_attention
from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model
from vllm.sequence import IntermediateTensors

# GLM-5.3: each group of 4 layers from layer 3 reuses its first layer's top-k.
GLM5_TOPK = dict(index_topk_freq=4, index_skip_topk_offset=3)


class TestPPCarriesTopk:
    @pytest.mark.parametrize(
        "config,pp_size,expected",
        [
            (dict(num_hidden_layers=78, **GLM5_TOPK), 2, True),
            (dict(num_hidden_layers=76, **GLM5_TOPK), 2, False),
            (dict(num_hidden_layers=78, **GLM5_TOPK), 1, False),
            (dict(num_hidden_layers=4, index_topk_pattern="FFSS"), 2, True),
            (dict(num_hidden_layers=4, index_topk_pattern="FSFS"), 2, False),
            (dict(num_hidden_layers=78), 2, False),
        ],
        ids=[
            "split-inside-group",
            "split-on-group-start",
            "no-pp",
            "pattern-split-on-shared",
            "pattern-split-on-full",
            "no-topk-reuse",
        ],
    )
    def test_stage_boundaries(self, monkeypatch, config, pp_size, expected):
        monkeypatch.setattr(
            deepseek_v32_attention,
            "get_pp_group",
            lambda: SimpleNamespace(world_size=pp_size),
        )
        config = SimpleNamespace(**config)
        assert deepseek_v32_attention.pp_carries_topk(config) == expected


@pytest.mark.parametrize("carries_topk", [True, False], ids=["topk", "no-topk"])
def test_empty_intermediate_tensors(carries_topk):
    config = SimpleNamespace(hidden_size=4, index_topk=8)
    make_empty = deepseek_v32_attention.make_dsa_intermediate_tensors_factory(
        config, carries_topk
    )
    tensors = make_empty(3, torch.bfloat16, torch.device("cpu"))

    assert ("topk_indices" in tensors.tensors) == carries_topk
    if carries_topk:
        assert tensors["topk_indices"].shape == (3, 8)
        assert tensors["topk_indices"].dtype == torch.int32


class _TopkReusingLayer(nn.Module):
    """Reads the shared top-k like a reusing layer, then indexes like the next."""

    def __init__(self, topk_indices_buffer: torch.Tensor) -> None:
        super().__init__()
        self.topk_indices_buffer = topk_indices_buffer
        self.seen_topk: torch.Tensor | None = None

    def forward(self, positions, hidden_states, residual, attn_in=None):
        num_tokens = hidden_states.shape[0]
        self.seen_topk = self.topk_indices_buffer[:num_tokens].clone()
        self.topk_indices_buffer[:num_tokens] += 10
        return hidden_states, residual


def test_middle_pp_stage_receives_and_forwards_topk(monkeypatch):
    model = object.__new__(deepseek_v32_model.DeepseekV32Model)
    nn.Module.__init__(model)
    model.topk_indices_buffer = torch.full((8, 2), -1, dtype=torch.int32)
    layer = _TopkReusingLayer(model.topk_indices_buffer)
    model.layers = nn.ModuleList([layer])
    model.start_layer, model.end_layer = 0, 1
    model.use_sequence_parallel = False
    model.aux_hidden_state_layers = ()
    model.pp_carries_topk = True
    monkeypatch.setattr(
        deepseek_v32_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )

    received_topk = torch.arange(6, dtype=torch.int32).view(3, 2)
    output = model(
        None,
        torch.arange(3),
        IntermediateTensors(
            {
                "hidden_states": torch.zeros(3, 4),
                "residual": torch.zeros(3, 4),
                "topk_indices": received_topk,
            }
        ),
    )

    torch.testing.assert_close(layer.seen_topk, received_topk)
    torch.testing.assert_close(output["topk_indices"], received_topk + 10)
