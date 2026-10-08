# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.model_executor.models import llama


def test_tpsp_rejects_pipeline_parallelism():
    config = ModelConfig.__new__(ModelConfig)
    config.enable_tpsp = True
    with pytest.raises(ValueError, match="enable-tpsp.*pipeline-parallel-size"):
        config.verify_with_parallel_config(SimpleNamespace(pipeline_parallel_size=2))
    with pytest.raises(ValueError, match="enable-tpsp.*pipeline-parallel-size"):
        llama.TPSPLlamaForCausalLM(
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(pipeline_parallel_size=2)
            )
        )


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_tpsp_biased_projection_uses_cuda_fused_path(monkeypatch, device_type):
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(llama, "get_tp_group", lambda: group)
    monkeypatch.setattr(llama, "select_sp_config", lambda profile, tokens: True)
    layer = llama.TPSPLlamaDecoderLayer.__new__(llama.TPSPLlamaDecoderLayer)
    nn.Module.__init__(layer)
    layer.hidden_size = 2

    class Projection(nn.Module):
        input_size_per_partition = 2

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(2, dtype=torch.bfloat16))
            self.bias = nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.bfloat16))

        def forward(self, x):
            return x + self.bias, None

    class Norm:
        weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        variance_epsilon = 1e-5

        def __call__(self, x, residual):
            return x + residual, x + residual

    class Backend:
        device = torch.device(device_type)

        def fused(
            self, x, weight, norm_weight, residual, eps, config, *, projection_bias
        ):
            assert device_type == "cuda"
            torch.testing.assert_close(projection_bias, projection.bias)
            output = x + projection_bias + residual
            return output, None, output

    x = torch.ones((2, 2), dtype=torch.bfloat16)
    projection = Projection()
    profile = SimpleNamespace(tp_size=1, hidden_size=2, input_width=2, config=object())
    output, residual = layer._project_and_normalize(
        x, projection, x, Norm(), profile, Backend(), False
    )
    expected = x + torch.tensor([2.0, 3.0], dtype=x.dtype) + x
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(residual, expected)


class _TPSPLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.input_layernorm = nn.Identity()

    def forward_sp(self, positions, hidden_states, residual, *args):
        local_residual = torch.full(
            (2, hidden_states.size(-1)),
            float(self.index + 1),
            dtype=hidden_states.dtype,
        )
        return hidden_states + 1, local_residual


@pytest.mark.parametrize("taps", [(), (0, 1, 2), (2,)])
def test_tpsp_aux_states_gather_only_requested_layers(monkeypatch, taps):
    group = SimpleNamespace(world_size=2, device_group=object())
    monkeypatch.setattr(llama, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(llama, "get_tp_group", lambda: group)
    gathered = []

    def all_gather(output, local, *, group):
        gathered.append(local.clone())
        output[: local.size(0)] = local
        output[local.size(0) :] = local + 10

    monkeypatch.setattr(llama.dist, "all_gather_into_tensor", all_gather)
    model = llama.TPSPLlamaModel.__new__(llama.TPSPLlamaModel)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.config = SimpleNamespace(hidden_size=2)
    model.embed_tokens = nn.Identity()
    model.layers = nn.ModuleList([_TPSPLayer(0), _TPSPLayer(1)])
    model.norm = nn.Identity()
    model.tpsp_profile = SimpleNamespace(
        profiles={
            "o": SimpleNamespace(enabled=True),
            "down": SimpleNamespace(enabled=True),
        },
        backend=None,
    )
    model.aux_hidden_state_layers = taps

    inputs = torch.zeros((3, 2))
    result = model.forward(None, None, None, inputs_embeds=inputs)
    if not taps:
        torch.testing.assert_close(result, inputs + 2)
    else:
        output, aux = result
        torch.testing.assert_close(output, inputs + 2)
        expected = {
            0: inputs,
            1: torch.tensor([[1.0, 1.0], [1.0, 1.0], [11.0, 11.0]]),
            2: torch.tensor([[2.0, 2.0], [2.0, 2.0], [12.0, 12.0]]),
        }
        for state, tap in zip(aux, taps, strict=True):
            torch.testing.assert_close(state, expected[tap])
    assert len(gathered) == sum(tap > 0 for tap in taps)


def test_tpsp_disabled_profiles_use_regular_llama_forward(monkeypatch):
    monkeypatch.setattr(llama, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    expected = object()
    monkeypatch.setattr(
        llama.LlamaModel, "forward", lambda self, *args, **kwargs: expected
    )
    model = llama.TPSPLlamaModel.__new__(llama.TPSPLlamaModel)
    nn.Module.__init__(model)
    model.tpsp_profile = SimpleNamespace(
        profiles={
            "o": SimpleNamespace(enabled=False),
            "down": SimpleNamespace(enabled=True),
        }
    )
    assert model.forward(None, None, None) is expected
