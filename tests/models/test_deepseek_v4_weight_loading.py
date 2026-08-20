# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.deepseek_v4.nvidia.model as model_module
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model


def test_attn_sink_uses_weight_loader_with_padded_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    param = nn.Parameter(
        torch.full((4,), -float("inf"), dtype=torch.float32),
        requires_grad=False,
    )
    calls = []

    def weight_loader(target, loaded_weight):
        calls.append(loaded_weight.clone())
        assert target is param
        assert loaded_weight.shape == target.shape
        target.data.copy_(loaded_weight)

    param.weight_loader = weight_loader
    model = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=3),
        quant_config=None,
        use_sequence_parallel=False,
        named_parameters=lambda: [("layers.0.attn.attn_sink", param)],
        get_expert_mapping=lambda: [],
    )
    monkeypatch.setattr(
        model_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        model_module, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        model_module, "is_pp_missing_parameter", lambda name, module: False
    )

    loaded = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result = DeepseekV4Model.load_weights(
        model, [("layers.0.attn.attn_sink", loaded)]
    )

    assert result == {"layers.0.attn.attn_sink"}
    assert len(calls) == 1
    torch.testing.assert_close(
        calls[0],
        torch.tensor([1.0, 2.0, 3.0, -float("inf")]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(param, calls[0], rtol=0, atol=0)
