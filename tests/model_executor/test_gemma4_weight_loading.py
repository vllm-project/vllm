# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.models.gemma4 import Gemma4Model


class _FakeParam:
    def __init__(self) -> None:
        self.weight_loader = Mock()


class _FakeGemma4Model:
    def __init__(self, param_names: list[str]) -> None:
        self.config = SimpleNamespace(num_experts=1)
        self.quant_config = None
        self._params = {name: _FakeParam() for name in param_names}

    def named_parameters(self):
        return list(self._params.items())

    def named_buffers(self):
        return []

    def named_modules(self):
        return []


@pytest.mark.parametrize(
    ("name", "loaded_weight", "expected_param", "expected_shard"),
    [
        (
            "layers.0.moe.experts.0.down_proj",
            torch.randn(4, 8),
            "layers.0.moe.experts.w2_weight",
            "w2",
        ),
        (
            "layers.0.moe.experts.0.down_proj.weight_scale",
            torch.tensor(1.0),
            "layers.0.moe.experts.w2_weight_scale",
            "w2",
        ),
        (
            "layers.0.moe.experts.0.down_proj.weight_scale_2",
            torch.tensor(1.0),
            "layers.0.moe.experts.w2_weight_scale_2",
            "w2",
        ),
        (
            "layers.0.moe.experts.0.down_proj.input_scale",
            torch.tensor(1.0),
            "layers.0.moe.experts.w2_input_scale",
            "w2",
        ),
        (
            "layers.0.moe.experts.0.gate_proj.weight_scale_2",
            torch.tensor(1.0),
            "layers.0.moe.experts.w13_weight_scale_2",
            "w1",
        ),
        (
            "layers.0.moe.experts.0.up_proj.input_scale",
            torch.tensor(1.0),
            "layers.0.moe.experts.w13_input_scale",
            "w3",
        ),
    ],
)
def test_gemma4_load_weights_maps_expert_scale_suffixes(
    name: str,
    loaded_weight: torch.Tensor,
    expected_param: str,
    expected_shard: str,
) -> None:
    model = _FakeGemma4Model([expected_param])
    param = model._params[expected_param]

    loaded_params = Gemma4Model.load_weights(model, [(name, loaded_weight)])

    param.weight_loader.assert_called_once_with(
        param,
        loaded_weight,
        expected_param,
        shard_id=expected_shard,
        expert_id=0,
    )
    assert expected_param in loaded_params
