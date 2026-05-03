# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, TypedDict

import pytest
import torch

from vllm.model_executor.models.gemma4 import Gemma4Model

pytestmark = pytest.mark.cpu_test


class WeightLoaderCall(TypedDict):
    target: torch.Tensor
    loaded_weight: torch.Tensor
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def _make_recording_param() -> tuple[torch.nn.Parameter, list[WeightLoaderCall]]:
    # Capture weight_loader calls so tests can assert the chosen remapping.
    param = torch.nn.Parameter(torch.zeros(1))
    calls: list[WeightLoaderCall] = []

    def weight_loader(
        target: torch.Tensor,
        loaded_weight: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        calls.append(
            {
                "target": target,
                "loaded_weight": loaded_weight.clone(),
                "args": args,
                "kwargs": kwargs,
            }
        )

    param.weight_loader = weight_loader
    return param, calls


class FakeGemma4Model(torch.nn.Module):
    # Reuse Gemma4Model.load_weights with a minimal module focused on MoE
    # expert-name remapping instead of constructing a full Gemma4 model.
    load_weights = Gemma4Model.load_weights

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_experts=1)
        self.quant_config = None

        self.experts_w13_weight_packed, self.w13_packed_calls = _make_recording_param()
        self.experts_w13_scales, self.w13_scales_calls = _make_recording_param()
        self.experts_w2_weight_scale, self.w2_scale_calls = _make_recording_param()

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        yield "experts.w13_weight_packed", self.experts_w13_weight_packed
        yield "experts.w13_scales", self.experts_w13_scales
        yield "experts.w2_weight_scale", self.experts_w2_weight_scale


def test_gemma4_load_weights_maps_dot_suffix_expert_names() -> None:
    model = FakeGemma4Model()

    loaded = model.load_weights(
        [
            ("experts.0.gate_proj.scales", torch.tensor([1.0])),
            ("experts.0.up_proj.scales", torch.tensor([2.0])),
        ]
    )

    assert "experts.w13_scales" in loaded
    assert len(model.w13_scales_calls) == 2
    assert [call["kwargs"]["shard_id"] for call in model.w13_scales_calls] == [
        "w1",
        "w3",
    ]
    assert all(call["kwargs"]["expert_id"] == 0 for call in model.w13_scales_calls)
    assert [call["args"][0] for call in model.w13_scales_calls] == [
        "experts.w13_scales",
        "experts.w13_scales",
    ]


def test_gemma4_load_weights_maps_underscore_suffix_expert_names() -> None:
    model = FakeGemma4Model()

    loaded = model.load_weights(
        [
            ("experts.0.gate_proj_packed", torch.tensor([3.0])),
            ("experts.0.down_proj_scale", torch.tensor([4.0])),
        ]
    )

    assert "experts.w13_weight_packed" in loaded
    assert "experts.w2_weight_scale" in loaded

    assert len(model.w13_packed_calls) == 1
    assert model.w13_packed_calls[0]["args"] == ("experts.w13_weight_packed",)
    assert model.w13_packed_calls[0]["kwargs"] == {"shard_id": "w1", "expert_id": 0}

    assert len(model.w2_scale_calls) == 1
    assert model.w2_scale_calls[0]["args"] == ("experts.w2_weight_scale",)
    assert model.w2_scale_calls[0]["kwargs"] == {"shard_id": "w2", "expert_id": 0}
