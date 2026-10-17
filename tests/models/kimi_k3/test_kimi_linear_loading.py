# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Kimi-K3 linear layer O(1) expert mapping and weight loader."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm.models.kimi_k3.amd.linear import KimiLinearModel

pytestmark = pytest.mark.cpu_test


class _MockExpertParam(nn.Parameter):
    """Parameter mock that tracks calls to its weight_loader."""

    def __new__(cls, data: torch.Tensor):
        return super().__new__(cls, data, requires_grad=False)

    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.loaded_calls: list[dict[str, Any]] = []

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        name: str = "",
        expert_id: int | None = None,
        shard_id: str | None = None,
        **kwargs: Any,
    ):
        self.loaded_calls.append(
            {
                "param": param,
                "weight": loaded_weight,
                "name": name,
                "expert_id": expert_id,
                "shard_id": shard_id,
            }
        )


def _create_minimal_kimi_model(num_experts: int = 4) -> KimiLinearModel:
    """Instantiate a minimal KimiLinearModel with routed expert parameters."""
    config = SimpleNamespace(
        is_moe=True,
        num_experts=num_experts,
        tie_word_embeddings=False,
        q_lora_rank=None,
        linear_attn_config={},
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
    )
    # Instantiate without running __init__ to avoid full transformer construction
    model = object.__new__(KimiLinearModel)
    nn.Module.__init__(model)
    model.config = config

    # Construct minimal submodule tree: model.layers.0.mlp.experts.routed_experts
    sub_model = nn.Module()
    layer0 = nn.Module()
    mlp = nn.Module()
    experts = nn.Module()
    routed_experts = nn.Module()

    # Routed experts register w13_weight and w2_weight
    w13_param = _MockExpertParam(torch.zeros(num_experts, 32, 16))
    w2_param = _MockExpertParam(torch.zeros(num_experts, 16, 32))

    routed_experts.register_parameter("w13_weight", w13_param)
    routed_experts.register_parameter("w2_weight", w2_param)

    experts.routed_experts = routed_experts
    mlp.experts = experts
    layer0.mlp = mlp
    sub_model.layers = nn.ModuleList([layer0])
    model.model = sub_model

    return model


def test_kimi_k3_load_weights_o1_expert_routing():
    """Verify that KimiLinearModel.load_weights dispatches all expert
    projections via O(1) lookup."""
    num_experts = 4
    model = _create_minimal_kimi_model(num_experts=num_experts)

    # Generate synthetic weights for all experts across w1, w2, w3 projections
    weights_to_load = []
    for exp_id in range(num_experts):
        w1_tensor = torch.randn(16, 16)
        w2_tensor = torch.randn(16, 16)
        w3_tensor = torch.randn(16, 16)

        weights_to_load.append(
            (f"model.layers.0.mlp.experts.{exp_id}.w1.weight", w1_tensor)
        )
        weights_to_load.append(
            (f"model.layers.0.mlp.experts.{exp_id}.w2.weight", w2_tensor)
        )
        weights_to_load.append(
            (f"model.layers.0.mlp.experts.{exp_id}.w3.weight", w3_tensor)
        )

    model.load_weights(weights_to_load)

    w13_param = model.model.layers[0].mlp.experts.routed_experts.w13_weight
    w2_param = model.model.layers[0].mlp.experts.routed_experts.w2_weight

    # Each expert has w1 and w3 routed to w13_param (total 2 * num_experts)
    assert len(w13_param.loaded_calls) == num_experts * 2
    # Each expert has w2 routed to w2_param (total num_experts)
    assert len(w2_param.loaded_calls) == num_experts

    # Verify per-expert dispatch correctness
    exp_w13_name = "model.layers.0.mlp.experts.routed_experts.w13_weight"
    exp_w2_name = "model.layers.0.mlp.experts.routed_experts.w2_weight"
    for exp_id in range(num_experts):
        w1_calls = [
            c
            for c in w13_param.loaded_calls
            if c["expert_id"] == exp_id and c["shard_id"] == "w1"
        ]
        assert len(w1_calls) == 1
        assert w1_calls[0]["name"] == exp_w13_name

        w3_calls = [
            c
            for c in w13_param.loaded_calls
            if c["expert_id"] == exp_id and c["shard_id"] == "w3"
        ]
        assert len(w3_calls) == 1
        assert w3_calls[0]["name"] == exp_w13_name

        w2_calls = [
            c
            for c in w2_param.loaded_calls
            if c["expert_id"] == exp_id and c["shard_id"] == "w2"
        ]
        assert len(w2_calls) == 1
        assert w2_calls[0]["name"] == exp_w2_name


def test_kimi_k3_load_weights_unpacked_weight_packed_substitution():
    """Verify that .weight_packed is substituted with .weight when experts
    are unpacked."""
    model = _create_minimal_kimi_model(num_experts=2)

    weights_to_load = [
        (
            "model.layers.0.mlp.experts.0.w1.weight_packed",
            torch.randn(16, 16),
        ),
        (
            "model.layers.0.mlp.experts.1.w2.weight_packed",
            torch.randn(16, 16),
        ),
    ]

    model.load_weights(weights_to_load)

    w13_param = model.model.layers[0].mlp.experts.routed_experts.w13_weight
    w2_param = model.model.layers[0].mlp.experts.routed_experts.w2_weight

    assert len(w13_param.loaded_calls) == 1
    assert w13_param.loaded_calls[0]["expert_id"] == 0
    assert w13_param.loaded_calls[0]["shard_id"] == "w1"

    assert len(w2_param.loaded_calls) == 1
    assert w2_param.loaded_calls[0]["expert_id"] == 1
    assert w2_param.loaded_calls[0]["shard_id"] == "w2"


def test_kimi_k3_load_weights_scale_keys_not_corrupted():
    """Verify that scale keys (e.g. .weight_scale) match the exact expert projection."""
    model = _create_minimal_kimi_model(num_experts=2)

    # Attach weight_scale parameters
    w13_scale = _MockExpertParam(torch.zeros(2, 32))
    w2_scale = _MockExpertParam(torch.zeros(2, 16))
    model.model.layers[0].mlp.experts.routed_experts.register_parameter(
        "w13_weight_scale", w13_scale
    )
    model.model.layers[0].mlp.experts.routed_experts.register_parameter(
        "w2_weight_scale", w2_scale
    )

    weights_to_load = [
        (
            "model.layers.0.mlp.experts.0.w1.weight_scale",
            torch.tensor([1.0]),
        ),
        (
            "model.layers.0.mlp.experts.1.w2.weight_scale",
            torch.tensor([2.0]),
        ),
    ]

    model.load_weights(weights_to_load)

    assert len(w13_scale.loaded_calls) == 1
    assert w13_scale.loaded_calls[0]["expert_id"] == 0
    assert w13_scale.loaded_calls[0]["shard_id"] == "w1"

    assert len(w2_scale.loaded_calls) == 1
    assert w2_scale.loaded_calls[0]["expert_id"] == 1
    assert w2_scale.loaded_calls[0]["shard_id"] == "w2"


def test_kimi_k3_load_weights_large_expert_count_o1_scaling():
    """Verify O(1) expert mapping handles large expert counts
    (e.g. 128 experts) correctly."""
    num_experts = 128
    model = _create_minimal_kimi_model(num_experts=num_experts)

    # Load 5 distinct expert tensors out of 128
    target_experts = [0, 17, 42, 99, 127]
    weights_to_load = []
    for exp_id in target_experts:
        weights_to_load.append(
            (f"model.layers.0.mlp.experts.{exp_id}.w1.weight", torch.randn(16, 16))
        )
        weights_to_load.append(
            (f"model.layers.0.mlp.experts.{exp_id}.w2.weight", torch.randn(16, 16))
        )

    model.load_weights(weights_to_load)

    w13_param = model.model.layers[0].mlp.experts.routed_experts.w13_weight
    w2_param = model.model.layers[0].mlp.experts.routed_experts.w2_weight

    assert len(w13_param.loaded_calls) == len(target_experts)
    assert len(w2_param.loaded_calls) == len(target_experts)

    loaded_eids_w13 = {c["expert_id"] for c in w13_param.loaded_calls}
    loaded_eids_w2 = {c["expert_id"] for c in w2_param.loaded_calls}

    assert loaded_eids_w13 == set(target_experts)
    assert loaded_eids_w2 == set(target_experts)


def test_kimi_k3_load_weights_non_expert_fallthrough():
    """Verify that non-expert parameters fall through correctly to standard
    weight loading."""
    model = _create_minimal_kimi_model(num_experts=2)

    norm_param = _MockExpertParam(torch.zeros(16))
    model.model.layers[0].register_parameter("input_layernorm", norm_param)

    weights_to_load = [
        ("model.layers.0.input_layernorm", torch.randn(16)),
    ]
    model.load_weights(weights_to_load)

    assert len(norm_param.loaded_calls) == 1
    assert norm_param.loaded_calls[0]["expert_id"] is None
    assert norm_param.loaded_calls[0]["shard_id"] is None
