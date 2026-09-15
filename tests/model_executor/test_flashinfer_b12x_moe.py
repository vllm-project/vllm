# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import sys
import weakref
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
import torch

import vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe as b12x
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _make_experts(
    activation: MoEActivation,
    max_num_tokens: int = 16,
) -> b12x.FlashInferB12xExperts:
    quant_config = cast(
        FusedMoEQuantConfig,
        SimpleNamespace(
            quant_dtype="nvfp4",
            g1_alphas=torch.full((2,), 1.0),
            g2_alphas=torch.full((2,), 2.0),
            w1_scale=torch.full((2,), 3.0),
            w2_scale=torch.full((2,), 4.0),
        ),
    )
    moe_config = cast(
        FusedMoEConfig,
        SimpleNamespace(
            in_dtype=torch.bfloat16,
            num_experts=2,
            experts_per_token=2,
            num_local_experts=2,
            hidden_dim=4,
            intermediate_size_per_partition=8,
            max_num_tokens=max_num_tokens,
            dp_size=2,
            device=torch.device("cpu"),
            activation=activation,
        ),
    )
    experts = b12x.FlashInferB12xExperts(moe_config, quant_config)
    experts._fc2_input_scale = torch.full((2,), 5.0)
    experts.w1_sf_mma = torch.full((2,), 6.0)
    experts.w2_sf_mma = torch.full((2,), 7.0)
    return experts


@pytest.mark.parametrize(
    ("activation", "expected_activation"),
    [
        (MoEActivation.SILU, "silu"),
        (MoEActivation.RELU2_NO_MUL, "relu2"),
    ],
)
def test_b12x_layers_share_owned_workspace_and_preserve_call_contract(
    monkeypatch: pytest.MonkeyPatch,
    activation: MoEActivation,
    expected_activation: str,
):
    """Reuse one fixed-capacity wrapper while passing each layer's weights."""
    wrappers = []
    calls = []

    class FakeB12xMoEWrapper:
        def __init__(self, **kwargs):
            self.config = kwargs
            wrappers.append(self)

        def run(self, **kwargs):
            calls.append(kwargs)
            return torch.full_like(kwargs["x"], len(calls))

    fake_fused_moe = ModuleType("flashinfer.fused_moe")
    fake_fused_moe.__dict__["B12xMoEWrapper"] = FakeB12xMoEWrapper
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fake_fused_moe)

    expert_instances = (_make_experts(activation), _make_experts(activation))
    assert len(wrappers) == 1
    assert wrappers[0].config == {
        "num_experts": 2,
        "top_k": 2,
        "hidden_size": 4,
        "intermediate_size": 8,
        "use_cuda_graph": True,
        "max_num_tokens": 32,
        "num_local_experts": 2,
        "output_dtype": torch.bfloat16,
        "device": "cpu",
        "activation": expected_activation,
    }

    output = torch.empty(3, 4, dtype=torch.bfloat16)
    hidden_states = torch.empty_like(output)
    w1 = torch.full((2, 1, 1), 11.0)
    w2 = torch.full((2, 1, 1), 12.0)
    topk_weights = torch.empty(3, 2)
    topk_ids = torch.zeros(3, 2, dtype=torch.int64)

    for experts in expert_instances:
        experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=2,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    assert len(calls) == 2
    assert torch.equal(output, torch.full_like(output, 2.0))
    for call, experts in zip(calls, expert_instances):
        assert call["x"] is hidden_states
        assert call["token_final_scales"] is topk_weights
        assert call["token_selected_experts"].dtype == torch.int32
        assert torch.equal(call["token_selected_experts"], topk_ids)
        assert call["w1_weight"] is w1
        assert call["w2_weight"] is w2
        assert call["w1_weight_sf"] is experts.w1_sf_mma
        assert call["w2_weight_sf"] is experts.w2_sf_mma
        assert call["w1_alpha"] is experts.g1_alphas
        assert call["w2_alpha"] is experts.g2_alphas
        assert call["fc2_input_scale"] is experts._fc2_input_scale


def test_b12x_owned_workspace_is_released_with_model(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeB12xMoEWrapper:
        def __init__(self, **kwargs):
            pass

    fake_fused_moe = ModuleType("flashinfer.fused_moe")
    fake_fused_moe.__dict__["B12xMoEWrapper"] = FakeB12xMoEWrapper
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fake_fused_moe)

    experts = _make_experts(MoEActivation.SILU, max_num_tokens=17)
    wrapper_ref = weakref.ref(experts._wrapper)

    del experts
    gc.collect()

    assert wrapper_ref() is None


def test_b12x_workspace_is_not_shared_between_model_owners(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeB12xMoEWrapper:
        def __init__(self, **kwargs):
            pass

    fake_fused_moe = ModuleType("flashinfer.fused_moe")
    fake_fused_moe.__dict__["B12xMoEWrapper"] = FakeB12xMoEWrapper
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fake_fused_moe)

    owners = iter((object(), object()))
    monkeypatch.setattr(b12x, "get_current_vllm_config_or_none", lambda: next(owners))

    first = _make_experts(MoEActivation.SILU, max_num_tokens=18)
    second = _make_experts(MoEActivation.SILU, max_num_tokens=18)

    assert first._wrapper is not second._wrapper
