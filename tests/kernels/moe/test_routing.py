# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router_factory import create_fused_moe_router
from vllm.model_executor.models.llama4 import Llama4MoE

MK_S = ()
TOP_KS = ()
NUM_EXPERTS = ()


def setup_eplb_state(enable_eplb: bool) -> EplbLayerState:
    return EplbLayerState() if enable_eplb else EplbLayerState()


def make_test_data(
    m: int, k: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = torch.rand((m, k))
    logits = torch.rand((m, num_experts))
    return hidden_states, logits


def make_e_score_correction_bias(
    e_score_correction_bias_val: float,
    num_experts: int,
) -> torch.Tensor:
    # TODO
    return torch.rand(num_experts, min=0, max=e_score_correction_bias_val)


def baseline_routing(
    score: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    return topk_weight, topk_ids


@pytest.mark.parametrize("m,k", MK_S)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("global_num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("enable_eplb", [False, True])
def test_fused_topk(
    m: int,
    k: int,
    top_k: int,
    global_num_experts: int,
    renormalize: bool,
    enable_eplb: bool,
):
    eplb_state = setup_eplb_state(enable_eplb)
    router = create_fused_moe_router(
        top_k=top_k,
        global_num_experts=global_num_experts,
        renormalize=renormalize,
        enable_eplb=enable_eplb,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)


@pytest.mark.parametrize("m,k", MK_S)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("global_num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("enable_eplb", [False, True])
@pytest.mark.parametrize("e_score_correction_bias_val", [1.0])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 1.1])
def test_fused_topk_bias(
    m: int,
    k: int,
    top_k: int,
    global_num_experts: int,
    renormalize: bool,
    enable_eplb: bool,
    e_score_correction_bias_val: float,
    routed_scaling_factor: float,
):
    eplb_state = setup_eplb_state(enable_eplb)

    e_score_correction_bias = make_e_score_correction_bias(
        e_score_correction_bias_val,
        global_num_experts,
    )

    router = create_fused_moe_router(
        e_score_correction_bias=e_score_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        top_k=top_k,
        global_num_experts=global_num_experts,
        renormalize=renormalize,
        enable_eplb=enable_eplb,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)


@pytest.mark.parametrize("m,k", MK_S)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("global_num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("enable_eplb", [False, True])
@pytest.mark.parametrize("e_score_correction_bias_val", [1.0])
@pytest.mark.parametrize("num_expert_group", [])
@pytest.mark.parametrize("topk_group", [])
@pytest.mark.parametrize("scoring_func", ["llama4", "signmoid", "softmax"])
def test_grouped_topk(
    m: int,
    k: int,
    top_k: int,
    global_num_experts: int,
    renormalize: bool,
    enable_eplb: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    e_score_correction_bias_val: float,
    routed_scaling_factor: float,
):
    eplb_state = setup_eplb_state(enable_eplb)

    e_score_correction_bias = make_e_score_correction_bias(
        e_score_correction_bias_val,
        global_num_experts,
    )

    if scoring_func == "llama4":
        routing_method_type = RoutingMethodType.Llama4
        scoring_func = "sigmoid"

    router = create_fused_moe_router(
        use_grouped_topk=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routing_method_type=routing_method_type,
        e_score_correction_bias=e_score_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        top_k=top_k,
        global_num_experts=global_num_experts,
        renormalize=renormalize,
        enable_eplb=enable_eplb,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)


@pytest.mark.parametrize("m,k", MK_S)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("global_num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("enable_eplb", [False, True])
@pytest.mark.parametrize("custom_routing_function", [Llama4MoE.custom_routing_function])
def test_custom(
    m: int,
    k: int,
    top_k: int,
    global_num_experts: int,
    renormalize: bool,
    enable_eplb: bool,
    custom_routing_function: Callable,
):
    eplb_state = setup_eplb_state(enable_eplb)

    router = create_fused_moe_router(
        top_k=top_k,
        global_num_experts=global_num_experts,
        custom_routing_function=custom_routing_function,
        renormalize=renormalize,
        enable_eplb=enable_eplb,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)


# See tests/test_routing_simulatator.py
@pytest.mark.parametrize("m,k", MK_S)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("global_num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("enable_eplb", [False, True])
@pytest.mark.parameterize("strategy", ["uniform_random", "normal_routing"])
def test_simulated(
    m: int,
    k: int,
    top_k: int,
    global_num_experts: int,
    renormalize: bool,
    enable_eplb: bool,
    strategy: str,
    monkeypatch,
):
    eplb_state = setup_eplb_state(enable_eplb)

    monkeypatch.setenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", strategy)
    router = create_fused_moe_router(
        top_k=top_k,
        global_num_experts=global_num_experts,
        enable_eplb=enable_eplb,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)
