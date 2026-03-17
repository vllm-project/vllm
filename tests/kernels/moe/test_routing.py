# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.models.llama4 import Llama4MoE

# Test parameters
MK_S = [(32, 256), (64, 512)]
TOP_KS = [2, 4, 6]
NUM_EXPERTS = [8, 16, 64]


def setup_eplb_state(enable_eplb: bool, global_num_experts: int) -> EplbLayerState:
    if not enable_eplb:
        return EplbLayerState()

    # Initialize EPLB state with proper tensors for testing
    # For testing purposes, we use a simple 1:1 mapping (no redundant experts)
    # expert_load_view: tracks load on each expert (shape: num_experts)
    expert_load_view = torch.zeros(global_num_experts, dtype=torch.int32, device="cuda")

    # logical_to_physical_map: maps logical experts to physical experts
    # Shape: (num_logical_experts, max_slots)
    # For testing, use simple 1:1 mapping with single slot per expert
    logical_to_physical_map = torch.arange(
        global_num_experts, dtype=torch.int64, device="cuda"
    ).unsqueeze(-1)

    # logical_replica_count: number of replicas per logical expert
    # Shape: (num_logical_experts,)
    # For testing, each logical expert has exactly 1 replica
    logical_replica_count = torch.ones(
        global_num_experts, dtype=torch.int64, device="cuda"
    )

    return EplbLayerState(
        expert_load_view=expert_load_view,
        logical_to_physical_map=logical_to_physical_map,
        logical_replica_count=logical_replica_count,
    )


def make_test_data(
    m: int, k: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn((m, k), device="cuda") / 10
    logits = torch.randn((m, num_experts), device="cuda")
    return hidden_states, logits


def make_e_score_correction_bias(
    e_score_correction_bias_val: float,
    num_experts: int,
) -> torch.Tensor:
    # return torch.randn(num_experts, device="cuda") * e_score_correction_bias_val
    return torch.full(
        (num_experts,), e_score_correction_bias_val, device="cuda", dtype=torch.float32
    )


def assert_routing_results_close(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    baseline_weights: torch.Tensor,
    baseline_ids: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """
    Compare routing results, sorting by expert ID first to handle non-deterministic
    ordering from sorted=False in topk.
    """
    # Sort both results by expert IDs for consistent comparison
    sorted_indices_actual = torch.argsort(topk_ids, dim=-1)
    sorted_indices_baseline = torch.argsort(baseline_ids.to(topk_ids.dtype), dim=-1)

    # Gather the sorted values
    topk_ids_sorted = torch.gather(topk_ids, 1, sorted_indices_actual)
    topk_weights_sorted = torch.gather(topk_weights, 1, sorted_indices_actual)
    baseline_ids_sorted = torch.gather(
        baseline_ids.to(topk_ids.dtype), 1, sorted_indices_baseline
    )
    baseline_weights_sorted = torch.gather(baseline_weights, 1, sorted_indices_baseline)

    # Compare
    torch.testing.assert_close(topk_ids_sorted, baseline_ids_sorted)
    torch.testing.assert_close(
        topk_weights_sorted, baseline_weights_sorted, rtol=rtol, atol=atol
    )


def baseline_fused_topk(
    router_logits: torch.Tensor, top_k: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline for standard fused top-k routing.

    Algorithm:
    1. Apply softmax to router logits
    2. Select top-k experts
    3. Optionally renormalize the weights
    """
    scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    # Use sorted=False to match vllm implementation (vllm_is_batch_invariant
    # defaults to False)
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def baseline_fused_topk_bias(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline for fused top-k with bias correction.

    Algorithm:
    1. Apply softmax to router logits
    2. Add bias to scores for expert selection
    3. Select top-k experts using biased scores
    4. Get weights from original (unbiased) scores
    5. Apply routed scaling factor
    6. Optionally renormalize the weights
    """
    # Apply softmax to get scores
    scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)

    # Add bias for expert selection
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)

    # Select top-k using biased scores (sorted=False to match implementation)
    topk_ids = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]

    # Get weights from original scores (not biased)
    topk_weights = scores.gather(1, topk_ids)

    # Renormalize if needed (BEFORE applying scaling factor)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Apply scaling factor (AFTER renormalization, if applicable)
    if routed_scaling_factor != 1.0:
        topk_weights *= routed_scaling_factor

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def baseline_grouped_topk(
    router_logits: torch.Tensor,
    top_k: int,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline for grouped top-k routing (e.g., DeepSeek).

    Algorithm:
    1. Apply scoring function (softmax or sigmoid)
    2. Optionally add bias
    3. Select top-k groups based on max scores within each group
    4. Mask scores to only include selected groups
    5. Select top-k experts from masked scores
    6. Apply scaling factor
    7. Optionally renormalize
    """
    num_token = router_logits.shape[0]

    # Apply scoring function
    if scoring_func == "softmax":
        scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(router_logits.float())
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    # Handle bias correction
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        # For bias case, use sum of top-2 scores in each group
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        # Use max score in each group
        group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values

    # Select top-k groups
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]

    # Create mask for selected groups
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Expand mask to all experts
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )

    # Mask scores (set non-selected to -inf)
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

    # Select top-k experts
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)

    # Renormalize if needed
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Apply scaling factor
    if routed_scaling_factor != 1.0:
        topk_weights *= routed_scaling_factor

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def baseline_custom_llama4(
    router_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline for Llama4 custom routing.

    Algorithm:
    1. Select top-k expert indices (without softmax)
    2. Apply sigmoid to the selected scores
    """
    router_scores, router_indices = torch.topk(router_logits, top_k, dim=-1)
    router_scores = torch.sigmoid(router_scores.float())
    return router_scores.to(torch.float32), router_indices.to(torch.int32)


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
    if top_k > global_num_experts:
        pytest.skip(f"top_k ({top_k}) > global_num_experts ({global_num_experts})")

    eplb_state = setup_eplb_state(enable_eplb, global_num_experts)
    router = create_fused_moe_router(
        top_k=top_k,
        global_num_experts=global_num_experts,
        renormalize=renormalize,
        enable_eplb=enable_eplb,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)

    # Get router output
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)

    # Compute baseline
    baseline_weights, baseline_ids = baseline_fused_topk(
        router_logits, top_k, renormalize
    )

    # Compare results
    assert_routing_results_close(topk_weights, topk_ids, baseline_weights, baseline_ids)


@pytest.mark.parametrize("m,k", MK_S)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("global_num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("enable_eplb", [False, True])
@pytest.mark.parametrize("e_score_correction_bias_val", [0.9])
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
    if top_k > global_num_experts:
        pytest.skip(f"top_k ({top_k}) > global_num_experts ({global_num_experts})")

    eplb_state = setup_eplb_state(enable_eplb, global_num_experts)

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

    # Get router output
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)

    # Compute baseline
    baseline_weights, baseline_ids = baseline_fused_topk_bias(
        router_logits,
        top_k,
        renormalize,
        e_score_correction_bias,
        routed_scaling_factor,
    )

    # Compare results
    assert_routing_results_close(topk_weights, topk_ids, baseline_weights, baseline_ids)


@pytest.mark.parametrize("m,k", MK_S)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize(
    "global_num_experts,num_expert_group,topk_group",
    [
        (64, 8, 4),  # 8 groups of 8 experts, select 4 groups
        (32, 4, 2),  # 4 groups of 8 experts, select 2 groups
    ],
)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("enable_eplb", [False, True])
@pytest.mark.parametrize("e_score_correction_bias_val", [0.9])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 1.1])
@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
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
    if top_k > global_num_experts:
        pytest.skip(f"top_k ({top_k}) > global_num_experts ({global_num_experts})")

    eplb_state = setup_eplb_state(enable_eplb, global_num_experts)

    e_score_correction_bias = make_e_score_correction_bias(
        e_score_correction_bias_val,
        global_num_experts,
    )

    router = create_fused_moe_router(
        use_grouped_topk=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        top_k=top_k,
        global_num_experts=global_num_experts,
        renormalize=renormalize,
        enable_eplb=enable_eplb,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)

    # Get router output
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)

    # Compute baseline
    baseline_weights, baseline_ids = baseline_grouped_topk(
        router_logits,
        top_k,
        num_expert_group,
        topk_group,
        scoring_func,
        renormalize,
        e_score_correction_bias,
        routed_scaling_factor,
    )

    # Compare results
    assert_routing_results_close(topk_weights, topk_ids, baseline_weights, baseline_ids)


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
    if top_k > global_num_experts:
        pytest.skip(f"top_k ({top_k}) > global_num_experts ({global_num_experts})")

    eplb_state = setup_eplb_state(enable_eplb, global_num_experts)

    router = create_fused_moe_router(
        top_k=top_k,
        global_num_experts=global_num_experts,
        custom_routing_function=custom_routing_function,
        renormalize=renormalize,
        enable_eplb=enable_eplb,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)

    # Get router output
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)

    # Compute baseline (Llama4 uses sigmoid)
    baseline_weights, baseline_ids = baseline_custom_llama4(router_logits, top_k)

    # Compare results
    assert_routing_results_close(topk_weights, topk_ids, baseline_weights, baseline_ids)


# TODO: is other test sufficient?
# # See tests/test_routing_simulatator.py
# @pytest.mark.parametrize("m,k", MK_S)
# @pytest.mark.parametrize("top_k", TOP_KS)
# @pytest.mark.parametrize("global_num_experts", NUM_EXPERTS)
# @pytest.mark.parametrize("renormalize", [False, True])
# @pytest.mark.parametrize("enable_eplb", [False, True])
# @pytest.mark.parameterize("strategy", ["uniform_random", "normal_routing"])
# def test_simulated(
#     m: int,
#     k: int,
#     top_k: int,
#     global_num_experts: int,
#     renormalize: bool,
#     enable_eplb: bool,
#     strategy: str,
#     monkeypatch,
# ):
#     eplb_state = setup_eplb_state(enable_eplb)

#     monkeypatch.setenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", strategy)
#     router = create_fused_moe_router(
#         top_k=top_k,
#         global_num_experts=global_num_experts,
#         enable_eplb=enable_eplb,
#         eplb_state=eplb_state,
#     )

#     hidden_states, router_logits = make_test_data(m, k, global_num_experts)
#     topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)
