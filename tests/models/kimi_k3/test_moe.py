# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExperts
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.models.kimi_k3.xpu.linear import KimiMoE


@pytest.mark.parametrize("distribution", ["random", "skewed"])
def test_kimi_k3_grouped_topk_896_expert_routing(distribution: str) -> None:
    torch.manual_seed(17)
    num_tokens = 8
    num_experts = 896
    top_k = 16
    hidden_states = torch.randn(num_tokens, 64)
    correction_bias = torch.randn(num_experts) * 0.1
    if distribution == "random":
        router_logits = torch.randn(num_tokens, num_experts)
    else:
        base = torch.linspace(-8.0, 8.0, num_experts)
        router_logits = torch.stack(
            [base.roll(token * 37) for token in range(num_tokens)]
        )

    router = create_fused_moe_router(
        top_k=top_k,
        global_num_experts=num_experts,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=1.0,
        e_score_correction_bias=correction_bias,
    )

    topk_weights, topk_ids = router.select_experts(
        hidden_states,
        router_logits,
    )

    scores = router_logits.sigmoid()
    reference_ids = torch.topk(
        scores + correction_bias.unsqueeze(0),
        k=top_k,
        dim=-1,
        sorted=False,
    ).indices
    reference_weights = scores.gather(1, reference_ids)
    reference_weights /= reference_weights.sum(dim=-1, keepdim=True)

    assert isinstance(router, GroupedTopKRouter)
    assert router.routing_method_type == RoutingMethodType.DeepSeekV3
    torch.testing.assert_close(topk_ids.to(torch.int64), reference_ids)
    torch.testing.assert_close(topk_weights, reference_weights)


def test_xpu_experts_propagates_dynamic_expert_map() -> None:
    _, initial_map, _ = determine_expert_map(4, 0, 896)
    _, updated_map, _ = determine_expert_map(4, 1, 896)
    assert initial_map is not None
    assert updated_map is not None
    native_impl = MagicMock(expert_map=initial_map)
    experts = SimpleNamespace(
        fused_moe_impl=native_impl,
        moe_config=SimpleNamespace(num_local_experts=224),
    )
    tensor = torch.empty(0)

    apply_args = dict(
        output=tensor,
        hidden_states=tensor,
        w1=tensor,
        w2=tensor,
        topk_weights=tensor,
        topk_ids=torch.empty(0, dtype=torch.int32),
        activation=MoEActivation.SITU,
        global_num_experts=896,
        a1q_scale=None,
        a2_scale=None,
        workspace13=tensor,
        workspace2=tensor,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    XPUExperts.apply(experts, expert_map=None, **apply_args)
    assert native_impl.expert_map is initial_map

    XPUExperts.apply(experts, expert_map=updated_map, **apply_args)
    assert native_impl.expert_map is updated_map
    assert native_impl.apply.call_count == 2


@pytest.mark.parametrize(
    "expert_map,match",
    [
        (torch.zeros(896, dtype=torch.int64), "dtype torch.int32"),
        (torch.zeros(895, dtype=torch.int32), "must have shape"),
        (torch.zeros(896, 1, dtype=torch.int32), "must have shape"),
        (torch.full((896,), -2, dtype=torch.int32), "local expert IDs"),
        (torch.full((896,), 224, dtype=torch.int32), "local expert IDs"),
    ],
)
def test_xpu_experts_rejects_invalid_expert_map(
    expert_map: torch.Tensor,
    match: str,
) -> None:
    native_impl = MagicMock()
    experts = SimpleNamespace(
        fused_moe_impl=native_impl,
        moe_config=SimpleNamespace(num_local_experts=224),
    )
    tensor = torch.empty(0)

    with pytest.raises(ValueError, match=match):
        XPUExperts.apply(
            experts,
            output=tensor,
            hidden_states=tensor,
            w1=tensor,
            w2=tensor,
            topk_weights=tensor,
            topk_ids=torch.empty(0, dtype=torch.int32),
            activation=MoEActivation.SITU,
            global_num_experts=896,
            expert_map=expert_map,
            a1q_scale=None,
            a2_scale=None,
            workspace13=tensor,
            workspace2=tensor,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
    native_impl.apply.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_eplb": True},
        {"num_redundant_experts": 1},
    ],
)
def test_xpu_kimi_moe_rejects_unsupported_eplb(kwargs: dict[str, object]) -> None:
    with pytest.raises(NotImplementedError, match="does not yet support EPLB"):
        KimiMoE(SimpleNamespace(), **kwargs)
