# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia import model as deepseek_v4_model
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MoE


class _FakeFusedMoE:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _make_config(*, scoring_func: str, n_group: int | None, topk_group: int | None):
    config = SimpleNamespace(
        n_shared_experts=None,
        n_routed_experts=8,
        num_experts_per_tok=2,
        hidden_size=16,
        moe_intermediate_size=32,
        norm_topk_prob=True,
        swiglu_limit=None,
        n_group=n_group,
        topk_group=topk_group,
    )
    gate = SimpleNamespace(
        e_score_correction_bias=torch.zeros(config.n_routed_experts),
        tid2eid=None,
    )
    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.tp_size = 1
    moe.n_routed_experts = config.n_routed_experts
    moe.shared_experts = None
    moe.gate = gate
    moe.scoring_func = scoring_func
    moe.routed_scaling_factor = 2.5
    moe.swiglu_limit = None
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            enable_eplb=False,
        )
    )
    return moe, vllm_config, config


@pytest.mark.parametrize(
    ("n_group", "topk_group"),
    [
        (1, 1),
        (2, 1),
    ],
)
def test_deepseek_v4_sigmoid_preserves_grouped_routing(
    monkeypatch,
    n_group,
    topk_group,
):
    monkeypatch.setattr(deepseek_v4_model, "FusedMoE", _FakeFusedMoE)
    monkeypatch.setattr(deepseek_v4_model, "get_tensor_model_parallel_rank", lambda: 0)
    moe, vllm_config, config = _make_config(
        scoring_func="sigmoid",
        n_group=n_group,
        topk_group=topk_group,
    )

    moe._init_fused_moe_experts(
        vllm_config,
        config,
        quant_config=None,
        prefix="model.layers.0.mlp",
    )

    assert moe.experts.kwargs["use_grouped_topk"] is True
    assert moe.experts.kwargs["num_expert_group"] == n_group
    assert moe.experts.kwargs["topk_group"] == topk_group


@pytest.mark.parametrize(
    ("n_group", "topk_group"),
    [
        (None, 1),
        (1, None),
        (0, 1),
        (1, 0),
    ],
)
def test_deepseek_v4_sigmoid_clears_invalid_group_routing(
    monkeypatch,
    n_group,
    topk_group,
):
    monkeypatch.setattr(deepseek_v4_model, "FusedMoE", _FakeFusedMoE)
    monkeypatch.setattr(deepseek_v4_model, "get_tensor_model_parallel_rank", lambda: 0)
    moe, vllm_config, config = _make_config(
        scoring_func="sigmoid",
        n_group=n_group,
        topk_group=topk_group,
    )

    moe._init_fused_moe_experts(
        vllm_config,
        config,
        quant_config=None,
        prefix="model.layers.0.mlp",
    )

    assert moe.experts.kwargs["use_grouped_topk"] is False
    assert moe.experts.kwargs["num_expert_group"] is None
    assert moe.experts.kwargs["topk_group"] is None


@pytest.mark.parametrize("scoring_func", ["softmax", "sqrtsoftplus"])
def test_deepseek_v4_non_sigmoid_clears_single_group_routing(monkeypatch, scoring_func):
    monkeypatch.setattr(deepseek_v4_model, "FusedMoE", _FakeFusedMoE)
    monkeypatch.setattr(deepseek_v4_model, "get_tensor_model_parallel_rank", lambda: 0)
    moe, vllm_config, config = _make_config(
        scoring_func=scoring_func,
        n_group=1,
        topk_group=1,
    )

    moe._init_fused_moe_experts(
        vllm_config,
        config,
        quant_config=None,
        prefix="model.layers.0.mlp",
    )

    assert moe.experts.kwargs["use_grouped_topk"] is False
    assert moe.experts.kwargs["num_expert_group"] is None
    assert moe.experts.kwargs["topk_group"] is None
