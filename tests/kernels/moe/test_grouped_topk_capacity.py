# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the grouped-topk capacity contract.

Run `pytest tests/kernels/moe/test_grouped_topk_capacity.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.cpu_router import _grouped_topk
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    grouped_topk,
)

pytestmark = pytest.mark.cpu_test

_IMPLS = ["grouped_topk", "cpu_grouped_topk"]
_LOGITS = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])


def _run(impl, logits, topk, num_expert_group, topk_group, bias):
    if impl == "grouped_topk":
        return grouped_topk(
            hidden_states=torch.empty((logits.shape[0], 0)),
            gating_output=logits,
            topk=topk,
            renormalize=True,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func="sigmoid",
            e_score_correction_bias=bias,
        )
    return _grouped_topk(
        gating_output=logits,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        e_score_correction_bias=bias,
    )


@pytest.mark.parametrize("impl", _IMPLS)
@pytest.mark.parametrize("bias", [None, torch.zeros(8)], ids=["no-bias", "with-bias"])
def test_grouped_topk_rejects_topk_exceeding_selected_groups(impl, bias):
    """A topk beyond the unmasked selected-group experts must raise."""
    with pytest.raises(ValueError, match="exceeds"):
        _run(impl, _LOGITS, topk=4, num_expert_group=4, topk_group=1, bias=bias)


@pytest.mark.parametrize("impl", _IMPLS)
def test_grouped_topk_allows_topk_at_capacity(impl):
    """A topk equal to the eligible count (2 groups x 2 experts) is allowed."""
    weights, _ = _run(
        impl, _LOGITS, topk=4, num_expert_group=4, topk_group=2, bias=None
    )
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))


@pytest.mark.parametrize("impl", _IMPLS)
def test_grouped_topk_valid_config_keeps_group_constraint(impl):
    """A DeepSeek-V3-shaped config must select only within its chosen groups."""
    torch.manual_seed(0)
    num_experts, num_groups, topk_group, topk = 256, 8, 4, 8
    logits = torch.randn(3, num_experts)
    weights, ids = _run(
        impl,
        logits,
        topk=topk,
        num_expert_group=num_groups,
        topk_group=topk_group,
        bias=None,
    )

    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3))
    experts_per_group = num_experts // num_groups
    selected_groups = torch.topk(
        logits.sigmoid().view(3, num_groups, experts_per_group).max(dim=-1).values,
        k=topk_group,
        dim=-1,
    )[1]
    id_groups = ids.long() // experts_per_group
    for row in range(3):
        assert all(
            int(group) in selected_groups[row].tolist() for group in id_groups[row]
        )
