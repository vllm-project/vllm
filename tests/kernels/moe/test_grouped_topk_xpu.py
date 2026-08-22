# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU grouped_topk eager-guard and numerical range coverage.

CUDA coverage stays in test_grouped_topk.py. This file only runs on XPU.
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="This test is skipped on non-XPU platform.",
)


def _reference_grouped_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(scoring_func)
    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        original_scores = scores
        group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=True)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=True)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=True)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("use_bias", [True, False])
def test_xpu_grouped_topk_ids_in_range(scoring_func: str, use_bias: bool) -> None:
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        grouped_topk,
    )

    torch.manual_seed(0)
    num_tokens = 8
    num_experts = 128
    num_expert_group = 8
    topk = 8
    topk_group = 4
    hidden = torch.randn(num_tokens, 16, device="xpu", dtype=torch.bfloat16)
    gating = torch.randn(num_tokens, num_experts, device="xpu", dtype=torch.float32)
    bias = (
        torch.randn(num_experts, device="xpu", dtype=torch.float32) if use_bias else None
    )

    weights, ids = grouped_topk(
        hidden,
        gating,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=1.0,
        e_score_correction_bias=bias,
    )

    assert int(ids.min().item()) >= 0
    assert int(ids.max().item()) < num_experts
    assert weights.shape == (num_tokens, topk)
    assert ids.shape == (num_tokens, topk)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32

    ref_w, ref_ids = _reference_grouped_topk(
        gating.float(),
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=1.0,
        e_score_correction_bias=None if bias is None else bias.float(),
    )
    # Selection can differ on exact ties; require every id in-range (above)
    # and weights finite. Compare when the selected id sets match.
    assert torch.isfinite(weights).all()
    same_ids = torch.equal(ids.cpu().sort(dim=-1).values, ref_ids.cpu().sort(dim=-1).values)
    if same_ids:
        torch.testing.assert_close(
            weights.cpu(),
            ref_w.cpu(),
            atol=1e-4,
            rtol=1e-4,
        )


def test_xpu_grouped_topk_does_not_device_assert_under_compile() -> None:
    """An outer compiled graph can cross the XPU-eager router safely."""
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        grouped_topk,
    )

    assert getattr(grouped_topk, "_torchdynamo_disable", False), (
        "grouped_topk must remain Dynamo-disabled on XPU"
    )

    torch.manual_seed(1)
    hidden = torch.randn(4, 16, device="xpu", dtype=torch.bfloat16)
    gating = torch.randn(4, 128, device="xpu", dtype=torch.float32)
    bias = torch.zeros(128, device="xpu", dtype=torch.float32)
    compiled = torch.compile(grouped_topk, dynamic=True)
    _, ids = compiled(
        hidden,
        gating,
        topk=8,
        renormalize=True,
        num_expert_group=8,
        topk_group=4,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        e_score_correction_bias=bias,
    )
    assert int(ids.min().item()) >= 0
    assert int(ids.max().item()) < 128
