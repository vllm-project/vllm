# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE grouped topk kernel

Run `pytest tests/kernels/moe/test_grouped_topk.py`.
"""
import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import (fused_grouped_topk,
                                                            grouped_topk)
from vllm.platforms import current_platform


@pytest.mark.parametrize("n_token", [1, 33, 64, 100])
@pytest.mark.parametrize("n_hidden", [1024, 2048])
@pytest.mark.parametrize("n_expert", [16])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("num_expert_group", [8])
@pytest.mark.parametrize("topk_group", [2])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_grouped_topk(n_token: int, n_hidden: int, n_expert: int, topk: int,
                      renormalize: bool, num_expert_group: int,
                      topk_group: int, scoring_func: str, dtype: torch.dtype):
    current_platform.seed_everything(0)
    hidden_states = torch.randn((n_token, n_hidden),
                                dtype=dtype,
                                device="cuda")
    gating_output = torch.randn((n_token, n_expert),
                                dtype=dtype,
                                device="cuda")
    e_score_correction_bias = torch.randn((n_expert, ),
                                          dtype=torch.float32,
                                          device="cuda")

    baseline_topk_weights, baseline_topk_ids = grouped_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias)

    test_topk_weights, test_topk_ids = fused_grouped_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias)

    if renormalize:
        torch.testing.assert_close(baseline_topk_weights,
                                   test_topk_weights,
                                   atol=2e-2,
                                   rtol=0)
    torch.testing.assert_close(baseline_topk_ids,
                               test_topk_ids,
                               atol=0,
                               rtol=0)
