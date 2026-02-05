# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE fused topk kernel

Run `pytest tests/kernels/moe/test_fused_topk.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.platforms import current_platform


def torch_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output.float(), dim=-1)
    else:
        assert scoring_func == "sigmoid"
        scores = torch.sigmoid(gating_output.float())

    if e_score_correction_bias is not None:
        num_experts = gating_output.shape[-1]
        scores_for_choice = scores.view(
            -1, num_experts
        ) + e_score_correction_bias.unsqueeze(0)
        _, topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1)
        topk_weights = scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_tokens", [1, 33, 56])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [6, 16])
@pytest.mark.parametrize("topk", [3, 4])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

    topk_weights_ref, topk_ids_ref = torch_topk(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )

    topk_weights, topk_ids, _ = fused_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )

    torch.testing.assert_close(
        topk_weights_ref.to(torch.float32), topk_weights, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(topk_ids_ref.to(torch.int32), topk_ids, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_tokens", [1, 33, 56])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [6, 16])
@pytest.mark.parametrize("topk", [3, 4])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_bias(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device="cuda"
    )

    topk_weights_ref, topk_ids_ref = torch_topk(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        e_score_correction_bias=e_score_correction_bias,
        scoring_func=scoring_func,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )

    torch.testing.assert_close(
        topk_weights_ref.to(torch.float32), topk_weights, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(topk_ids_ref.to(torch.int32), topk_ids, atol=0, rtol=0)
