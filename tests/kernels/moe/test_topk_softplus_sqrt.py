# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe import layer as moe_layer
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
    fused_topk_bias,
)
from vllm.platforms import current_platform


def _torch_topk_softplus_sqrt(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
):
    scores = F.softplus(gating_output.float()).sqrt()
    original_scores = scores
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores

    if hash_indices_table is not None:
        assert input_ids is not None
        topk_ids = hash_indices_table[input_ids.long()]
    else:
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=True)[1]

    topk_weights = original_scores.gather(1, topk_ids.long())
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def test_sqrtsoftplus_bias_uses_deepseek_v4_routing_method():
    assert (
        get_routing_method_type(
            scoring_func="sqrtsoftplus",
            top_k=8,
            renormalize=True,
            num_expert_group=None,
            has_e_score_bias=True,
        )
        == RoutingMethodType.DeepseekV4
    )
    assert (
        get_routing_method_type(
            scoring_func="sqrtsoftplus",
            top_k=8,
            renormalize=False,
            num_expert_group=None,
            has_e_score_bias=True,
        )
        == RoutingMethodType.Unspecified
    )


def test_deepseek_v4_fused_shared_expert_is_appended_after_routing(monkeypatch):
    routed_weights = torch.tensor(
        [
            [0.25, 0.50, 0.25, 0.50, 0.50, 0.50],
            [0.125, 0.375, 0.25, 0.75, 0.50, 0.50],
        ],
        dtype=torch.float32,
    )
    routed_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]],
        dtype=torch.int32,
    )

    def fake_fused_topk_bias(**kwargs):
        return routed_weights.clone(), routed_ids.clone()

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.router."
        "fused_topk_bias_router.fused_topk_bias",
        fake_fused_topk_bias,
    )
    router = FusedTopKBiasRouter(
        top_k=6,
        global_num_experts=384,
        e_score_correction_bias=torch.zeros(384),
        renormalize=True,
        routed_scaling_factor=2.5,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=1,
        shared_expert_weight=1.0,
    )

    weights, ids = router.select_experts(
        torch.empty(2, 8),
        torch.empty(2, 384),
        torch.int32,
    )

    assert weights.shape == ids.shape == (2, 7)
    torch.testing.assert_close(weights[:, :6], routed_weights, rtol=0, atol=0)
    torch.testing.assert_close(ids[:, :6], routed_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        weights[:, :6].sum(-1), torch.full((2,), 2.5), rtol=0, atol=0
    )
    torch.testing.assert_close(weights[:, 6], torch.ones(2), rtol=0, atol=0)
    torch.testing.assert_close(
        ids[:, 6],
        torch.full((2,), 384, dtype=torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("aiter_enabled", "env_enabled", "is_act_and_mul"),
    [
        (aiter_enabled, env_enabled, is_act_and_mul)
        for aiter_enabled in (False, True)
        for env_enabled in (False, True)
        for is_act_and_mul in (False, True)
    ],
)
def test_shared_expert_count_preserves_aiter_and_backend_neutral_gates(
    monkeypatch, aiter_enabled, env_enabled, is_act_and_mul
):
    monkeypatch.setattr(
        moe_layer.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: aiter_enabled,
    )
    monkeypatch.setattr(
        moe_layer.envs,
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS",
        env_enabled,
    )
    expected_fused_experts = int((aiter_enabled or env_enabled) and is_act_and_mul)
    assert moe_layer.determine_expert_counts(384, 0, 1, is_act_and_mul) == (
        384,
        384,
        expected_fused_experts,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("num_tokens", [1, 33, 128])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [128, 256, 384, 512])
@pytest.mark.parametrize("topk", [6, 8, 16])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 1.5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_softplus_sqrt(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device="cuda"
    )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )

    # Different kernels may return the topk experts in different orders when
    # scores tie; sort by expert id before comparing.
    sorted_ref_ids, idx_ref = topk_ids_ref.sort(dim=-1)
    sorted_ids, idx_ops = topk_ids.sort(dim=-1)
    torch.testing.assert_close(sorted_ref_ids, sorted_ids, atol=0, rtol=0)

    sorted_w_ref = topk_weights_ref.gather(1, idx_ref)
    sorted_w = topk_weights.gather(1, idx_ops)
    torch.testing.assert_close(sorted_w_ref, sorted_w, atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("num_tokens", [1, 33, 128])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [256, 384, 512])
@pytest.mark.parametrize("topk", [6, 8, 16])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_softplus_sqrt_hash(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    vocab_size = 1024
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    # Per-token fixed expert selection: for each vocab id pick `topk` distinct
    # experts.
    hash_indices_table = torch.stack(
        [torch.randperm(num_experts)[:topk] for _ in range(vocab_size)]
    ).to(device="cuda", dtype=torch.long)
    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.long, device="cuda"
    )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        input_ids=input_ids,
        hash_indices_table=hash_indices_table,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=None,
        topk=topk,
        renormalize=renormalize,
        input_tokens=input_ids,
        hash_indices_table=hash_indices_table,
        routed_scaling_factor=routed_scaling_factor,
    )

    sorted_ref_ids, idx_ref = topk_ids_ref.sort(dim=-1)
    sorted_ids, idx_ops = topk_ids.sort(dim=-1)
    torch.testing.assert_close(sorted_ref_ids, sorted_ids, atol=0, rtol=0)

    sorted_w_ref = topk_weights_ref.gather(1, idx_ref)
    sorted_w = topk_weights.gather(1, idx_ops)
    torch.testing.assert_close(sorted_w_ref, sorted_w, atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="The DeepSeek V4 fast path is CUDA-only.",
)
@pytest.mark.parametrize(
    ("num_tokens", "num_experts", "indices_type"),
    [
        (0, 256, torch.uint32),
        (17, 256, torch.uint32),
        (17, 384, torch.int64),
    ],
)
def test_dsv4_fast_topk(
    num_tokens: int,
    num_experts: int,
    indices_type: torch.dtype,
):
    torch.manual_seed(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=6,
        renormalize=True,
        routed_scaling_factor=1.5,
        e_score_correction_bias=correction_bias,
    )
    topk_weights, topk_ids = dsv4_topk(
        gating_output, correction_bias, indices_type, 1.5
    )

    assert topk_ids.dtype == indices_type
    torch.testing.assert_close(topk_ids_ref.to(indices_type), topk_ids, atol=0, rtol=0)
    torch.testing.assert_close(
        topk_weights_ref,
        topk_weights,
        atol=2e-5,
        rtol=2e-5,
    )
