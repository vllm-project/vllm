# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
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


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("use_hash", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("use_padding_mask", [False, True])
@pytest.mark.parametrize("pad_with_nan", [False, True])
@pytest.mark.parametrize("num_experts", [128, 256, 384])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_softplus_sqrt_padding(
    use_hash: bool,
    use_bias: bool,
    use_padding_mask: bool,
    pad_with_nan: bool,
    num_experts: int,
    dtype: torch.dtype,
):
    """Verify explicit padding and NaN-padded rows do not affect real rows."""
    torch.manual_seed(0)
    num_tokens = 8
    topk = 6
    indices_dtype = torch.int32

    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

    padding_rows = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    padding_rows[1::2] = True
    if pad_with_nan:
        gating_output[padding_rows] = float("nan")
    is_padding = padding_rows if use_padding_mask else None

    # A negative correction bias makes explicit pad rows look selectable unless
    # the kernel uses the is_padding guard.
    e_score_correction_bias = None
    if use_bias:
        e_score_correction_bias = (
            -torch.rand((num_experts,), dtype=torch.float32, device="cuda") - 1.0
        )

    input_ids = None
    hash_indices_table = None
    if use_hash:
        vocab_size = 64
        hash_indices_table = torch.stack(
            [torch.randperm(num_experts)[:topk] for _ in range(vocab_size)]
        ).to(device="cuda", dtype=indices_dtype)
        input_ids = torch.randint(
            0, vocab_size, (num_tokens,), dtype=indices_dtype, device="cuda"
        )

    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_ids = torch.empty(num_tokens, topk, dtype=indices_dtype, device="cuda")
    token_expert_indices = torch.empty(
        num_tokens, topk, dtype=torch.int32, device="cuda"
    )

    ops.topk_hash_softplus_sqrt(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize=True,
        routed_scaling_factor=1.0,
        e_score_correction_bias=e_score_correction_bias,
        input_tokens=input_ids,
        hash_indices_table=hash_indices_table,
        is_padding=is_padding,
    )

    if use_padding_mask:
        pad_ids = topk_ids[padding_rows]
        pad_weights = topk_weights[padding_rows]
        assert torch.equal(pad_ids, torch.full_like(pad_ids, -1)), (
            f"Explicit pad rows should contain only -1 ids, got {pad_ids.tolist()}"
        )
        assert (pad_weights == 0).all(), (
            "Explicit pad rows should have all-zero weights, "
            f"got {pad_weights.tolist()}"
        )

    if pad_with_nan:
        nan_pad_weights = topk_weights[padding_rows]
        assert torch.isfinite(nan_pad_weights).all(), (
            f"NaN-padded rows have non-finite weights, got {nan_pad_weights.tolist()}"
        )
        assert (nan_pad_weights == 0).all(), (
            "NaN-padded rows should have all-zero weights, "
            f"got {nan_pad_weights.tolist()}"
        )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=True,
        routed_scaling_factor=1.0,
        e_score_correction_bias=e_score_correction_bias,
        input_ids=input_ids,
        hash_indices_table=hash_indices_table,
    )

    rows_to_compare = torch.ones(num_tokens, dtype=torch.bool, device="cuda")
    if use_padding_mask or pad_with_nan:
        rows_to_compare = ~padding_rows

    sorted_ref_ids, idx_ref = topk_ids_ref[rows_to_compare].sort(dim=-1)
    sorted_ids, idx_ops = topk_ids[rows_to_compare].sort(dim=-1)
    torch.testing.assert_close(
        sorted_ref_ids, sorted_ids.to(sorted_ref_ids.dtype), atol=0, rtol=0
    )

    sorted_w_ref = topk_weights_ref[rows_to_compare].gather(1, idx_ref)
    sorted_w = topk_weights[rows_to_compare].gather(1, idx_ops)
    torch.testing.assert_close(sorted_w_ref, sorted_w, atol=2e-2, rtol=1e-2)
