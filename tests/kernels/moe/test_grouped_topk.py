# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE grouped topk kernel

Run `pytest tests/kernels/moe/test_grouped_topk.py`.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.config import (
    CompilationConfig,
    VllmConfig,
    get_cached_compilation_config,
    set_current_vllm_config,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopk,
    fused_grouped_topk,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("n_token", [1, 33, 64])
@pytest.mark.parametrize("n_hidden", [1024, 2048])
@pytest.mark.parametrize(
    "n_expert,topk,num_expert_group,topk_group",
    [
        (16, 2, 8, 2),
        (128, 2, 8, 2),
        (256, 8, 8, 4),
        (384, 8, 1, 1),
        (512, 22, 1, 1),
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [torch.float32])
def test_grouped_topk(
    monkeypatch: pytest.MonkeyPatch,
    n_token: int,
    n_hidden: int,
    n_expert: int,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    scoring_func: str,
    routed_scaling_factor: float,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype,
):
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all", "+grouped_topk"])
    )
    get_cached_compilation_config.cache_clear()

    set_random_seed(0)
    hidden_states = torch.randn((n_token, n_hidden), dtype=input_dtype, device="cuda")
    gating_output = torch.randn((n_token, n_expert), dtype=input_dtype, device="cuda")
    e_score_correction_bias = torch.randn((n_expert,), dtype=bias_dtype, device="cuda")

    with set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        m.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
        m.setattr(envs, "VLLM_BATCH_INVARIANT", True)
        grouped_topk = GroupedTopk(
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )
        assert grouped_topk._forward_method.__name__ == "forward_cuda"
        baseline_topk_weights, baseline_topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            e_score_correction_bias=e_score_correction_bias,
        )

        test_topk_weights, test_topk_ids = fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        torch.testing.assert_close(
            baseline_topk_weights, test_topk_weights, atol=2e-2, rtol=0
        )
        torch.testing.assert_close(baseline_topk_ids, test_topk_ids, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("n_token", [1, 25, 128])
@pytest.mark.parametrize("n_hidden", [7168])
@pytest.mark.parametrize(
    "n_expert,topk,num_expert_group,topk_group",
    [
        # Mistral Large 3 shape: multi-group max scoring
        (64, 8, 8, 4),
        # single-group (plain top-k, no group structure)
        (64, 8, 1, 1),
        # larger expert count, multi-group
        (256, 8, 8, 4),
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_grouped_topk_no_bias(
    monkeypatch: pytest.MonkeyPatch,
    n_token: int,
    n_hidden: int,
    n_expert: int,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    routed_scaling_factor: float,
    input_dtype: torch.dtype,
):
    """Fused path with e_score_correction_bias=None must match the Python
    fallback (torch.compile grouped_topk) exactly on expert IDs and closely
    on renormalized weights.

    This covers the case where VLLM_USE_FUSED_MOE_GROUPED_TOPK was previously
    gated out for bias-free softmax models (e.g. Mistral Large 3).  The fix
    adds a max-per-group scoring mode to the C++ kernel so the fused path
    produces identical routing decisions to the Python path.
    """
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all", "+grouped_topk"])
    )
    get_cached_compilation_config.cache_clear()

    set_random_seed(0)
    hidden_states = torch.randn((n_token, n_hidden), dtype=input_dtype, device="cuda")
    gating_output = torch.randn((n_token, n_expert), dtype=input_dtype, device="cuda")

    with set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        # Reference: Python fallback path (no fused kernel)
        m.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
        m.setattr(envs, "VLLM_BATCH_INVARIANT", True)
        grouped_topk_op = GroupedTopk(
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func="softmax",
            routed_scaling_factor=routed_scaling_factor,
        )
        ref_weights, ref_ids = grouped_topk_op(
            hidden_states=hidden_states,
            gating_output=gating_output,
            e_score_correction_bias=None,
        )

    # Fused path: calls the C++ kernel with group_scoring_func=1 (max per group)
    fused_weights, fused_ids = fused_grouped_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func="softmax",
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=None,
    )

    # Expert IDs must match exactly (routing decisions identical)
    ref_ids_sorted = ref_ids.sort(dim=-1).values
    fused_ids_sorted = fused_ids.sort(dim=-1).values
    assert torch.all(ref_ids_sorted == fused_ids_sorted), (
        f"Expert ID mismatch: fused path selected different experts than the "
        f"Python fallback.  Match rate: "
        f"{(ref_ids_sorted == fused_ids_sorted).float().mean():.2%}"
    )

    # Weights should be close (float32 renorm rounding only)
    torch.testing.assert_close(ref_weights, fused_weights, atol=1e-5, rtol=0)
