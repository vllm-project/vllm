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
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.router import (
    grouped_topk_router as grouped_topk_module,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopk,
    GroupedTopKRouter,
    fused_grouped_topk,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def test_grouped_topk_masks_padding(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(
        grouped_topk_module.rocm_aiter_ops,
        "is_fused_moe_enabled",
        lambda: False,
    )
    weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    ids = torch.arange(10, dtype=torch.int32).reshape(5, 2)
    expected_ids = ids.clone()
    is_padding = torch.tensor([False, False, False, True, True])
    monkeypatch.setattr(
        grouped_topk_module,
        "grouped_topk",
        lambda **_: (weights, ids),
    )
    router = GroupedTopKRouter(
        top_k=2,
        global_num_experts=4,
        num_expert_group=2,
        topk_group=1,
    )

    with set_forward_context(None, VllmConfig(), is_padding=is_padding):
        masked_weights, masked_ids = router._compute_routing(
            hidden_states=torch.zeros(5, 2),
            router_logits=torch.zeros(5, 4),
            indices_type=None,
        )

    torch.testing.assert_close(masked_weights, weights)
    assert masked_ids is ids
    torch.testing.assert_close(masked_ids[:3], expected_ids[:3])
    assert torch.all(masked_ids[3:] == -1)


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
