# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE grouped topk kernel.

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
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopk,
    GroupedTopKRouter,
    fused_grouped_topk,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _run_single_group_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    *,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fused_grouped_topk(
        hidden_states=torch.empty(
            (logits.shape[0], 0), dtype=logits.dtype, device=logits.device
        ),
        gating_output=logits,
        topk=topk,
        renormalize=renormalize,
        e_score_correction_bias=bias,
        num_expert_group=1,
        topk_group=1,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
    )


def _single_group_reference(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    *,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func == "sigmoid":
        scores = 0.5 * torch.tanh(0.5 * logits.float()) + 0.5
    else:
        scores = torch.softmax(logits, dim=-1).float()
    indices = torch.argsort(
        scores + bias.float(), dim=-1, descending=True, stable=True
    )[:, :topk]
    values = scores.gather(1, indices)
    if renormalize:
        values /= values.sum(dim=-1, keepdim=True) + 1e-20
    values *= routed_scaling_factor
    return values, indices.to(torch.int32)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("skip_padding", [False, True])
@pytest.mark.parametrize("use_fused", [False, True])
def test_kimi_grouped_topk_masks_padding_when_enabled(
    skip_padding: bool, use_fused: bool, monkeypatch
):
    set_random_seed(0)
    monkeypatch.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", str(int(use_fused)))
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    hidden_states = torch.randn((2, 16), device="cuda")
    gating_output = torch.randn((2, 896), device="cuda")
    is_padding = torch.tensor([False, True], device="cuda")
    router = GroupedTopKRouter(
        top_k=16,
        global_num_experts=896,
        num_expert_group=1,
        topk_group=1,
        skip_padding=skip_padding,
        e_score_correction_bias=torch.randn((896,), device="cuda"),
    )

    baseline_weights, baseline_ids = router._compute_routing(
        hidden_states, gating_output, torch.int32
    )
    with set_forward_context(None, VllmConfig(), is_padding=is_padding):
        topk_weights, topk_ids = router._compute_routing(
            hidden_states, gating_output, torch.int32
        )

    torch.testing.assert_close(topk_weights[0], baseline_weights[0], rtol=0, atol=0)
    assert torch.equal(topk_ids[0], baseline_ids[0])
    if skip_padding:
        assert torch.all(topk_ids[1] == -1)
        assert torch.all(topk_weights[1] == 0)
    else:
        assert torch.all(topk_ids[1] >= 0)
        assert torch.all(topk_weights[1] > 0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize(
    "num_experts,num_groups,topk_group,topk",
    [(128, 1, 1, 8), (896, 1, 1, 16), (256, 8, 4, 8), (512, 8, 4, 8)],
)
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
def test_grouped_topk_padding_changes_on_graph_replay(
    num_experts: int, num_groups: int, topk_group: int, topk: int, scoring_func: str
):
    """Every kernel family must read the current mask on each graph replay."""
    set_random_seed(0)
    logits = torch.randn((33, num_experts), device="cuda")
    bias = torch.randn((num_experts,), device="cuda")
    hidden = torch.empty((33, 0), device="cuda")
    padding = torch.zeros(33, dtype=torch.bool, device="cuda")

    def route(mask):
        return fused_grouped_topk(
            hidden,
            logits,
            topk,
            True,
            bias,
            num_groups,
            topk_group,
            scoring_func,
            2.5,
            is_padding=mask,
        )

    baseline_weights, baseline_ids = route(None)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        route(padding)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        weights, ids = route(padding)
    for pattern in (
        torch.arange(33, device="cuda") % 2 == 0,
        torch.ones_like(padding),
        torch.zeros_like(padding),
    ):
        padding.copy_(pattern)
        graph.replay()
        torch.testing.assert_close(
            weights, baseline_weights.masked_fill(pattern[:, None], 0), rtol=0, atol=0
        )
        assert torch.equal(ids, baseline_ids.masked_fill(pattern[:, None], -1))


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
def test_grouped_topk_padding_opcheck():
    torch.library.opcheck(
        torch.ops._moe_C.grouped_topk,
        (
            torch.randn((4, 256), device="cuda"),
            8,
            4,
            8,
            True,
            2.5,
            torch.randn(256, device="cuda"),
            1,
            torch.tensor([False, True, False, True], device="cuda"),
        ),
    )


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
def test_grouped_topk_single_group_large_batch():
    set_random_seed(0)
    logits = torch.randn((1536, 896), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((896,), dtype=torch.float32, device="cuda")

    expected_values, expected_ids = _single_group_reference(
        logits, bias, 16, scoring_func="sigmoid", renormalize=True
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits, bias, 16, scoring_func="sigmoid", renormalize=True
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize(
    "num_experts,topk,input_dtype,bias_dtype",
    [
        (512, 9, torch.bfloat16, torch.float32),
        (512, 16, torch.float16, torch.float16),
        (513, 9, torch.float32, torch.bfloat16),
        (513, 16, torch.bfloat16, torch.float32),
        (895, 9, torch.float16, torch.bfloat16),
        (896, 16, torch.float32, torch.float16),
        (897, 9, torch.bfloat16, torch.bfloat16),
        (897, 16, torch.float16, torch.float32),
        (1024, 9, torch.float32, torch.bfloat16),
        (1024, 16, torch.bfloat16, torch.float16),
    ],
)
@pytest.mark.parametrize(
    "scoring_func,renormalize,routed_scaling_factor",
    [
        ("sigmoid", True, 1.0),
        ("sigmoid", False, 2.5),
        ("softmax", True, 2.5),
        ("softmax", False, 1.0),
    ],
)
def test_grouped_topk_single_group_tiers(
    num_experts: int,
    topk: int,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float,
):
    set_random_seed(7)
    logits = torch.randn((17, num_experts), dtype=input_dtype, device="cuda")
    bias = torch.randn((num_experts,), dtype=bias_dtype, device="cuda")

    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize(
    "num_experts,topk,scoring_func",
    [
        (128, 8, "sigmoid"),
        (129, 8, "sigmoid"),
        (257, 8, "sigmoid"),
        (385, 8, "sigmoid"),
        (512, 9, "sigmoid"),
        (513, 9, "sigmoid"),
        (769, 9, "sigmoid"),
        (897, 16, "sigmoid"),
        (1024, 16, "sigmoid"),
        (128, 4, "softmax"),
        (128, 5, "softmax"),
        (129, 8, "softmax"),
        (161, 8, "softmax"),
        (256, 9, "softmax"),
        (257, 8, "softmax"),
        (512, 9, "softmax"),
        (512, 17, "softmax"),
        (512, 23, "softmax"),
        (513, 8, "softmax"),
        (577, 9, "softmax"),
        (769, 9, "softmax"),
        (897, 9, "softmax"),
        (1024, 16, "softmax"),
    ],
)
def test_grouped_topk_single_group_capacity_tiers(
    num_experts: int,
    topk: int,
    scoring_func: str,
):
    set_random_seed(11)
    logits = torch.randn((3, num_experts), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda")
    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=True,
        routed_scaling_factor=2.5,
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_experts", [512, 896, 1024])
def test_grouped_topk_single_group_stable_ties(num_experts: int):
    logits = torch.zeros((1, num_experts), dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros((num_experts,), dtype=torch.float32, device="cuda")

    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        16,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
    )

    expected_ids = torch.arange(16, dtype=torch.int32, device="cuda")[None]
    expected_values = torch.full((1, 16), 2.5 / 16, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_experts", [512, 896, 1024])
@pytest.mark.parametrize("num_finite", [0, 15])
@pytest.mark.parametrize("renormalize", [False, True])
def test_grouped_topk_single_group_nonfinite_scores(
    num_experts: int, num_finite: int, renormalize: bool
):
    logits = torch.full(
        (1, num_experts), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    if num_finite:
        logits[0, :num_finite] = torch.arange(
            num_finite, dtype=torch.bfloat16, device="cuda"
        )
    logits[0, num_finite] = torch.inf
    logits[0, num_finite + 1] = -torch.inf
    bias = torch.zeros((num_experts,), dtype=torch.float32, device="cuda")

    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        16,
        scoring_func="sigmoid",
        renormalize=renormalize,
        routed_scaling_factor=2.5,
    )

    if num_finite == 0:
        expected_ids = torch.arange(16, dtype=torch.int32, device="cuda")[None]
        if renormalize:
            expected_values = torch.full(
                (1, 16), 1 / 16, dtype=torch.float32, device="cuda"
            )
        else:
            expected_values = torch.zeros((1, 16), dtype=torch.float32, device="cuda")
    else:
        expected_ids = torch.cat(
            (
                torch.arange(num_finite - 1, -1, -1, dtype=torch.int32, device="cuda"),
                torch.tensor([num_finite], dtype=torch.int32, device="cuda"),
            )
        )[None]
        finite_values = logits[0, :num_finite].float().sigmoid().flip(0)
        if renormalize:
            finite_values /= finite_values.sum()
        finite_values *= 2.5
        expected_values = torch.cat(
            (finite_values, torch.zeros(1, dtype=torch.float32, device="cuda"))
        )[None]

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)
