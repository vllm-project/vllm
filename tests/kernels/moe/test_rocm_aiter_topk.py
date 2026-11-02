# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# This is a test for the AITER ops.
# It tests if the AITER ops are
# 1. correctly registered as custom ops
# 2. correctly defined the relationship between
#    implementation and fake function
# 3. can be used with torch.compile
# This file will be skipped if AITER is not installed
# and the platform is not ROCm.

import importlib.util

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)

# this import statement is needed to ensure the ops are registered
import vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe  # noqa: F401
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe
from vllm.model_executor.layers.fused_moe.router.aiter_shared_routed_fused_moe_router import (  # noqa: E501
    AiterSharedRoutedFusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)

# need to import once to ensure the ops are registered
# Check if aiter package is installed
aiter_available = importlib.util.find_spec("aiter") is not None

if not aiter_available:
    pytest.skip("These tests require AITER to run.", allow_module_level=True)


def test_rocm_aiter_biased_grouped_topk_custom_op_registration():
    """Test that the custom op is correctly registered."""
    # Check if the op exists in torch.ops.vllm
    assert hasattr(torch.ops.vllm, "rocm_aiter_biased_grouped_topk")

    # Check if the op is callable
    assert callable(torch.ops.vllm.rocm_aiter_biased_grouped_topk)


def test_rocm_aiter_grouped_topk_custom_op_registration():
    """Test that the custom op is correctly registered."""
    # Check if the op exists in torch.ops.vllm
    assert hasattr(torch.ops.vllm, "rocm_aiter_grouped_topk")

    # Check if the op is callable
    assert callable(torch.ops.vllm.rocm_aiter_grouped_topk)


def test_rocm_aiter_biased_grouped_topk_torch_compile_compatibility():
    """Test that the op can be used with torch.compile."""
    # Create test tensors
    token = 64
    expert = 256
    num_expert_group = 8
    topk = 8
    topk_group = 4
    renormalize = True
    scale_factor = 1.0

    gating_output = torch.randn((token, expert), dtype=torch.bfloat16, device="cuda")
    e_score_correction_bias = torch.randn(
        (expert,), dtype=torch.bfloat16, device="cuda"
    )

    device = gating_output.device
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)

    # Define a function that uses the op
    def biased_grouped_topk_fn(
        gating_output, e_score_correction_bias, topk_weights, topk_ids
    ):
        return torch.ops.vllm.rocm_aiter_biased_grouped_topk(
            gating_output,
            e_score_correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            scale_factor,
        )

    # Verify the op's fake implementation
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_biased_grouped_topk,
        (gating_output, e_score_correction_bias, topk_weights, topk_ids),
        kwargs={
            "num_expert_group": num_expert_group,
            "topk_group": topk_group,
            "need_renorm": renormalize,
            "routed_scaling_factor": scale_factor,
        },
        test_utils=("test_faketensor"),
    )

    # Compile the function with appropriate settings
    compiled_fn = torch.compile(
        biased_grouped_topk_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False,
    )

    topk_weights_original = torch.empty(
        (token, topk), dtype=torch.float32, device=device
    )
    topk_ids_original = torch.empty((token, topk), dtype=torch.int32, device=device)

    topk_weights_compiled = torch.empty(
        (token, topk), dtype=torch.float32, device=device
    )
    topk_ids_compiled = torch.empty((token, topk), dtype=torch.int32, device=device)

    # Run both compiled (V1 graph mode) and uncompiled versions (V1 eager mode)
    biased_grouped_topk_fn(
        gating_output, e_score_correction_bias, topk_weights_original, topk_ids_original
    )
    compiled_fn(
        gating_output, e_score_correction_bias, topk_weights_compiled, topk_ids_compiled
    )

    # Sort the results for comparison since the order might not be deterministic
    topk_ids_original, indices_original = torch.sort(topk_ids_original)
    topk_weights_original = torch.gather(topk_weights_original, 1, indices_original)

    topk_ids_compiled, indices_compiled = torch.sort(topk_ids_compiled)
    topk_weights_compiled = torch.gather(topk_weights_compiled, 1, indices_compiled)

    # Verify results match
    assert torch.allclose(
        topk_weights_original, topk_weights_compiled, rtol=1e-2, atol=1e-2
    )
    assert torch.allclose(topk_ids_original, topk_ids_compiled)


def test_rocm_aiter_grouped_topk_torch_compile_compatibility():
    """Test that the op can be used with torch.compile."""
    # Create test tensors
    token = 64
    expert = 256
    num_expert_group = 8
    topk = 8
    topk_group = 4
    renormalize = True
    scoring_func = "softmax"
    scale_factor = 1.0

    gating_output = torch.randn((token, expert), dtype=torch.bfloat16, device="cuda")

    device = gating_output.device
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)

    # Define a function that uses the op
    def grouped_topk_fn(gating_output, topk_weights, topk_ids, scoring_func):
        return torch.ops.vllm.rocm_aiter_grouped_topk(
            gating_output,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            scoring_func,
            scale_factor,
        )

    # Verify the op's fake implementation
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_grouped_topk,
        (gating_output, topk_weights, topk_ids),
        kwargs={
            "num_expert_group": num_expert_group,
            "topk_group": topk_group,
            "need_renorm": renormalize,
            "scoring_func": scoring_func,
            "routed_scaling_factor": scale_factor,
        },
        test_utils=("test_faketensor"),
    )

    # Compile the function with appropriate settings
    compiled_fn = torch.compile(
        grouped_topk_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False,
    )

    topk_weights_original = torch.empty(
        (token, topk), dtype=torch.float32, device=device
    )
    topk_ids_original = torch.empty((token, topk), dtype=torch.int32, device=device)

    topk_weights_compiled = torch.empty(
        (token, topk), dtype=torch.float32, device=device
    )
    topk_ids_compiled = torch.empty((token, topk), dtype=torch.int32, device=device)

    # Run both compiled (V1 graph mode) and uncompiled versions (V1 eager mode)
    grouped_topk_fn(
        gating_output, topk_weights_original, topk_ids_original, scoring_func
    )
    compiled_fn(gating_output, topk_weights_compiled, topk_ids_compiled, scoring_func)

    # Sort the results for comparison since the order might not be deterministic
    topk_ids_original, indices_original = torch.sort(topk_ids_original)
    topk_weights_original = torch.gather(topk_weights_original, 1, indices_original)

    topk_ids_compiled, indices_compiled = torch.sort(topk_ids_compiled)
    topk_weights_compiled = torch.gather(topk_weights_compiled, 1, indices_compiled)

    # Verify results match
    assert torch.allclose(
        topk_weights_original, topk_weights_compiled, rtol=1e-2, atol=1e-2
    )
    assert torch.allclose(topk_ids_original, topk_ids_compiled)


def test_rocm_aiter_topk_gating_custom_op_registration():
    assert callable(torch.ops.vllm.rocm_aiter_topk_gating)


def _sort_routing(weights: torch.Tensor, ids: torch.Tensor):
    ids_sorted, order = torch.sort(ids, dim=-1)
    weights_sorted = torch.gather(weights, 1, order)
    return weights_sorted, ids_sorted


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64])
@pytest.mark.parametrize("num_experts", [256, 512])
@pytest.mark.parametrize("topk", [8, 10])
@pytest.mark.parametrize("renormalize", [True, False])
def test_rocm_aiter_topk_gating_matches_softmax_reference(
    num_tokens: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
):
    """Numerical check vs softmax + topk, including Qwen3.8 (E=512, k=10)."""
    torch.manual_seed(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.bfloat16, device="cuda"
    )
    scores = torch.softmax(gating_output.float(), dim=-1)
    cutoff = torch.topk(scores, k=topk, dim=-1).values[:, -1:]

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    torch.ops.vllm.rocm_aiter_topk_gating(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        0,
        "",
    )

    # BF16 logits can tie at the top-k boundary, so a different tied expert is
    # valid. Check selection against the cutoff and weights against the scores
    # of the experts the kernel actually selected.
    assert ((topk_ids >= 0) & (topk_ids < num_experts)).all()
    sorted_ids = torch.sort(topk_ids, dim=-1).values
    assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()
    selected_scores = torch.gather(scores, 1, topk_ids.long())
    assert (selected_scores >= cutoff).all()
    expected_weights = selected_scores
    if renormalize:
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(topk_weights, expected_weights, atol=2e-2, rtol=2e-2)


def test_rocm_aiter_topk_gating_shared_expert_suffix():
    """Fused shared expert: softmax top-k on the prefix, sigmoid appended.

    Matches the Qwen3.8 FSE launch: 512 routed experts, top-10, one shared
    column, weights wide enough for the suffix, and the shared id prefilled.
    """
    torch.manual_seed(0)
    num_tokens, num_experts, topk, num_shared = 4, 512, 10, 1
    routed = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16, device="cuda")
    shared = torch.randn((num_tokens, num_shared), dtype=torch.bfloat16, device="cuda")
    shared[0, 0] = 40
    gating_output = torch.cat([routed, shared], dim=-1).contiguous()
    width = topk + num_shared
    shared_ids = torch.arange(
        num_experts, num_experts + num_shared, device="cuda", dtype=torch.int32
    )
    topk_ids = torch.full((num_tokens, width), -1, dtype=torch.int32, device="cuda")
    topk_ids[:, topk:] = shared_ids
    topk_weights = torch.empty((num_tokens, width), dtype=torch.float32, device="cuda")
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device="cuda"
    )

    torch.ops.vllm.rocm_aiter_topk_gating(
        topk_weights,
        topk_ids[:, :topk],
        token_expert_indices,
        gating_output,
        True,
        num_shared,
        "sigmoid",
    )

    scores = torch.softmax(routed.float(), dim=-1)
    cutoff = torch.topk(scores, k=topk, dim=-1).values[:, -1:]
    routed_ids = topk_ids[:, :topk]
    assert ((routed_ids >= 0) & (routed_ids < num_experts)).all()
    sorted_ids = torch.sort(routed_ids, dim=-1).values
    assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()
    selected = torch.gather(scores, 1, routed_ids.long())
    assert (selected >= cutoff).all()
    expected = selected / selected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(topk_weights[:, :topk], expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        topk_weights[:, topk:],
        torch.sigmoid(shared.float()),
        atol=2e-2,
        rtol=2e-2,
    )
    assert torch.equal(topk_ids[:, topk:], shared_ids.expand_as(topk_ids[:, topk:]))


def _assert_softmax_topk(
    weights: torch.Tensor,
    ids: torch.Tensor,
    logits: torch.Tensor,
    renormalize: bool,
):
    # BF16 logits can tie at the top-k boundary, so check the selection against
    # the cutoff and the weights against the scores of the selected experts.
    ids = ids.long()
    sorted_ids = ids.sort(dim=-1).values
    assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()
    scores = torch.softmax(logits.float(), dim=-1)
    cutoff = scores.topk(ids.shape[-1], dim=-1).values[:, -1:]
    selected = scores.gather(1, ids)
    assert (selected >= cutoff).all()
    if renormalize:
        selected = selected / selected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(weights, selected)


@pytest.fixture
def aiter_shared_expert_fusion(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_MOE",
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS",
    ):
        monkeypatch.setenv(name, "1")
    rocm_aiter_ops.refresh_env_variables()
    monkeypatch.setattr(rocm_aiter_moe, "aiter_topK_meta_data", None)
    rocm_aiter_moe.init_aiter_topK_meta_data.cache_clear()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()
    rocm_aiter_moe.init_aiter_topK_meta_data.cache_clear()


@pytest.mark.parametrize(
    ("num_experts", "topk", "num_shared", "renormalize", "shared_expert_weight"),
    [
        pytest.param(128, 4, 1, True, 1.0, id="mistral_large_4"),
        pytest.param(64, 6, 2, False, 0.5, id="deepseek_v2_lite_scaled"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_rocm_aiter_shared_routed_router_without_shared_expert_gate(
    aiter_shared_expert_fusion,
    num_experts: int,
    topk: int,
    num_shared: int,
    renormalize: bool,
    shared_expert_weight: float,
    dtype: torch.dtype,
):
    """Fused shared experts without a gate: every routed expert stays
    selectable, and the shared slots keep their ids and constant weight."""
    rocm_aiter_moe.init_aiter_topK_meta_data(
        n_routed_experts=num_experts,
        n_shared_experts=num_shared,
        top_k=topk,
        tp_rank=0,
        tp_size=1,
        max_num_tokens=64,
    )
    router = create_fused_moe_router(
        top_k=topk,
        global_num_experts=num_experts,
        renormalize=renormalize,
        num_fused_shared_experts=num_shared,
        shared_expert_weight=shared_expert_weight,
    )
    assert isinstance(router, AiterSharedRoutedFusedMoERouter)

    set_random_seed(0)
    num_tokens = 33
    logits = (torch.randn((num_tokens, num_experts), device="cuda") * 3).to(dtype)
    # The last routed experts must stay selectable.
    last_experts = range(num_experts - num_shared, num_experts)
    logits[0, -num_shared:] = 20.0
    hidden_states = torch.empty((num_tokens, 16), dtype=dtype, device="cuda")
    weights, ids = router.select_experts(hidden_states, logits)

    assert set(last_experts) <= set(ids[0, :topk].tolist())
    _assert_softmax_topk(weights[:, :topk], ids[:, :topk], logits, renormalize)
    shared_ids = torch.arange(
        num_experts, num_experts + num_shared, device="cuda", dtype=torch.int32
    )
    assert torch.equal(ids[:, topk:], shared_ids.expand(num_tokens, -1))
    assert (weights[:, topk:] == shared_expert_weight).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_rocm_aiter_shared_routed_router_with_shared_expert_gate(
    aiter_shared_expert_fusion, dtype: torch.dtype
):
    """Qwen3-Next: the shared-expert gate logits follow the routed logits, and
    the shared slot weight is their sigmoid."""
    num_experts, topk, num_shared = 512, 10, 1
    rocm_aiter_moe.init_aiter_topK_meta_data(
        n_routed_experts=num_experts,
        n_shared_experts=num_shared,
        top_k=topk,
        tp_rank=0,
        tp_size=1,
        max_num_tokens=64,
    )
    router = create_fused_moe_router(
        top_k=topk,
        global_num_experts=num_experts,
        renormalize=True,
        num_fused_shared_experts=num_shared,
        has_shared_expert_gate=True,
    )
    assert isinstance(router, AiterSharedRoutedFusedMoERouter)

    set_random_seed(0)
    num_tokens = 33
    logits = torch.randn((num_tokens, num_experts + num_shared), device="cuda") * 3
    logits = logits.to(dtype)
    hidden_states = torch.empty((num_tokens, 16), dtype=dtype, device="cuda")
    weights, ids = router.select_experts(hidden_states, logits)

    _assert_softmax_topk(
        weights[:, :topk], ids[:, :topk], logits[:, :num_experts], renormalize=True
    )
    assert (ids[:, topk:] == num_experts).all()
    torch.testing.assert_close(
        weights[:, topk:], torch.sigmoid(logits[:, num_experts:].float())
    )


def test_rocm_aiter_topk_gating_torch_compile_compatibility():
    token, expert, topk, renormalize = 4, 512, 10, True
    gating_output = torch.randn((token, expert), dtype=torch.bfloat16, device="cuda")
    device = gating_output.device

    def gating_fn(gating_output, topk_weights, topk_ids, token_expert_indices):
        return torch.ops.vllm.rocm_aiter_topk_gating(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            renormalize,
            0,
            "",
        )

    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_topk_gating,
        (
            torch.empty((token, topk), dtype=torch.float32, device=device),
            torch.empty((token, topk), dtype=torch.int32, device=device),
            torch.empty((token, topk), dtype=torch.int32, device=device),
            gating_output,
        ),
        kwargs={
            "renormalize": renormalize,
            "num_shared_experts": 0,
            "shared_expert_scoring_func": "",
        },
        test_utils=("test_faketensor"),
    )

    compiled_fn = torch.compile(
        gating_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False,
    )

    def alloc():
        return (
            torch.empty((token, topk), dtype=torch.float32, device=device),
            torch.empty((token, topk), dtype=torch.int32, device=device),
            torch.empty((token, topk), dtype=torch.int32, device=device),
        )

    w0, i0, t0 = alloc()
    w1, i1, t1 = alloc()
    gating_fn(gating_output, w0, i0, t0)
    compiled_fn(gating_output, w1, i1, t1)
    w0, i0 = _sort_routing(w0, i0)
    w1, i1 = _sort_routing(w1, i1)
    assert torch.allclose(w0, w1, rtol=1e-2, atol=1e-2)
    assert torch.equal(i0, i1)
