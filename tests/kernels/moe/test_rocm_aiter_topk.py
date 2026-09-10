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

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)

# this import statement is needed to ensure the ops are registered
import vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe  # noqa: F401

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
    assert hasattr(torch.ops.vllm, "rocm_aiter_topk_gating")
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
    from vllm._aiter_ops import rocm_aiter_ops

    if not rocm_aiter_ops.topk_gating_available():
        pytest.skip("aiter.ops.topk.topk_gating is not available")

    torch.manual_seed(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.bfloat16, device="cuda"
    )
    scores = torch.softmax(gating_output.float(), dim=-1)
    ref_weights, ref_ids = torch.topk(scores, k=topk, dim=-1)
    if renormalize:
        ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)

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

    got_w, got_ids = _sort_routing(topk_weights, topk_ids)
    ref_w, ref_ids_sorted = _sort_routing(ref_weights, ref_ids.to(torch.int32))
    assert torch.equal(got_ids, ref_ids_sorted)
    torch.testing.assert_close(got_w, ref_w, atol=2e-2, rtol=2e-2)


def test_rocm_aiter_topk_gating_torch_compile_compatibility():
    from vllm._aiter_ops import rocm_aiter_ops

    if not rocm_aiter_ops.topk_gating_available():
        pytest.skip("aiter.ops.topk.topk_gating is not available")

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
