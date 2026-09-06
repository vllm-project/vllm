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

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    rocm_aiter_grouped_topk,
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


def test_rocm_aiter_topk_gating_custom_op_registration():
    """Test that the custom op is correctly registered."""
    assert hasattr(torch.ops.vllm, "rocm_aiter_topk_gating")
    assert callable(torch.ops.vllm.rocm_aiter_topk_gating)


def test_rocm_aiter_topk_gating_torch_compile_compatibility():
    """Test biased sigmoid top-k through eager and compiled custom-op paths."""
    torch.manual_seed(54594)
    token = 4
    expert = 256
    topk = 8
    scale_factor = 2.827
    gating_output = torch.randn((token, expert), dtype=torch.float32, device="cuda")
    correction_bias = torch.randn((expert,), dtype=torch.float32, device="cuda")

    def allocate_outputs():
        weights = torch.empty((token, topk + 1), dtype=torch.float32, device="cuda")[
            :, :topk
        ]
        ids = torch.empty((token, topk + 1), dtype=torch.int32, device="cuda")[:, :topk]
        assert not weights.is_contiguous()
        assert not ids.is_contiguous()
        return weights, ids

    def topk_gating_fn(gating_output, correction_bias, topk_weights, topk_ids):
        return torch.ops.vllm.rocm_aiter_topk_gating(
            gating_output,
            correction_bias,
            topk_weights,
            topk_ids,
            True,
            scale_factor,
            "sigmoid",
        )

    opcheck_weights, opcheck_ids = allocate_outputs()
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_topk_gating,
        (gating_output, correction_bias, opcheck_weights, opcheck_ids),
        kwargs={
            "need_renorm": True,
            "routed_scaling_factor": scale_factor,
            "scoring_func": "sigmoid",
        },
        test_utils=("test_faketensor",),
    )

    eager_weights, eager_ids = allocate_outputs()
    compiled_weights, compiled_ids = allocate_outputs()
    topk_gating_fn(gating_output, correction_bias, eager_weights, eager_ids)
    compiled_fn = torch.compile(
        topk_gating_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False,
    )
    compiled_fn(gating_output, correction_bias, compiled_weights, compiled_ids)

    scores = gating_output.sigmoid()
    expected_ids = torch.topk(scores + correction_bias, topk, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    expected_weights *= scale_factor

    for actual_weights, actual_ids in (
        (eager_weights, eager_ids),
        (compiled_weights, compiled_ids),
    ):
        actual_ids, order = torch.sort(actual_ids)
        actual_weights = torch.gather(actual_weights, 1, order)
        expected_ids_sorted, expected_order = torch.sort(expected_ids.to(torch.int32))
        expected_weights_sorted = torch.gather(expected_weights, 1, expected_order)
        torch.testing.assert_close(actual_ids, expected_ids_sorted)
        torch.testing.assert_close(
            actual_weights, expected_weights_sorted, rtol=0.0, atol=1e-6
        )


def test_hy4_grouped_topk_dispatches_to_topk_gating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden_states = torch.empty((4, 6144), dtype=torch.bfloat16, device="cuda")
    gating_output = torch.empty((4, 256), dtype=torch.float32, device="cuda")
    correction_bias = torch.empty((256,), dtype=torch.float32, device="cuda")
    calls = 0

    def topk_gating(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1

    def fail_grouped_topk(*args, **kwargs) -> None:
        raise AssertionError("Hy4 used biased_grouped_topk")

    monkeypatch.setattr(rocm_aiter_ops, "topk_gating", topk_gating)
    monkeypatch.setattr(rocm_aiter_ops, "biased_grouped_topk", fail_grouped_topk)
    rocm_aiter_grouped_topk(
        hidden_states,
        gating_output,
        topk=8,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.827,
        e_score_correction_bias=correction_bias,
    )

    assert calls == 1


def test_non_hy4_grouped_topk_preserves_biased_grouped_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden_states = torch.empty((4, 6144), dtype=torch.bfloat16, device="cuda")
    gating_output = torch.empty((4, 128), dtype=torch.float32, device="cuda")
    correction_bias = torch.empty((128,), dtype=torch.float32, device="cuda")
    calls = 0

    def fail_topk_gating(*args, **kwargs) -> None:
        raise AssertionError("non-Hy4 shape used topk_gating")

    def biased_grouped_topk(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(rocm_aiter_ops, "topk_gating", fail_topk_gating)
    monkeypatch.setattr(rocm_aiter_ops, "biased_grouped_topk", biased_grouped_topk)
    rocm_aiter_grouped_topk(
        hidden_states,
        gating_output,
        topk=8,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.827,
        e_score_correction_bias=correction_bias,
    )

    assert calls == 1


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
