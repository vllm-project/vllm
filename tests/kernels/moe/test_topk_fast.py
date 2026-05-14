# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the moeTopKFast kernel optimization.

Run `pytest tests/kernels/moe/test_topk_fast.py`.
"""

import pytest
import torch

from vllm._custom_ops import topk_softmax
from vllm.platforms import current_platform


def torch_topk_softmax(gating_output, topk, renormalize):
    scores = torch.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def run_topk_softmax(gating_output, topk, renormalize):
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device="cuda")
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_softmax(topk_weights, topk_ids, token_expert_indices,
                 gating_output.clone(), renormalize)
    return topk_weights, topk_ids


def values_match(gating_output, ids_ref, ids_test):
    scores = torch.softmax(gating_output.float(), dim=-1)
    vals_ref = scores.gather(1, ids_ref.long())
    vals_test = scores.gather(1, ids_test.long())
    return torch.equal(vals_ref, vals_test)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA-only test.")
@pytest.mark.parametrize("num_tokens", [1, 16, 128, 512, 1024, 2048])
@pytest.mark.parametrize("num_experts", [4, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_topk_softmax(num_tokens, num_experts, topk):
    torch.manual_seed(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda")

    ref_weights, ref_ids = torch_topk_softmax(gating_output, topk, False)
    test_weights, test_ids = run_topk_softmax(gating_output, topk, False)

    torch.testing.assert_close(ref_weights, test_weights, atol=1e-3, rtol=1e-3)
    assert values_match(gating_output, ref_ids.int(), test_ids)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA-only test.")
@pytest.mark.parametrize("num_tokens", [1, 16, 128, 512, 1024, 2048])
@pytest.mark.parametrize("num_experts", [512])
@pytest.mark.parametrize("topk", [1, 2, 3, 4, 5, 8])
def test_topk_fast_large_experts(num_tokens, num_experts, topk):
    torch.manual_seed(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda")

    ref_weights, ref_ids = torch_topk_softmax(gating_output, topk, False)
    test_weights, test_ids = run_topk_softmax(gating_output, topk, False)

    torch.testing.assert_close(ref_weights, test_weights, atol=1e-3, rtol=1e-3)
    assert values_match(gating_output, ref_ids.int(), test_ids)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA-only test.")
@pytest.mark.parametrize("num_tokens", [1, 16, 128, 512, 1024, 2048])
@pytest.mark.parametrize("num_experts", [4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_topk_softmax_dtype(num_tokens, num_experts, topk, dtype):
    torch.manual_seed(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=dtype, device="cuda")

    ref_weights, ref_ids = torch_topk_softmax(gating_output, topk, False)
    test_weights, test_ids = run_topk_softmax(gating_output, topk, False)

    torch.testing.assert_close(ref_weights, test_weights, atol=1e-3, rtol=1e-3)
    assert values_match(gating_output.float(), ref_ids.int(), test_ids)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA-only test.")
@pytest.mark.parametrize("num_tokens", [1, 16, 128, 512, 1024, 2048])
@pytest.mark.parametrize("num_experts", [4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_topk_softmax_renormalize(num_tokens, num_experts, topk):
    torch.manual_seed(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda")

    weights_no, ids_no = run_topk_softmax(gating_output, topk, False)
    weights_yes, ids_yes = run_topk_softmax(gating_output, topk, True)

    expected = weights_no / weights_no.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(expected, weights_yes, atol=1e-3, rtol=1e-3)
    assert values_match(gating_output, ids_no, ids_yes)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
